//! Phase 14 benchmark harness: scalar vs SIMD kernel timings.
//!
//! Usage: `cargo run --release --bin bench [-- N_ITER_SCALE]`
//!
//! Measures the hot kernels on realistic dimensions with synthetic but
//! structurally valid quantized data, running each once through the runtime
//! dispatch (SIMD when available) and once forced scalar, then prints the
//! speedup. `QWEN_NO_SIMD=1` globally disables the fast path.

use std::hint::black_box;
use std::time::Instant;

use qwen3_5_397b_in_rust::gguf::GGmlType;
use qwen3_5_397b_in_rust::model::kernels::{gemv, quantize_row_q8_0, rms_norm, rope_multi_imrope, swiglu, RopeConfig};
use qwen3_5_397b_in_rust::model::simd;

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
    }
}

fn le16(v: f32) -> [u8; 2] {
    qwen3_5_397b_in_rust::model::quant::f32_to_fp16(v).to_le_bytes()
}

/// One structurally valid random K-quant weight block.
fn k_block(kind: GGmlType, rng: &mut Lcg) -> Vec<u8> {
    let mut b = Vec::new();
    match kind {
        GGmlType::Q4_K => {
            b.extend_from_slice(&le16(rng.next_f32(1.0)));
            b.extend_from_slice(&le16(rng.next_f32(0.1)));
            b.extend((0..12).map(|_| rng.next_u8()));
            b.extend((0..128).map(|_| rng.next_u8()));
        }
        GGmlType::Q5_K => {
            b.extend_from_slice(&le16(rng.next_f32(1.0)));
            b.extend_from_slice(&le16(rng.next_f32(0.1)));
            b.extend((0..12).map(|_| rng.next_u8()));
            b.extend((0..32).map(|_| rng.next_u8()));
            b.extend((0..128).map(|_| rng.next_u8()));
        }
        GGmlType::Q6_K => {
            b.extend((0..128).map(|_| rng.next_u8()));
            b.extend((0..64).map(|_| rng.next_u8()));
            b.extend((0..16).map(|_| rng.next_u8()));
            b.extend_from_slice(&le16(rng.next_f32(1.0)));
        }
        _ => unreachable!(),
    }
    b
}

fn time_it<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    // Warmup
    for _ in 0..(iters / 10).max(2) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() / iters as f64 * 1e6 // µs per call
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);

    println!("== Phase 14 kernel benchmark ==");
    println!(
        "simd available: {} (avx2+fma or NEON), forced-scalar runs use force_scalar(true)\n",
        simd::use_simd()
    );

    let mut rng = Lcg(0xDEADBEEF);
    let n_in = 4096usize;
    let n_out = 4096usize;
    let x: Vec<f32> = (0..n_in).map(|_| rng.next_f32(1.0)).collect();

    struct Case {
        name: String,
        iters: usize,
        run: Box<dyn Fn()>,
    }
    let mut cases: Vec<Case> = Vec::new();

    // Quantized gemv: activation quantized inside the timed closure? No —
    // gemv re-quantizes x every call; hoist it out by timing gemv as-is
    // (that IS the real cost profile per layer).
    for (ty, label) in [
        (GGmlType::Q4_K, "q4_K"),
        (GGmlType::Q6_K, "q6_K"),
        (GGmlType::Q8_0, "q8_0"),
    ] {
        let w = match ty {
            GGmlType::Q8_0 => (0..n_out)
                .flat_map(|r| {
                    let row: Vec<f32> = (0..n_in).map(|i| ((r * 31 + i) % 97) as f32 * 0.01 - 0.5).collect();
                    quantize_row_q8_0(&row)
                })
                .collect(),
            other => {
                let mut w = Vec::new();
                for _ in 0..n_out {
                    for _ in 0..n_in / 256 {
                        w.extend(k_block(other, &mut rng));
                    }
                }
                w
            }
        };
        let x_c = x.clone();
        cases.push(Case {
            name: format!("gemv {label} {n_in}x{n_out}"),
            iters: 20 * scale,
            run: Box::new(move || {
                black_box(gemv(ty, &w, n_in, n_out, &x_c).unwrap());
            }),
        });
    }

    // F32 gemv
    {
        let wf: Vec<u8> = (0..n_out * n_in)
            .flat_map(|i| ((i % 251) as f32 * 0.01 - 1.2).to_le_bytes())
            .collect();
        let x_c = x.clone();
        cases.push(Case {
            name: format!("gemv f32  {n_in}x{n_out}"),
            iters: 20 * scale,
            run: Box::new(move || {
                black_box(gemv(GGmlType::F32, &wf, n_in, n_out, &x_c).unwrap());
            }),
        });
    }

    // rms_norm / swiglu at hidden size
    {
        let v: Vec<f32> = (0..n_in).map(|i| (i % 89) as f32 * 0.03 - 1.3).collect();
        let w: Vec<f32> = (0..n_in).map(|i| (i % 53) as f32 * 0.02 - 0.5).collect();
        let up: Vec<f32> = (0..n_in).map(|i| (i % 71) as f32 * 0.02 - 0.7).collect();
        cases.push(Case {
            name: format!("rms_norm {n_in}"),
            iters: 20000 * scale,
            run: {
                let (v, w) = (v.clone(), w.clone());
                Box::new(move || { black_box(rms_norm(&v, &w, 1e-6)); })
            },
        });
        cases.push(Case {
            name: format!("swiglu   {n_in}"),
            iters: 20000 * scale,
            run: {
                let (v, up) = (v.clone(), up.clone());
                Box::new(move || { black_box(swiglu(&v, &up)); })
            },
        });

        // RoPE on a full-attn head row (128 channels, n_rot = 64)
        let row: Vec<f32> = (0..128).map(|i| (i % 37) as f32 * 0.05 - 0.9).collect();
        let cfg = RopeConfig {
            freq_base: 1_000_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        };
        cases.push(Case {
            name: "rope imrope 128".to_string(),
            iters: 50000 * scale,
            run: Box::new(move || {
                black_box(rope_multi_imrope(
                    black_box(&row),
                    [5, 0, 0, 0],
                    64,
                    [24, 24, 24, 24],
                    4096,
                    &cfg,
                ));
            }),
        });
    }

    // Header
    println!(
        "{:<26} {:>12} {:>12} {:>9} {:>10}",
        "kernel", "scalar µs", "simd µs", "speedup", "Mrows/s"
    );
    println!("{}", "-".repeat(75));

    for case in &cases {
        simd::force_scalar(true);
        let t_scalar = time_it(case.iters, || (case.run)());
        simd::force_scalar(false);
        let t_simd = time_it(case.iters, || (case.run)());

        let speedup = t_scalar / t_simd;
        // Rough rows/s for gemv-like ops (n_out dots per call).
        let mrows = if case.name.starts_with("gemv") {
            n_out as f64 / t_simd / 1e6 * 1e6
        } else {
            f64::NAN
        };
        println!(
            "{:<26} {:>12.1} {:>12.1} {:>8.1}x {:>10.1}",
            case.name,
            t_scalar,
            t_simd,
            speedup,
            mrows
        );
    }

    simd::force_scalar(false);
}
