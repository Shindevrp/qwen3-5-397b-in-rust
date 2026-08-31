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
use qwen3_5_397b_in_rust::model::kernels::{
    RopeConfig, gemv, quantize_row_q8_0, rms_norm, rope_multi_imrope, swiglu,
};
use qwen3_5_397b_in_rust::model::simd;

struct Lcg(u64);
impl Lcg {
    fn next_u8(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
    fn next_f32(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--e2e") {
        e2e_main(&args[1..]);
    } else {
        let scale: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(1);
        kernel_main(scale);
    }
}

fn usage_e2e() {
    eprintln!(
        "Usage: bench --e2e [model.gguf] [options]\n\
         \n\
         Benchmarks the full generation pipeline end-to-end.\n\
         Without a path, a synthetic model is built in-memory.\n\
         \n\
         Options:\n\
         \x20 --preset NAME     tiny | medium (default) | large\n\
         \x20 --steps N         decode steps per timing run (default 128)\n\
         \x20 --batches LIST    comma-separated batch sizes (default 1,2,4,8)\n\
         \x20 --scale N         multiplier for iteration counts\n\
         \x20 --json            print machine-readable latency summary"
    );
}

/// End-to-end pipeline benchmark: prefill throughput, single-stream decode,
/// multi-sequence batch scaling, and cache memory footprint.
fn e2e_main(args: &[String]) {
    use qwen3_5_397b_in_rust::model::loader::ModelLoader;
    use qwen3_5_397b_in_rust::model::pipeline::{
        GenerationState, ModelWeights, generate_token, generate_token_batch, prefill, prefill_batch,
    };
    use qwen3_5_397b_in_rust::model::synth::{SynthConfig, write_temp};
    use std::time::Instant;

    if args.iter().any(|a| a == "-h" || a == "--help") {
        usage_e2e();
        return;
    }

    // ---- parse args ----
    let mut model_path: Option<String> = None;
    let mut preset = "medium".to_string();
    let mut steps = 128usize;
    let mut batches: Vec<usize> = vec![1, 2, 4, 8];
    let mut scale = 1usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--preset" => {
                i += 1;
                preset = args.get(i).cloned().unwrap_or_else(|| "medium".into());
            }
            "--steps" => {
                i += 1;
                steps = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(128);
            }
            "--batches" => {
                i += 1;
                batches = args
                    .get(i)
                    .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                    .unwrap_or(vec![1, 2, 4, 8]);
            }
            "--scale" => {
                i += 1;
                scale = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "--json" => {
                // Flag consumed directly in the latency section below.
            }
            p if !p.starts_with("--") => model_path = Some(p.to_string()),
            other => {
                eprintln!("unknown option {other}");
                usage_e2e();
                std::process::exit(2);
            }
        }
        i += 1;
    }
    let steps = (steps * scale).max(32);

    println!("== Phase 17 end-to-end benchmark ==");
    println!(
        "simd available: {}, threads: {}\n",
        simd::use_simd(),
        rayon::current_num_threads()
    );

    // ---- load model ----
    let _keepalive;
    let (cfg_summary, weight_bytes): (String, u64);
    let loader_tmp;
    let model = match &model_path {
        Some(path) => {
            loader_tmp = ModelLoader::open(path).expect("open gguf");
            let m = ModelWeights::load(&loader_tmp).expect("load weights");
            weight_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            _keepalive = None;
            cfg_summary = String::new();
            m
        }
        None => {
            let cfg =
                SynthConfig::preset(&preset).unwrap_or_else(|| panic!("unknown preset {preset}"));
            let t0 = Instant::now();
            let tmp = write_temp(&cfg).expect("build synthetic gguf");
            println!(
                "built synthetic '{preset}' model in {:.1}s",
                t0.elapsed().as_secs_f64()
            );
            loader_tmp = ModelLoader::open(tmp.path()).expect("open gguf");
            let m = ModelWeights::load(&loader_tmp).expect("load weights");
            weight_bytes = tmp.as_file().metadata().map(|md| md.len()).unwrap_or(0);
            _keepalive = Some(tmp);
            cfg_summary = format!(" ({preset})");
            m
        }
    };

    let c = &model.cfg;
    let n_ctx = c.context_length as usize;
    let n_embd = c.embedding_length as usize;
    let n_vocab = model.output_weight.n_elements / n_embd;
    let n_attn_layers = (0..c.block_count as usize)
        .filter(|&l| (l + 1) % c.full_attention_interval as usize == 0)
        .count();

    println!(
        "model{}: {} layers ({} full-attn @ every {}), embd {}, heads {}+{}kv (head {}), vocab {}, ctx {}, experts {}",
        cfg_summary,
        c.block_count,
        n_attn_layers,
        c.full_attention_interval,
        n_embd,
        c.attention_head_count,
        c.attention_head_count_kv,
        c.attention_key_length,
        n_vocab,
        n_ctx,
        c.expert_count,
    );
    println!(
        "weights: {:.1} MiB   kv-cache: {} B/token/seq (f32 K+V)\n",
        weight_bytes as f64 / (1024.0 * 1024.0),
        2 * c.attention_head_count_kv as usize
            * c.attention_key_length as usize
            * 4
            * n_attn_layers
    );

    // ---- prefill throughput ----
    println!("prefill (single sequence):");
    println!("{:>10} {:>10} {:>12}", "tokens", "ms", "tok/s");
    for len in [64usize, 256, 1024] {
        let len = len.min(n_ctx / 2);
        if len < 8 {
            continue;
        }
        let prompt: Vec<u32> = (0..len).map(|t| (t % n_vocab) as u32).collect();
        // warmup on a short prefix to fault in caches/mmaps
        {
            let mut w = GenerationState::new(&model);
            prefill(&mut w, &prompt[..len.min(16)], &model).expect("warmup prefill");
        }
        let mut state = GenerationState::new(&model);
        let t0 = Instant::now();
        prefill(&mut state, &prompt, &model).expect("prefill");
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        println!("{len:>10} {ms:>10.1} {:>12.0}", len as f64 / ms * 1e3);
    }

    // ---- single-stream decode ----
    println!("\ndecode (single stream, argmax):");
    let prompt_len = 32.min(n_ctx / 4);
    let prompt: Vec<u32> = (0..prompt_len).map(|t| (t % n_vocab) as u32).collect();
    let mut state = GenerationState::new(&model);
    prefill(&mut state, &prompt, &model).expect("prefill");
    let ttft_ms = state.last_timing.total_us as f64 / 1e3;
    let mut last = *prompt.last().unwrap();

    // warmup
    for _ in 0..16 {
        let (_h, next) = generate_token(&mut state, last, &model).expect("generate");
        last = next;
    }
    let t0 = Instant::now();
    let mut stage_sums = [0u64; 5]; // embed, delta_net, full_attn, lm_head, total
    for _ in 0..steps {
        let (_h, next) = generate_token(&mut state, last, &model).expect("generate");
        last = next;
        let t = &state.last_timing;
        stage_sums[0] += t.embed_us;
        stage_sums[1] += t.delta_net_us;
        stage_sums[2] += t.full_attn_us;
        stage_sums[3] += t.lm_head_us;
        stage_sums[4] += t.total_us;
    }
    let decode_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!(
        "{steps} steps in {decode_ms:.0} ms -> {:.1} tok/s",
        steps as f64 / decode_ms * 1e3
    );

    // ---- Phase 29: latency breakdown + JSON summary ----
    let itl_us = decode_ms * 1e3 / steps.max(1) as f64;
    println!("\nlatency breakdown (avg per decode step):");
    println!("{:>12} {:>10}", "stage", "us");
    for (name, us) in [
        ("embed", stage_sums[0]),
        ("delta_net", stage_sums[1]),
        ("full_attn", stage_sums[2]),
        ("lm_head", stage_sums[3]),
        ("total", stage_sums[4]),
    ] {
        println!("{name:>12} {:>10.1}", us as f64 / steps.max(1) as f64);
    }
    println!("TTFT (prefill of {prompt_len}): {ttft_ms:.1} ms | ITL: {itl_us:.1} us");

    if args.iter().any(|a| a == "--json") {
        println!(
            "{{\n  \"ttft_ms\": {ttft_ms:.2},\n  \"itl_us\": {itl_us:.2},\n  \"decode_tokps\": {:.2},\n  \"steps\": {steps},\n  \"stages_us\": {{\"embed\": {}, \"delta_net\": {}, \"full_attn\": {}, \"lm_head\": {}, \"total\": {}}}\n}}",
            steps as f64 / decode_ms * 1e3,
            stage_sums[0] / steps.max(1) as u64,
            stage_sums[1] / steps.max(1) as u64,
            stage_sums[2] / steps.max(1) as u64,
            stage_sums[3] / steps.max(1) as u64,
            stage_sums[4] / steps.max(1) as u64,
        );
    }

    // ---- batch scaling ----
    println!("\nbatch decode (lockstep, aggregate throughput):");
    println!(
        "{:>7} {:>10} {:>12} {:>9} {:>9}",
        "batch", "total tok", "agg tok/s", "speedup", "eff"
    );
    let b_prompt_len = 16.min(n_ctx / 4).max(1);
    let mut base_tps = 0.0f64;
    for &b in &batches {
        assert!(b_prompt_len + steps <= n_ctx, "batch run exceeds ctx");
        let prompts: Vec<Vec<u32>> = (0..b)
            .map(|s| {
                (0..b_prompt_len)
                    .map(|t| ((s * 31 + t) % n_vocab) as u32)
                    .collect()
            })
            .collect();
        let mut states: Vec<GenerationState> =
            (0..b).map(|_| GenerationState::new(&model)).collect();
        let refs: Vec<&[u32]> = prompts.iter().map(|v| v.as_slice()).collect();
        prefill_batch(&mut states, &refs, &model).expect("batch prefill");
        let mut lasts: Vec<u32> = prompts.iter().map(|p| *p.last().unwrap()).collect();

        // warmup
        let w_last = generate_token_batch(&mut states, &lasts, &model).expect("warmup");
        lasts = w_last;

        let total_tokens = b * steps;
        let t0 = Instant::now();
        for _ in 0..steps {
            lasts = generate_token_batch(&mut states, &lasts, &model).expect("batch step");
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let agg_tps = total_tokens as f64 / ms * 1e3;
        if b == batches[0] || base_tps == 0.0 {
            base_tps = agg_tps;
        }
        let speedup = agg_tps / base_tps;
        let eff = speedup / b as f64;
        println!(
            "{b:>7} {total_tokens:>10} {agg_tps:>12.0} {:>8.2}x {:>8.1}",
            speedup, eff
        );
    }

    // ---- memory footprint ----
    fn state_bytes(state: &GenerationState) -> usize {
        let kv: usize = state.kv_caches.iter().map(|c| c.allocated_bytes()).sum();
        let conv: usize = state.conv_states.iter().map(|v| v.len() * 4).sum();
        let ssm: usize = state.ssm_states.iter().map(|v| v.len() * 4).sum();
        kv + conv + ssm
    }

    println!("\nper-sequence state footprint:");
    let empty = GenerationState::new(&model);
    println!(
        "  fresh:      {:>8.2} MiB (kv capacity {} tokens/layer)",
        state_bytes(&empty) as f64 / (1024.0 * 1024.0),
        empty
            .kv_caches
            .first()
            .map(|c| c.capacity_tokens())
            .unwrap_or(0)
    );
    let filled_prompt: Vec<u32> = (0..1024.min(n_ctx / 2))
        .map(|t| (t % n_vocab) as u32)
        .collect();
    let mut filled = GenerationState::new(&model);
    prefill(&mut filled, &filled_prompt, &model).expect("prefill");
    println!(
        "  after {} tokens: {:>8.2} MiB (pos {})",
        filled.pos,
        state_bytes(&filled) as f64 / (1024.0 * 1024.0),
        filled.pos
    );
    println!(
        "  projected @ full ctx {}: ~{:.1} MiB/sequence (kv only)",
        n_ctx,
        (2 * c.attention_head_count_kv as usize
            * c.attention_key_length as usize
            * 4
            * n_attn_layers
            * n_ctx) as f64
            / (1024.0 * 1024.0)
    );
}

/// Phase 14 kernel micro-benchmarks: scalar vs SIMD on synthetic K-blocks.
fn kernel_main(scale: usize) {
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
                    let row: Vec<f32> = (0..n_in)
                        .map(|i| ((r * 31 + i) % 97) as f32 * 0.01 - 0.5)
                        .collect();
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
                Box::new(move || {
                    black_box(rms_norm(&v, &w, 1e-6));
                })
            },
        });
        cases.push(Case {
            name: format!("swiglu   {n_in}"),
            iters: 20000 * scale,
            run: {
                let (v, up) = (v.clone(), up.clone());
                Box::new(move || {
                    black_box(swiglu(&v, &up));
                })
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
            case.name, t_scalar, t_simd, speedup, mrows
        );
    }

    simd::force_scalar(false);
}
