//! SIMD-accelerated math kernels (Phase 14).
//!
//! Every function in the arch modules is `unsafe` because it requires AVX2+FMA
//! (x86) or is NEON-only math (aarch64); callers must gate on
//! [`use_simd()`]. Intrinsic bodies therefore run inside `unsafe fn`s without
//! per-statement blocks.
#![allow(unsafe_op_in_unsafe_fn)]
//!
//! AVX2+FMA paths for x86-64 and NEON paths for aarch64, selected at runtime.
//! The public entry points in [`crate::model::kernels`] dispatch here when the
//! CPU supports the required features and fall back to their original scalar
//! bodies otherwise (or when `QWEN_NO_SIMD` is set / [`force_scalar`] is used,
//! which the benchmark harness flips to measure both paths).
//!
//! Numerical contract vs the scalar ports:
//!
//! * Quantized integer dot products are **exact** — i8×i8 products are summed
//!   in i16→i32 with the same associativity-free integer arithmetic, so only
//!   the final per-block float scaling order differs from the scalar code.
//! * `rms_norm` keeps the f64 tail semantics; the bulk sum-of-squares moves to
//!   f32 lanes (deviation ~1e-7 relative, far inside engine tolerances).
//! * `swiglu` uses a degree-5 polynomial vectorized expf (max rel err ≈ 2e-7
//!   on the sigmoid's argument range). Extreme arguments clamp to ±126/127
//!   exponents, so `silu(g)` for |g| > ~80 decays to ±g·2⁻¹²⁶ instead of the
//!   true ±g·e⁻|g| — an absolute error below 1e-30 that is invisible after
//!   summation with the rest of the FFN output.
//! * RoPE rotation is restructured as two half-vector FMAs; identical math.

/// Runtime override used by the benchmark harness (`--scalar` mode).
static FORCE_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn force_scalar(v: bool) {
    FORCE_SCALAR.store(v, std::sync::atomic::Ordering::Relaxed);
}

fn env_disabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| {
        std::env::var("QWEN_NO_SIMD").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// True when the accelerated path may be taken on this machine.
#[inline]
pub fn use_simd() -> bool {
    if FORCE_SCALAR.load(std::sync::atomic::Ordering::Relaxed) || env_disabled() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        // FMA is required alongside AVX2 by every kernel below.
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true // NEON is mandatory on aarch64
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// Safe dispatch helpers (usable without cfg gymnastics at call sites)
// ---------------------------------------------------------------------------

/// Dispatched f32 dot product with a built-in scalar fallback.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if use_simd() {
        return unsafe { x86::dot_f32(a, b) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_simd() {
        return unsafe { arm::dot_f32(a, b) };
    }
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

/// Dispatched RoPE half-rotation with a built-in scalar fallback.
pub fn rotate_halves(x: &[f32], cos: &[f32], sin: &[f32], out: &mut [f32], n_offset: usize) {
    #[cfg(target_arch = "x86_64")]
    if use_simd() {
        return unsafe { x86::rotate_halves(x, cos, sin, out, n_offset) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_simd() {
        return unsafe { arm::rotate_halves(x, cos, sin, out, n_offset) };
    }
    for i in 0..n_offset {
        let (a, b) = (x[i], x[n_offset + i]);
        out[i] = a * cos[i] - b * sin[i];
        out[n_offset + i] = a * sin[i] + b * cos[i];
    }
}

// ---------------------------------------------------------------------------
// x86-64 — AVX2 + FMA
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use std::arch::x86_64::*;

    use crate::model::quant::QK_K;
    use crate::model::quant::fp16_to_f32;

    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn hsum_ps(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps::<1>(v);
        let lo = _mm256_castps256_ps128(v);
        let s4 = _mm_add_ps(hi, lo);
        let s64 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s32 = _mm_add_ss(s64, _mm_shuffle_ps::<0x55>(s64, s64));
        _mm_cvtss_f32(s32)
    }

    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn hsum_epi32(v: __m256i) -> i32 {
        let hi = _mm256_extracti128_si256::<1>(v);
        let lo = _mm256_castsi256_si128(v);
        let s4 = _mm_add_epi32(hi, lo);
        let s64 = _mm_add_epi32(s4, _mm_unpackhi_epi64(s4, s4));
        let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32::<0x55>(s64));
        _mm_cvtsi128_si32(s32)
    }

    /// f32 dot product, 8-wide FMA with scalar tail.
    ///
    /// # Safety
    /// Requires AVX2+FMA (checked by the caller via [`super::use_simd`]).
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let n8 = n & !7;
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut acc = _mm256_setzero_ps();
        for i in (0..n8).step_by(8) {
            acc = _mm256_fmadd_ps(
                _mm256_loadu_ps(ap.add(i)),
                _mm256_loadu_ps(bp.add(i)),
                acc,
            );
        }
        let mut sum = hsum_ps(acc);
        for i in n8..n {
            sum += *ap.add(i) * *bp.add(i);
        }
        sum
    }

    /// `ggml_vec_dot_q8_0_q8_0`: exact i32 core, per-block fp16 scale via FMA.
    ///
    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(32));
        let nb = n / 32;
        let mut acc = _mm256_setzero_ps();
        for i in 0..nb {
            let xb = x.as_ptr().add(i * 34);
            let yb = y.as_ptr().add(i * 34);
            let dx = fp16_to_f32(u16::from_le_bytes([*xb, *xb.add(1)]));
            let dy = fp16_to_f32(u16::from_le_bytes([*yb, *yb.add(1)]));
            let xl = _mm_loadu_si128(xb.add(2) as *const __m128i);
            let xh = _mm_loadu_si128(xb.add(18) as *const __m128i);
            let yl = _mm_loadu_si128(yb.add(2) as *const __m128i);
            let yh = _mm_loadu_si128(yb.add(18) as *const __m128i);
            let pl = _mm256_madd_epi16(_mm256_cvtepi8_epi16(xl), _mm256_cvtepi8_epi16(yl));
            let ph = _mm256_madd_epi16(_mm256_cvtepi8_epi16(xh), _mm256_cvtepi8_epi16(yh));
            let isum = _mm256_add_epi32(pl, ph);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(isum), _mm256_set1_ps(dx * dy), acc);
        }
        hsum_ps(acc)
    }

    /// Exact integer dot of one 256-element K-quant block: 16 groups of 16,
    /// each scaled by `sc[g]`. All arithmetic in i32 — no overflow (each lane
    /// accumulates ≤ 16 × 32258 × 127 < 2^27).
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn block_isums(a: &[i8], q8: &[u8], sc: &[i8]) -> __m256i {
        let mut acc = _mm256_setzero_si256();
        for (g, &scale) in sc.iter().enumerate().take(16) {
            let av = _mm_loadu_si128(a.as_ptr().add(g * 16) as *const __m128i);
            let qv = _mm_loadu_si128(q8.as_ptr().add(g * 16) as *const __m128i);
            let prod = _mm256_madd_epi16(_mm256_cvtepi8_epi16(av), _mm256_cvtepi8_epi16(qv));
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(prod, _mm256_set1_epi32(scale as i32)));
        }
        acc
    }

    /// Decode the 6-bit scales/mins shared by Q4_K and Q5_K blocks.
    fn k_scales(xb: &[u8]) -> ([i8; 8], [i8; 8]) {
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes(xb[4..8].try_into().unwrap());
        utmp[1] = u32::from_le_bytes(xb[8..12].try_into().unwrap());
        utmp[2] = u32::from_le_bytes(xb[12..16].try_into().unwrap());
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f_0f0f) | (((utmp[1] >> 6) & 0x0303_0303) << 4);
        let uaux = utmp[1] & 0x3f3f_3f3f;
        utmp[1] = (utmp[2] & 0x0f0f_0f0f) | (((utmp[0] >> 6) & 0x0303_0303) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f_3f3f;
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        scales[0..4].copy_from_slice(&utmp[0].to_le_bytes());
        scales[4..8].copy_from_slice(&utmp[1].to_le_bytes());
        mins[0..4].copy_from_slice(&utmp[2].to_le_bytes());
        mins[4..8].copy_from_slice(&utmp[3].to_le_bytes());
        (
            scales.map(|v| v as i8),
            mins.map(|v| v as i8),
        )
    }

    fn unpack_q4_k(xb: &[u8], a: &mut [i8; QK_K]) {
        for j in 0..(QK_K / 64) {
            let q4 = &xb[16 + j * 32..16 + (j + 1) * 32];
            for l in 0..32 {
                a[j * 64 + l] = (q4[l] & 0x0F) as i8;
                a[j * 64 + 32 + l] = (q4[l] >> 4) as i8;
            }
        }
    }

    fn unpack_q5_k(xb: &[u8], a: &mut [i8; QK_K]) {
        let q4 = &xb[48..176];
        let hm = &xb[16..48];
        let mut m = 1u8;
        for j in 0..(QK_K / 64) {
            let q = &q4[j * 32..(j + 1) * 32];
            for l in 0..32 {
                a[j * 64 + l] = (q[l] & 0x0F) as i8 + if hm[l] & m != 0 { 16 } else { 0 };
                let m2 = m << 1;
                a[j * 64 + 32 + l] = (q[l] >> 4) as i8 + if hm[l] & m2 != 0 { 16 } else { 0 };
            }
            m <<= 2;
        }
    }

    fn unpack_q6_k(xb: &[u8], a: &mut [i8; QK_K]) {
        let q4 = &xb[0..128];
        let qh = &xb[128..192];
        for j in 0..(QK_K / 128) {
            let q = &q4[j * 64..(j + 1) * 64];
            let h = &qh[j * 32..(j + 1) * 32];
            for l in 0..32 {
                a[j * 128 + l] = (((q[l] & 0x0F) | ((h[l] & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 32 + l] = (((q[l + 32] & 0x0F) | (((h[l] >> 2) & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 64 + l] = (((q[l] >> 4) | (((h[l] >> 4) & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 96 + l] = (((q[l + 32] >> 4) | (((h[l] >> 6) & 0x03) << 4)) as i8) - 32;
            }
        }
    }

    /// Per-block min-correction Σ bsums[j]·mins[j/2] (Q4_K/Q5_K).
    fn min_correction(yb: &[u8], mins: &[i8; 8]) -> i32 {
        let mut sumi = 0i32;
        for j in 0..(QK_K / 16) {
            let bsums = i16::from_le_bytes([yb[258 + 2 * j], yb[259 + 2 * j]]);
            sumi += bsums as i32 * mins[j / 2] as i32;
        }
        sumi
    }

    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q4_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut accf = _mm256_setzero_ps();
        let mut min_sum = 0.0f32;
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 144..(i + 1) * 144];
            let yb = &y[i * 290..(i + 1) * 290];
            let dy = fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * dy;
            let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * dy;
            let (scales, mins) = k_scales(xb);
            unpack_q4_k(xb, &mut a);
            let mut sc = [0i8; 16];
            for g in 0..16 {
                sc[g] = scales[g / 2];
            }
            accf = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_isums(&a, &yb[2..258], &sc)),
                _mm256_set1_ps(d),
                accf,
            );
            min_sum -= dmin * min_correction(yb, &mins) as f32;
        }
        hsum_ps(accf) + min_sum
    }

    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q5_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut accf = _mm256_setzero_ps();
        let mut min_sum = 0.0f32;
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 176..(i + 1) * 176];
            let yb = &y[i * 290..(i + 1) * 290];
            let dy = fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * dy;
            let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * dy;
            let (scales, mins) = k_scales(xb);
            unpack_q5_k(xb, &mut a);
            let mut sc = [0i8; 16];
            for g in 0..16 {
                sc[g] = scales[g / 2];
            }
            accf = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_isums(&a, &yb[2..258], &sc)),
                _mm256_set1_ps(d),
                accf,
            );
            min_sum -= dmin * min_correction(yb, &mins) as f32;
        }
        hsum_ps(accf) + min_sum
    }

    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn vec_dot_q6_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut accf = _mm256_setzero_ps();
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 210..(i + 1) * 210];
            let yb = &y[i * 290..(i + 1) * 290];
            let d = fp16_to_f32(u16::from_le_bytes([xb[208], xb[209]]))
                * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            unpack_q6_k(xb, &mut a);
            let sc: [i8; 16] = xb[192..208]
                .iter()
                .map(|&v| v as i8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            accf = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(block_isums(&a, &yb[2..258], &sc)),
                _mm256_set1_ps(d),
                accf,
            );
        }
        hsum_ps(accf)
    }

    /// Vectorized expf: z = x·log2e split into integer exponent n (via the
    /// 1.5·2²³ magic-number rounding trick) and fraction t = (z−n)·ln2 ∈
    /// [−ln2/2, ln2/2], evaluated with a degree-5 Taylor polynomial, then
    /// scaled by 2^n through exponent-bit construction. n clamps to
    /// [−126, 127].
    ///
    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn v_expf(x: __m256) -> __m256 {
        const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
        let z = _mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E));
        let magic = _mm256_set1_ps(MAGIC);
        let nf = _mm256_sub_ps(_mm256_add_ps(magic, z), magic); // round-to-nearest int
        let t = _mm256_mul_ps(_mm256_sub_ps(z, nf), _mm256_set1_ps(std::f32::consts::LN_2));

        // e^t ≈ 1 + t + t²/2 + t³/6 + t⁴/24 + t⁵/120 (Horner)
        let mut p = _mm256_fmadd_ps(_mm256_set1_ps(1.0 / 120.0), t, _mm256_set1_ps(1.0 / 24.0));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0 / 6.0));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(0.5));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0));
        p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(1.0));

        let ni = _mm256_max_epi32(
            _mm256_min_epi32(_mm256_cvtps_epi32(nf), _mm256_set1_epi32(127)),
            _mm256_set1_epi32(-126),
        );
        let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
            _mm256_add_epi32(ni, _mm256_set1_epi32(127)),
            23,
        ));
        _mm256_mul_ps(p, scale)
    }

    /// SwiGLU: `out[i] = silu(gate[i]) · up[i]`.
    ///
    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        assert_eq!(gate.len(), up.len());
        let n = gate.len();
        let n8 = n & !7;
        let (gp, upp) = (gate.as_ptr(), up.as_ptr());
        let mut out = vec![0f32; n];
        let op = out.as_mut_ptr();
        let one = _mm256_set1_ps(1.0);
        let sign_flip = _mm256_set1_ps(-0.0);
        for i in (0..n8).step_by(8) {
            let g = _mm256_loadu_ps(gp.add(i));
            let e = v_expf(_mm256_xor_ps(g, sign_flip)); // e^(-g)
            let s = _mm256_div_ps(one, _mm256_add_ps(one, e));
            _mm256_storeu_ps(
                op.add(i),
                _mm256_mul_ps(_mm256_mul_ps(g, s), _mm256_loadu_ps(upp.add(i))),
            );
        }
        for i in n8..n {
            let g = *gp.add(i);
            let s = 1.0 / (1.0 + (-g).exp());
            *op.add(i) = g * s * *upp.add(i);
        }
        out
    }

    /// RMS norm: f32-lane sum-of-squares (f64 tail), then broadcast-scale.
    ///
    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        assert_eq!(x.len(), w.len());
        let n = x.len();
        let n8 = n & !7;
        let xp = x.as_ptr();
        let mut acc = _mm256_setzero_ps();
        for i in (0..n8).step_by(8) {
            let v = _mm256_loadu_ps(xp.add(i));
            acc = _mm256_fmadd_ps(v, v, acc);
        }
        let mut sum = hsum_ps(acc) as f64;
        for i in n8..n {
            sum += (*xp.add(i) * *xp.add(i)) as f64;
        }
        let mean = (sum / n as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();

        let wp = w.as_ptr();
        let mut out = vec![0f32; n];
        let op = out.as_mut_ptr();
        let vs = _mm256_set1_ps(scale);
        for i in (0..n8).step_by(8) {
            let r = _mm256_mul_ps(
                _mm256_mul_ps(_mm256_loadu_ps(xp.add(i)), vs),
                _mm256_loadu_ps(wp.add(i)),
            );
            _mm256_storeu_ps(op.add(i), r);
        }
        for i in n8..n {
            *op.add(i) = *xp.add(i) * scale * *wp.add(i);
        }
        out
    }

    /// NEOX-style half-split rotation over the first `2·n_offset` channels:
    /// `out[i] = x[i]·cos[i] − x[n_offset+i]·sin[i]`,
    /// `out[n_offset+i] = x[i]·sin[i] + x[n_offset+i]·cos[i]`.
    /// Other channels must already be copied into `out` by the caller.
    ///
    /// # Safety
    /// Requires AVX2+FMA.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn rotate_halves(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        out: &mut [f32],
        n_offset: usize,
    ) {
        let n = n_offset;
        let (xp, op) = (x.as_ptr(), out.as_mut_ptr());
        let n8 = n & !7;
        for i in (0..n8).step_by(8) {
            let va = _mm256_loadu_ps(xp.add(i));
            let vb = _mm256_loadu_ps(xp.add(n + i));
            let vc = _mm256_loadu_ps(cos.as_ptr().add(i));
            let vs = _mm256_loadu_ps(sin.as_ptr().add(i));
            _mm256_storeu_ps(op.add(i), _mm256_fnmadd_ps(vb, vs, _mm256_mul_ps(va, vc)));
            _mm256_storeu_ps(op.add(n + i), _mm256_fmadd_ps(va, vs, _mm256_mul_ps(vb, vc)));
        }
        for i in n8..n {
            let a = *xp.add(i);
            let b = *xp.add(n + i);
            *op.add(i) = a * cos[i] - b * sin[i];
            *op.add(n + i) = a * sin[i] + b * cos[i];
        }
    }
}

// ---------------------------------------------------------------------------
// aarch64 — NEON
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
pub mod arm {
    use std::arch::aarch64::*;

    use crate::model::quant::QK_K;
    use crate::model::quant::fp16_to_f32;

    #[inline(always)]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        let n4 = n & !3;
        let (ap, bp) = (a.as_ptr(), b.as_ptr());
        let mut acc = vdupq_n_f32(0.0);
        for i in (0..n4).step_by(4) {
            acc = vfmaq_f32(acc, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
        }
        let mut sum = vaddvq_f32(acc);
        for i in n4..n {
            sum += *ap.add(i) * *bp.add(i);
        }
        sum
    }

    #[inline(always)]
    pub unsafe fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(32));
        let nb = n / 32;
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..nb {
            let xb = x.as_ptr().add(i * 34);
            let yb = y.as_ptr().add(i * 34);
            let dx = fp16_to_f32(u16::from_le_bytes([*xb, *xb.add(1)]));
            let dy = fp16_to_f32(u16::from_le_bytes([*yb, *yb.add(1)]));
            let xv = vld1q_s8(xb.add(2) as *const i8);
            let yv = vld1q_s8(yb.add(2) as *const i8);
            let pl = vmull_s8(vget_low_s8(xv), vget_low_s8(yv));
            let ph = vmull_high_s8(xv, yv);
            let mut isum = vdupq_n_s32(0);
            isum = vpadalq_s16(isum, pl);
            isum = vpadalq_s16(isum, ph);
            acc = vfmaq_f32(acc, vcvtq_f32_s32(isum), vdupq_n_f32(dx * dy));
        }
        vaddvq_f32(acc)
    }

    /// Exact integer dot of one 256-element K-quant block (see x86 twin).
    #[inline(always)]
    pub unsafe fn block_isums(a: &[i8], q8: &[u8], sc: &[i8]) -> int32x4_t {
        let mut acc = vdupq_n_s32(0);
        for (g, &scale) in sc.iter().enumerate().take(16) {
            let av = vld1q_s8(a.as_ptr().add(g * 16));
            let qv = vld1q_s8(q8.as_ptr().add(g * 16));
            let pl = vmull_s8(vget_low_s8(av), vget_low_s8(qv));
            let ph = vmull_high_s8(av, qv);
            let mut t = vdupq_n_s32(0);
            t = vpadalq_s16(t, pl);
            t = vpadalq_s16(t, ph);
            acc = vaddq_s32(acc, vmulq_n_s32(t, scale as i32));
        }
        acc
    }

    fn k_scales(xb: &[u8]) -> ([i8; 8], [i8; 8]) {
        let mut utmp = [0u32; 4];
        utmp[0] = u32::from_le_bytes(xb[4..8].try_into().unwrap());
        utmp[1] = u32::from_le_bytes(xb[8..12].try_into().unwrap());
        utmp[2] = u32::from_le_bytes(xb[12..16].try_into().unwrap());
        utmp[3] = ((utmp[2] >> 4) & 0x0f0f_0f0f) | (((utmp[1] >> 6) & 0x0303_0303) << 4);
        let uaux = utmp[1] & 0x3f3f_3f3f;
        utmp[1] = (utmp[2] & 0x0f0f_0f0f) | (((utmp[0] >> 6) & 0x0303_0303) << 4);
        utmp[2] = uaux;
        utmp[0] &= 0x3f3f_3f3f;
        let mut scales = [0u8; 8];
        let mut mins = [0u8; 8];
        scales[0..4].copy_from_slice(&utmp[0].to_le_bytes());
        scales[4..8].copy_from_slice(&utmp[1].to_le_bytes());
        mins[0..4].copy_from_slice(&utmp[2].to_le_bytes());
        mins[4..8].copy_from_slice(&utmp[3].to_le_bytes());
        (scales.map(|v| v as i8), mins.map(|v| v as i8))
    }

    fn unpack_q4_k(xb: &[u8], a: &mut [i8; QK_K]) {
        for j in 0..(QK_K / 64) {
            let q4 = &xb[16 + j * 32..16 + (j + 1) * 32];
            for l in 0..32 {
                a[j * 64 + l] = (q4[l] & 0x0F) as i8;
                a[j * 64 + 32 + l] = (q4[l] >> 4) as i8;
            }
        }
    }

    fn unpack_q5_k(xb: &[u8], a: &mut [i8; QK_K]) {
        let q4 = &xb[48..176];
        let hm = &xb[16..48];
        let mut m = 1u8;
        for j in 0..(QK_K / 64) {
            let q = &q4[j * 32..(j + 1) * 32];
            for l in 0..32 {
                a[j * 64 + l] = (q[l] & 0x0F) as i8 + if hm[l] & m != 0 { 16 } else { 0 };
                let m2 = m << 1;
                a[j * 64 + 32 + l] = (q[l] >> 4) as i8 + if hm[l] & m2 != 0 { 16 } else { 0 };
            }
            m <<= 2;
        }
    }

    fn unpack_q6_k(xb: &[u8], a: &mut [i8; QK_K]) {
        let q4 = &xb[0..128];
        let qh = &xb[128..192];
        for j in 0..(QK_K / 128) {
            let q = &q4[j * 64..(j + 1) * 64];
            let h = &qh[j * 32..(j + 1) * 32];
            for l in 0..32 {
                a[j * 128 + l] = (((q[l] & 0x0F) | ((h[l] & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 32 + l] = (((q[l + 32] & 0x0F) | (((h[l] >> 2) & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 64 + l] = (((q[l] >> 4) | (((h[l] >> 4) & 0x03) << 4)) as i8) - 32;
                a[j * 128 + 96 + l] = (((q[l + 32] >> 4) | (((h[l] >> 6) & 0x03) << 4)) as i8) - 32;
            }
        }
    }

    fn min_correction(yb: &[u8], mins: &[i8; 8]) -> i32 {
        let mut sumi = 0i32;
        for j in 0..(QK_K / 16) {
            let bsums = i16::from_le_bytes([yb[258 + 2 * j], yb[259 + 2 * j]]);
            sumi += bsums as i32 * mins[j / 2] as i32;
        }
        sumi
    }

    #[inline(always)]
    pub unsafe fn vec_dot_q4_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut acc = vdupq_n_f32(0.0);
        let mut min_sum = 0.0f32;
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 144..(i + 1) * 144];
            let yb = &y[i * 290..(i + 1) * 290];
            let dy = fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * dy;
            let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * dy;
            let (scales, mins) = k_scales(xb);
            unpack_q4_k(xb, &mut a);
            let mut sc = [0i8; 16];
            for g in 0..16 {
                sc[g] = scales[g / 2];
            }
            acc = vfmaq_f32(
                acc,
                vcvtq_f32_s32(block_isums(&a, &yb[2..258], &sc)),
                vdupq_n_f32(d),
            );
            min_sum -= dmin * min_correction(yb, &mins) as f32;
        }
        vaddvq_f32(acc) + min_sum
    }

    #[inline(always)]
    pub unsafe fn vec_dot_q5_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut acc = vdupq_n_f32(0.0);
        let mut min_sum = 0.0f32;
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 176..(i + 1) * 176];
            let yb = &y[i * 290..(i + 1) * 290];
            let dy = fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * dy;
            let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * dy;
            let (scales, mins) = k_scales(xb);
            unpack_q5_k(xb, &mut a);
            let mut sc = [0i8; 16];
            for g in 0..16 {
                sc[g] = scales[g / 2];
            }
            acc = vfmaq_f32(
                acc,
                vcvtq_f32_s32(block_isums(&a, &yb[2..258], &sc)),
                vdupq_n_f32(d),
            );
            min_sum -= dmin * min_correction(yb, &mins) as f32;
        }
        vaddvq_f32(acc) + min_sum
    }

    #[inline(always)]
    pub unsafe fn vec_dot_q6_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
        assert!(n.is_multiple_of(QK_K));
        let nb = n / QK_K;
        let mut acc = vdupq_n_f32(0.0);
        let mut a = [0i8; QK_K];
        for i in 0..nb {
            let xb = &x[i * 210..(i + 1) * 210];
            let yb = &y[i * 290..(i + 1) * 290];
            let d = fp16_to_f32(u16::from_le_bytes([xb[208], xb[209]]))
                * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
            unpack_q6_k(xb, &mut a);
            let sc: [i8; 16] = xb[192..208]
                .iter()
                .map(|&v| v as i8)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();
            acc = vfmaq_f32(
                acc,
                vcvtq_f32_s32(block_isums(&a, &yb[2..258], &sc)),
                vdupq_n_f32(d),
            );
        }
        vaddvq_f32(acc)
    }

    /// Vectorized expf (magic-number rounding; see x86 twin for details).
    #[inline(always)]
    pub unsafe fn v_expf(x: float32x4_t) -> float32x4_t {
        const MAGIC: f32 = 12582912.0; // 1.5 * 2^23
        let z = vmulq_n_f32(x, std::f32::consts::LOG2_E);
        let magic = vdupq_n_f32(MAGIC);
        let nf = vsubq_f32(vaddq_f32(magic, z), magic);
        let t = vmulq_n_f32(vsubq_f32(z, nf), std::f32::consts::LN_2);

        let mut p = vfmaq_n_f32(vdupq_n_f32(1.0 / 24.0), t, 1.0 / 120.0);
        p = vfmaq_n_f32(vdupq_n_f32(1.0 / 6.0), t, p);
        p = vfmaq_n_f32(vdupq_n_f32(0.5), t, p);
        p = vfmaq_n_f32(vdupq_n_f32(1.0), t, p);
        p = vfmaq_n_f32(vdupq_n_f32(1.0), t, p);

        let ni = vmaxq_s32(
            vminq_s32(vcvtq_s32_f32(nf), vdupq_n_s32(127)),
            vdupq_n_s32(-126),
        );
        let bits = vshlq_n_u32::<23>(vreinterpretq_u32_s32(vaddq_s32(ni, vdupq_n_s32(127))));
        vmulq_f32(p, vreinterpretq_f32_u32(bits))
    }

    #[inline(always)]
    pub unsafe fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        assert_eq!(gate.len(), up.len());
        let n = gate.len();
        let n4 = n & !3;
        let (gp, upp) = (gate.as_ptr(), up.as_ptr());
        let mut out = vec![0f32; n];
        let op = out.as_mut_ptr();
        let one = vdupq_n_f32(1.0);
        for i in (0..n4).step_by(4) {
            let g = vld1q_f32(gp.add(i));
            let e = v_expf(vnegq_f32(g));
            let s = vdivq_f32(one, vaddq_f32(one, e));
            vst1q_f32(op.add(i), vmulq_f32(vmulq_f32(g, s), vld1q_f32(upp.add(i))));
        }
        for i in n4..n {
            let g = *gp.add(i);
            let s = 1.0 / (1.0 + (-g).exp());
            *op.add(i) = g * s * *upp.add(i);
        }
        out
    }

    #[inline(always)]
    pub unsafe fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
        assert_eq!(x.len(), w.len());
        let n = x.len();
        let n4 = n & !3;
        let xp = x.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for i in (0..n4).step_by(4) {
            let v = vld1q_f32(xp.add(i));
            acc = vfmaq_f32(acc, v, v);
        }
        let mut sum = vaddvq_f32(acc) as f64;
        for i in n4..n {
            sum += (*xp.add(i) * *xp.add(i)) as f64;
        }
        let mean = (sum / n as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();

        let wp = w.as_ptr();
        let mut out = vec![0f32; n];
        let op = out.as_mut_ptr();
        let vs = vdupq_n_f32(scale);
        for i in (0..n4).step_by(4) {
            let r = vmulq_f32(vmulq_f32(vld1q_f32(xp.add(i)), vs), vld1q_f32(wp.add(i)));
            vst1q_f32(op.add(i), r);
        }
        for i in n4..n {
            *op.add(i) = *xp.add(i) * scale * *wp.add(i);
        }
        out
    }

    #[inline(always)]
    pub unsafe fn rotate_halves(
        x: &[f32],
        cos: &[f32],
        sin: &[f32],
        out: &mut [f32],
        n_offset: usize,
    ) {
        let n = n_offset;
        let (xp, op) = (x.as_ptr(), out.as_mut_ptr());
        let n4 = n & !3;
        for i in (0..n4).step_by(4) {
            let va = vld1q_f32(xp.add(i));
            let vb = vld1q_f32(xp.add(n + i));
            let vc = vld1q_f32(cos.as_ptr().add(i));
            let vs = vld1q_f32(sin.as_ptr().add(i));
            // out_a = a*c - b*s ; out_b = a*s + b*c
            let oa = vfmsq_f32(vmulq_f32(va, vc), vb, vs);
            let ob = vfmaq_f32(vmulq_f32(vb, vc), va, vs);
            vst1q_f32(op.add(i), oa);
            vst1q_f32(op.add(n + i), ob);
        }
        for i in n4..n {
            let a = *xp.add(i);
            let b = *xp.add(n + i);
            *op.add(i) = a * cos[i] - b * sin[i];
            *op.add(n + i) = a * sin[i] + b * cos[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so both dispatch paths see identical data.
    struct Lcg(u64);
    impl Lcg {
        fn next_f32(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0 * scale
        }
        fn next_i8(&mut self) -> i8 {
            self.next_f32(200.0).round().clamp(-127.0, 127.0) as i8
        }
    }

    fn close(got: f32, expect: f32, tol_rel: f32) {
        let tol = tol_rel * expect.abs().max(1.0);
        assert!(
            (got - expect).abs() <= tol,
            "got {got} expected {expect} (diff {})",
            (got - expect).abs()
        );
    }

    #[test]
    fn dot_f32_matches_scalar() {
        if !use_simd() {
            return;
        }
        let mut rng = Lcg(42);
        let a: Vec<f32> = (0..4096).map(|_| rng.next_f32(3.0)).collect();
        let b: Vec<f32> = (0..4096).map(|_| rng.next_f32(3.0)).collect();
        let scalar: f32 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
        #[cfg(target_arch = "x86_64")]
        let got = unsafe { x86::dot_f32(&a, &b) };
        #[cfg(target_arch = "aarch64")]
        let got = unsafe { arm::dot_f32(&a, &b) };
        close(got, scalar, 1e-5);
    }

    #[test]
    fn rms_norm_simd_matches_scalar() {
        if !use_simd() {
            return;
        }
        let mut rng = Lcg(7);
        let x: Vec<f32> = (0..1024).map(|_| rng.next_f32(2.0)).collect();
        let w: Vec<f32> = (0..1024).map(|_| rng.next_f32(1.0)).collect();
        let eps = 1e-5f32;
        let mut sum = 0.0f64;
        for &v in &x {
            sum += (v * v) as f64;
        }
        let scale = 1.0f32 / ((sum / 1024.0) as f32 + eps).sqrt();
        let expect: Vec<f32> = x.iter().zip(&w).map(|(&xi, &wi)| xi * scale * wi).collect();
        #[cfg(target_arch = "x86_64")]
        let got = unsafe { x86::rms_norm(&x, &w, eps) };
        #[cfg(target_arch = "aarch64")]
        let got = unsafe { arm::rms_norm(&x, &w, eps) };
        for (&g, &e) in got.iter().zip(&expect) {
            close(g, e, 1e-4);
        }
    }

    #[test]
    fn swiglu_simd_matches_scalar() {
        if !use_simd() {
            return;
        }
        let mut rng = Lcg(99);
        let gate: Vec<f32> = (0..1000).map(|_| rng.next_f32(10.0)).collect();
        let up: Vec<f32> = (0..1000).map(|_| rng.next_f32(10.0)).collect();
        let expect: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| {
                let s = 1.0 / (1.0 + (-g).exp());
                g * s * u
            })
            .collect();
        #[cfg(target_arch = "x86_64")]
        let got = unsafe { x86::swiglu(&gate, &up) };
        #[cfg(target_arch = "aarch64")]
        let got = unsafe { arm::swiglu(&gate, &up) };
        for (&g, &e) in got.iter().zip(&expect) {
            close(g, e, 1e-4);
        }
    }

    #[test]
    fn rotate_halves_matches_scalar() {
        if !use_simd() {
            return;
        }
        let mut rng = Lcg(123);
        let n_offset = 64;
        let x: Vec<f32> = (0..2 * n_offset).map(|_| rng.next_f32(5.0)).collect();
        let cos: Vec<f32> = (0..n_offset).map(|_| rng.next_f32(1.0)).collect();
        let sin: Vec<f32> = (0..n_offset).map(|_| rng.next_f32(1.0)).collect();
        let mut expect = x.clone();
        for i in 0..n_offset {
            let (a, b) = (x[i], x[n_offset + i]);
            expect[i] = a * cos[i] - b * sin[i];
            expect[n_offset + i] = a * sin[i] + b * cos[i];
        }
        let mut got = x.clone();
        #[cfg(target_arch = "x86_64")]
        unsafe { x86::rotate_halves(&x, &cos, &sin, &mut got, n_offset) };
        #[cfg(target_arch = "aarch64")]
        unsafe { arm::rotate_halves(&x, &cos, &sin, &mut got, n_offset) };
        for (&g, &e) in got.iter().zip(&expect) {
            close(g, e, 1e-5);
        }
    }

    /// Build one synthetic K-quant weight block plus its Q8_K activation.
    fn synth_k_block(kind: &str, seed: u64) -> (Vec<u8>, Vec<u8>) {
        use crate::model::kernels::{quantize_row_q8_k, quantize_row_q8_0};
        let mut rng = Lcg(seed);
        match kind {
            "q8_0" => {
                let w: Vec<f32> = (0..32).map(|_| rng.next_f32(2.0)).collect();
                let x: Vec<f32> = (0..32).map(|_| rng.next_f32(2.0)).collect();
                (quantize_row_q8_0(&w), quantize_row_q8_0(&x))
            }
            _ => {
                // Random valid K-blocks: random nibbles/scales, fp16 scales.
                let le16 = |v: f32| crate::model::quant::f32_to_fp16(v).to_le_bytes();
                let mut wb = Vec::new();
                match kind {
                    "q4_k" => {
                        wb.extend_from_slice(&le16(rng.next_f32(1.0)));
                        wb.extend_from_slice(&le16(rng.next_f32(0.1)));
                        wb.extend((0..12).map(|_| rng.0 as u8));
                        wb.extend((0..128).map(|_| rng.next_i8() as u8));
                        assert_eq!(wb.len(), 144);
                    }
                    "q5_k" => {
                        wb.extend_from_slice(&le16(rng.next_f32(1.0)));
                        wb.extend_from_slice(&le16(rng.next_f32(0.1)));
                        wb.extend((0..12).map(|_| rng.0 as u8));
                        wb.extend((0..32).map(|_| rng.0 as u8));
                        wb.extend((0..128).map(|_| rng.next_i8() as u8));
                        assert_eq!(wb.len(), 176);
                    }
                    "q6_k" => {
                        wb.extend((0..128).map(|_| rng.next_i8() as u8));
                        wb.extend((0..64).map(|_| rng.0 as u8));
                        wb.extend((0..16).map(|_| rng.0 as u8));
                        wb.extend_from_slice(&le16(rng.next_f32(1.0)));
                        assert_eq!(wb.len(), 210);
                    }
                    _ => unreachable!(),
                }
                // Activation: random q8_K block.
                let act_f: Vec<f32> = (0..256).map(|_| rng.next_f32(50.0)).collect();
                (wb, quantize_row_q8_k(&act_f))
            }
        }
    }

    #[test]
    fn vec_dots_simd_match_scalar_reference() {
        if !use_simd() {
            return;
        }
        // Independent dequantize-based reference (same approach as kernels tests).
        use crate::model::quant::dequantize;
        fn dequant_q8k(act: &[u8]) -> Vec<f32> {
            const QK_K: usize = crate::model::quant::QK_K;
            let mut v = Vec::new();
            for b in act.chunks_exact(290) {
                let d = crate::model::quant::fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for j in 0..QK_K {
                    v.push(b[2 + j] as i8 as f32 * d);
                }
            }
            v
        }
        let cases = [
            ("q8_0", 32usize, 34usize),
            ("q4_k", 256, 144),
            ("q5_k", 256, 176),
            ("q6_k", 256, 210),
        ];
        for (kind, n, row_bytes) in cases {
            for seed in [1u64, 2, 3] {
                println!("case {kind} seed {seed}");
                let (w, act) = synth_k_block(kind, seed);
                // Multi-block rows for the K-quants.
                let (w_row, act_row, nn) = if kind == "q8_0" {
                    (w.clone(), act.clone(), n)
                } else {
                    let mut wr = w.clone();
                    let mut ar = act.clone();
                    for s in [10u64, 20] {
                        let (w2, a2) = synth_k_block(kind, seed * 100 + s);
                        wr.extend(w2);
                        ar.extend(a2);
                    }
                    (wr, ar, n * 3)
                };
                #[cfg(target_arch = "x86_64")]
                let got = match kind {
                    "q8_0" => unsafe { x86::vec_dot_q8_0_q8_0(nn, &w_row, &act_row) },
                    "q4_k" => unsafe { x86::vec_dot_q4_k_q8_k(nn, &w_row, &act_row) },
                    "q5_k" => unsafe { x86::vec_dot_q5_k_q8_k(nn, &w_row, &act_row) },
                    _ => unsafe { x86::vec_dot_q6_k_q8_k(nn, &w_row, &act_row) },
                };
                #[cfg(target_arch = "aarch64")]
                let got = match kind {
                    "q8_0" => unsafe { arm::vec_dot_q8_0_q8_0(nn, &w_row, &act_row) },
                    "q4_k" => unsafe { arm::vec_dot_q4_k_q8_k(nn, &w_row, &act_row) },
                    "q5_k" => unsafe { arm::vec_dot_q5_k_q8_k(nn, &w_row, &act_row) },
                    _ => unsafe { arm::vec_dot_q6_k_q8_k(nn, &w_row, &act_row) },
                };
                // Reference: dequantize both sides, naive f64 dot.
                let w_ty = match kind {
                    "q8_0" => crate::gguf::GGmlType::Q8_0,
                    "q4_k" => crate::gguf::GGmlType::Q4_K,
                    "q5_k" => crate::gguf::GGmlType::Q5_K,
                    _ => crate::gguf::GGmlType::Q6_K,
                };
                let wq = dequantize(w_ty, &w_row, nn as u64).unwrap();
                let aq = if kind == "q8_0" {
                    dequantize(crate::gguf::GGmlType::Q8_0, &act_row, nn as u64).unwrap()
                } else {
                    dequant_q8k(&act_row)
                };
                let expect: f32 =
                    wq.iter().zip(&aq).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() as f32;
                assert_eq!(w_row.len(), row_bytes * (nn / n));
                close(got, expect, 1e-2);
            }
        }
    }
}

