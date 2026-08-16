//! Math kernels for the inference engine.
//!
//! These are scalar ports of ggml's CPU reference implementations so the
//! results are reproducible against the C sources in `ref/` and can be
//! cross-checked against independent numpy references bit-for-bit (modulo
//! float accumulation order, which is noted where relevant).

use crate::gguf::GGmlType;
use crate::model::quant::{QuantError, QK_K, fp16_to_f32};

pub const QK8_0: usize = 32;

/// Fused RMS norm, mirroring `ggml_compute_forward_rms_norm_f32` with
/// `GGML_RMS_NORM_FUSE_OP_MUL`:
///
/// ```text
/// y[i] = x[i] * scale * w[i],   scale = 1/sqrt(mean(x^2) + eps)
/// ```
///
/// The sum of squares accumulates in f64 exactly like ggml's `ggml_float`;
/// `mean` and `scale` are computed in f32.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len(), w.len(), "rms_norm weights must match input length");
    let n = x.len();
    let mut sum = 0.0f64;
    for &xi in x {
        sum += (xi * xi) as f64;
    }
    let mean = (sum / n as f64) as f32;
    let scale = 1.0f32 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(&xi, &wi)| xi * scale * wi).collect()
}

// ---- activation quantization -------------------------------------------------

/// `quantize_row_q8_0_ref` (ggml-quants.c): per 32 elements,
/// `d = amax/127`, `qs[j] = roundf(x[j] * (1/d))`, d stored as fp16.
/// Block size 34 bytes.
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<u8> {
    assert!(x.len().is_multiple_of(QK8_0), "Q8_0 quantization needs a multiple of 32 elements");
    let mut out = Vec::with_capacity(x.len() / QK8_0 * 34);
    for block in x.chunks_exact(QK8_0) {
        let amax = block.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&crate::model::quant::f32_to_fp16(d).to_le_bytes());
        for &v in block {
            out.push((v * id).round() as i8 as u8);
        }
    }
    out
}

/// `quantize_row_q8_K_ref` (ggml-quants.c): per 256 elements, signed max,
/// `iscale = -127/max`, `qs[j] = MIN(127, nearest_int(iscale*x[j]))`,
/// `bsums[j] = sum(qs[16j..16j+16])`, `d = 1/iscale` (fp16).
/// `nearest_int` is round-half-to-even, so Rust uses `round_ties_even`.
/// Block size 290 bytes (d f16 + qs[256] + bsums[16]).
pub fn quantize_row_q8_k(x: &[f32]) -> Vec<u8> {
    assert!(x.len().is_multiple_of(QK_K), "Q8_K quantization needs a multiple of 256 elements");
    let mut out = Vec::with_capacity(x.len() / QK_K * 290);
    for block in x.chunks_exact(QK_K) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in block {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            out.extend_from_slice(&0u16.to_le_bytes());
            out.extend_from_slice(&[0u8; QK_K]);
            out.extend_from_slice(&[0u8; 16]);
            continue;
        }
        let iscale = -127.0f32 / max;
        let qs_start = out.len();
        out.extend_from_slice(&0u16.to_le_bytes());
        for &v in block {
            let q = (iscale * v).round_ties_even() as i32;
            out.push(127.min(q) as i8 as u8);
        }
        for j in 0..(QK_K / 16) {
            let mut sum = 0i16;
            for ii in 0..16 {
                sum += out[qs_start + 2 + j * 16 + ii] as i8 as i16;
            }
            out.extend_from_slice(&sum.to_le_bytes());
        }
        let d = 1.0f32 / iscale;
        let dh = crate::model::quant::f32_to_fp16(d);
        out[qs_start..qs_start + 2].copy_from_slice(&dh.to_le_bytes());
    }
    out
}

// ---- vec_dot: scalar ports of the `*_generic` functions in quants.c ---------

/// `ggml_vec_dot_q8_0_q8_0_generic`.
pub fn vec_dot_q8_0_q8_0(n: usize, x: &[u8], y: &[u8]) -> f32 {
    assert!(n.is_multiple_of(QK8_0));
    let nb = n / QK8_0;
    let mut sumf = 0.0f32;
    for i in 0..nb {
        let xb = &x[i * 34..(i + 1) * 34];
        let yb = &y[i * 34..(i + 1) * 34];
        let dx = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let dy = fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut sumi = 0i32;
        for j in 0..QK8_0 {
            sumi += (xb[2 + j] as i8 as i32) * (yb[2 + j] as i8 as i32);
        }
        sumf += sumi as f32 * (dx * dy);
    }
    sumf
}

/// `ggml_vec_dot_q4_K_q8_K_generic`.
pub fn vec_dot_q4_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
    assert!(n.is_multiple_of(QK_K));
    let nb = n / QK_K;
    let mut sumf = 0.0f32;
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];
    let mut aux16 = [0i16; 8];
    let mut utmp = [0u32; 4];
    for i in 0..nb {
        let xb = &x[i * 144..(i + 1) * 144];
        let yb = &y[i * 290..(i + 1) * 290];
        let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));

        let mut a = [0i8; QK_K];
        for j in 0..(QK_K / 64) {
            let q4 = &xb[16 + j * 32..16 + (j + 1) * 32];
            for l in 0..32 {
                a[j * 64 + l] = (q4[l] & 0x0F) as i8;
                a[j * 64 + 32 + l] = (q4[l] >> 4) as i8;
            }
        }

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

        let mut sumi = 0i32;
        for j in 0..(QK_K / 16) {
            let bsums = i16::from_le_bytes([yb[258 + 2 * j], yb[259 + 2 * j]]);
            sumi += bsums as i32 * mins[j / 2] as i32;
        }

        aux32.iter_mut().for_each(|v| *v = 0);
        let mut q8 = &yb[2..258];
        let mut aptr = &a[..];
        for &sv in &scales {
            let scale = sv as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = (q8[l] as i8 as i16) * (aptr[l] as i16);
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8 = &q8[8..];
                aptr = &aptr[8..];
            }
        }
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        sumf -= dmin * sumi as f32;
    }
    for &s in &sums {
        sumf += s;
    }
    sumf
}

/// `ggml_vec_dot_q5_K_q8_K_generic`.
pub fn vec_dot_q5_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
    assert!(n.is_multiple_of(QK_K));
    let nb = n / QK_K;
    let mut sumf = 0.0f32;
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];
    let mut aux16 = [0i16; 8];
    let mut utmp = [0u32; 4];
    for i in 0..nb {
        let xb = &x[i * 176..(i + 1) * 176];
        let yb = &y[i * 290..(i + 1) * 290];
        let d = fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]])) * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let dmin = fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]])) * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));

        let q4 = &xb[48..176];
        let hm = &xb[16..48];
        let mut a = [0i8; QK_K];
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

        let mut sumi = 0i32;
        for j in 0..(QK_K / 16) {
            let bsums = i16::from_le_bytes([yb[258 + 2 * j], yb[259 + 2 * j]]);
            sumi += bsums as i32 * mins[j / 2] as i32;
        }

        aux32.iter_mut().for_each(|v| *v = 0);
        let mut q8 = &yb[2..258];
        let mut aptr = &a[..];
        for &sv in &scales {
            let scale = sv as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux16[l] = (q8[l] as i8 as i16) * (aptr[l] as i16);
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8 = &q8[8..];
                aptr = &aptr[8..];
            }
        }
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
        sumf -= dmin * sumi as f32;
    }
    for &s in &sums {
        sumf += s;
    }
    sumf
}

/// `ggml_vec_dot_q6_K_q8_K_generic`.
pub fn vec_dot_q6_k_q8_k(n: usize, x: &[u8], y: &[u8]) -> f32 {
    assert!(n.is_multiple_of(QK_K));
    let nb = n / QK_K;
    let mut sumf = 0.0f32;
    let mut sums = [0.0f32; 8];
    let mut aux32 = [0i32; 8];
    let mut aux16 = [0i16; 8];
    for i in 0..nb {
        let xb = &x[i * 210..(i + 1) * 210];
        let yb = &y[i * 290..(i + 1) * 290];
        let d = fp16_to_f32(u16::from_le_bytes([xb[208], xb[209]])) * fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));

        let q4 = &xb[0..128];
        let qh = &xb[128..192];
        let sc = &xb[192..208];
        let mut a = [0i8; QK_K];
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

        aux32.iter_mut().for_each(|v| *v = 0);
        let mut q8 = &yb[2..258];
        let mut aptr = &a[..];
        for &sv in sc {
            let scale = sv as i8 as i32;
            for _ in 0..2 {
                for l in 0..8 {
                    aux16[l] = (q8[l] as i8 as i16) * (aptr[l] as i16);
                }
                for l in 0..8 {
                    aux32[l] += scale * aux16[l] as i32;
                }
                q8 = &q8[8..];
                aptr = &aptr[8..];
            }
        }
        for l in 0..8 {
            sums[l] += d * aux32[l] as f32;
        }
    }
    for &s in &sums {
        sumf += s;
    }
    sumf
}

// ---- gemv ---------------------------------------------------------------

/// Quantized matrix-vector product: `out[r] = dot(w_row_r, x)`.
///
/// `w` holds `n_out` rows of `n_in` elements in GGUF order (each row is a
/// multiple of the block size). The activation vector is quantized once:
/// Q8_0 for Q8_0 weights, Q8_K for the K-types, matching ggml's matmul
/// dispatch. F32 weights use a plain f32 dot product.
pub fn gemv(ty: GGmlType, w: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Result<Vec<f32>, QuantError> {
    assert_eq!(x.len(), n_in);
    let row_bytes = crate::model::quant::tensor_size(ty, n_in as u64)? as usize;
    assert!(w.len() >= n_out * row_bytes);
    let mut out = Vec::with_capacity(n_out);
    match ty {
        GGmlType::F32 => {
            for r in 0..n_out {
                let row = &w[r * row_bytes..(r + 1) * row_bytes];
                let mut sum = 0.0f32;
                for (i, c) in row.chunks_exact(4).enumerate() {
                    sum += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * x[i];
                }
                out.push(sum);
            }
        }
        GGmlType::Q8_0 => {
            let act = quantize_row_q8_0(x);
            for r in 0..n_out {
                out.push(vec_dot_q8_0_q8_0(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act));
            }
        }
        GGmlType::Q4_K => {
            let act = quantize_row_q8_k(x);
            for r in 0..n_out {
                out.push(vec_dot_q4_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act));
            }
        }
        GGmlType::Q5_K => {
            let act = quantize_row_q8_k(x);
            for r in 0..n_out {
                out.push(vec_dot_q5_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act));
            }
        }
        GGmlType::Q6_K => {
            let act = quantize_row_q8_k(x);
            for r in 0..n_out {
                out.push(vec_dot_q6_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act));
            }
        }
        other => return Err(QuantError::UnsupportedType(other.name())),
    }
    Ok(out)
}

// ---- IMRoPE (rope_multi) -------------------------------------------------

/// Rotary scaling parameters for `ggml_rope_ext` / `ggml_rope_multi`.
#[derive(Debug, Clone, Copy)]
pub struct RopeConfig {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            freq_base: 10_000_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// `rope_yarn_ramp` from ggml ops.cpp.
#[inline]
fn rope_yarn_ramp(low: f32, high: f32, i0: usize) -> f32 {
    let y = (i0 as f32 / 2.0 - low) / 0.001f32.max(high - low);
    1.0 - 1.0f32.min(0.0f32.max(y))
}

/// `rope_yarn` from ggml ops.cpp: computes (cos_theta, sin_theta).
#[inline]
fn rope_yarn(theta_extrap: f32, freq_scale: f32, corr: [f32; 2], i0: usize, ext_factor: f32, mscale: f32) -> (f32, f32) {
    let theta_interp = freq_scale * theta_extrap;
    let mut theta = theta_interp;
    let mut mscale = mscale;
    if ext_factor != 0.0 {
        let ramp_mix = rope_yarn_ramp(corr[0], corr[1], i0) * ext_factor;
        theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0 + 0.1f32 * (1.0 / freq_scale).ln();
    }
    (theta.cos() * mscale, theta.sin() * mscale)
}

/// `ggml_rope_yarn_corr_dims`: `start = floor(n_ctx_orig/beta_fast)`,
/// `end = floor(n_ctx_orig/beta_slow)`.
#[inline]
fn rope_yarn_corr_dims(n_ctx_orig: usize, beta_fast: f32, beta_slow: f32) -> [f32; 2] {
    [
        (n_ctx_orig as f32 / beta_fast).floor(),
        (n_ctx_orig as f32 / beta_slow).floor(),
    ]
}

/// Partial IMRoPE (interleaved multi-position rope) applied to one head row.
///
/// Port of `ggml_compute_forward_rope_flt` for `GGML_ROPE_TYPE_IMROPE`:
/// the cache is built with `ggml_mrope_cache_init` (interleaved sector
/// mapping) using the four per-token positions `pos = [p_t, p_h, p_w, p_e]`,
/// then `rotate_pairs` applies NEOX-style half-split rotation over the first
/// `n_dims` channels. The remaining `x.len() - n_dims` channels are copied.
pub fn rope_multi_imrope(
    x: &[f32],
    pos: [i32; 4],
    n_dims: usize,
    sections: [i32; 4],
    n_ctx_orig: usize,
    cfg: &RopeConfig,
) -> Vec<f32> {
    let ne0 = x.len();
    assert!(n_dims <= ne0 && n_dims.is_multiple_of(2), "n_dims {n_dims} out of range for row of {ne0}");
    let theta_scale = cfg.freq_base.powf(-2.0 / n_dims as f32);
    let corr = rope_yarn_corr_dims(n_ctx_orig, cfg.beta_fast, cfg.beta_slow);

    let sect_dims = sections.iter().sum::<i32>();
    assert!(sect_dims > 0 && (sect_dims as usize) <= ne0, "sections {sections:?} incompatible with row of {ne0}");
    let sec_w = sections[1] + sections[0];
    let _sec_e = sections[2] + sec_w;

    let mut theta_t = pos[0] as f32;
    let mut theta_h = pos[1] as f32;
    let mut theta_w = pos[2] as f32;
    let mut theta_e = pos[3] as f32;

    let mut cache = vec![0f32; ne0];
    let mut i0 = 0usize;
    while i0 < ne0 {
        let sector = (i0 as i32 / 2) % sect_dims;
        let theta = if sector % 3 == 1 && sector < 3 * sections[1] {
            theta_h
        } else if sector % 3 == 2 && sector < 3 * sections[2] {
            theta_w
        } else if sector % 3 == 0 && sector < 3 * sections[0] {
            theta_t
        } else {
            theta_e
        };
        let (c, s) = rope_yarn(theta, cfg.freq_scale, corr, i0, cfg.ext_factor, cfg.attn_factor);
        cache[i0] = c;
        cache[i0 + 1] = s;

        theta_t *= theta_scale;
        theta_w *= theta_scale;
        theta_h *= theta_scale;
        theta_e *= theta_scale;
        i0 += 2;
    }

    let mut out = x.to_vec();
    let n_offset = n_dims / 2;
    let mut i0 = 0usize;
    while i0 < n_dims {
        let ic = i0 / 2;
        let x0 = x[ic];
        let x1 = x[ic + n_offset];
        let cos = cache[i0];
        let sin = cache[i0 + 1];
        out[ic] = x0 * cos - x1 * sin;
        out[ic + n_offset] = x0 * sin + x1 * cos;
        i0 += 2;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::quant::dequantize;

    fn le16(v: f32) -> [u8; 2] {
        crate::model::quant::f32_to_fp16(v).to_le_bytes()
    }

    #[test]
    fn rms_norm_known_value() {
        let x = [3.0f32, -4.0];
        let w = [1.0f32, 1.0];
        let out = rms_norm(&x, &w, 1e-6);
        // sum = 25, mean = 12.5, scale = 1/sqrt(12.5)
        let scale = 1.0f32 / (12.5f32 + 1e-6).sqrt();
        assert!((out[0] - 3.0 * scale).abs() < 1e-6);
        assert!((out[1] + 4.0 * scale).abs() < 1e-6);
    }

    #[test]
    fn rms_norm_applies_weight() {
        let x = [1.0f32, 2.0, 3.0];
        let w = [2.0f32, 3.0, 4.0];
        let out = rms_norm(&x, &w, 0.0);
        let scale = 1.0f32 / (14.0f32 / 3.0).sqrt();
        for (i, v) in out.iter().enumerate() {
            assert!((v - x[i] * scale * w[i]).abs() < 1e-6);
        }
    }

    /// Dequantize a whole Q8_K byte buffer (290-byte blocks) to f32.
    fn dequant_q8k(act: &[u8]) -> Vec<f32> {
        let mut v = Vec::new();
        for b in act.chunks_exact(290) {
            let d = fp16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for j in 0..QK_K {
                v.push(b[2 + j] as i8 as f32 * d);
            }
        }
        v
    }

    /// Naive dot product of a dequantized weight row and a dequantized
    /// activation vector (both quantized), used as an independent reference.
    fn naive_quantized_dot(ty: GGmlType, w: &[u8], act_ty: GGmlType, act: &[u8], n: usize) -> f32 {
        let wq = dequantize(ty, w, n as u64).unwrap();
        let aq = match act_ty {
            GGmlType::Q8_0 => dequantize(act_ty, act, n as u64).unwrap(),
            GGmlType::Q8_K => dequant_q8k(act),
            _ => unreachable!(),
        };
        let mut sum = 0.0f64;
        for (a, b) in wq.iter().zip(aq.iter()) {
            sum += (*a as f64) * (*b as f64);
        }
        sum as f32
    }

    fn assert_close(got: f32, expect: f32) {
        let tol = 1e-2 * expect.abs().max(1.0);
        assert!((got - expect).abs() <= tol, "got {got} expected {expect} (diff {})", (got - expect).abs());
    }

    #[test]
    fn vec_dot_q8_0_matches_naive() {
        let w: Vec<f32> = (0..32).map(|i| (i as f32 * 0.7).sin()).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 1.1).cos() * 3.0).collect();
        let wq = quantize_row_q8_0(&w);
        let xq = quantize_row_q8_0(&x);
        let got = vec_dot_q8_0_q8_0(w.len(), &wq, &xq);
        let expect = naive_quantized_dot(GGmlType::Q8_0, &wq, GGmlType::Q8_0, &xq, w.len());
        assert_close(got, expect);
    }

    fn k_blocks(kind: &str, seed: u64) -> (Vec<u8>, Vec<f32>) {
        // Deterministic pseudo-random blocks with varied scale patterns.
        let mut state = seed;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut raw = Vec::new();
        let mut refv = Vec::new();
        for _ in 0..2 {
            let (block, values) = match kind {
                "Q4_K" => {
                    let d = next() % 16;
                    let dmin = next() % 16;
                    let mut b = Vec::new();
                    b.extend_from_slice(&le16((d + 1) as f32 * 0.25));
                    b.extend_from_slice(&le16((dmin + 1) as f32));
                    let mut scales = [0u8; 12];
                    for j in 0..8 {
                        let ls = (next() % 15) as u8;
                        let lm = (next() % 15) as u8;
                        if j < 4 {
                            scales[j] = ls;
                            scales[j + 4] = lm;
                        } else {
                            scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                            scales[j - 4] |= (ls >> 4) << 6;
                            scales[j] |= (lm >> 4) << 6;
                        }
                    }
                    b.extend_from_slice(&scales);
                    for _ in 0..128 {
                        b.push((next() & 0xFF) as u8);
                    }
                    let vals = crate::model::quant::dequantize(GGmlType::Q4_K, &b, QK_K as u64).unwrap();
                    (b, vals)
                }
                "Q5_K" => {
                    let mut b = Vec::new();
                    b.extend_from_slice(&le16(0.5));
                    b.extend_from_slice(&le16(3.0));
                    let mut scales = [0u8; 12];
                    for j in 0..8 {
                        let ls = (next() % 15) as u8;
                        let lm = (next() % 15) as u8;
                        if j < 4 {
                            scales[j] = ls;
                            scales[j + 4] = lm;
                        } else {
                            scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                            scales[j - 4] |= (ls >> 4) << 6;
                            scales[j] |= (lm >> 4) << 6;
                        }
                    }
                    b.extend_from_slice(&scales);
                    for _ in 0..32 {
                        b.push((next() & 0xFF) as u8);
                    }
                    for _ in 0..128 {
                        b.push((next() & 0xFF) as u8);
                    }
                    let vals = crate::model::quant::dequantize(GGmlType::Q5_K, &b, QK_K as u64).unwrap();
                    (b, vals)
                }
                "Q6_K" => {
                    let d = next() % 16;
                    let mut b = Vec::new();
                    for _ in 0..128 {
                        b.push((next() & 0xFF) as u8);
                    }
                    for _ in 0..64 {
                        b.push((next() & 0xFF) as u8);
                    }
                    for _ in 0..16 {
                        b.push((next() % 64) as u8);
                    }
                    b.extend_from_slice(&le16(1.0 / (d + 1) as f32));
                    let vals = crate::model::quant::dequantize(GGmlType::Q6_K, &b, QK_K as u64).unwrap();
                    (b, vals)
                }
                _ => unreachable!(),
            };
            raw.extend_from_slice(&block);
            refv.extend_from_slice(&values);
        }
        (raw, refv)
    }

    #[test]
    fn vec_dot_q4_k_matches_naive() {
        let (wq, _) = k_blocks("Q4_K", 7);
        let x = vec![0.75f32; QK_K * 2];
        let xq = quantize_row_q8_k(&x);
        let got = vec_dot_q4_k_q8_k(x.len(), &wq, &xq);
        let expect = naive_quantized_dot(GGmlType::Q4_K, &wq, GGmlType::Q8_K, &xq, x.len());
        assert_close(got, expect);
    }

    #[test]
    fn vec_dot_q5_k_matches_naive() {
        let (wq, _) = k_blocks("Q5_K", 8);
        let x: Vec<f32> = (0..QK_K * 2).map(|i| (i as f32 * 0.01).sin()).collect();
        let xq = quantize_row_q8_k(&x);
        let got = vec_dot_q5_k_q8_k(x.len(), &wq, &xq);
        let expect = naive_quantized_dot(GGmlType::Q5_K, &wq, GGmlType::Q8_K, &xq, x.len());
        assert_close(got, expect);
    }

    #[test]
    fn vec_dot_q6_k_matches_naive() {
        let (wq, _) = k_blocks("Q6_K", 9);
        let x: Vec<f32> = (0..QK_K * 2).map(|i| ((i * 7) % 61) as f32 * 0.1 - 3.0).collect();
        let xq = quantize_row_q8_k(&x);
        let got = vec_dot_q6_k_q8_k(x.len(), &wq, &xq);
        let expect = naive_quantized_dot(GGmlType::Q6_K, &wq, GGmlType::Q8_K, &xq, x.len());
        assert_close(got, expect);
    }

    #[test]
    fn gemv_matches_dequant_dot() {
        let w: Vec<f32> = (0..32).map(|i| (i as f32 * 0.7).sin()).collect();
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 1.1).cos() * 3.0).collect();
        let wq = quantize_row_q8_0(&w);
        let xq = quantize_row_q8_0(&x);
        let out = gemv(GGmlType::Q8_0, &wq, w.len(), 1, &x).unwrap();
        let expect = naive_quantized_dot(GGmlType::Q8_0, &wq, GGmlType::Q8_0, &xq, w.len());
        assert_close(out[0], expect);
    }

    #[test]
    fn gemv_multi_row() {
        let n_in = QK_K * 2;
        let n_out = 3;
        let mut w = Vec::new();
        for r in 0..n_out {
            let row: Vec<f32> = (0..n_in).map(|i| (r as f32 + 1.0) * (i as f32 * 0.01).cos()).collect();
            w.extend_from_slice(&quantize_row_q8_0(&row));
        }
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.02).sin()).collect();
        let out = gemv(GGmlType::Q8_0, &w, n_in, n_out, &x).unwrap();
        assert_eq!(out.len(), n_out);
        let row_bytes = crate::model::quant::tensor_size(GGmlType::Q8_0, n_in as u64).unwrap() as usize;
        let xq = quantize_row_q8_0(&x);
        for (r, v) in out.iter().enumerate() {
            let expect = naive_quantized_dot(GGmlType::Q8_0, &w[r * row_bytes..(r + 1) * row_bytes], GGmlType::Q8_0, &xq, n_in);
            assert_close(*v, expect);
        }
    }

    #[test]
    fn rope_identity_at_zero() {
        let x: Vec<f32> = (0..8).map(|i| i as f32 + 0.5).collect();
        let cfg = RopeConfig::default();
        let out = rope_multi_imrope(&x, [0, 0, 0, 0], 8, [4, 4, 0, 0], 0, &cfg);
        for (a, b) in x.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_known_rotation() {
        // n_dims=4, sections=[1,0,0,0]: every pair uses theta_t
        // (sector = (i0/2) % 1 = 0 always). freq_base=1e7 ->
        // theta_scale = 1e7^(-2/4) ~ 3.16e-4. pos p_t=1:
        // pair0 theta = 1 rad, pair1 theta = theta_scale.
        // Interleaved layout: pair ic = (x[ic], x[ic+2]), so x=[1,1,0,0]
        // gives pairs (1,0) and (1,0).
        let x = [1.0f32, 1.0, 0.0, 0.0];
        let cfg = RopeConfig::default();
        let out = rope_multi_imrope(&x, [1, 0, 0, 0], 4, [1, 0, 0, 0], 0, &cfg);
        let ts = 10_000_000.0f32.powf(-2.0 / 4.0);
        let (c0, s0) = (1.0f32.cos(), 1.0f32.sin());
        let (c1, s1) = (ts.cos(), ts.sin());
        let expect = [c0, c1, s0, s1];
        for (i, (a, b)) in out.iter().zip(expect.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx {i}: got {a} expected {b}");
        }
    }

    #[test]
    fn rope_section_mapping() {
        // sections [11,11,10,0], freq_base=1 -> theta_scale=1, so each
        // sector keeps its initial theta. Mapping: sectors 0,3,6 -> theta_t
        // (pos 1), 1,4 -> theta_h (pos 2), 2,5 -> theta_w (pos 3).
        // ne0 = n_dims = 64 (real model geometry); pairs are (x[ic], x[ic+32]),
        // so x = [1]*32 + [0]*32 gives pairs (1,0).
        let mut x = [0.0f32; 64];
        x[..32].fill(1.0);
        let cfg = RopeConfig { freq_base: 1.0, ..Default::default() }; // theta_scale = 1
        let out = rope_multi_imrope(&x, [1, 2, 3, 0], 64, [11, 11, 10, 0], 0, &cfg);
        let theta = [1.0f32, 2.0, 3.0];
        for ic in 0..7 {
            let expected_theta = match ic % 3 {
                0 => theta[0],
                1 => theta[1],
                _ => theta[2],
            };
            let (c, s) = (expected_theta.cos(), expected_theta.sin());
            assert!((out[ic] - c).abs() < 1e-5, "ic {ic}: got {} expected {}", out[ic], c);
            assert!((out[ic + 32] - s).abs() < 1e-5, "ic {ic}: got {} expected {}", out[ic + 32], s);
        }
    }
}
