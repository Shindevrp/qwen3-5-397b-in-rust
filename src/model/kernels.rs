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

/// Batched matrix-vector product: `out[row * n_batch + b] = dot(w_row, x[b])`.
///
/// `x` has `n_in * n_batch` floats laid out as `n_batch` contiguous vectors.
/// Output is `n_out * n_batch` floats, row-major by weight row.
pub fn gemm(
    ty: GGmlType,
    w: &[u8],
    n_in: usize,
    n_out: usize,
    n_batch: usize,
    x: &[f32],
) -> Result<Vec<f32>, QuantError> {
    assert_eq!(
        x.len(),
        n_in * n_batch,
        "x must have n_in × n_batch = {n_in}×{n_batch} = {} floats",
        n_in * n_batch,
    );
    let row_bytes = crate::model::quant::tensor_size(ty, n_in as u64)? as usize;
    assert!(w.len() >= n_out * row_bytes);
    let mut out = vec![0.0f32; n_out * n_batch];
    match ty {
        GGmlType::F32 => {
            for b in 0..n_batch {
                let xb = &x[b * n_in..(b + 1) * n_in];
                for r in 0..n_out {
                    let row = &w[r * row_bytes..(r + 1) * row_bytes];
                    let mut sum = 0.0f32;
                    for (i, c) in row.chunks_exact(4).enumerate() {
                        sum += f32::from_le_bytes([c[0], c[1], c[2], c[3]]) * xb[i];
                    }
                    out[r * n_batch + b] = sum;
                }
            }
        }
        GGmlType::Q8_0 => {
            for b in 0..n_batch {
                let act = quantize_row_q8_0(&x[b * n_in..(b + 1) * n_in]);
                for r in 0..n_out {
                    out[r * n_batch + b] =
                        vec_dot_q8_0_q8_0(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act);
                }
            }
        }
        GGmlType::Q4_K => {
            for b in 0..n_batch {
                let act = quantize_row_q8_k(&x[b * n_in..(b + 1) * n_in]);
                for r in 0..n_out {
                    out[r * n_batch + b] =
                        vec_dot_q4_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act);
                }
            }
        }
        GGmlType::Q5_K => {
            for b in 0..n_batch {
                let act = quantize_row_q8_k(&x[b * n_in..(b + 1) * n_in]);
                for r in 0..n_out {
                    out[r * n_batch + b] =
                        vec_dot_q5_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act);
                }
            }
        }
        GGmlType::Q6_K => {
            for b in 0..n_batch {
                let act = quantize_row_q8_k(&x[b * n_in..(b + 1) * n_in]);
                for r in 0..n_out {
                    out[r * n_batch + b] =
                        vec_dot_q6_k_q8_k(n_in, &w[r * row_bytes..(r + 1) * row_bytes], &act);
                }
            }
        }
        other => return Err(QuantError::UnsupportedType(other.name())),
    }
    Ok(out)
}

// ---- Softmax + Attention ---------------------------------------------------

/// In-place softmax over each row.  `x` has `n_rows * n_cols` elements;
/// `n_cols` is the number of elements per row that are softmaxed together.
pub fn softmax_in_place(x: &mut [f32], n_cols: usize) {
    assert!(x.len().is_multiple_of(n_cols));
    for row in x.chunks_exact_mut(n_cols) {
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Scaled dot-product attention with optional causal mask.
///
/// **Layout** (ggml row-major): `ne[0] = head_dim, ne[1] = tokens, ne[2] = heads`.
///
/// - `q`: `[n_q × n_heads × head_dim]`  — query tokens.
/// - `k`: `[n_kv × n_kv_heads × head_dim]`  — key cache (or full keys).
/// - `v`: `[n_kv × n_kv_heads × head_dim]`  — value cache.
/// - Returns `[n_q × n_heads × head_dim]`  — one float per (token, head, dim).
///
/// Each query token `qt` attends to key tokens `0..=min(qt, n_kv-1)` when
/// `causal` is true, or `0..n_kv` otherwise (for decode, n_kv == pos+1 and
/// the entire cache is valid).
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_q: usize,
    n_kv: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    assert!(n_heads.is_multiple_of(n_kv_heads), "n_heads must be a multiple of n_kv_heads");
    let gqa = n_heads / n_kv_heads;
    let q_stride = head_dim; // stride between heads within one query token
    let kv_head_stride = head_dim; // stride between tokens for one kv head
    let kv_tok_stride = head_dim * n_kv_heads; // stride between kv tokens

    let mut out = vec![0.0f32; n_q * n_heads * head_dim];
    // scores buffer: n_kv elements per (qt, qh) pair
    let mut scores = vec![0.0f32; n_kv];

    for qt in 0..n_q {
        let q_base = qt * n_heads * head_dim;
        let o_base = qt * n_heads * head_dim;
        for qh in 0..n_heads {
            let kv_h = qh / gqa;
            let qh_base = q_base + qh * q_stride;
            let oh_base = o_base + qh * head_dim;
            let kv_h_base = kv_h * kv_head_stride;

            // Upper bound of positions this query can attend to.
            let max_kv = if causal {
                // For prefill (n_q > 1): qt is the query position, so we can
                // attend to key positions 0..=qt (clamped to n_kv).
                // For decode (n_q == 1, autoregressive): attend to all n_kv.
                if n_q > 1 { qt.min(n_kv - 1) } else { n_kv - 1 }
            } else {
                n_kv - 1
            };

            // QK^T / sqrt(d)
            for t in 0..=max_kv {
                let k_base = kv_h_base + t * kv_tok_stride;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qh_base + d] * k[k_base + d];
                }
                scores[t] = dot * scale;
            }
            // Mask future positions to -inf
            for t in (max_kv + 1)..n_kv {
                scores[t] = f32::NEG_INFINITY;
            }

            softmax_in_place(&mut scores[..n_kv], n_kv);

            // scores × V
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..n_kv {
                    let v_base = kv_h_base + t * kv_tok_stride;
                    acc += scores[t] * v[v_base + d];
                }
                out[oh_base + d] = acc;
            }
        }
    }
    out
}

// ---- Delta-net linear attention (autoregressive) --------------------------

/// Delta-net autoregressive linear attention (GDA mode).
///
/// Layout: per-head vectors are contiguous: `q[hi * S_k .. (hi+1) * S_k]`.
///
/// Steps:
/// 1. L2-normalize q, k
/// 2. Scale q by `1/sqrt(S_v)`
/// 3. `beta = sigmoid(beta)`
/// 4. `state *= exp(g)` (per-head scalar gate, GDA)
/// 5. `k_state = state^T @ k`
/// 6. `v_diff = v - k_state`
/// 7. `state += outer(v_diff, k * beta)`
/// 8. `output = state^T @ q`
#[allow(clippy::too_many_arguments)]
pub fn delta_net_autoregressive(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    s_k: usize,
    s_v: usize,
    n_heads: usize,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(q.len(), s_k * n_heads);
    assert_eq!(k.len(), s_k * n_heads);
    assert_eq!(v.len(), s_v * n_heads);
    assert_eq!(g.len(), n_heads);
    assert_eq!(beta.len(), n_heads);
    assert_eq!(state.len(), s_v * s_v * n_heads);
    let scale = 1.0 / (s_v as f32).sqrt();
    let mut out = vec![0.0f32; s_v * n_heads];

    for hi in 0..n_heads {
        let q_base = hi * s_k;
        let v_base = hi * s_v;
        let st_base = hi * s_v * s_v;

        // L2-normalize q
        let q_slice = &q[q_base..q_base + s_k];
        let q_norm_sq: f32 = q_slice.iter().map(|x| x * x).sum();
        let q_factor = scale / (q_norm_sq.sqrt() + eps);

        // L2-normalize k
        let k_slice = &k[q_base..q_base + s_k];
        let k_norm_sq: f32 = k_slice.iter().map(|x| x * x).sum();
        let k_factor = 1.0 / (k_norm_sq.sqrt() + eps);

        // Sigmoid beta
        let b = 1.0 / (1.0 + (-beta[hi]).exp());

        // State decay: state *= exp(g)
        let decay = g[hi].exp();
        for val in state[st_base..st_base + s_v * s_v].iter_mut() {
            *val *= decay;
        }

        // k_state = state^T @ k_norm  → [S_v]
        // state^T[i,j] = state[j,i] = state[st_base + i * s_v + j]
        let mut k_state = vec![0.0f32; s_v];
        for i in 0..s_v {
            let mut acc = 0.0f32;
            for j in 0..s_k {
                acc += state[st_base + i * s_v + j] * k_slice[j];
            }
            k_state[i] = acc * k_factor;
        }

        // v_diff = v - k_state
        let mut v_diff = vec![0.0f32; s_v];
        for i in 0..s_v {
            v_diff[i] = v[v_base + i] - k_state[i];
        }

        // State update: state += outer(v_diff, k_norm * beta)
        for i in 0..s_v {
            for j in 0..s_k {
                state[st_base + i * s_v + j] += v_diff[i] * k_slice[j] * k_factor * b;
            }
        }

        // output = state^T @ q_norm  → [S_v]
        let q_norm_factor = q_factor; // already includes scale
        for i in 0..s_v {
            let mut acc = 0.0f32;
            for j in 0..s_k {
                acc += state[st_base + i * s_v + j] * q_slice[j];
            }
            out[v_base + i] = acc * q_norm_factor;
        }
    }
    out
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

// ---------------------------------------------------------------------------
// Phase 3a – small utility kernels for MoE FFN / full-layer forward pass
// ---------------------------------------------------------------------------

/// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`.
///
/// Port of `ggml_swiglu_split` for the common case where gate and up are
/// already separate tensors (Qwen3.5 MoE path).
#[inline]
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    assert_eq!(gate.len(), up.len(), "swiglu: gate and up must have same length");
    gate.iter().zip(up.iter()).map(|(&g, &u)| {
        let s = 1.0 / (1.0 + (-g).exp());
        g * s * u
    }).collect()
}

/// Per-head RMS norm.
///
/// Normalizes each contiguous `head_size` chunk of `x` independently, applying
/// the weight `w` (also `head_size` long, tiled for every head).
///
/// Port of applying `LLM_NORM_RMS` to a `[head_size, n_heads]` tensor.
pub fn rms_norm_per_head(x: &[f32], w: &[f32], head_size: usize, eps: f32) -> Vec<f32> {
    assert!(x.len().is_multiple_of(head_size), "rms_norm_per_head: x.len must be multiple of head_size");
    assert_eq!(w.len(), head_size, "rms_norm_per_head: w must be head_size long");
    let n_heads = x.len() / head_size;
    let mut out = vec![0.0f32; x.len()];
    for h in 0..n_heads {
        let base = h * head_size;
        let mut sum = 0.0f64;
        for i in 0..head_size {
            let v = x[base + i];
            sum += (v * v) as f64;
        }
        let mean = (sum / head_size as f64) as f32;
        let scale = 1.0f32 / (mean + eps).sqrt();
        for i in 0..head_size {
            out[base + i] = x[base + i] * scale * w[i];
        }
    }
    out
}

/// Softmax + top-k selection + renormalize.
///
/// Given `logits` of length `n_experts`, computes softmax, selects the `k`
/// largest experts, and renormalizes the selected weights to sum to 1.
///
/// Returns `(selected_weights, selected_indices)` where both are length `k`.
///
/// Port of `ggml_soft_max` → `ggml_argsort_top_k` → weight normalization.
pub fn softmax_topk(logits: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
    let n = logits.len();
    assert!(k <= n, "softmax_topk: k must be <= n_experts");

    // Softmax in f64 for numerical stability
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps_f64: Vec<f64> = logits.iter().map(|&l| ((l - max_l) as f64).exp()).collect();
    let sum_exp: f64 = exps_f64.iter().sum();
    let probs: Vec<f32> = exps_f64.iter().map(|&e| (e / sum_exp) as f32).collect();

    // Partial sort to find top-k indices, then sort within top-k (descending)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.select_nth_unstable_by(k, |&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
    idx.truncate(k);
    idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut weights = Vec::with_capacity(k);
    let mut sum_w = 0.0f64;
    for &i in &idx {
        sum_w += probs[i] as f64;
    }
    for &i in &idx {
        weights.push((probs[i] as f64 / sum_w) as f32);
    }

    (weights, idx)
}

/// Causal 1D convolution over the channel dimension with SiLU activation.
///
/// `input` has shape `[channels, seq_len]` (row-major: `input[c * seq_len + t]`).
/// `kernel` has shape `[channels, kernel_size]` (same layout as ggml `ssm_conv1d`).
/// `state_in` has shape `[channels, kernel_size - 1]` — the causal history.
///
/// Returns `(output, state_out)` where:
/// - `output` has shape `[channels, seq_len]` with SiLU applied
/// - `state_out` has shape `[channels, kernel_size - 1]` — updated causal state
///
/// Port of `ggml_ssm_conv` + `ggml_silu` for the Qwen3.5 delta-net path.
/// The kernel slides over the concatenated `[state | input]` sequence:
///   t=0: dot(kernel, [state[0], state[1], ..., input[0]])
///   t=1: dot(kernel, [state[1], ..., input[0], input[1]])
///   ...
pub fn conv1d_silu(
    input: &[f32],
    kernel: &[f32],
    state_in: &[f32],
    channels: usize,
    seq_len: usize,
    kernel_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    let pad = kernel_size - 1;
    assert_eq!(input.len(), channels * seq_len);
    assert_eq!(kernel.len(), channels * kernel_size);
    assert_eq!(state_in.len(), channels * pad);

    let mut output = vec![0.0f32; channels * seq_len];
    let mut state_out = vec![0.0f32; channels * pad];

    for c in 0..channels {
        for t in 0..seq_len {
            let mut acc = 0.0f32;
            // For position t in the output, the kernel window covers
            // the pad state elements + the first (t+1) input elements.
            // window[k] = state_in[c*pad + t+k - (kernel_size-1)] for k where in range,
            //              or input[c*seq_len + t+k - (kernel_size-1)] otherwise.
            for k in 0..kernel_size {
                let src_idx = t as isize + k as isize - pad as isize;
                let val = if src_idx < 0 {
                    state_in[c * pad + (src_idx + pad as isize) as usize]
                } else {
                    input[c * seq_len + src_idx as usize]
                };
                acc += val * kernel[c * kernel_size + k];
            }
            // Apply SiLU
            let s = 1.0 / (1.0 + (-acc).exp());
            output[c * seq_len + t] = acc * s;
        }
        // New state: last `pad` elements of input
        for s in 0..pad {
            state_out[c * pad + s] = input[c * seq_len + seq_len - pad + s];
        }
    }

    (output, state_out)
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
    fn gemm_matches_naive() {
        let n_in = QK_K;
        let n_out = 4;
        let n_batch = 3;
        // craft n_out rows of Q8_0 weights
        let mut state = 7u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut w = Vec::new();
        for _ in 0..n_out {
            let row: Vec<f32> = (0..n_in).map(|_| ((next() as f32) - 1000.0) * 0.001).collect();
            w.extend_from_slice(&quantize_row_q8_0(&row));
        }
        let mut x = Vec::new();
        for b in 0..n_batch {
            for i in 0..n_in {
                x.push(((b as f32 * 100.0 + i as f32) * 0.03).sin());
            }
        }
        let out = gemm(GGmlType::Q8_0, &w, n_in, n_out, n_batch, &x).unwrap();
        assert_eq!(out.len(), n_out * n_batch);
        let row_bytes = crate::model::quant::tensor_size(GGmlType::Q8_0, n_in as u64).unwrap() as usize;
        for b in 0..n_batch {
            let xb = &x[b * n_in..(b + 1) * n_in];
            let xq = quantize_row_q8_0(xb);
            for r in 0..n_out {
                let expect = naive_quantized_dot(
                    GGmlType::Q8_0,
                    &w[r * row_bytes..(r + 1) * row_bytes],
                    GGmlType::Q8_0,
                    &xq,
                    n_in,
                );
                assert_close(out[r * n_batch + b], expect);
            }
        }
    }

    #[test]
    fn softmax_known_values() {
        // Uniform input → equal probs
        let mut x = vec![1.0f32; 4];
        softmax_in_place(&mut x, 4);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
        // Known distribution: [1.0, 2.0, 3.0]
        let mut x2 = vec![1.0f32, 2.0, 3.0];
        softmax_in_place(&mut x2, 3);
        let sum: f32 = x2.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(x2[2] > x2[1]);
        assert!(x2[1] > x2[0]);
    }

    #[test]
    fn attention_causal_matches_numpy() {
        // Small test: head_dim=8, n_heads=2, n_kv_heads=2 (no GQA), n_q=4, n_kv=4
        let hd = 8;
        let n_heads = 2;
        let n_kv_heads = 2;
        let n_q = 4;
        let n_kv = 4;
        let scale = 1.0 / (hd as f32).sqrt();

        // Craft deterministic Q, K, V
        let q: Vec<f32> = (0..n_q * n_heads * hd)
            .map(|i| (i as f32 * 0.1).sin() * 2.0 - 1.0)
            .collect();
        let k: Vec<f32> = (0..n_kv * n_kv_heads * hd)
            .map(|i| (i as f32 * 0.15 + 5.0).cos() * 1.5)
            .collect();
        let v: Vec<f32> = (0..n_kv * n_kv_heads * hd)
            .map(|i| (i as f32 * 0.2 + 10.0).sin())
            .collect();

        let out = attention_forward(&q, &k, &v, n_heads, n_kv_heads, hd, n_q, n_kv, scale, true);

        // Reference: naive numpy-style
        let causal = true;
        for qt in 0..n_q {
            for qh in 0..n_heads {
                let mut scores = vec![0.0f32; n_kv];
                for t in 0..n_kv {
                    if causal && t > qt {
                        scores[t] = f32::NEG_INFINITY;
                    } else {
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q[qt * n_heads * hd + qh * hd + d]
                                * k[t * n_kv_heads * hd + qh * hd + d];
                        }
                        scores[t] = dot * scale;
                    }
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_e: f32 = exps.iter().sum();
                let probs: Vec<f32> = exps.iter().map(|e| e / sum_e).collect();
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..n_kv {
                        acc += probs[t] * v[t * n_kv_heads * hd + qh * hd + d];
                    }
                    let idx = qt * n_heads * hd + qh * hd + d;
                    assert_close(out[idx], acc);
                }
            }
        }
    }

    #[test]
    fn attention_gqa() {
        // GQA: n_heads=4, n_kv_heads=2, hd=8, n_q=2, n_kv=3
        let hd = 8;
        let n_heads = 4;
        let n_kv_heads = 2;
        let n_q = 2;
        let n_kv = 3;
        let scale = 1.0 / (hd as f32).sqrt();

        let q: Vec<f32> = (0..n_q * n_heads * hd).map(|i| (i as f32 * 0.3).sin()).collect();
        let k: Vec<f32> = (0..n_kv * n_kv_heads * hd).map(|i| (i as f32 * 0.5 + 1.0).cos()).collect();
        let v: Vec<f32> = (0..n_kv * n_kv_heads * hd).map(|i| (i as f32 * 0.2 + 2.0).sin()).collect();

        let out = attention_forward(&q, &k, &v, n_heads, n_kv_heads, hd, n_q, n_kv, scale, false);

        // Verify: qh 0 and qh 1 use kv_h 0; qh 2 and qh 3 use kv_h 1
        // So out[0,0,:] should equal out[0,1,:] (same KV, same Q? no, different Q rows)
        // But out[:,0,:] and out[:,1,:] use same K/V cache
        for qh in 0..n_heads {
            let kv_h = qh / 2; // gqa = 4/2 = 2
            for qt in 0..n_q {
                let mut scores = vec![0.0f32; n_kv];
                for t in 0..n_kv {
                    let mut dot = 0.0f32;
                    for d in 0..hd {
                        dot += q[qt * n_heads * hd + qh * hd + d]
                            * k[t * n_kv_heads * hd + kv_h * hd + d];
                    }
                    scores[t] = dot * scale;
                }
                softmax_in_place(&mut scores, n_kv);
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..n_kv {
                        acc += scores[t] * v[t * n_kv_heads * hd + kv_h * hd + d];
                    }
                    assert_close(out[qt * n_heads * hd + qh * hd + d], acc);
                }
            }
        }
    }

    #[test]
    fn delta_net_matches_naive() {
        let s_k = 16;
        let s_v = 16;
        let n_heads = 2;
        let eps = 1e-6f32;

        let mut state = vec![0.0f32; s_v * s_v * n_heads];

        // Run two tokens to exercise state accumulation
        let mut out_all = Vec::new();
        for step in 0..2 {
            let q: Vec<f32> = (0..s_k * n_heads)
                .map(|i| (i as f32 * 0.3 + step as f32 * 7.0).sin())
                .collect();
            let k: Vec<f32> = (0..s_k * n_heads)
                .map(|i| (i as f32 * 0.5 + step as f32 * 3.0 + 1.0).cos())
                .collect();
            let v: Vec<f32> = (0..s_v * n_heads)
                .map(|i| (i as f32 * 0.2 + step as f32 * 5.0 + 2.0).sin())
                .collect();
            let g: Vec<f32> = vec![-0.1, -0.3];
            let beta: Vec<f32> = vec![0.5, 0.8];

            let out = delta_net_autoregressive(
                &q, &k, &v, &g, &beta, &mut state, s_k, s_v, n_heads, eps,
            );
            out_all.push(out);
        }

        // Naive reference: same computation
        let mut state_ref = vec![0.0f32; s_v * s_v * n_heads];
        let mut out_ref_all = Vec::new();
        for step in 0..2 {
            let q: Vec<f32> = (0..s_k * n_heads)
                .map(|i| (i as f32 * 0.3 + step as f32 * 7.0).sin())
                .collect();
            let k: Vec<f32> = (0..s_k * n_heads)
                .map(|i| (i as f32 * 0.5 + step as f32 * 3.0 + 1.0).cos())
                .collect();
            let v: Vec<f32> = (0..s_v * n_heads)
                .map(|i| (i as f32 * 0.2 + step as f32 * 5.0 + 2.0).sin())
                .collect();
            let g_vals = [-0.1f32, -0.3f32];
            let beta_vals = [0.5f32, 0.8f32];
            let scale = 1.0 / (s_v as f32).sqrt();

            let mut out_h = vec![0.0f32; s_v * n_heads];
            for hi in 0..n_heads {
                // L2 normalize q
                let qh = &q[hi * s_k..(hi + 1) * s_k];
                let qn: f32 = qh.iter().map(|x| x * x).sum();
                let qf = scale / (qn.sqrt() + eps);
                // L2 normalize k
                let kh = &k[hi * s_k..(hi + 1) * s_k];
                let kn: f32 = kh.iter().map(|x| x * x).sum();
                let kf = 1.0 / (kn.sqrt() + eps);
                // sigmoid beta
                let b = 1.0 / (1.0 + (-beta_vals[hi]).exp());
                // decay
                let decay = g_vals[hi].exp();
                let sb = hi * s_v * s_v;
                for val in state_ref[sb..sb + s_v * s_v].iter_mut() {
                    *val *= decay;
                }
                // k_state = state^T @ (k * kf)
                let mut kst = vec![0.0f32; s_v];
                for i in 0..s_v {
                    for j in 0..s_k {
                        kst[i] += state_ref[sb + i * s_v + j] * kh[j];
                    }
                    kst[i] *= kf;
                }
                // v_diff
                let vh = &v[hi * s_v..(hi + 1) * s_v];
                let mut vd = vec![0.0f32; s_v];
                for i in 0..s_v {
                    vd[i] = vh[i] - kst[i];
                }
                // state += outer(vd, kh * kf * b)
                for i in 0..s_v {
                    for j in 0..s_k {
                        state_ref[sb + i * s_v + j] += vd[i] * kh[j] * kf * b;
                    }
                }
                // output = state^T @ (qh * qf)
                for i in 0..s_v {
                    for j in 0..s_k {
                        out_h[hi * s_v + i] += state_ref[sb + i * s_v + j] * qh[j];
                    }
                    out_h[hi * s_v + i] *= qf;
                }
            }
            out_ref_all.push(out_h);
        }

        // Compare state and outputs
        for (step, (got, expect)) in out_all.iter().zip(out_ref_all.iter()).enumerate() {
            for (i, (a, b)) in got.iter().zip(expect.iter()).enumerate() {
                let tol = 1e-3 * b.abs().max(1.0);
                assert!(
                    (a - b).abs() <= tol,
                    "step={step} i={i}: got {a} expect {b} diff {}",
                    (a - b).abs()
                );
            }
        }
        // State must also match
        for (i, (a, b)) in state.iter().zip(state_ref.iter()).enumerate() {
            let tol = 1e-3 * b.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "state[{i}]: got {a} expect {b} diff {}",
                (a - b).abs()
            );
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

    // ---- Phase 3a kernel tests -----------------------------------------------

    #[test]
    fn swiglu_known_values() {
        // silu(1.0) = 1 / (1 + e^-1) ≈ 0.7310585786
        // silu(0.0) = 0.5
        // silu(-1.0) = 1 / (1 + e^1) ≈ 0.2689414214
        let gate = [1.0f32, 0.0, -1.0];
        let up = [2.0f32, 3.0, 4.0];
        let out = swiglu(&gate, &up);
        // silu(g) = g * sigmoid(g)
        let s1 = 1.0f32 * 1.0 / (1.0 + (-1.0f32).exp());  // silu(1)
        let s0 = 0.0f32 * 0.5f32;                           // silu(0) = 0
        let sm1 = -1.0f32 / (1.0 + (1.0f32).exp()); // silu(-1)
        assert!((out[0] - s1 * 2.0).abs() < 1e-6);
        assert!((out[1] - s0 * 3.0).abs() < 1e-6);
        assert!((out[2] - sm1 * 4.0).abs() < 1e-6);
    }

    #[test]
    fn swiglu_matches_numpy() {
        let rng_swi = |i: usize| (i as f32 * 0.15 + 1.7).sin() * 2.0 - 1.0;
        let gate: Vec<f32> = (0..64).map(rng_swi).collect();
        let up: Vec<f32> = (0..64).map(|i| (i as f32 * 0.23 - 0.5).cos()).collect();
        let out = swiglu(&gate, &up);
        // Cross-check against manual numpy-style computation
        for i in 0..64 {
            let s = 1.0f32 / (1.0 + (-gate[i]).exp());
            let expect = gate[i] * s * up[i];
            assert!((out[i] - expect).abs() < 1e-6, "swiglu[{i}]: got {} expected {}", out[i], expect);
        }
    }

    #[test]
    fn rms_norm_per_head_known() {
        // 2 heads, head_size=4
        let x = [1.0f32, -2.0, 3.0, -4.0,   2.0, 0.0, -1.0, 3.0];
        let w = [1.0f32, 1.0, 1.0, 1.0];
        let out = rms_norm_per_head(&x, &w, 4, 1e-6);
        // Head 0: sum_sq = 1+4+9+16=30, mean=7.5, scale=1/sqrt(7.5+eps)
        let s0 = 1.0f32 / (7.5f32 + 1e-6f32).sqrt();
        assert!((out[0] - 1.0 * s0).abs() < 1e-5);
        assert!((out[1] + 2.0 * s0).abs() < 1e-5);
        // Head 1: sum_sq = 4+0+1+9=14, mean=3.5, scale=1/sqrt(3.5+eps)
        let s1 = 1.0f32 / (3.5f32 + 1e-6f32).sqrt();
        assert!((out[4] - 2.0 * s1).abs() < 1e-5);
        assert!((out[6] + 1.0 * s1).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_per_head_weighted() {
        let x = [3.0f32, -4.0, 5.0, -6.0];
        let w = [2.0f32, 0.5, 1.0, 3.0];
        let out = rms_norm_per_head(&x, &w, 4, 1e-6);
        // sum_sq = 9+16+25+36=86, mean=21.5, scale=1/sqrt(21.5+eps)
        let s = 1.0f32 / (21.5f32 + 1e-6f32).sqrt();
        assert!((out[0] - 3.0 * s * 2.0).abs() < 1e-5);
        assert!((out[1] + 4.0 * s * 0.5).abs() < 1e-5);
        assert!((out[2] - 5.0 * s * 1.0).abs() < 1e-5);
        assert!((out[3] + 6.0 * s * 3.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_topk_basic() {
        let logits = [1.0f32, 2.0, 0.5, 3.0, -1.0];
        let (weights, indices) = softmax_topk(&logits, 2);
        // Top-2 should be indices 3 (3.0) and 1 (2.0)
        assert_eq!(indices[0], 3);
        assert_eq!(indices[1], 1);
        // Weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "weights sum: {sum}");
        // They should be proportional to softmax probs
        let max_l = 3.0f32;
        let e3 = (3.0 - max_l).exp();
        let e1 = (2.0 - max_l).exp();
        let total: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let expected_w0 = e3 / total;
        let expected_w1 = e1 / total;
        // After renormalization within top-2:
        let norm = expected_w0 + expected_w1;
        assert!((weights[0] - expected_w0 / norm).abs() < 1e-5);
        assert!((weights[1] - expected_w1 / norm).abs() < 1e-5);
    }

    #[test]
    fn softmax_topk_single() {
        let logits = [0.1f32, -0.5, 0.3, 0.2];
        let (weights, indices) = softmax_topk(&logits, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 2); // highest logit
        assert!((weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax_topk_all_equal() {
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let (weights, indices) = softmax_topk(&logits, 2);
        assert_eq!(indices.len(), 2);
        // All equal -> softmax gives 0.25 each, renormalized top-2 -> 0.5 each
        assert!((weights[0] - 0.5).abs() < 1e-5);
        assert!((weights[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn conv1d_silu_basic() {
        // 2 channels, kernel_size=3, seq_len=4
        let channels = 2;
        let kernel_size = 3;
        let seq_len = 4;
        let input: Vec<f32> = (0..channels * seq_len).map(|i| (i as f32 + 1.0) * 0.1).collect();
        // input: ch0=[0.1, 0.2, 0.3, 0.4], ch1=[0.5, 0.6, 0.7, 0.8]
        let kernel: Vec<f32> = (0..channels * kernel_size).map(|i| (i as f32 + 1.0) * 0.05).collect();
        // kernel: ch0=[0.05, 0.10, 0.15], ch1=[0.20, 0.25, 0.30]
        let state_in = vec![0.0f32; channels * (kernel_size - 1)];

        let (out, state_out) = conv1d_silu(&input, &kernel, &state_in, channels, seq_len, kernel_size);

        // ch0, t=0: acc = 0*0.05 + 0*0.10 + 0.1*0.15 = 0.015, silu(0.015) ≈ 0.015 * 0.50375
        let acc00 = 0.1f32 * 0.15f32;
        let expected00 = acc00 / (1.0f32 + (-acc00).exp());
        assert!((out[0] - expected00).abs() < 1e-5, "out[0]: got {} expected {}", out[0], expected00);

        // ch0, t=1: acc = 0*0.05 + 0.1*0.10 + 0.2*0.15 = 0.04
        let acc01 = 0.1f32 * 0.10f32 + 0.2f32 * 0.15f32;
        let expected01 = acc01 / (1.0f32 + (-acc01).exp());
        assert!((out[1] - expected01).abs() < 1e-5, "out[1]: got {} expected {}", out[1], expected01);

        // State update: ch0 state = [input[2], input[3]] = [0.3, 0.4]
        assert!((state_out[0] - 0.3).abs() < 1e-6);
        assert!((state_out[1] - 0.4).abs() < 1e-6);
        // ch1 state = [input[6], input[7]] = [0.7, 0.8]
        assert!((state_out[2] - 0.7).abs() < 1e-6);
        assert!((state_out[3] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn conv1d_silu_state_carry() {
        // Verify that state is properly used in the convolution
        let channels = 1;
        let kernel_size = 3;
        let seq_len = 2;
        let input = [0.1f32, 0.2];
        let kernel = [1.0f32, 1.0, 1.0]; // simple sum kernel
        let state_in = [0.3f32, 0.4]; // causal history

        let (out, state_out) = conv1d_silu(&input, &kernel, &state_in, channels, seq_len, kernel_size);

        // t=0: acc = 0.3*1 + 0.4*1 + 0.1*1 = 0.8, silu(0.8) = 0.8 / (1 + e^-0.8)
        let acc0 = 0.8f32;
        let expected0 = acc0 / (1.0 + (-acc0).exp());
        assert!((out[0] - expected0).abs() < 1e-5);

        // t=1: acc = 0.4*1 + 0.1*1 + 0.2*1 = 0.7, silu(0.7)
        let acc1 = 0.7f32;
        let expected1 = acc1 / (1.0 + (-acc1).exp());
        assert!((out[1] - expected1).abs() < 1e-5);

        // New state = [input[0], input[1]] = [0.1, 0.2]
        assert!((state_out[0] - 0.1).abs() < 1e-6);
        assert!((state_out[1] - 0.2).abs() < 1e-6);
    }
}
