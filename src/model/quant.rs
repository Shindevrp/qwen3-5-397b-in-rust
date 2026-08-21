//! Dequantization of GGUF tensor payloads into f32.
//!
//! Byte layouts and formulas are ported 1:1 from ggml's reference
//! `ggml-quants.c` / `ggml-common.h` (see `ref/` for the sources).
//! This model only uses F32, Q8_0, Q4_K, Q5_K, Q6_K, so those are the
//! only types implemented here; everything else is rejected.

use crate::gguf::GGmlType;
use std::sync::Arc;
use memmap2::Mmap;

pub const QK_K: usize = 256;

/// Raw quantized tensor — zero-copy reference into mmap'd GGUF data.
/// Holds an Arc<Mmap> so the underlying mapping stays alive.
#[derive(Clone)]
pub struct RawTensor {
    pub ty: GGmlType,
    pub n_elements: usize,
    mmap: Arc<Mmap>,
    pub offset: usize,
    pub len: usize,
}

impl RawTensor {
    /// Create a zero-copy reference into mmap'd GGUF data.
    pub fn from_mmap(mmap: Arc<Mmap>, ty: GGmlType, offset: usize, len: usize, n_elements: usize) -> Self {
        Self { ty, n_elements, mmap, offset, len }
    }

    /// Create an owned RawTensor from a byte vector (for tests / synthetic models).
    /// Copies data into a temp file and mmaps it to keep the API uniform.
    pub fn new(ty: GGmlType, data: Vec<u8>, n_elements: usize) -> Self {
        let len = data.len();
        let tmp = tempfile::tempfile().expect("create tempfile for RawTensor");
        use std::io::Write;
        (&tmp).write_all(&data).expect("write to tempfile");
        let mmap = Arc::new(unsafe { Mmap::map(&tmp).expect("mmap tempfile") });
        Self { ty, n_elements, mmap, offset: 0, len }
    }

    /// Access the raw quantized bytes.
    pub fn data(&self) -> &[u8] {
        &self.mmap[self.offset..self.offset + self.len]
    }

    /// Dequantize to f32 on demand.
    pub fn dequant(&self) -> Result<Vec<f32>, QuantError> {
        dequantize(self.ty, self.data(), self.n_elements as u64)
    }

    /// Row size in bytes for this tensor's quantization type and inner dimension.
    pub fn row_bytes(&self, n_in: usize) -> Result<usize, QuantError> {
        tensor_size(self.ty, n_in as u64).map(|b| b as usize)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum QuantError {
    #[error("unsupported quant type {0}")]
    UnsupportedType(String),
    #[error("tensor has {dims} elements, which is not a multiple of block size {block}")]
    NotMultiple { dims: u64, block: u64 },
    #[error("tensor needs {expected} bytes but data slice has {bytes}")]
    BadLength { expected: u64, bytes: usize },
}

/// Number of f32 elements decoded from one block.
pub const fn block_size(ty: GGmlType) -> Option<usize> {
    match ty {
        GGmlType::F32 => Some(1),
        GGmlType::Q8_0 => Some(32),
        GGmlType::Q4_K | GGmlType::Q5_K | GGmlType::Q6_K => Some(QK_K),
        _ => None,
    }
}

/// Size in bytes of one block on disk.
pub const fn block_bytes(ty: GGmlType) -> Option<usize> {
    match ty {
        GGmlType::F32 => Some(4),
        GGmlType::Q8_0 => Some(34),        // f16 d + 32 x i8
        GGmlType::Q4_K => Some(144),       // d, dmin (f16) + scales[12] + qs[128]
        GGmlType::Q5_K => Some(176),       // d, dmin (f16) + scales[12] + qh[32] + qs[128]
        GGmlType::Q6_K => Some(210),       // ql[128] + qh[64] + scales[16] + d (f16)
        _ => None,
    }
}

/// Bytes required to store `n` elements of `ty` on disk.
pub fn tensor_size(ty: GGmlType, n: u64) -> Result<u64, QuantError> {
    let (bs, bb) = match (block_size(ty), block_bytes(ty)) {
        (Some(bs), Some(bb)) => (bs as u64, bb as u64),
        _ => return Err(QuantError::UnsupportedType(ty.name())),
    };
    if !n.is_multiple_of(bs) {
        return Err(QuantError::NotMultiple { dims: n, block: bs });
    }
    Ok(n / bs * bb)
}

/// Dequantize a tensor's raw bytes into `n` f32 values (row-major, matching
/// the dequantize_row_* order in ggml-quants.c).
pub fn dequantize(ty: GGmlType, data: &[u8], n: u64) -> Result<Vec<f32>, QuantError> {
    let bs = block_size(ty).ok_or_else(|| QuantError::UnsupportedType(ty.name()))? as u64;
    if !n.is_multiple_of(bs) {
        return Err(QuantError::NotMultiple { dims: n, block: bs });
    }
    let expected = tensor_size(ty, n)?;
    if (data.len() as u64) < expected {
        return Err(QuantError::BadLength { expected, bytes: data.len() });
    }
    let nb = (n / bs) as usize;
    let mut out = Vec::with_capacity(n as usize);
    match ty {
        GGmlType::F32 => {
            for chunk in data[..(n as usize) * 4].chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        GGmlType::Q8_0 => {
            for b in data.chunks_exact(34).take(nb) {
                out.extend_from_slice(&dequantize_q8_0(b));
            }
        }
        GGmlType::Q4_K => {
            for b in data.chunks_exact(144).take(nb) {
                out.extend_from_slice(&dequantize_q4_k(b));
            }
        }
        GGmlType::Q5_K => {
            for b in data.chunks_exact(176).take(nb) {
                out.extend_from_slice(&dequantize_q5_k(b));
            }
        }
        GGmlType::Q6_K => {
            for b in data.chunks_exact(210).take(nb) {
                out.extend_from_slice(&dequantize_q6_k(b));
            }
        }
        other => return Err(QuantError::UnsupportedType(other.name())),
    }
    Ok(out)
}

// ---- fp16 helpers -----------------------------------------------------------

/// IEEE 754 half-precision -> f32.
pub fn fp16_to_f32(h: u16) -> f32 {
    let sign = (u32::from(h) >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let frac = h & 0x03ff;
    let bits = match exp {
        0 => {
            if frac == 0 {
                sign << 31
            } else {
                // normalize the subnormal mantissa
                let mut f = u32::from(frac);
                let mut e = 127 - 15 + 1;
                while f & 0x0400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                (sign << 31) | (e << 23) | ((f & 0x03ff) << 13)
            }
        }
        0x1f => {
            if frac == 0 {
                (sign << 31) | 0x7f80_0000 // +/- inf
            } else {
                (sign << 31) | 0x7fc0_0000 | (u32::from(frac) << 13) // NaN
            }
        }
        e => {
            let e = i32::from(e) + (127 - 15);
            (sign << 31) | ((e as u32) << 23) | (u32::from(frac) << 13)
        }
    };
    f32::from_bits(bits)
}

/// f32 -> IEEE 754 half-precision (round-to-nearest-even).
pub fn f32_to_fp16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // inf / NaN
        let inf = sign | 0x7c00;
        if frac == 0 {
            return inf;
        }
        // NaN: keep a quiet bit so the result stays NaN
        let payload = (frac >> 13) as u16;
        return inf | 0x0200 | if payload != 0 { payload } else { 1 };
    }

    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        // subnormal or zero
        if e < -10 {
            return sign; // rounds to zero
        }
        let m = frac | 0x0080_0000; // 24-bit mantissa with implicit bit
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let half_up = rem > (1u32 << (shift - 1))
            || (rem == (1u32 << (shift - 1)) && (half & 1) == 1);
        return sign | if half_up { half + 1 } else { half };
    }

    let half = (((e as u32) << 10) | (frac >> 13)) as u16;
    let rem = frac & 0x1fff;
    let half_up = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    sign | if half_up { half + 1 } else { half }
}

// ---- per-type dequant (ported from ggml-quants.c) ---------------------------

fn dequantize_q8_0(block: &[u8]) -> [f32; 32] {
    debug_assert_eq!(block.len(), 34);
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut out = [0f32; 32];
    for (i, v) in out.iter_mut().enumerate() {
        *v = (block[2 + i] as i8) as f32 * d;
    }
    out
}

/// Q4_K: x = d*sc*(nibble) - dmin*m  over 8 groups of 32.
fn dequantize_q4_k(block: &[u8]) -> [f32; QK_K] {
    debug_assert_eq!(block.len(), 144);
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    let mut out = [0f32; QK_K];
    let mut is = 0;
    for chunk in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = dmin * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = dmin * f32::from(m);
        let q = &qs[chunk * 32..chunk * 32 + 32];
        for l in 0..32 {
            out[chunk * 64 + l] = d1 * f32::from(q[l] & 0x0F) - m1;
            out[chunk * 64 + 32 + l] = d2 * f32::from(q[l] >> 4) - m2;
        }
        is += 2;
    }
    out
}

/// Q5_K: like Q4_K but with a 5th bit stored per 64-element group in `qh`.
fn dequantize_q5_k(block: &[u8]) -> [f32; QK_K] {
    debug_assert_eq!(block.len(), 176);
    let d = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = fp16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut out = [0f32; QK_K];
    let mut is = 0;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    for chunk in 0..4 {
        let (sc, m) = get_scale_min_k4(is, scales);
        let d1 = d * f32::from(sc);
        let m1 = dmin * f32::from(m);
        let (sc, m) = get_scale_min_k4(is + 1, scales);
        let d2 = d * f32::from(sc);
        let m2 = dmin * f32::from(m);
        let q = &qs[chunk * 32..chunk * 32 + 32];
        for l in 0..32 {
            let h1 = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
            let h2 = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
            out[chunk * 64 + l] = d1 * (f32::from(q[l] & 0x0F) + h1) - m1;
            out[chunk * 64 + 32 + l] = d2 * (f32::from(q[l] >> 4) + h2) - m2;
        }
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
    out
}

/// Q6_K: x = d*sc*(6-bit quant - 32).
fn dequantize_q6_k(block: &[u8]) -> [f32; QK_K] {
    debug_assert_eq!(block.len(), 210);
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];
    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));

    let mut out = [0f32; QK_K];
    for n in 0..2 {
        let ql = &ql[n * 64..n * 64 + 64];
        let qh = &qh[n * 32..n * 32 + 32];
        let sc = &sc[n * 8..n * 8 + 8];
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from(ql[l] & 0x0F) | (i32::from(qh[l] & 0x03) << 4);
            let q2 = i32::from(ql[l + 32] & 0x0F) | (i32::from((qh[l] >> 2) & 0x03) << 4);
            let q3 = i32::from(ql[l] >> 4) | (i32::from((qh[l] >> 4) & 0x03) << 4);
            let q4 = i32::from(ql[l + 32] >> 4) | (i32::from((qh[l] >> 6) & 0x03) << 4);
            // Q6_K scales are SIGNED int8 in ggml (block_q6_K.scales), unlike
            // the 6-bit unsigned scales of Q4_K/Q5_K.
            out[n * 128 + l] = d * f32::from(sc[is] as i8) * (q1 - 32) as f32;
            out[n * 128 + l + 32] = d * f32::from(sc[is + 2] as i8) * (q2 - 32) as f32;
            out[n * 128 + l + 64] = d * f32::from(sc[is + 4] as i8) * (q3 - 32) as f32;
            out[n * 128 + l + 96] = d * f32::from(sc[is + 6] as i8) * (q4 - 32) as f32;
        }
    }
    out
}

/// Decode the 6-bit scale and min for group `j` (0..8) of a Q4_K/Q5_K block.
#[inline]
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    debug_assert!(j < 8);
    debug_assert!(scales.len() >= 12);
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn le16(v: f32) -> [u8; 2] {
        f32_to_fp16(v).to_le_bytes()
    }

    /// Mirror of the Q4_K/Q5_K scale packing in `quantize_row_q4_K_ref`:
    /// group j<4 stores 6-bit ls/lm in bytes j and j+4; group j>=4 packs the
    /// low 4 bits into byte j+4 and the high 2 bits into bytes j-4 and j.
    fn pack_scales(groups: [(u8, u8); 8]) -> [u8; 12] {
        let mut scales = [0u8; 12];
        for (j, &(ls, lm)) in groups.iter().enumerate() {
            if j < 4 {
                scales[j] = ls;
                scales[j + 4] = lm;
            } else {
                scales[j + 4] = (ls & 0x0F) | ((lm & 0x0F) << 4);
                scales[j - 4] |= (ls >> 4) << 6;
                scales[j] |= (lm >> 4) << 6;
            }
        }
        scales
    }

    #[test]
    fn fp16_round_trip() {
        for v in [0.0f32, -0.0, 1.0, -1.0, 0.5, 2.0, -2.5, 65504.0, 0.00006097555] {
            let h = f32_to_fp16(v);
            let back = fp16_to_f32(h);
            assert_eq!(back, v, "round trip of {v} via 0x{h:04x}");
        }
    }

    #[test]
    fn fp16_special_values() {
        assert_eq!(fp16_to_f32(0x0001), 2.0_f32.powi(-24));
        assert_eq!(fp16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(fp16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(fp16_to_f32(0x7e00).is_nan());
        assert_eq!(fp16_to_f32(0x3c00), 1.0);
        assert_eq!(fp16_to_f32(0xc000), -2.0);
    }

    #[test]
    fn q8_0_block() {
        // d = 0.5, quants -16..16
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&le16(0.5));
        for (i, b) in block[2..].iter_mut().enumerate() {
            *b = (i as i8 - 16) as u8;
        }
        let out = dequantize_q8_0(&block);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, (i as f32 - 16.0) * 0.5);
        }
    }

    #[test]
    fn q8_0_negative_scale() {
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&le16(-3.0));
        block[2] = 7u8;
        block[3] = 255u8; // -1
        let out = dequantize_q8_0(&block);
        assert_eq!(out[0], -21.0);
        assert_eq!(out[1], 3.0);
    }

    #[test]
    fn q4_k_identity_scale() {
        // d = 1, dmin = 2, all group scales = 1, all mins = 1, nibbles 0..15
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&le16(1.0));
        block[2..4].copy_from_slice(&le16(2.0));
        block[4..16].copy_from_slice(&pack_scales([(1, 1); 8]));
        let qs = &mut block[16..144];
        for (i, b) in qs.iter_mut().enumerate() {
            // group g (0..8), position p (0..31): nibble = p % 16
            let nib = (i % 32 % 16) as u8;
            *b = nib | (nib << 4);
        }
        let out = dequantize_q4_k(&block);
        for g in 0..8 {
            for p in 0..32 {
                let nib = p % 16;
                let expected = 1.0 * 1.0 * nib as f32 - 2.0 * 1.0;
                assert_eq!(out[g * 32 + p], expected, "group {g} pos {p}");
            }
        }
    }

    #[test]
    fn q4_k_scale_variation() {
        // d = 0.25, dmin = 4; group0 scale=4 (d1=1.0), group1 scale=2 (d2=0.5)
        // mins all 1 (m1=m2=4)
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&le16(0.25));
        block[2..4].copy_from_slice(&le16(4.0));
        let mut groups = [(1u8, 1u8); 8];
        groups[0].0 = 4;
        groups[1].0 = 2;
        block[4..16].copy_from_slice(&pack_scales(groups));
        let qs = &mut block[16..144];
        for b in qs.iter_mut() {
            *b = 0xAA; // nibble 10 low, 10 high
        }
        let out = dequantize_q4_k(&block);
        // group 0: d1=1.0, m1=4 -> 1*10 - 4 = 6
        assert!(out[0..32].iter().all(|&v| v == 6.0));
        // group 1: d2=0.5, m2=4 -> 0.5*10 - 4 = 1
        assert!(out[32..64].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn q5_k_high_bit() {
        // d=1, dmin=2, scales/mins all 1; qs nibbles 0; qh toggles high bit
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&le16(1.0));
        block[2..4].copy_from_slice(&le16(2.0));
        block[4..16].copy_from_slice(&pack_scales([(1, 1); 8]));
        // qh: byte l holds bits for the 4 groups: bits 0,1 group0; 2,3 group1; ...
        // group0 spans all 32 qh bytes, so set every byte to bits 0|1
        for b in block[16..48].iter_mut() {
            *b = 0b0000_0011; // group0: low nibble +16, high nibble +16
        }
        let out = dequantize_q5_k(&block);
        // group 0, low: 1*(0+16)-2 = 14 ; high: 1*(0+16)-2 = 14
        assert!(out[0..64].iter().all(|&v| v == 14.0));
        // groups 1..3: no high bit -> 1*0 - 2 = -2
        for g in 1..4 {
            assert!(out[g * 64..g * 64 + 64].iter().all(|&v| v == -2.0));
        }
    }

    #[test]
    fn q6_k_quant_values() {
        // d = 3, all scales = 2, ql/qh = 0 -> all quants -32 -> -192
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&le16(3.0));
        for b in block[192..208].iter_mut() {
            *b = 2;
        }
        let out = dequantize_q6_k(&block);
        assert!(out.iter().all(|&v| v == -192.0));

        // ql[0] = ql[32] = 0x5 and qh[0] = 0xFF
        block[0] = 0x05;
        block[32] = 0x05;
        block[128] = 0xff;
        let out = dequantize_q6_k(&block);
        // q1 = (5 | (0xff&3)<<4) = 5 | 48 = 53 -> 21
        assert_eq!(out[0], 3.0 * 2.0 * 21.0);
        // q2 = (ql[32]&0xF=5 | (0xff>>2 &3)<<4) = 53 -> 21
        assert_eq!(out[32], 3.0 * 2.0 * 21.0);
        // q3 = (ql[0]>>4=0 | (0xff>>4 &3)<<4) = 48 -> 16
        assert_eq!(out[64], 3.0 * 2.0 * 16.0);
        // q4 = (ql[32]>>4=0 | (0xff>>6 &3)<<4) = 48 -> 16
        assert_eq!(out[96], 3.0 * 2.0 * 16.0);

        // Scales are SIGNED int8: byte 0x81 = -127, not +129.
        let mut block = vec![0u8; 210];
        block[208..210].copy_from_slice(&le16(1.0));
        for b in block[192..208].iter_mut() {
            *b = 0x81;
        }
        let out = dequantize_q6_k(&block);
        assert!(out.iter().all(|&v| v == -127.0 * -32.0));
    }

    #[test]
    fn dequantize_f32() {
        let data = 1.0f32.to_le_bytes()
            .into_iter()
            .chain((-2.0f32).to_le_bytes())
            .collect::<Vec<_>>();
        let out = dequantize(GGmlType::F32, &data, 2).unwrap();
        assert_eq!(out, vec![1.0, -2.0]);
    }

    #[test]
    fn tensor_size_matches_quantized() {
        assert_eq!(tensor_size(GGmlType::Q8_0, 32).unwrap(), 34);
        assert_eq!(tensor_size(GGmlType::Q4_K, 256).unwrap(), 144);
        assert_eq!(tensor_size(GGmlType::Q5_K, 256).unwrap(), 176);
        assert_eq!(tensor_size(GGmlType::Q6_K, 256).unwrap(), 210);
        assert_eq!(tensor_size(GGmlType::F32, 4).unwrap(), 16);
        assert!(tensor_size(GGmlType::Q8_0, 33).is_err());
        assert!(tensor_size(GGmlType::F16, 1).is_err());
    }
}
