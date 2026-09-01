"""Generate synthetic kernel inputs and independent numpy reference outputs for
the Rust kernels (rms_norm, quantized gemv, IMRoPE rope_multi).

Weight blocks are crafted with structured d/dmin/scales + random payload and
dequantized with independent numpy ports of the ggml dequantize_row_* functions.
Activations are quantized with numpy ports of quantize_row_q8_0_ref /
quantize_row_q8_K_ref (which mirror the Rust quantizers). The reference gemv is
the exact quantized computation: dot(dequant(w_row), dequant(xq)).

Run `cargo run --release --bin kern_check` on the same directory afterwards,
then `check_kern.py` compares ref_* vs rust_* outputs.
"""

import os
import struct
import math

import numpy as np

np.set_printoptions(suppress=True, linewidth=200)

QK8_0 = 32
QK_K = 256
OUT = "/tmp/opencode/kern_check"


def fp16(f):
    return struct.pack("<e", np.float32(f))


def ref_f16(h):
    return float(np.frombuffer(struct.pack("<H", h), dtype=np.float16)[0])


def fp16_bytes(v):
    return struct.unpack("<H", fp16(v))[0]


def round_half_away(x):
    """C roundf: round half away from zero."""
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


# ---- activation quantizers (mirror Rust quantize_row_q8_0 / q8_k) ----------

def quantize_q8_0(x):
    """quantize_row_q8_0_ref: d = amax/127 (fp16), qs = roundf(x*id)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    out = bytearray()
    for i in range(0, x.size, QK8_0):
        block = x[i:i + QK8_0]
        amax = np.abs(block).max()
        d = np.float32(amax) / np.float32(127)
        id_ = np.float32(1) / d if d != 0 else np.float32(0)
        q = np.where(block * id_ >= 0, np.floor(block * id_ + 0.5), np.ceil(block * id_ - 0.5))
        out += fp16(float(d))
        out += bytes(struct.pack("<32b", *q.astype(np.int8).tolist()))
    return bytes(out)


def quantize_q8_k(x):
    """quantize_row_q8_K_ref: iscale = -127/max, qs = min(127, nearest_int),
    bsums = sums of 16, d = 1/iscale (fp16). nearest_int rounds half-to-even."""
    x = np.asarray(x, dtype=np.float32).ravel()
    out = bytearray()
    for i in range(0, x.size, QK_K):
        block = x[i:i + QK_K]
        ax = np.abs(block)
        idx = int(np.argmax(ax))
        amax = float(ax[idx])
        if amax == 0.0:
            out += b"\x00\x00" + b"\x00" * QK_K + b"\x00" * 16
            continue
        mx = float(block[idx])
        iscale = np.float32(-127.0) / np.float32(mx)
        v = np.round(iscale * block)
        qs = np.minimum(127, v).astype(np.int8)
        d = np.float32(1.0) / iscale
        out += fp16(float(d))
        out += bytes(struct.pack("<256b", *qs.tolist()))
        bsums = qs.reshape(-1, 16).sum(axis=1).astype(np.int16)
        out += bytes(struct.pack("<16h", *bsums.tolist()))
    return bytes(out)


def dequant_q8_0(data, n):
    vals = np.empty(n, dtype=np.float64)
    for i in range(0, n, QK8_0):
        off = i // QK8_0 * 34
        d = ref_f16(struct.unpack_from("<H", data, off)[0])
        qs = np.frombuffer(data, dtype=np.int8, count=QK8_0, offset=off + 2)
        vals[i:i + QK8_0] = d * qs.astype(np.float64)
    return vals


def dequant_q8_k(data, n):
    vals = np.empty(n, dtype=np.float64)
    for i in range(0, n, QK_K):
        off = i // QK_K * 290
        d = ref_f16(struct.unpack_from("<H", data, off)[0])
        qs = np.frombuffer(data, dtype=np.int8, count=QK_K, offset=off + 2)
        vals[i:i + QK_K] = d * qs.astype(np.float64)
    return vals


# ---- weight dequant ports (ggml dequantize_row_*) --------------------------

def get_scale_min_k4(scales, j):
    if j < 4:
        return scales[j] & 63, scales[j + 4] & 63
    return (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4), \
           (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)


def dequant_q4_k(data, n):
    vals = np.empty(n, dtype=np.float64)
    for b in range(0, n, QK_K):
        off = b // QK_K * 144
        d = ref_f16(struct.unpack_from("<H", data, off)[0])
        dmin = ref_f16(struct.unpack_from("<H", data, off + 2)[0])
        scales = list(data[off + 4:off + 16])
        qs = data[off + 16:off + 144]
        is_ = 0
        for chunk in range(4):
            sc, m = get_scale_min_k4(scales, is_)
            d1 = d * sc
            m1 = dmin * m
            sc, m = get_scale_min_k4(scales, is_ + 1)
            d2 = d * sc
            m2 = dmin * m
            for l in range(32):
                vals[b + chunk * 64 + l] = d1 * (qs[chunk * 32 + l] & 0x0F) - m1
                vals[b + chunk * 64 + 32 + l] = d2 * (qs[chunk * 32 + l] >> 4) - m2
            is_ += 2
    return vals


def dequant_q5_k(data, n):
    vals = np.empty(n, dtype=np.float64)
    for b in range(0, n, QK_K):
        off = b // QK_K * 176
        d = ref_f16(struct.unpack_from("<H", data, off)[0])
        dmin = ref_f16(struct.unpack_from("<H", data, off + 2)[0])
        scales = list(data[off + 4:off + 16])
        qh = data[off + 16:off + 48]
        qs = data[off + 48:off + 176]
        is_ = 0
        u1, u2 = 1, 2
        for chunk in range(4):
            sc, m = get_scale_min_k4(scales, is_)
            d1 = d * sc
            m1 = dmin * m
            sc, m = get_scale_min_k4(scales, is_ + 1)
            d2 = d * sc
            m2 = dmin * m
            for l in range(32):
                h1 = 16 if qh[l] & u1 else 0
                h2 = 16 if qh[l] & u2 else 0
                vals[b + chunk * 64 + l] = d1 * ((qs[chunk * 32 + l] & 0x0F) + h1) - m1
                vals[b + chunk * 64 + 32 + l] = d2 * ((qs[chunk * 32 + l] >> 4) + h2) - m2
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return vals


def dequant_q6_k(data, n):
    vals = np.empty(n, dtype=np.float64)
    for b in range(0, n, QK_K):
        off = b // QK_K * 210
        d = ref_f16(struct.unpack_from("<H", data, off + 208)[0])
        ql = data[off:off + 128]
        qh = data[off + 128:off + 192]
        sc = list(data[off + 192:off + 208])
        for n2 in range(2):
            ql2 = ql[n2 * 64:(n2 + 1) * 64]
            qh2 = qh[n2 * 32:(n2 + 1) * 32]
            sc2 = sc[n2 * 8:(n2 + 1) * 8]
            for l in range(32):
                is_ = l // 16
                q1 = (ql2[l] & 0x0F) | ((qh2[l] & 0x03) << 4)
                q2 = (ql2[l + 32] & 0x0F) | ((qh2[l] >> 2) & 0x03) << 4
                q3 = (ql2[l] >> 4) | ((qh2[l] >> 4) & 0x03) << 4
                q4 = (ql2[l + 32] >> 4) | ((qh2[l] >> 6) & 0x03) << 4
                vals[b + n2 * 128 + l] = d * sc2[is_] * (q1 - 32)
                vals[b + n2 * 128 + 32 + l] = d * sc2[is_ + 2] * (q2 - 32)
                vals[b + n2 * 128 + 64 + l] = d * sc2[is_ + 4] * (q3 - 32)
                vals[b + n2 * 128 + 96 + l] = d * sc2[is_ + 6] * (q4 - 32)
    return vals


# ---- weight crafting --------------------------------------------------------

def craft_q8_0(nblocks, rng):
    raw = bytearray()
    for blk in range(nblocks):
        d = np.float32(0.125 * (blk % 8 + 1))
        q = rng.integers(-127, 128, QK8_0, dtype=np.int32).astype(np.int8)
        raw += fp16(float(d))
        raw += bytes(struct.pack("<32b", *q.tolist()))
    return bytes(raw)


def craft_q4_k(nblocks, rng):
    raw = bytearray()
    for blk in range(nblocks):
        d = np.float32(0.25 * (blk % 4 + 1))
        dmin = np.float32(1.5 * (blk % 3 + 1))
        raw += fp16(float(d))
        raw += fp16(float(dmin))
        ls = rng.integers(0, 16, 8, dtype=np.uint8)
        lm = rng.integers(0, 16, 8, dtype=np.uint8)
        scales = [0] * 12
        for j in range(8):
            if j < 4:
                scales[j] = int(ls[j])
                scales[j + 4] = int(lm[j])
            else:
                scales[j + 4] = (int(ls[j]) & 0x0F) | ((int(lm[j]) & 0x0F) << 4)
                scales[j - 4] |= (int(ls[j]) >> 4) << 6
                scales[j] |= (int(lm[j]) >> 4) << 6
        raw += bytes(scales)
        raw += bytes(rng.integers(0, 256, 128, dtype=np.uint8).tolist())
    return bytes(raw)


def craft_q5_k(nblocks, rng):
    raw = bytearray()
    for blk in range(nblocks):
        d = np.float32(0.5 * (blk % 4 + 1))
        dmin = np.float32(3.0 * (blk % 2 + 1))
        raw += fp16(float(d))
        raw += fp16(float(dmin))
        ls = rng.integers(0, 16, 8, dtype=np.uint8)
        lm = rng.integers(0, 16, 8, dtype=np.uint8)
        scales = [0] * 12
        for j in range(8):
            if j < 4:
                scales[j] = int(ls[j])
                scales[j + 4] = int(lm[j])
            else:
                scales[j + 4] = (int(ls[j]) & 0x0F) | ((int(lm[j]) & 0x0F) << 4)
                scales[j - 4] |= (int(ls[j]) >> 4) << 6
                scales[j] |= (int(lm[j]) >> 4) << 6
        raw += bytes(scales)
        raw += bytes(rng.integers(0, 256, 32, dtype=np.uint8).tolist())
        raw += bytes(rng.integers(0, 256, 128, dtype=np.uint8).tolist())
    return bytes(raw)


def craft_q6_k(nblocks, rng):
    raw = bytearray()
    for blk in range(nblocks):
        d = np.float32(1.0 / (blk % 8 + 1))
        raw += bytes(rng.integers(0, 256, 128, dtype=np.uint8).tolist())
        raw += bytes(rng.integers(0, 256, 64, dtype=np.uint8).tolist())
        raw += bytes(rng.integers(0, 64, 16, dtype=np.uint8).tolist())
        raw += fp16(float(d))
    return bytes(raw)


DEQUANT = {
    "Q8_0": dequant_q8_0,
    "Q4_K": dequant_q4_k,
    "Q5_K": dequant_q5_k,
    "Q6_K": dequant_q6_k,
}
ROW_BYTES = {
    "Q8_0": 34 * QK8_0,
    "Q4_K": 144 * (QK_K // QK_K),
    "Q5_K": 176,
    "Q6_K": 210,
}
NBLOCKS = {
    "Q8_0": QK8_0,
    "Q4_K": QK_K,
    "Q5_K": QK_K,
    "Q6_K": QK_K,
}


# ---- numpy references for the kernels --------------------------------------

def ref_rms_norm(x, w, eps):
    x = np.asarray(x, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    sum_sq = np.float64(0.0)
    for v in x:
        sum_sq += float(v) * float(v)
    mean = np.float32(sum_sq / x.size)
    scale = np.float32(1.0) / np.sqrt(np.float32(mean) + np.float32(eps))
    return x.astype(np.float64) * float(scale) * w.astype(np.float64)


def ref_rope(x, pos, n_dims, sections, n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor):
    """Port of kernels::rope_multi_imrope (partial IMRoPE)."""
    x = np.asarray(x, dtype=np.float64)
    ne0 = x.size
    theta_scale = freq_base ** (-2.0 / n_dims)
    beta_fast, beta_slow = 32.0, 1.0
    corr = [math.floor(n_ctx_orig / beta_fast), math.floor(n_ctx_orig / beta_slow)]
    sect_dims = sum(sections)
    theta_t, theta_h, theta_w, theta_e = [float(p) for p in pos]

    cache = np.zeros(ne0, dtype=np.float64)
    i0 = 0
    while i0 < ne0:
        sector = (i0 // 2) % sect_dims
        if sector % 3 == 1 and sector < 3 * sections[1]:
            theta = theta_h
        elif sector % 3 == 2 and sector < 3 * sections[2]:
            theta = theta_w
        elif sector % 3 == 0 and sector < 3 * sections[0]:
            theta = theta_t
        else:
            theta = theta_e
        theta_interp = freq_scale * theta
        theta_f = theta_interp
        mscale = attn_factor
        if ext_factor != 0.0:
            y = (i0 / 2.0 - corr[0]) / max(0.001, corr[1] - corr[0])
            ramp = 1.0 - min(1.0, max(0.0, y))
            mix = ramp * ext_factor
            theta_f = theta_interp * (1.0 - mix) + theta * mix
            mscale *= 1.0 + 0.1 * math.log(1.0 / freq_scale)
        cache[i0] = math.cos(theta_f) * mscale
        cache[i0 + 1] = math.sin(theta_f) * mscale
        theta_t *= theta_scale
        theta_w *= theta_scale
        theta_h *= theta_scale
        theta_e *= theta_scale
        i0 += 2

    out = x.copy()
    n_offset = n_dims // 2
    i0 = 0
    while i0 < n_dims:
        ic = i0 // 2
        x0, x1 = x[ic], x[ic + n_offset]
        c, s = cache[i0], cache[i0 + 1]
        out[ic] = x0 * c - x1 * s
        out[ic + n_offset] = x0 * s + x1 * c
        i0 += 2
    return out


def ref_attention(q, k, v, n_heads, n_kv_heads, head_dim, n_q, n_kv, scale, causal):
    """Numpy reference for scaled dot-product attention (ggml layout)."""
    out = np.zeros((n_q, n_heads, head_dim), dtype=np.float64)
    gqa = n_heads // n_kv_heads
    for qt in range(n_q):
        for qh in range(n_heads):
            kv_h = qh // gqa
            scores = np.zeros(n_kv, dtype=np.float64)
            # For decode (n_q == 1), the KV cache only contains valid past
            # positions, so attend to all of them regardless of causal flag.
            max_kv = n_kv - 1 if (not causal or n_q == 1) else min(qt, n_kv - 1)
            for t in range(max_kv + 1):
                dot = np.dot(q[qt, qh, :], k[t, kv_h, :])
                scores[t] = dot * scale
            if causal and max_kv + 1 < n_kv:
                scores[max_kv + 1:] = -np.inf
            mx = scores.max()
            ex = np.exp(scores - mx)
            probs = ex / ex.sum()
            for t in range(n_kv):
                out[qt, qh, :] += probs[t] * v[t, kv_h, :]
    return out.ravel(order="C")


def ref_delta_net(q, k, v, g, beta, state, s_k, s_v, eps):
    """Numpy reference for delta-net autoregressive (GDA mode).

    In ggml, mul_mat(A, B) computes A^T @ B.  So:
      mul_mat(state_t, k) = state_t^T @ k = state @ k
    The C code transposes state then passes to mul_mat, yielding state @ k
    (not state^T @ k).
    """
    h = q.shape[0]
    out = np.zeros((h, s_v), dtype=np.float64)
    state = state.astype(np.float64).copy()
    scale = 1.0 / (s_v ** 0.5)
    for hi in range(h):
        qh = q[hi].astype(np.float64)
        kh = k[hi].astype(np.float64)
        vh = v[hi].astype(np.float64)
        # L2 normalize
        qn = np.linalg.norm(qh) + eps
        kn = np.linalg.norm(kh) + eps
        qh_n = qh / qn * scale
        kh_n = kh / kn
        b = 1.0 / (1.0 + np.exp(-beta[hi]))
        # State decay
        state[hi] *= np.exp(g[hi])
        # k_state = state @ k  (ggml mul_mat(state_t, k) = state_t^T @ k = state @ k)
        kst = state[hi] @ kh_n
        # v_diff
        vd = vh - kst
        # State update: state += outer(vd, kh * b)
        state[hi] += np.outer(vd, kh_n * b)
        # Output = state @ q
        out[hi] = state[hi] @ qh_n
    return out.ravel(order="C"), state


# ---- Phase 3a references ----------------------------------------------------

def ref_swiglu(gate, up):
    """SwiGLU: silu(gate) * up."""
    gate = np.asarray(gate, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    s = 1.0 / (1.0 + np.exp(-gate))
    return (gate * s * up).astype(np.float32)


def ref_rms_norm_per_head(x, w, head_size, eps):
    """RMS norm applied independently per head."""
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    n_heads = x.size // head_size
    out = np.empty_like(x, dtype=np.float64)
    for h in range(n_heads):
        base = h * head_size
        chunk = x[base:base + head_size]
        ss = float(np.sum(chunk ** 2))
        mean = ss / head_size
        scale = 1.0 / np.sqrt(mean + eps)
        out[base:base + head_size] = chunk * scale * w
    return out.astype(np.float32)


def ref_softmax_topk(logits, k):
    """Softmax + top-k + renormalize. Returns (weights, indices)."""
    logits = np.asarray(logits, dtype=np.float64).ravel()
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    # argpartition for top-k
    idx = np.argpartition(-probs, k)[:k]
    # sort within top-k by descending prob
    idx = idx[np.argsort(-probs[idx])]
    weights = probs[idx] / probs[idx].sum()
    return weights.astype(np.float32), idx.astype(int)


def ref_conv1d_silu(input_arr, kernel, state_in, channels, seq_len, kernel_size):
    """Causal conv1d + SiLU over [state | input]."""
    input_arr = np.asarray(input_arr, dtype=np.float64).ravel()
    kernel = np.asarray(kernel, dtype=np.float64).ravel()
    state_in = np.asarray(state_in, dtype=np.float64).ravel()
    pad = kernel_size - 1
    out = np.empty(channels * seq_len, dtype=np.float64)
    for c in range(channels):
        for t in range(seq_len):
            acc = 0.0
            for k in range(kernel_size):
                src = t + k - pad
                if src < 0:
                    val = state_in[c * pad + src + pad]
                else:
                    val = input_arr[c * seq_len + src]
                acc += val * kernel[c * kernel_size + k]
            # SiLU
            s = 1.0 / (1.0 + np.exp(-acc))
            out[c * seq_len + t] = acc * s
    state_out = np.empty(channels * pad, dtype=np.float64)
    for c in range(channels):
        state_out[c*pad:(c+1)*pad] = input_arr[c*seq_len + seq_len - pad : c*seq_len + seq_len]
    return out.astype(np.float32), state_out.astype(np.float32)


def ref_moe_ffn(input_arr, router_w, gate_up_w, down_w,
                n_embd, n_ff, n_expert, n_expert_used, n_tokens):
    """Numpy reference for MoE FFN: router → softmax → top-k → expert SwiGLU → weighted sum."""
    input_arr = np.asarray(input_arr, dtype=np.float64).reshape(n_tokens, n_embd)
    router_w = np.asarray(router_w, dtype=np.float64).reshape(n_expert, n_embd)
    gate_up_w = np.asarray(gate_up_w, dtype=np.float64).reshape(n_expert, 2 * n_ff, n_embd)
    down_w = np.asarray(down_w, dtype=np.float64).reshape(n_expert, n_embd, n_ff)
    output = np.zeros((n_tokens, n_embd), dtype=np.float64)
    for t in range(n_tokens):
        x = input_arr[t]  # [n_embd]
        logits = router_w @ x  # [n_expert]
        # Softmax
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        # Top-k
        idx = np.argsort(-probs)[:n_expert_used]
        w = probs[idx]
        w = w / w.sum()
        for slot, e in enumerate(idx):
            gate_up = gate_up_w[e] @ x  # [2*n_ff]
            gate = gate_up[:n_ff]
            up = gate_up[n_ff:]
            # SwiGLU
            s = 1.0 / (1.0 + np.exp(-gate))
            ff_out = gate * s * up  # [n_ff]
            # Down
            expert_out = down_w[e] @ ff_out  # [n_embd]
            output[t] += expert_out * w[slot]
    return output.ravel(order="C").astype(np.float32)


def ref_full_layer_forward(
    input_arr, attn_norm_w, wq, wk, wv, wo, q_norm_w, k_norm_w,
    post_norm_w, ffn_gate_w, ffn_up_w, ffn_down_w,
    pos, n_embd, n_heads, n_kv_heads, head_size, n_ff, n_tokens,
    eps=1e-6, rope_sections=(11, 11, 10, 0), freq_base=1e7,
):
    """Numpy reference for full single-layer forward pass."""
    input_arr = np.asarray(input_arr, dtype=np.float64).reshape(n_tokens, n_embd)
    attn_norm_w = np.asarray(attn_norm_w, dtype=np.float64)
    wq = np.asarray(wq, dtype=np.float64).reshape(2 * n_heads * head_size, n_embd)
    wk = np.asarray(wk, dtype=np.float64).reshape(n_kv_heads * head_size, n_embd)
    wv = np.asarray(wv, dtype=np.float64).reshape(n_kv_heads * head_size, n_embd)
    wo = np.asarray(wo, dtype=np.float64).reshape(n_embd, n_heads * head_size)
    q_norm_w = np.asarray(q_norm_w, dtype=np.float64)
    k_norm_w = np.asarray(k_norm_w, dtype=np.float64)
    post_norm_w = np.asarray(post_norm_w, dtype=np.float64)
    ffn_gate_w = np.asarray(ffn_gate_w, dtype=np.float64).reshape(n_ff, n_embd)
    ffn_up_w = np.asarray(ffn_up_w, dtype=np.float64).reshape(n_ff, n_embd)
    ffn_down_w = np.asarray(ffn_down_w, dtype=np.float64).reshape(n_embd, n_ff)

    def rms_norm_ref(x, w, eps):
        var = np.mean(x ** 2) + eps
        return x / np.sqrt(var) * w

    def rms_norm_per_head_ref(x, w, head_size, eps):
        """x: [total_features] where total_features = n_heads * head_size"""
        out = np.zeros_like(x)
        n_heads = len(x) // head_size
        for h in range(n_heads):
            chunk = x[h * head_size:(h + 1) * head_size]
            var = np.mean(chunk ** 2) + eps
            out[h * head_size:(h + 1) * head_size] = chunk / np.sqrt(var) * w
        return out

    def rope_partial_imrope(x_2d, positions, n_rot, rope_sections, context_len=4096, freq_base=1e7, freq_scale=1.0, attn_factor=1.0):
        """x_2d: [n_tokens, dim], interleaved pair layout matching Rust rope_multi_imrope."""
        x = x_2d.copy().astype(np.float64)
        n_tokens, n_dims = x.shape
        n_full_pairs = n_rot // 2
        n_pairs_to_process = min(n_full_pairs, n_dims // 2)
        n_offset = n_dims // 2
        sect_dims = sum(rope_sections)
        theta_scale = freq_base ** (-2.0 / n_dims)
        for t in range(n_tokens):
            pos_arr = [float(positions[t][0]), float(positions[t][1]),
                       float(positions[t][2]), float(positions[t][3])]
            for ic in range(n_pairs_to_process):
                sector = ic % sect_dims
                if sector % 3 == 1 and sector < 3 * rope_sections[1]:
                    pos_idx = 1
                elif sector % 3 == 2 and sector < 3 * rope_sections[2]:
                    pos_idx = 2
                elif sector % 3 == 0 and sector < 3 * rope_sections[0]:
                    pos_idx = 0
                else:
                    pos_idx = 3
                theta = pos_arr[pos_idx] * (theta_scale ** ic)
                c = float(np.cos(theta)) * attn_factor
                s = float(np.sin(theta)) * attn_factor
                x0, x1 = x[t, ic], x[t, ic + n_offset]
                x[t, ic] = x0 * c - x1 * s
                x[t, ic + n_offset] = x0 * s + x1 * c
        return x.astype(np.float32)

    def attention_ref(Q, K, V, scale, is_causal, gqa_ratio):
        """Q: [n_tokens, n_heads, head_size], K: [n_tokens, n_kv_heads, head_size]"""
        n_tokens_q, n_heads, head_size = Q.shape
        _, n_kv_heads, _ = K.shape
        out = np.zeros((n_tokens_q, n_heads, head_size), dtype=np.float64)
        for t in range(n_tokens_q):
            for h in range(n_heads):
                kv_h = h // gqa_ratio
                scores = np.zeros(n_tokens_q, dtype=np.float64)
                for t2 in range(n_tokens_q):
                    if is_causal and t2 > t:
                        continue
                    dot = 0.0
                    for d in range(head_size):
                        dot += Q[t, h, d] * K[t2, kv_h, d]
                    scores[t2] = dot * scale
                # Softmax
                scores -= scores.max()
                exps = np.exp(scores)
                if is_causal:
                    for t2 in range(t + 1, n_tokens_q):
                        exps[t2] = 0.0
                w_sum = exps.sum()
                if w_sum > 0:
                    probs = exps / w_sum
                else:
                    probs = exps
                for d in range(head_size):
                    acc = 0.0
                    for t2 in range(n_tokens_q):
                        acc += probs[t2] * V[t2, kv_h, d]
                    out[t, h, d] = acc
        return out

    # 1. Attention norm
    normed = np.zeros_like(input_arr)
    for t in range(n_tokens):
        normed[t] = rms_norm_ref(input_arr[t], attn_norm_w, eps)

    # 2. QKV projections
    Q_full = normed @ wq.T  # [n_tokens, 2*n_heads*head_size]
    K = normed @ wk.T       # [n_tokens, n_kv_heads*head_size]
    V = normed @ wv.T       # [n_tokens, n_kv_heads*head_size]

    # 3. Split Q
    q_size = n_heads * head_size
    Q = Q_full[:, :q_size]
    gate = Q_full[:, q_size:]

    # 4. QK norm per head
    Q_n = np.zeros_like(Q)
    K_n = np.zeros_like(K)
    for t in range(n_tokens):
        Q_n[t] = rms_norm_per_head_ref(Q[t], q_norm_w, head_size, eps)
        K_n[t] = rms_norm_per_head_ref(K[t], k_norm_w, head_size, eps)

    # 5. RoPE
    n_rot = sum(max(s, 0) for s in rope_sections) * 2
    Q_3d = Q_n.reshape(n_tokens, n_heads, head_size)
    K_3d = K_n.reshape(n_tokens, n_kv_heads, head_size)
    Q_r = np.zeros_like(Q_3d)
    K_r = np.zeros_like(K_3d)
    for t in range(n_tokens):
        positions = [[pos[0] + t, pos[1], pos[2], pos[3]]]
        for h in range(n_heads):
            head_slice = Q_3d[t, h:h+1, :]  # [1, head_size]
            Q_r[t, h] = rope_partial_imrope(head_slice, positions, n_rot, rope_sections, freq_base=freq_base).ravel()
        for h in range(n_kv_heads):
            head_slice = K_3d[t, h:h+1, :]  # [1, head_size]
            K_r[t, h] = rope_partial_imrope(head_slice, positions, n_rot, rope_sections, freq_base=freq_base).ravel()

    # 6. GQA attention
    gqa_ratio = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_size)
    attn_out = attention_ref(Q_r, K_r, V.reshape(n_tokens, n_kv_heads, head_size), scale, True, gqa_ratio)

    # 7. Gate sigmoid + multiply
    attn_flat = attn_out.reshape(n_tokens, q_size)
    gated = attn_flat * (1.0 / (1.0 + np.exp(-gate)))

    # 8. Output projection
    attn_proj = gated @ wo.T  # [n_tokens, n_embd]

    # 9. Residual
    residual1 = input_arr + attn_proj

    # 10. Post-attention norm
    post_normed = np.zeros_like(residual1)
    for t in range(n_tokens):
        post_normed[t] = rms_norm_ref(residual1[t], post_norm_w, eps)

    # 11. Dense SwiGLU FFN
    gate_out = post_normed @ ffn_gate_w.T  # [n_tokens, n_ff]
    up_out = post_normed @ ffn_up_w.T      # [n_tokens, n_ff]
    swiglu = gate_out * (1.0 / (1.0 + np.exp(-gate_out))) * up_out
    ffn_out = swiglu @ ffn_down_w.T        # [n_tokens, n_embd]

    # 12. Final residual
    output = residual1 + ffn_out
    return output.ravel().astype(np.float32)


def ref_delta_layer_forward(
    input_arr, attn_norm_w, wqkv, wqkv_gate, conv_kernel, alpha_bias,
    ssm_a, ssm_norm_w, ssm_out, post_norm_w, ffn_gate_w, ffn_up_w, ffn_down_w,
    n_embd, n_ff, conv_dim, conv_kernel_size, ba_dim,
    s_k, s_v, n_heads_k, n_heads_v, eps=1e-6,
):
    """Numpy reference for delta-net layer forward pass (single token)."""
    input_arr = np.asarray(input_arr, dtype=np.float64).ravel()
    attn_norm_w = np.asarray(attn_norm_w, dtype=np.float64).ravel()
    wqkv = np.asarray(wqkv, dtype=np.float64).reshape(conv_dim, n_embd)
    wqkv_gate = np.asarray(wqkv_gate, dtype=np.float64).reshape(ba_dim, n_embd)
    conv_kernel = np.asarray(conv_kernel, dtype=np.float64).reshape(conv_dim, conv_kernel_size)
    alpha_bias = np.asarray(alpha_bias, dtype=np.float64).ravel()
    ssm_a = np.asarray(ssm_a, dtype=np.float64).ravel()
    ssm_norm_w = np.asarray(ssm_norm_w, dtype=np.float64).ravel()
    ssm_out = np.asarray(ssm_out, dtype=np.float64).reshape(n_embd, s_v * n_heads_v)
    post_norm_w = np.asarray(post_norm_w, dtype=np.float64).ravel()
    ffn_gate_w = np.asarray(ffn_gate_w, dtype=np.float64).reshape(n_ff, n_embd)
    ffn_up_w = np.asarray(ffn_up_w, dtype=np.float64).reshape(n_ff, n_embd)
    ffn_down_w = np.asarray(ffn_down_w, dtype=np.float64).reshape(n_embd, n_ff)

    def rms_norm_ref(x, w, eps):
        var = np.mean(x ** 2) + eps
        return x / np.sqrt(var) * w

    def softplus(x):
        return np.log1p(np.exp(x))

    # 1. Attention norm
    normed = rms_norm_ref(input_arr, attn_norm_w, eps)

    # 2. QKV projection
    qkv = wqkv @ normed  # [conv_dim]

    # 3. Gate projection
    gate_proj = wqkv_gate @ normed  # [ba_dim]

    # 4. Split into beta and alpha
    q_size = s_k * n_heads_k
    k_size = s_k * n_heads_k
    v_size = s_v * n_heads_v
    ratio = n_heads_v // n_heads_k
    alpha = np.zeros(n_heads_v, dtype=np.float64)
    beta = np.zeros(n_heads_v, dtype=np.float64)
    for hi in range(n_heads_k):
        for j in range(ratio):
            v_hi = hi * ratio + j
            beta[v_hi] = gate_proj[hi * ratio + j]
            alpha[v_hi] = gate_proj[n_heads_k * ratio + hi * ratio + j]

    # 5. Decay gate
    decay_gate = softplus(alpha + alpha_bias) * ssm_a

    # 6. Conv1d (seq_len=1, state is zero)
    pad = conv_kernel_size - 1
    conv_state_in = np.zeros(conv_dim * pad, dtype=np.float64)
    conv_out = np.zeros(conv_dim, dtype=np.float64)
    for c in range(conv_dim):
        acc = 0.0
        for k in range(conv_kernel_size):
            src = k - pad
            if src < 0:
                val = float(conv_state_in[c * pad + src + pad])
            else:
                val = float(qkv[c + src])
            acc += val * float(conv_kernel[c, k])
        s = 1.0 / (1.0 + np.exp(-acc))
        conv_out[c] = acc * s

    # 7. Split conv → q_conv, k_conv, v_conv
    q_conv_raw = conv_out[0:q_size]
    k_conv_raw = conv_out[q_size:q_size + k_size]
    v_conv = conv_out[q_size + k_size:q_size + k_size + v_size]

    # 8. Repeat Q and K to match V-head count
    q_conv = np.zeros(s_k * n_heads_v, dtype=np.float64)
    k_conv = np.zeros(s_k * n_heads_v, dtype=np.float64)
    for hi in range(n_heads_k):
        for r in range(ratio):
            dst_hi = hi * ratio + r
            q_conv[dst_hi * s_k:(dst_hi + 1) * s_k] = q_conv_raw[hi * s_k:(hi + 1) * s_k]
            k_conv[dst_hi * s_k:(dst_hi + 1) * s_k] = k_conv_raw[hi * s_k:(hi + 1) * s_k]

    # 9. Beta sigmoid
    b = 1.0 / (1.0 + np.exp(-beta))

    # 10. Delta-net autoregressive
    ssm_state = np.zeros((n_heads_v, s_v, s_v), dtype=np.float64)
    attn_out = np.zeros(n_heads_v * s_v, dtype=np.float64)
    scale = 1.0 / np.sqrt(float(s_v))
    for hi in range(n_heads_v):
        qh = q_conv[hi * s_k:(hi + 1) * s_k]
        kh = k_conv[hi * s_k:(hi + 1) * s_k]
        vh = v_conv[hi * s_v:(hi + 1) * s_v]
        qn = np.linalg.norm(qh) + eps
        kn = np.linalg.norm(kh) + eps
        qh_n = qh / qn * scale
        kh_n = kh / kn
        state_h = ssm_state[hi]  # [s_v, s_v]
        state_h *= np.exp(decay_gate[hi])
        # k_state = state^T @ k_norm — Rust: state[st_base + i*s_v + j] * k[j] for j in 0..s_k
        kst = state_h[:, :s_k] @ kh_n  # [s_v]
        vd = vh - kst
        # state += outer(vd, kh_n * b)
        state_h[:, :s_k] += np.outer(vd, kh_n * b[hi])
        attn_out[hi * s_v:(hi + 1) * s_v] = state_h[:, :s_k] @ qh_n

    # 11. Gated norm
    normed_attn = rms_norm_ref(attn_out, ssm_norm_w, eps)
    gated_attn = np.zeros(v_size, dtype=np.float64)
    for i in range(v_size):
        v_hi = i // s_v
        s = 1.0 / (1.0 + np.exp(-alpha[v_hi]))
        gated_attn[i] = normed_attn[i] * s

    # 12. Output projection
    attn_residual = ssm_out @ gated_attn

    # 13. Residual
    residual1 = input_arr + attn_residual

    # 14. Post-attention norm
    post_normed = rms_norm_ref(residual1, post_norm_w, eps)

    # 15. Dense SwiGLU FFN
    gate_out = ffn_gate_w @ post_normed  # [n_ff]
    up_out = ffn_up_w @ post_normed      # [n_ff]
    swiglu = gate_out * (1.0 / (1.0 + np.exp(-gate_out))) * up_out
    ffn_out = ffn_down_w @ swiglu        # [n_embd]

    # 16. Final residual
    output = residual1 + ffn_out
    return output.ravel().astype(np.float32)


# ---- main -------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    spec = []

    # RMS norm cases
    rng = np.random.default_rng(101)
    x = rng.standard_normal(4096).astype(np.float32)
    w = (0.5 + rng.random(4096)).astype(np.float32)
    x.tofile(os.path.join(OUT, "rms_1.bin"))
    w.tofile(os.path.join(OUT, "rms_1_w.bin"))
    np.savetxt(os.path.join(OUT, "ref_rms_1.txt"), ref_rms_norm(x, w, 1e-6), fmt="%.9e")
    spec.append("rms 1 4096 0.000001")

    rng = np.random.default_rng(102)
    x = (rng.standard_normal(512) * 3.0 - 1.0).astype(np.float32)
    w = rng.random(512).astype(np.float32)
    x.tofile(os.path.join(OUT, "rms_2.bin"))
    w.tofile(os.path.join(OUT, "rms_2_w.bin"))
    np.savetxt(os.path.join(OUT, "ref_rms_2.txt"), ref_rms_norm(x, w, 1e-5), fmt="%.9e")
    spec.append("rms 2 512 0.00001")

    # GEMV cases (weights crafted, activations quantized by the ref quantizers)
    gemv_cases = [
        ("Q8_0", 4096, 6, "gemv_x1.bin", 201),
        ("Q4_K", 4096, 6, "gemv_x2.bin", 202),
        ("Q5_K", 4096, 6, "gemv_x3.bin", 203),
        ("Q6_K", 4096, 6, "gemv_x4.bin", 204),
        ("F32", 1024, 8, "gemv_x5.bin", 205),
    ]
    for ty, n_in, n_out, xname, seed in gemv_cases:
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(n_in) * 2.0 - 0.5).astype(np.float32)
        x.tofile(os.path.join(OUT, xname))
        spec.append(f"gemv {ty} {n_in} {n_out} {xname}")

        if ty == "F32":
            w = rng.standard_normal((n_out, n_in)).astype(np.float32)
            w.tofile(os.path.join(OUT, "gemv_F32_w.bin"))
            ref = w.astype(np.float64) @ x.astype(np.float64)
            np.savetxt(os.path.join(OUT, "ref_gemv_F32.txt"), ref, fmt="%.9e")
            continue

        nblocks = n_in // (QK8_0 if ty == "Q8_0" else QK_K)
        dequant = DEQUANT[ty]
        craft = globals()["craft_" + ty.lower()]
        xq = quantize_q8_0(x) if ty == "Q8_0" else quantize_q8_k(x)
        xd = dequant_q8_0(xq, n_in) if ty == "Q8_0" else dequant_q8_k(xq, n_in)
        refs = []
        rows = bytearray()
        for r in range(n_out):
            raw = craft(nblocks, rng)
            rows += raw
            wd = dequant(raw, n_in)
            refs.append(float((wd * xd).sum()))
        with open(os.path.join(OUT, f"gemv_{ty}_w.bin"), "wb") as f:
            f.write(bytes(rows))
        np.savetxt(os.path.join(OUT, f"ref_gemv_{ty}.txt"), np.array(refs), fmt="%.9e")

    # GEMM cases (batched multi-row)
    gemm_cases = [
        ("Q8_0", 256, 4, 3, "gemm_x1.bin", 401),
        ("Q4_K", 256, 4, 3, "gemm_x2.bin", 402),
        ("Q5_K", 256, 4, 3, "gemm_x3.bin", 403),
        ("Q6_K", 256, 4, 3, "gemm_x4.bin", 404),
        ("F32",  128, 3, 4, "gemm_x5.bin", 405),
    ]
    for ty, n_in, n_out, n_batch, xname, seed in gemm_cases:
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(n_in * n_batch) * 2.0 - 0.5).astype(np.float32)
        x.tofile(os.path.join(OUT, xname))
        spec.append(f"gemm {ty} {n_in} {n_out} {n_batch} {xname}")
        if ty == "F32":
            w = rng.standard_normal((n_out, n_in)).astype(np.float32)
            w.tofile(os.path.join(OUT, "gemm_F32_w.bin"))
            X = x.reshape(n_batch, n_in).T.astype(np.float64)     # n_in × n_batch
            W = w.astype(np.float64)                                # n_out × n_in
            ref = W @ X                                             # n_out × n_batch
            np.savetxt(os.path.join(OUT, "ref_gemm_F32.txt"), ref.ravel(order="C"), fmt="%.9e")
            continue
        nblocks = n_in // (QK8_0 if ty == "Q8_0" else QK_K)
        deq = DEQUANT[ty]
        craft = globals()["craft_" + ty.lower()]
        # quantize each batch element, then dequant to get activation reference
        xd_batch = []
        for b in range(n_batch):
            xb = x[b * n_in:(b + 1) * n_in]
            xq = quantize_q8_0(xb) if ty == "Q8_0" else quantize_q8_k(xb)
            xd_batch.append(dequant_q8_0(xq, n_in) if ty == "Q8_0" else dequant_q8_k(xq, n_in))
        rows = bytearray()
        ref = np.zeros((n_out, n_batch), dtype=np.float64)
        for r in range(n_out):
            raw = craft(nblocks, rng)
            rows += raw
            wd = deq(raw, n_in)
            for b in range(n_batch):
                ref[r, b] = (wd * xd_batch[b]).sum()
        with open(os.path.join(OUT, f"gemm_{ty}_w.bin"), "wb") as f:
            f.write(bytes(rows))
        np.savetxt(os.path.join(OUT, f"ref_gemm_{ty}.txt"), ref.ravel(order="C"), fmt="%.9e")

    # ATTENTION cases
    attn_cases = [
        # (n_heads, n_kv_heads, head_dim, n_q, n_kv, causal, seed)
        (4, 2, 16, 6, 8, True,  501),
        (4, 2, 16, 3, 3, False, 502),
        (8, 8, 32, 1, 12, True, 503),
    ]
    for qi, (nh, nkv, hd, nq, nkv_len, causal, seed) in enumerate(attn_cases, 1):
        rng = np.random.default_rng(seed)
        q = (rng.standard_normal((nq, nh, hd)) * 0.5).astype(np.float32)
        k = (rng.standard_normal((nkv_len, nkv, hd)) * 0.5).astype(np.float32)
        v = (rng.standard_normal((nkv_len, nkv, hd)) * 0.5).astype(np.float32)
        q.tofile(os.path.join(OUT, f"attn_{qi}_q.bin"))
        k.tofile(os.path.join(OUT, f"attn_{qi}_k.bin"))
        v.tofile(os.path.join(OUT, f"attn_{qi}_v.bin"))
        scale = 1.0 / (hd ** 0.5)
        ref = ref_attention(q, k, v, nh, nkv, hd, nq, nkv_len, scale, causal)
        np.savetxt(os.path.join(OUT, f"ref_attn_{qi}.txt"), ref, fmt="%.9e")
        spec.append(f"attn {qi} {nh} {nkv} {hd} {nq} {nkv_len} {int(causal)}")

    # DELTA-NET cases (autoregressive, GDA mode)
    dn_cases = [
        # (s_k, s_v, n_heads, n_steps, seed)
        (16, 16, 2, 3, 601),
        (32, 32, 4, 2, 602),
    ]
    for di, (sk, sv, nh, nsteps, seed) in enumerate(dn_cases, 1):
        rng = np.random.default_rng(seed)
        state = np.zeros((nh, sv, sv), dtype=np.float32)
        all_refs = []
        all_inputs = []
        for step in range(nsteps):
            q = (rng.standard_normal((nh, sk)) * 0.5).astype(np.float32)
            k = (rng.standard_normal((nh, sk)) * 0.5).astype(np.float32)
            v = (rng.standard_normal((nh, sv)) * 0.5).astype(np.float32)
            g = (rng.standard_normal(nh) * 0.1 - 0.2).astype(np.float32)
            beta = (rng.standard_normal(nh) * 0.3 + 0.5).astype(np.float32)
            ref_out, state = ref_delta_net(q, k, v, g, beta, state, sk, sv, 1e-6)
            all_refs.append(ref_out)
            all_inputs.append((q, k, v, g, beta))
        # Save initial state (zeros) + inputs for each step
        # The Rust binary needs to replay the steps
        # Write state_init (all zeros) + per-step inputs + final ref outputs
        for step, ((q, k, v, g, beta), ref_out) in enumerate(zip(all_inputs, all_refs)):
            q.tofile(os.path.join(OUT, f"dn_{di}_s{step}_q.bin"))
            k.tofile(os.path.join(OUT, f"dn_{di}_s{step}_k.bin"))
            v.tofile(os.path.join(OUT, f"dn_{di}_s{step}_v.bin"))
            g.tofile(os.path.join(OUT, f"dn_{di}_s{step}_g.bin"))
            beta.tofile(os.path.join(OUT, f"dn_{di}_s{step}_beta.bin"))
            np.savetxt(os.path.join(OUT, f"ref_dn_{di}_s{step}.txt"), ref_out, fmt="%.9e")
        spec.append(f"delta {di} {sk} {sv} {nh} {nsteps}")

    # ROPE cases
    rng = np.random.default_rng(301)
    x = (rng.standard_normal(64) * 2.0).astype(np.float32)
    x.tofile(os.path.join(OUT, "rope_1.bin"))
    ref = ref_rope(x, (5, 2, 7, 0), 64, (11, 11, 10, 0), 4096,
                   10_000_000.0, 1.0, 0.0, 1.0)
    np.savetxt(os.path.join(OUT, "ref_rope_1.txt"), ref, fmt="%.9e")
    spec.append("rope 1 64 64 11 11 10 0 5 2 7 0 4096 10000000.0 1.0 0.0 1.0")

    rng = np.random.default_rng(302)
    x = np.concatenate([rng.standard_normal(64).astype(np.float32),
                        np.full(16, 3.0, dtype=np.float32)])
    x.tofile(os.path.join(OUT, "rope_2.bin"))
    ref = ref_rope(x, (3, 5, 9, 0), 64, (11, 11, 10, 0), 4096,
                   10_000_000.0, 1.0, 0.0, 1.0)
    np.savetxt(os.path.join(OUT, "ref_rope_2.txt"), ref, fmt="%.9e")
    spec.append("rope 2 80 64 11 11 10 0 3 5 9 0 4096 10000000.0 1.0 0.0 1.0")

    rng = np.random.default_rng(303)
    x = rng.standard_normal(64).astype(np.float32)
    x.tofile(os.path.join(OUT, "rope_3.bin"))
    ref = ref_rope(x, (1, 2, 3, 4), 64, (4, 3, 2, 1), 2048,
                   100_000.0, 1.0, 0.0, 1.0)
    np.savetxt(os.path.join(OUT, "ref_rope_3.txt"), ref, fmt="%.9e")
    spec.append("rope 3 64 64 4 3 2 1 1 2 3 4 2048 100000.0 1.0 0.0 1.0")

    # ---- Phase 3a: swiglu, rms_norm_per_head, softmax_topk, conv1d_silu ------

    # SwiGLU cases
    for si, (n, seed) in enumerate([(64, 701), (256, 702)], 1):
        rng = np.random.default_rng(seed)
        gate = (rng.standard_normal(n) * 0.5).astype(np.float32)
        up = (rng.standard_normal(n) * 0.5).astype(np.float32)
        gate.tofile(os.path.join(OUT, f"swiglu_{si}_gate.bin"))
        up.tofile(os.path.join(OUT, f"swiglu_{si}_up.bin"))
        ref = ref_swiglu(gate, up)
        np.savetxt(os.path.join(OUT, f"ref_swiglu_{si}.txt"), ref, fmt="%.9e")
        spec.append(f"swiglu {si} {n}")

    # rms_norm_per_head cases
    for ri, (n_heads, head_size, seed) in enumerate([(8, 64, 711), (4, 128, 712)], 1):
        rng = np.random.default_rng(seed)
        x = (rng.standard_normal(n_heads * head_size) * 2.0).astype(np.float32)
        w = (0.5 + rng.random(head_size)).astype(np.float32)
        x.tofile(os.path.join(OUT, f"rph_{ri}.bin"))
        w.tofile(os.path.join(OUT, f"rph_{ri}_w.bin"))
        ref = ref_rms_norm_per_head(x, w, head_size, 1e-6)
        np.savetxt(os.path.join(OUT, f"ref_rph_{ri}.txt"), ref, fmt="%.9e")
        spec.append(f"rph {ri} {n_heads * head_size} {head_size}")

    # softmax_topk cases
    for ti, (n_experts, k, seed) in enumerate([(128, 8, 721), (64, 4, 722)], 1):
        rng = np.random.default_rng(seed)
        logits = (rng.standard_normal(n_experts) * 2.0).astype(np.float32)
        logits.tofile(os.path.join(OUT, f"topk_{ti}.bin"))
        weights_ref, idx_ref = ref_softmax_topk(logits, k)
        # Save: weights then indices (as f32)
        np.savetxt(os.path.join(OUT, f"ref_topk_{ti}_w.txt"), weights_ref, fmt="%.9e")
        np.savetxt(os.path.join(OUT, f"ref_topk_{ti}_i.txt"), idx_ref.astype(np.float32), fmt="%.0f")
        spec.append(f"topk {ti} {n_experts} {k}")

    # conv1d_silu cases
    for ci, (channels, kernel_size, seq_len, seed) in enumerate([(16, 4, 8, 731), (32, 3, 12, 732)], 1):
        rng = np.random.default_rng(seed)
        inp = (rng.standard_normal(channels * seq_len) * 0.3).astype(np.float32)
        krn = (rng.standard_normal(channels * kernel_size) * 0.2).astype(np.float32)
        pad = kernel_size - 1
        st = np.zeros(channels * pad, dtype=np.float32)
        inp.tofile(os.path.join(OUT, f"conv_{ci}_inp.bin"))
        krn.tofile(os.path.join(OUT, f"conv_{ci}_krn.bin"))
        st.tofile(os.path.join(OUT, f"conv_{ci}_st.bin"))
        ref_out, ref_st = ref_conv1d_silu(inp, krn, st, channels, seq_len, kernel_size)
        np.savetxt(os.path.join(OUT, f"ref_conv_{ci}_out.txt"), ref_out, fmt="%.9e")
        np.savetxt(os.path.join(OUT, f"ref_conv_{ci}_st.txt"), ref_st, fmt="%.9e")
        spec.append(f"conv {ci} {channels} {kernel_size} {seq_len}")

    # MoE FFN cases
    for mi, (n_embd, n_ff, n_expert, n_expert_used, n_tokens, seed) in enumerate([
        (16, 12, 8, 2, 3, 741),
        (32, 24, 16, 4, 1, 742),
    ], 1):
        rng = np.random.default_rng(seed)
        inp = (rng.standard_normal(n_tokens * n_embd) * 0.3).astype(np.float32)
        rw = (rng.standard_normal(n_expert * n_embd) * 0.2).astype(np.float32)
        guw = (rng.standard_normal(n_expert * 2 * n_ff * n_embd) * 0.1).astype(np.float32)
        dw = (rng.standard_normal(n_expert * n_embd * n_ff) * 0.15).astype(np.float32)
        inp.tofile(os.path.join(OUT, f"moe_{mi}_inp.bin"))
        rw.tofile(os.path.join(OUT, f"moe_{mi}_rw.bin"))
        guw.tofile(os.path.join(OUT, f"moe_{mi}_guw.bin"))
        dw.tofile(os.path.join(OUT, f"moe_{mi}_dw.bin"))
        ref = ref_moe_ffn(inp, rw, guw, dw, n_embd, n_ff, n_expert, n_expert_used, n_tokens)
        np.savetxt(os.path.join(OUT, f"ref_moe_{mi}.txt"), ref, fmt="%.9e")
        spec.append(f"moe {mi} {n_embd} {n_ff} {n_expert} {n_expert_used} {n_tokens}")

    # Full layer cases: rope_sections must satisfy sum(sections) == head_size/2
    for fi, (n_embd, n_heads, n_kv_heads, head_size, n_ff, n_tokens, seed, sections) in enumerate([
        (64, 4, 2, 32, 48, 3, 901, [8, 8, 0, 0]),
        (64, 8, 4, 64, 64, 2, 902, [11, 11, 10, 0]),
    ], 1):
        rng = np.random.default_rng(seed)
        inp = (rng.standard_normal(n_tokens * n_embd) * 0.3).astype(np.float32)
        anw = (rng.standard_normal(n_embd) * 0.2 + 1.0).astype(np.float32)
        wq = (rng.standard_normal(2 * n_heads * head_size * n_embd) * 0.1).astype(np.float32)
        wk = (rng.standard_normal(n_kv_heads * head_size * n_embd) * 0.1).astype(np.float32)
        wv = (rng.standard_normal(n_kv_heads * head_size * n_embd) * 0.1).astype(np.float32)
        wo = (rng.standard_normal(n_embd * n_heads * head_size) * 0.1).astype(np.float32)
        qnw = (rng.standard_normal(head_size) * 0.1 + 1.0).astype(np.float32)
        knw = (rng.standard_normal(head_size) * 0.1 + 1.0).astype(np.float32)
        pnw = (rng.standard_normal(n_embd) * 0.2 + 1.0).astype(np.float32)
        fgw = (rng.standard_normal(n_ff * n_embd) * 0.1).astype(np.float32)
        fuw = (rng.standard_normal(n_ff * n_embd) * 0.1).astype(np.float32)
        fdw = (rng.standard_normal(n_embd * n_ff) * 0.1).astype(np.float32)
        inp.tofile(os.path.join(OUT, f"full_layer_{fi}_inp.bin"))
        anw.tofile(os.path.join(OUT, f"full_layer_{fi}_anw.bin"))
        wq.tofile(os.path.join(OUT, f"full_layer_{fi}_wq.bin"))
        wk.tofile(os.path.join(OUT, f"full_layer_{fi}_wk.bin"))
        wv.tofile(os.path.join(OUT, f"full_layer_{fi}_wv.bin"))
        wo.tofile(os.path.join(OUT, f"full_layer_{fi}_wo.bin"))
        qnw.tofile(os.path.join(OUT, f"full_layer_{fi}_qnw.bin"))
        knw.tofile(os.path.join(OUT, f"full_layer_{fi}_knw.bin"))
        pnw.tofile(os.path.join(OUT, f"full_layer_{fi}_pnw.bin"))
        fgw.tofile(os.path.join(OUT, f"full_layer_{fi}_fgw.bin"))
        fuw.tofile(os.path.join(OUT, f"full_layer_{fi}_fuw.bin"))
        fdw.tofile(os.path.join(OUT, f"full_layer_{fi}_fdw.bin"))
        pos = [5, 0, 0, 0]
        ref = ref_full_layer_forward(inp, anw, wq, wk, wv, wo, qnw, knw, pnw, fgw, fuw, fdw, pos, n_embd, n_heads, n_kv_heads, head_size, n_ff, n_tokens, rope_sections=sections)
        np.savetxt(os.path.join(OUT, f"ref_full_layer_{fi}.txt"), ref, fmt="%.9e")
        sect_str = ",".join(str(s) for s in sections)
        spec.append(f"full_layer {fi} {n_embd} {n_heads} {n_kv_heads} {head_size} {n_ff} {n_tokens} {sect_str}")

    # Delta layer cases: conv_dim = s_k * n_heads_k * 2 + s_v * n_heads_v
    #                     ba_dim = n_heads_v * 2, s_k == s_v (real model always has this)
    for di, (n_embd, n_ff, s_kv, n_heads_k, n_heads_v, seed) in enumerate([
        (32, 64, 8, 2, 4, 911),
        (64, 128, 16, 4, 8, 912),
    ], 1):
        s_k = s_kv
        s_v = s_kv
        conv_dim = s_k * n_heads_k * 2 + s_v * n_heads_v
        conv_kernel_size = 4
        ba_dim = n_heads_v * 2
        rng = np.random.default_rng(seed)
        inp = (rng.standard_normal(n_embd) * 0.3).astype(np.float32)
        anw = (rng.standard_normal(n_embd) * 0.2 + 1.0).astype(np.float32)
        wqkv = (rng.standard_normal(conv_dim * n_embd) * 0.1).astype(np.float32)
        wg = (rng.standard_normal(ba_dim * n_embd) * 0.1).astype(np.float32)
        ck = (rng.standard_normal(conv_dim * conv_kernel_size) * 0.1).astype(np.float32)
        ab = (rng.standard_normal(n_heads_v) * 0.1).astype(np.float32)
        sa = -(np.abs(rng.standard_normal(n_heads_v)) * 0.1).astype(np.float32)
        snw = np.ones(s_v * n_heads_v, dtype=np.float32)
        so = (rng.standard_normal(n_embd * s_v * n_heads_v) * 0.1).astype(np.float32)
        pnw = (rng.standard_normal(n_embd) * 0.2 + 1.0).astype(np.float32)
        fgw = (rng.standard_normal(n_ff * n_embd) * 0.1).astype(np.float32)
        fuw = (rng.standard_normal(n_ff * n_embd) * 0.1).astype(np.float32)
        fdw = (rng.standard_normal(n_embd * n_ff) * 0.1).astype(np.float32)

        inp.tofile(os.path.join(OUT, f"delta_layer_{di}_inp.bin"))
        anw.tofile(os.path.join(OUT, f"delta_layer_{di}_anw.bin"))
        wqkv.tofile(os.path.join(OUT, f"delta_layer_{di}_wqkv.bin"))
        wg.tofile(os.path.join(OUT, f"delta_layer_{di}_wg.bin"))
        ck.tofile(os.path.join(OUT, f"delta_layer_{di}_ck.bin"))
        ab.tofile(os.path.join(OUT, f"delta_layer_{di}_ab.bin"))
        sa.tofile(os.path.join(OUT, f"delta_layer_{di}_sa.bin"))
        snw.tofile(os.path.join(OUT, f"delta_layer_{di}_snw.bin"))
        so.tofile(os.path.join(OUT, f"delta_layer_{di}_so.bin"))
        pnw.tofile(os.path.join(OUT, f"delta_layer_{di}_pnw.bin"))
        fgw.tofile(os.path.join(OUT, f"delta_layer_{di}_fgw.bin"))
        fuw.tofile(os.path.join(OUT, f"delta_layer_{di}_fuw.bin"))
        fdw.tofile(os.path.join(OUT, f"delta_layer_{di}_fdw.bin"))

        ref = ref_delta_layer_forward(
            inp, anw, wqkv, wg, ck, ab, sa, snw, so, pnw, fgw, fuw, fdw,
            n_embd, n_ff, conv_dim, conv_kernel_size, ba_dim,
            s_k, s_v, n_heads_k, n_heads_v,
        )
        np.savetxt(os.path.join(OUT, f"ref_delta_layer_{di}.txt"), ref, fmt="%.9e")
        spec.append(f"delta_layer {di} {n_embd} {n_ff} {conv_dim} {conv_kernel_size} {ba_dim} {s_k} {s_v} {n_heads_k} {n_heads_v}")

    with open(os.path.join(OUT, "spec.txt"), "w") as f:
        f.write("\n".join(spec) + "\n")
    print("wrote", os.path.join(OUT, "spec.txt"))


if __name__ == "__main__":
    main()
