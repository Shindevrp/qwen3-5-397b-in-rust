//! End-to-end inference pipeline: GGUF weights → unified forward → token.
//!
//! Loads dequantised weight slices from a `ModelLoader`, wires them into the
//! kernel-level layer structs, and drives the token-by-token forward pass
//! through both recurrent (delta-net) and full-attention layers.

use rayon::prelude::*;
use crate::model::config::Qwen3_5Config;
use crate::model::kernels::{
    DeltaNetLayerWeights, FullAttnLayerWeights, RopeConfig,
    delta_net_layer_forward, embed_tokens,
    lm_head_argmax, lm_head_logits,
};
use crate::model::loader::ModelLoader;
use crate::model::quant::RawTensor;

/// GGUF tensor name helpers per layer.
struct TensorNames {
    prefix: String,
}

impl TensorNames {
    fn new(layer: usize) -> Self {
        Self { prefix: format!("blk.{layer}.") }
    }

    fn name(&self, suffix: &str) -> String {
        format!("{}{}", self.prefix, suffix)
    }

    // --- shared ---
    fn attn_norm(&self) -> String { self.name("attn_norm.weight") }
    fn post_attn_norm(&self) -> String { self.name("post_attention_norm.weight") }

    // --- full attention ---
    fn attn_q(&self) -> String { self.name("attn_q.weight") }
    fn attn_k(&self) -> String { self.name("attn_k.weight") }
    fn attn_v(&self) -> String { self.name("attn_v.weight") }
    fn attn_o(&self) -> String { self.name("attn_output.weight") }
    fn attn_q_norm(&self) -> String { self.name("attn_q_norm.weight") }
    fn attn_k_norm(&self) -> String { self.name("attn_k_norm.weight") }

    // --- delta-net ---
    fn attn_qkv(&self) -> String { self.name("attn_qkv.weight") }
    fn attn_gate(&self) -> String { self.name("attn_gate.weight") }
    fn ssm_conv1d(&self) -> String { self.name("ssm_conv1d.weight") }
    fn ssm_dt(&self) -> String { self.name("ssm_dt.bias") }
    fn ssm_a(&self) -> String { self.name("ssm_a") }
    fn ssm_norm(&self) -> String { self.name("ssm_norm.weight") }
    fn ssm_out(&self) -> String { self.name("ssm_out.weight") }

    // --- dense FFN ---
    fn ffn_gate(&self) -> String { self.name("ffn_gate.weight") }
    fn ffn_up(&self) -> String { self.name("ffn_up.weight") }
    fn ffn_down(&self) -> String { self.name("ffn_down.weight") }

    // --- MoE FFN ---
    #[allow(dead_code)]
    fn ffn_gate_inp(&self) -> String { self.name("ffn_gate_inp.weight") }
    #[allow(dead_code)]
    fn ffn_gate_exps(&self) -> String { self.name("ffn_gate_exps.weight") }
    #[allow(dead_code)]
    fn ffn_up_exps(&self) -> String { self.name("ffn_up_exps.weight") }
    #[allow(dead_code)]
    fn ffn_down_exps(&self) -> String { self.name("ffn_down_exps.weight") }

    // --- shared expert ---
    #[allow(dead_code)]
    fn ffn_gate_inp_shexp(&self) -> String { self.name("ffn_gate_inp_shexp.weight") }
    #[allow(dead_code)]
    fn ffn_gate_shexp(&self) -> String { self.name("ffn_gate_shexp.weight") }
    #[allow(dead_code)]
    fn ffn_up_shexp(&self) -> String { self.name("ffn_up_shexp.weight") }
    #[allow(dead_code)]
    fn ffn_down_shexp(&self) -> String { self.name("ffn_down_shexp.weight") }
}

// ---------------------------------------------------------------------------
// MoE FFN weights (fused gate+up for the kernel)
// ---------------------------------------------------------------------------

pub struct LoadedMoeFfn {
    pub router_w: RawTensor,      // [n_expert, n_embd]
    pub gate_up_w: RawTensor,     // [n_expert, 2*n_ff, n_embd] (fused gate+up)
    pub down_w: RawTensor,        // [n_expert, n_embd, n_ff]
    // Shared expert (shexp)
    pub shexp_gate_w: RawTensor,     // [n_ff_shexp, n_embd]
    pub shexp_up_w: RawTensor,       // [n_ff_shexp, n_embd]
    pub shexp_down_w: RawTensor,     // [n_embd, n_ff_shexp]
    pub shexp_gate_inp_w: RawTensor, // [n_embd]
    pub n_ff_shexp: usize,
}

// ---------------------------------------------------------------------------
// Full-attention layer weights (loaded as raw quantized bytes)
// ---------------------------------------------------------------------------

pub struct LoadedFullAttnLayer {
    pub attn_norm_w: RawTensor,
    pub wq: RawTensor,
    pub wk: RawTensor,
    pub wv: RawTensor,
    pub wo: RawTensor,
    pub q_norm_w: RawTensor,
    pub k_norm_w: RawTensor,
    pub post_norm_w: RawTensor,
    /// Dense FFN (when expert_count == 0)
    pub ffn_gate_w: RawTensor,
    pub ffn_up_w: RawTensor,
    pub ffn_down_w: RawTensor,
    /// MoE FFN (when expert_count > 0), None if dense
    pub moe_ffn: Option<LoadedMoeFfn>,
}

// ---------------------------------------------------------------------------
// Delta-net layer weights (loaded as raw quantized bytes)
// ---------------------------------------------------------------------------

pub struct LoadedDeltaNetLayer {
    pub attn_norm_w: RawTensor,
    pub wqkv: RawTensor,
    pub wqkv_gate: RawTensor,
    pub conv_kernel: RawTensor,
    pub alpha_bias: RawTensor,
    pub ssm_a: RawTensor,
    pub ssm_norm_w: RawTensor,
    pub ssm_out: RawTensor,
    pub post_norm_w: RawTensor,
    /// Dense FFN (when expert_count == 0)
    pub ffn_gate_w: RawTensor,
    pub ffn_up_w: RawTensor,
    pub ffn_down_w: RawTensor,
    /// MoE FFN (when expert_count > 0), None if dense
    pub moe_ffn: Option<LoadedMoeFfn>,
}

// ---------------------------------------------------------------------------
// Full model weights (raw quantized bytes, dequantized on demand)
// ---------------------------------------------------------------------------

pub struct ModelWeights {
    pub cfg: Qwen3_5Config,
    pub tok_embd: RawTensor,
    pub output_norm_w: RawTensor,
    pub output_weight: RawTensor,
    pub full_attn_layers: Vec<LoadedFullAttnLayer>,
    pub delta_net_layers: Vec<LoadedDeltaNetLayer>,
}

/// Helper: dequantize a `RawTensor` to `Vec<f32>`. Panics on quant error
/// (weights are validated at GGUF load time).
fn dq(rt: &RawTensor) -> Vec<f32> {
    rt.dequant().expect("dequant failed (tensor corrupted?)")
}

/// Parallel dequantization of two `RawTensor`s.
fn dq2(a: &RawTensor, b: &RawTensor) -> (Vec<f32>, Vec<f32>) {
    let mut ra = None;
    let mut rb = None;
    rayon::scope(|s| {
        s.spawn(|_| { ra = Some(dq(a)); });
        s.spawn(|_| { rb = Some(dq(b)); });
    });
    (ra.unwrap(), rb.unwrap())
}

/// Parallel dequantization of three `RawTensor`s.
fn dq3(a: &RawTensor, b: &RawTensor, c: &RawTensor) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut ra = None;
    let mut rb = None;
    let mut rc = None;
    rayon::scope(|s| {
        s.spawn(|_| { ra = Some(dq(a)); });
        s.spawn(|_| { rb = Some(dq(b)); });
        s.spawn(|_| { rc = Some(dq(c)); });
    });
    (ra.unwrap(), rb.unwrap(), rc.unwrap())
}

impl ModelWeights {
    pub fn load(loader: &ModelLoader) -> Result<Self, String> {
        let cfg = loader.cfg.clone();
        let n_embd = cfg.embedding_length as usize;
        let n_vocab = {
            let meta = loader.tensor_meta("token_embd.weight")
                .ok_or("missing token_embd.weight")?;
            meta.dims[0] as usize
        };

        // Global weights — stored as raw quantized bytes
        let tok_embd = loader.raw_tensor("token_embd.weight").map_err(|e| e.to_string())?;
        let output_norm_w = loader.raw_tensor("output_norm.weight").map_err(|e| e.to_string())?;
        let output_weight = if loader.tensor_meta("output.weight").is_some() {
            loader.raw_tensor("output.weight").map_err(|e| e.to_string())?
        } else {
            // Weight-tied: output shares token_embd
            tok_embd.clone()
        };

        let n_layers = cfg.block_count as usize;
        let n_expert = cfg.expert_count as usize;
        let n_ff = cfg.expert_feed_forward_length as usize;
        let n_ff_shexp = cfg.expert_shared_feed_forward_length as usize;
        let mut full_attn_layers = Vec::new();
        let mut delta_net_layers = Vec::new();

        let raw = |name: &str| -> Result<RawTensor, String> {
            loader.raw_tensor(name).map_err(|e| format!("{name}: {e}"))
        };

        // MoE FFN loader (shared by both layer types)
        // Fuses gate+up into a single F32 RawTensor at load time.
        let load_moe = |t: &TensorNames| -> Result<Option<LoadedMoeFfn>, String> {
            if n_expert == 0 {
                return Ok(None);
            }
            let router_w = raw(&t.ffn_gate_inp())?;
            let gate_exps = raw(&t.ffn_gate_exps())?;
            let up_exps = raw(&t.ffn_up_exps())?;
            let down_w = raw(&t.ffn_down_exps())?;

            // Fuse gate+up: dequant both, interleave, store as F32 RawTensor
            let gate_dq = gate_exps.dequant().map_err(|e| format!("gate_exps: {e}"))?;
            let up_dq = up_exps.dequant().map_err(|e| format!("up_exps: {e}"))?;
            let mut fused = vec![0.0f32; n_expert * 2 * n_ff * n_embd];
            for e in 0..n_expert {
                let gate_base = e * n_ff * n_embd;
                let fused_base = e * 2 * n_ff * n_embd;
                fused[fused_base..fused_base + n_ff * n_embd]
                    .copy_from_slice(&gate_dq[gate_base..gate_base + n_ff * n_embd]);
                fused[fused_base + n_ff * n_embd..fused_base + 2 * n_ff * n_embd]
                    .copy_from_slice(&up_dq[gate_base..gate_base + n_ff * n_embd]);
            }
            // Store fused result as F32 RawTensor
            let gate_up_w = RawTensor::new(
                crate::gguf::GGmlType::F32,
                fused.iter().flat_map(|f| f.to_le_bytes()).collect(),
                n_expert * 2 * n_ff * n_embd,
            );

            // Shared expert (shexp) — loaded when n_ff_shexp > 0
            let (shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w) = if n_ff_shexp > 0 {
                (
                    raw(&t.ffn_gate_shexp())?,
                    raw(&t.ffn_up_shexp())?,
                    raw(&t.ffn_down_shexp())?,
                    raw(&t.ffn_gate_inp_shexp())?,
                )
            } else {
                (
                    RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                    RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                    RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                    RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                )
            };

            Ok(Some(LoadedMoeFfn {
                router_w,
                gate_up_w,
                down_w,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
            }))
        };

        for i in 0..n_layers {
            let t = TensorNames::new(i);
            let attn_norm_w = raw(&t.attn_norm())?;
            let post_norm_w = raw(&t.post_attn_norm())?;

            if cfg.is_recurrent(i) {
                // Delta-net layer
                let wqkv = raw(&t.attn_qkv())?;
                let wqkv_gate = raw(&t.attn_gate())?;
                let conv_kernel = raw(&t.ssm_conv1d())?;
                let alpha_bias = raw(&t.ssm_dt())?;
                let ssm_a = raw(&t.ssm_a())?;
                let ssm_norm_w = raw(&t.ssm_norm())?;
                let ssm_out = raw(&t.ssm_out())?;

                // FFN: MoE or dense
                let moe_ffn = load_moe(&t)?;
                let (ffn_gate_w, ffn_up_w, ffn_down_w) = if moe_ffn.is_some() {
                    (
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                    )
                } else {
                    (raw(&t.ffn_gate())?, raw(&t.ffn_up())?, raw(&t.ffn_down())?)
                };

                delta_net_layers.push(LoadedDeltaNetLayer {
                    attn_norm_w,
                    wqkv,
                    wqkv_gate,
                    conv_kernel,
                    alpha_bias,
                    ssm_a,
                    ssm_norm_w,
                    ssm_out,
                    post_norm_w,
                    ffn_gate_w,
                    ffn_up_w,
                    ffn_down_w,
                    moe_ffn,
                });
            } else {
                // Full-attention layer
                let wq = raw(&t.attn_q())?;
                let wk = raw(&t.attn_k())?;
                let wv = raw(&t.attn_v())?;
                let wo = raw(&t.attn_o())?;
                let q_norm_w = raw(&t.attn_q_norm())?;
                let k_norm_w = raw(&t.attn_k_norm())?;

                // FFN: MoE or dense
                let moe_ffn = load_moe(&t)?;
                let (ffn_gate_w, ffn_up_w, ffn_down_w) = if moe_ffn.is_some() {
                    (
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                        RawTensor::new(crate::gguf::GGmlType::F32, vec![], 0),
                    )
                } else {
                    (raw(&t.ffn_gate())?, raw(&t.ffn_up())?, raw(&t.ffn_down())?)
                };

                full_attn_layers.push(LoadedFullAttnLayer {
                    attn_norm_w,
                    wq, wk, wv, wo,
                    q_norm_w, k_norm_w,
                    post_norm_w,
                    ffn_gate_w,
                    ffn_up_w,
                    ffn_down_w,
                    moe_ffn,
                });
            }
        }

        eprintln!(
            "loaded {} layers ({} full-attn, {} delta-net), n_embd={n_embd}, n_vocab={n_vocab}, n_expert={n_expert}",
            n_layers,
            full_attn_layers.len(),
            delta_net_layers.len(),
        );

        Ok(Self {
            cfg,
            tok_embd,
            output_norm_w,
            output_weight,
            full_attn_layers,
            delta_net_layers,
        })
    }
}

// ---------------------------------------------------------------------------
// Unified forward pass
// ---------------------------------------------------------------------------

/// Run a single token through the full model, dispatching each layer to
/// the correct kernel (full-attention or delta-net) based on `is_recurrent`.
///
/// Returns `(hidden_state, next_token_id)`.
pub fn forward_pass(
    token_id: u32,
    model: &ModelWeights,
) -> (Vec<f32>, u32) {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
    let n_vocab = model.tok_embd.n_elements / n_embd;
    let eps = cfg.attention_layer_norm_rms_epsilon;

    let rope_cfg = RopeConfig {
        freq_base: cfg.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let rope_sections = cfg.rope_sections;

    // Embed
    let tok_embd_dq = dq(&model.tok_embd);
    let mut hidden = embed_tokens(token_id, &tok_embd_dq, n_embd);

    // Layer state buffers (for delta-net recurrent states)
    let conv_dim = cfg.conv_dim as usize;
    let conv_kernel_size = cfg.ssm_conv_kernel as usize;
    let s_v = cfg.head_v_dim as usize;
    let n_heads_v = cfg.ssm_time_step_rank as usize;
    let s_v_dim = s_v * s_v * n_heads_v;
    let conv_buf_size = conv_dim * conv_kernel_size.saturating_sub(1);

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;
    let mut conv_state = vec![0.0f32; conv_buf_size];
    let mut ssm_state = vec![0.0f32; s_v_dim];

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let moe_router_w_dq;
            let moe_gate_up_w_dq;
            let moe_down_w_dq;
            let n_expert;
            let n_expert_used;
            let shexp_gate_w_dq;
            let shexp_up_w_dq;
            let shexp_down_w_dq;
            let shexp_gate_inp_w_dq;
            let n_ff_shexp;

            if let Some(ref moe) = layer.moe_ffn {
                moe_router_w_dq = dq(&moe.router_w);
                moe_gate_up_w_dq = dq(&moe.gate_up_w);
                moe_down_w_dq = dq(&moe.down_w);
                n_expert = cfg.expert_count as usize;
                n_expert_used = cfg.expert_used_count as usize;
                shexp_gate_w_dq = dq(&moe.shexp_gate_w);
                shexp_up_w_dq = dq(&moe.shexp_up_w);
                shexp_down_w_dq = dq(&moe.shexp_down_w);
                shexp_gate_inp_w_dq = dq(&moe.shexp_gate_inp_w);
                n_ff_shexp = moe.n_ff_shexp;
            } else {
                moe_router_w_dq = vec![];
                moe_gate_up_w_dq = vec![];
                moe_down_w_dq = vec![];
                n_expert = 0;
                n_expert_used = 0;
                shexp_gate_w_dq = vec![];
                shexp_up_w_dq = vec![];
                shexp_down_w_dq = vec![];
                shexp_gate_inp_w_dq = vec![];
                n_ff_shexp = 0;
            };

            let (dn_attn_norm_w, dn_wqkv, dn_wqkv_gate) = dq3(&layer.attn_norm_w, &layer.wqkv, &layer.wqkv_gate);
            let (dn_conv_kernel, dn_alpha_bias, dn_ssm_a) = dq3(&layer.conv_kernel, &layer.alpha_bias, &layer.ssm_a);
            let (dn_ssm_norm_w, dn_ssm_out, dn_post_norm_w) = dq3(&layer.ssm_norm_w, &layer.ssm_out, &layer.post_norm_w);
            let (dn_ffn_gate_w, dn_ffn_up_w, dn_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &dn_attn_norm_w,
                wqkv: &dn_wqkv,
                wqkv_gate: &dn_wqkv_gate,
                conv_kernel: &dn_conv_kernel,
                alpha_bias: &dn_alpha_bias,
                ssm_a: &dn_ssm_a,
                ssm_norm_w: &dn_ssm_norm_w,
                ssm_out: &dn_ssm_out,
                post_norm_w: &dn_post_norm_w,
                ffn_gate_w: &dn_ffn_gate_w,
                ffn_up_w: &dn_ffn_up_w,
                ffn_down_w: &dn_ffn_down_w,
                moe_router_w: &moe_router_w_dq,
                moe_gate_up_w: &moe_gate_up_w_dq,
                moe_down_w: &moe_down_w_dq,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w_dq,
                shexp_up_w: &shexp_up_w_dq,
                shexp_down_w: &shexp_down_w_dq,
                shexp_gate_inp_w: &shexp_gate_inp_w_dq,
                n_ff_shexp,
            };

            hidden = delta_net_layer_forward(
                &hidden,
                &layer_ref,
                &mut conv_state,
                &mut ssm_state,
                n_embd,
                n_ff,
                conv_dim,
                conv_kernel_size,
                cfg.ba_dim as usize,
                cfg.head_k_dim as usize,
                s_v,
                cfg.ssm_group_count as usize,
                n_heads_v,
                eps,
            );
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let moe_router_w_dq;
            let moe_gate_up_w_dq;
            let moe_down_w_dq;
            let n_expert;
            let n_expert_used;
            let shexp_gate_w_dq;
            let shexp_up_w_dq;
            let shexp_down_w_dq;
            let shexp_gate_inp_w_dq;
            let n_ff_shexp;

            if let Some(ref moe) = layer.moe_ffn {
                moe_router_w_dq = dq(&moe.router_w);
                moe_gate_up_w_dq = dq(&moe.gate_up_w);
                moe_down_w_dq = dq(&moe.down_w);
                n_expert = cfg.expert_count as usize;
                n_expert_used = cfg.expert_used_count as usize;
                shexp_gate_w_dq = dq(&moe.shexp_gate_w);
                shexp_up_w_dq = dq(&moe.shexp_up_w);
                shexp_down_w_dq = dq(&moe.shexp_down_w);
                shexp_gate_inp_w_dq = dq(&moe.shexp_gate_inp_w);
                n_ff_shexp = moe.n_ff_shexp;
            } else {
                moe_router_w_dq = vec![];
                moe_gate_up_w_dq = vec![];
                moe_down_w_dq = vec![];
                n_expert = 0;
                n_expert_used = 0;
                shexp_gate_w_dq = vec![];
                shexp_up_w_dq = vec![];
                shexp_down_w_dq = vec![];
                shexp_gate_inp_w_dq = vec![];
                n_ff_shexp = 0;
            };

            let (fa_attn_norm_w, fa_wq, fa_wk) = dq3(&layer.attn_norm_w, &layer.wq, &layer.wk);
            let (fa_wv, fa_wo, fa_q_norm_w) = dq3(&layer.wv, &layer.wo, &layer.q_norm_w);
            let (fa_k_norm_w, fa_post_norm_w) = dq2(&layer.k_norm_w, &layer.post_norm_w);
            let (fa_ffn_gate_w, fa_ffn_up_w, fa_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &fa_attn_norm_w,
                wq: &fa_wq,
                wk: &fa_wk,
                wv: &fa_wv,
                wo: &fa_wo,
                q_norm_w: &fa_q_norm_w,
                k_norm_w: &fa_k_norm_w,
                post_norm_w: &fa_post_norm_w,
                ffn_gate_w: &fa_ffn_gate_w,
                ffn_up_w: &fa_ffn_up_w,
                ffn_down_w: &fa_ffn_down_w,
                moe_router_w: &moe_router_w_dq,
                moe_gate_up_w: &moe_gate_up_w_dq,
                moe_down_w: &moe_down_w_dq,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w_dq,
                shexp_up_w: &shexp_up_w_dq,
                shexp_down_w: &shexp_down_w_dq,
                shexp_gate_inp_w: &shexp_gate_inp_w_dq,
                n_ff_shexp,
            };

            let pos = [layer_idx as i32, 0, 0, 0];
            hidden = crate::model::kernels::full_layer_forward(
                &hidden,
                layer_ref.attn_norm_w,
                layer_ref.wq,
                layer_ref.wk,
                layer_ref.wv,
                layer_ref.wo,
                layer_ref.q_norm_w,
                layer_ref.k_norm_w,
                pos,
                &rope_cfg,
                layer_ref.post_norm_w,
                layer_ref.ffn_gate_w,
                layer_ref.ffn_up_w,
                layer_ref.ffn_down_w,
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                1,
                eps,
                rope_sections,
                &moe_router_w_dq,
                &moe_gate_up_w_dq,
                &moe_down_w_dq,
                n_expert,
                n_expert_used,
                &shexp_gate_w_dq,
                &shexp_up_w_dq,
                &shexp_down_w_dq,
                &shexp_gate_inp_w_dq,
                n_ff_shexp,
                None,
            );
        }
    }

    // LM head
    let output_norm_dq = dq(&model.output_norm_w);
    let output_weight_dq = dq(&model.output_weight);
    let next_token = lm_head_argmax(
        &hidden,
        &output_norm_dq,
        &output_weight_dq,
        n_embd,
        n_vocab,
        eps,
    );

    (hidden, next_token)
}

// ---------------------------------------------------------------------------
// KV cache and generation
// ---------------------------------------------------------------------------

/// Per-layer KV cache for full-attention layers.
///
/// Memory is paged: only `capacity_tokens` positions are allocated up front
/// (a small window, not the full context). `ensure_capacity` grows the
/// allocation geometrically as generation proceeds, so memory tracks actual
/// usage instead of `context_length`.
pub struct LayerKvCache {
    pub k: Vec<f32>,  // [capacity_tokens, n_kv_heads * head_size]
    pub v: Vec<f32>,  // [capacity_tokens, n_kv_heads * head_size]
    pub n_used: usize, // number of positions currently in use
    capacity_tokens: usize,
    kv_dim: usize,
}

/// Initial per-layer KV allocation (tokens). Grown on demand.
pub const KV_INITIAL_CAPACITY_TOKENS: usize = 1024;

impl LayerKvCache {
    pub fn new(n_ctx: usize, n_kv_heads: usize, head_size: usize) -> Self {
        Self::with_capacity(n_ctx.min(KV_INITIAL_CAPACITY_TOKENS), n_kv_heads, head_size)
    }

    pub fn with_capacity(capacity_tokens: usize, n_kv_heads: usize, head_size: usize) -> Self {
        let kv_dim = n_kv_heads * head_size;
        Self {
            k: vec![0.0; capacity_tokens * kv_dim],
            v: vec![0.0; capacity_tokens * kv_dim],
            n_used: 0,
            capacity_tokens,
            kv_dim,
        }
    }

    /// Allocated token slots.
    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    /// Grow K/V allocations (geometric doubling) until `tokens` positions fit.
    /// Existing entries are preserved — indexing is by absolute position.
    pub fn ensure_capacity(&mut self, tokens: usize) {
        if tokens <= self.capacity_tokens {
            return;
        }
        let mut new_cap = self.capacity_tokens.max(1);
        while new_cap < tokens {
            new_cap *= 2;
        }
        self.k.resize(new_cap * self.kv_dim, 0.0);
        self.v.resize(new_cap * self.kv_dim, 0.0);
        self.capacity_tokens = new_cap;
    }
}

/// Persistent generation state across calls.
pub struct GenerationState {
    pub kv_caches: Vec<LayerKvCache>,
    pub conv_states: Vec<Vec<f32>>,
    pub ssm_states: Vec<Vec<f32>>,
    pub pos: usize,
}

impl GenerationState {
    pub fn new(model: &ModelWeights) -> Self {
        let cfg = &model.cfg;
        let n_ctx = cfg.context_length as usize;
        let n_kv_heads = cfg.attention_head_count_kv as usize;
        let head_size = cfg.attention_key_length as usize;

        let mut kv_caches = Vec::new();
        let mut conv_states = Vec::new();
        let mut ssm_states = Vec::new();

        let mut delta_net_count = 0;
        for i in 0..cfg.block_count as usize {
            if cfg.is_recurrent(i) {
                let _layer = &model.delta_net_layers[delta_net_count];
                delta_net_count += 1;
                // conv state: channels × (kernel_size - 1)
                let kernel_size = cfg.ssm_conv_kernel as usize;
                let channels = cfg.conv_dim as usize;
                conv_states.push(vec![0.0f32; channels * kernel_size.saturating_sub(1)]);
                // ssm state: [s_v * s_v * n_heads_v]
                let s_v = cfg.head_v_dim as usize;
                let n_heads_v = cfg.ssm_time_step_rank as usize;
                ssm_states.push(vec![0.0f32; s_v * s_v * n_heads_v]);
            } else {
                kv_caches.push(LayerKvCache::new(n_ctx, n_kv_heads, head_size));
            }
        }

        Self { kv_caches, conv_states, ssm_states, pos: 0 }
    }
}

/// Prefill: process all prompt tokens at once, populating the KV cache
/// and recurrent states. Returns the hidden state after the last token.
///
/// Errors if the tokens would exceed the model's context length.
pub fn prefill(
    state: &mut GenerationState,
    token_ids: &[u32],
    model: &ModelWeights,
) -> Result<Vec<f32>, String> {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
    let eps = cfg.attention_layer_norm_rms_epsilon;
    let n_ctx = cfg.context_length as usize;

    if state.pos + token_ids.len() > n_ctx {
        return Err(format!(
            "context overflow: pos {} + {} tokens > context_length {n_ctx}",
            state.pos,
            token_ids.len()
        ));
    }

    let rope_cfg = RopeConfig {
        freq_base: cfg.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let rope_sections = cfg.rope_sections;

    // Embed all prompt tokens at once — parallelized across tokens
    let n_tokens = token_ids.len();
    let tok_embd_dq = dq(&model.tok_embd);
    let mut hidden = vec![0.0f32; n_tokens * n_embd];
    hidden
        .par_chunks_mut(n_embd)
        .zip(token_ids.par_iter())
        .for_each(|(chunk, &tid)| {
            let emb = embed_tokens(tid, &tok_embd_dq, n_embd);
            chunk.copy_from_slice(&emb);
        });

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields(layer, cfg);

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let (dn_wqkv, dn_wqkv_gate, dn_conv_kernel) = dq3(&layer.wqkv, &layer.wqkv_gate, &layer.conv_kernel);
            let (dn_alpha_bias, dn_ssm_a, dn_ssm_norm_w) = dq3(&layer.alpha_bias, &layer.ssm_a, &layer.ssm_norm_w);
            let (dn_ssm_out, dn_post_norm_w) = dq2(&layer.ssm_out, &layer.post_norm_w);
            let (dn_ffn_gate_w, dn_ffn_up_w, dn_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &dn_attn_norm_w,
                wqkv: &dn_wqkv,
                wqkv_gate: &dn_wqkv_gate,
                conv_kernel: &dn_conv_kernel,
                alpha_bias: &dn_alpha_bias,
                ssm_a: &dn_ssm_a,
                ssm_norm_w: &dn_ssm_norm_w,
                ssm_out: &dn_ssm_out,
                post_norm_w: &dn_post_norm_w,
                ffn_gate_w: &dn_ffn_gate_w,
                ffn_up_w: &dn_ffn_up_w,
                ffn_down_w: &dn_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            // Process tokens one-by-one (delta-net is inherently sequential)
            let conv_dim = cfg.conv_dim as usize;
            let conv_kernel_size = cfg.ssm_conv_kernel as usize;
            let s_v = cfg.head_v_dim as usize;
            let n_heads_v = cfg.ssm_time_step_rank as usize;

            for t in 0..n_tokens {
                let token_hidden = &hidden[t * n_embd..(t + 1) * n_embd];
                let out = delta_net_layer_forward(
                    token_hidden,
                    &layer_ref,
                    &mut state.conv_states[delta_net_idx - 1],
                    &mut state.ssm_states[delta_net_idx - 1],
                    n_embd,
                    n_ff,
                    conv_dim,
                    conv_kernel_size,
                    cfg.ba_dim as usize,
                    cfg.head_k_dim as usize,
                    s_v,
                    cfg.ssm_group_count as usize,
                    n_heads_v,
                    eps,
                );
                hidden[t * n_embd..(t + 1) * n_embd].copy_from_slice(&out);
            }
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields_full(layer, cfg);

            let (fa_attn_norm_w, fa_wq, fa_wk) = dq3(&layer.attn_norm_w, &layer.wq, &layer.wk);
            let (fa_wv, fa_wo, fa_q_norm_w) = dq3(&layer.wv, &layer.wo, &layer.q_norm_w);
            let (fa_k_norm_w, fa_post_norm_w) = dq2(&layer.k_norm_w, &layer.post_norm_w);
            let (fa_ffn_gate_w, fa_ffn_up_w, fa_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &fa_attn_norm_w,
                wq: &fa_wq,
                wk: &fa_wk,
                wv: &fa_wv,
                wo: &fa_wo,
                q_norm_w: &fa_q_norm_w,
                k_norm_w: &fa_k_norm_w,
                post_norm_w: &fa_post_norm_w,
                ffn_gate_w: &fa_ffn_gate_w,
                ffn_up_w: &fa_ffn_up_w,
                ffn_down_w: &fa_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;
            cache.ensure_capacity(nc + n_tokens);

            hidden = crate::model::kernels::full_layer_forward(
                &hidden,
                layer_ref.attn_norm_w,
                layer_ref.wq,
                layer_ref.wk,
                layer_ref.wv,
                layer_ref.wo,
                layer_ref.q_norm_w,
                layer_ref.k_norm_w,
                pos,
                &rope_cfg,
                layer_ref.post_norm_w,
                layer_ref.ffn_gate_w,
                layer_ref.ffn_up_w,
                layer_ref.ffn_down_w,
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                n_tokens,
                eps,
                rope_sections,
                &moe_router_w,
                &moe_gate_up_w,
                &moe_down_w,
                n_expert,
                n_expert_used,
                &shexp_gate_w,
                &shexp_up_w,
                &shexp_down_w,
                &shexp_gate_inp_w,
                n_ff_shexp,
                Some(crate::model::kernels::KvCacheMut {
                    k: &mut cache.k,
                    v: &mut cache.v,
                    n_cached: nc,
                }),
            );
            cache.n_used = nc + n_tokens;
        }
    }

    state.pos += n_tokens;
    Ok(hidden)
}

/// Decode a single token using KV cache and recurrent states.
/// Returns `(hidden_state, next_token_id)` or an error on context overflow.
pub fn generate_token(
    state: &mut GenerationState,
    token_id: u32,
    model: &ModelWeights,
) -> Result<(Vec<f32>, u32), String> {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
    let n_vocab = model.tok_embd.n_elements / n_embd;
    let eps = cfg.attention_layer_norm_rms_epsilon;

    if state.pos + 1 > cfg.context_length as usize {
        return Err(format!(
            "context overflow: pos {} + 1 > context_length {}",
            state.pos, cfg.context_length
        ));
    }

    let rope_cfg = RopeConfig {
        freq_base: cfg.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let rope_sections = cfg.rope_sections;

    let tok_embd_dq = dq(&model.tok_embd);
    let mut hidden = embed_tokens(token_id, &tok_embd_dq, n_embd);

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields(layer, cfg);

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let dn_wqkv = dq(&layer.wqkv);
            let dn_wqkv_gate = dq(&layer.wqkv_gate);
            let dn_conv_kernel = dq(&layer.conv_kernel);
            let dn_alpha_bias = dq(&layer.alpha_bias);
            let dn_ssm_a = dq(&layer.ssm_a);
            let dn_ssm_norm_w = dq(&layer.ssm_norm_w);
            let dn_ssm_out = dq(&layer.ssm_out);
            let dn_post_norm_w = dq(&layer.post_norm_w);
            let dn_ffn_gate_w = dq(&layer.ffn_gate_w);
            let dn_ffn_up_w = dq(&layer.ffn_up_w);
            let dn_ffn_down_w = dq(&layer.ffn_down_w);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &dn_attn_norm_w,
                wqkv: &dn_wqkv,
                wqkv_gate: &dn_wqkv_gate,
                conv_kernel: &dn_conv_kernel,
                alpha_bias: &dn_alpha_bias,
                ssm_a: &dn_ssm_a,
                ssm_norm_w: &dn_ssm_norm_w,
                ssm_out: &dn_ssm_out,
                post_norm_w: &dn_post_norm_w,
                ffn_gate_w: &dn_ffn_gate_w,
                ffn_up_w: &dn_ffn_up_w,
                ffn_down_w: &dn_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            let conv_dim = cfg.conv_dim as usize;
            let conv_kernel_size = cfg.ssm_conv_kernel as usize;
            let s_v = cfg.head_v_dim as usize;
            let n_heads_v = cfg.ssm_time_step_rank as usize;

            hidden = delta_net_layer_forward(
                &hidden,
                &layer_ref,
                &mut state.conv_states[delta_net_idx - 1],
                &mut state.ssm_states[delta_net_idx - 1],
                n_embd,
                n_ff,
                conv_dim,
                conv_kernel_size,
                cfg.ba_dim as usize,
                cfg.head_k_dim as usize,
                s_v,
                cfg.ssm_group_count as usize,
                n_heads_v,
                eps,
            );
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields_full(layer, cfg);

            let (fa_attn_norm_w, fa_wq, fa_wk) = dq3(&layer.attn_norm_w, &layer.wq, &layer.wk);
            let (fa_wv, fa_wo, fa_q_norm_w) = dq3(&layer.wv, &layer.wo, &layer.q_norm_w);
            let (fa_k_norm_w, fa_post_norm_w) = dq2(&layer.k_norm_w, &layer.post_norm_w);
            let (fa_ffn_gate_w, fa_ffn_up_w, fa_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &fa_attn_norm_w,
                wq: &fa_wq,
                wk: &fa_wk,
                wv: &fa_wv,
                wo: &fa_wo,
                q_norm_w: &fa_q_norm_w,
                k_norm_w: &fa_k_norm_w,
                post_norm_w: &fa_post_norm_w,
                ffn_gate_w: &fa_ffn_gate_w,
                ffn_up_w: &fa_ffn_up_w,
                ffn_down_w: &fa_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;
            cache.ensure_capacity(nc + 1);

            hidden = crate::model::kernels::full_layer_forward(
                &hidden,
                layer_ref.attn_norm_w,
                layer_ref.wq,
                layer_ref.wk,
                layer_ref.wv,
                layer_ref.wo,
                layer_ref.q_norm_w,
                layer_ref.k_norm_w,
                pos,
                &rope_cfg,
                layer_ref.post_norm_w,
                layer_ref.ffn_gate_w,
                layer_ref.ffn_up_w,
                layer_ref.ffn_down_w,
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                1,
                eps,
                rope_sections,
                &moe_router_w,
                &moe_gate_up_w,
                &moe_down_w,
                n_expert,
                n_expert_used,
                &shexp_gate_w,
                &shexp_up_w,
                &shexp_down_w,
                &shexp_gate_inp_w,
                n_ff_shexp,
                Some(crate::model::kernels::KvCacheMut {
                    k: &mut cache.k,
                    v: &mut cache.v,
                    n_cached: nc,
                }),
            );
            cache.n_used = nc + 1;
        }
    }

    state.pos += 1;

    let output_norm_dq = dq(&model.output_norm_w);
    let output_weight_dq = dq(&model.output_weight);
    let next_token = lm_head_argmax(&hidden, &output_norm_dq, &output_weight_dq, n_embd, n_vocab, eps);
    Ok((hidden, next_token))
}

/// Same as `generate_token` but returns raw logits instead of argmax token.
/// Caller can apply custom sampling (temperature, top-k, top-p, etc.).
pub fn generate_token_logits(
    state: &mut GenerationState,
    token_id: u32,
    model: &ModelWeights,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
    let n_vocab = model.tok_embd.n_elements / n_embd;
    let eps = cfg.attention_layer_norm_rms_epsilon;

    if state.pos + 1 > cfg.context_length as usize {
        return Err(format!(
            "context overflow: pos {} + 1 > context_length {}",
            state.pos, cfg.context_length
        ));
    }

    let rope_cfg = RopeConfig {
        freq_base: cfg.rope_freq_base,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
    };
    let rope_sections = cfg.rope_sections;

    let tok_embd_dq = dq(&model.tok_embd);
    let mut hidden = embed_tokens(token_id, &tok_embd_dq, n_embd);

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields(layer, cfg);

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let dn_wqkv = dq(&layer.wqkv);
            let dn_wqkv_gate = dq(&layer.wqkv_gate);
            let dn_conv_kernel = dq(&layer.conv_kernel);
            let dn_alpha_bias = dq(&layer.alpha_bias);
            let dn_ssm_a = dq(&layer.ssm_a);
            let dn_ssm_norm_w = dq(&layer.ssm_norm_w);
            let dn_ssm_out = dq(&layer.ssm_out);
            let dn_post_norm_w = dq(&layer.post_norm_w);
            let dn_ffn_gate_w = dq(&layer.ffn_gate_w);
            let dn_ffn_up_w = dq(&layer.ffn_up_w);
            let dn_ffn_down_w = dq(&layer.ffn_down_w);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &dn_attn_norm_w,
                wqkv: &dn_wqkv,
                wqkv_gate: &dn_wqkv_gate,
                conv_kernel: &dn_conv_kernel,
                alpha_bias: &dn_alpha_bias,
                ssm_a: &dn_ssm_a,
                ssm_norm_w: &dn_ssm_norm_w,
                ssm_out: &dn_ssm_out,
                post_norm_w: &dn_post_norm_w,
                ffn_gate_w: &dn_ffn_gate_w,
                ffn_up_w: &dn_ffn_up_w,
                ffn_down_w: &dn_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            let conv_dim = cfg.conv_dim as usize;
            let conv_kernel_size = cfg.ssm_conv_kernel as usize;
            let s_v = cfg.head_v_dim as usize;
            let n_heads_v = cfg.ssm_time_step_rank as usize;

            hidden = delta_net_layer_forward(
                &hidden,
                &layer_ref,
                &mut state.conv_states[delta_net_idx - 1],
                &mut state.ssm_states[delta_net_idx - 1],
                n_embd,
                n_ff,
                conv_dim,
                conv_kernel_size,
                cfg.ba_dim as usize,
                cfg.head_k_dim as usize,
                s_v,
                cfg.ssm_group_count as usize,
                n_heads_v,
                eps,
            );
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields_full(layer, cfg);

            let (fa_attn_norm_w, fa_wq, fa_wk) = dq3(&layer.attn_norm_w, &layer.wq, &layer.wk);
            let (fa_wv, fa_wo, fa_q_norm_w) = dq3(&layer.wv, &layer.wo, &layer.q_norm_w);
            let (fa_k_norm_w, fa_post_norm_w) = dq2(&layer.k_norm_w, &layer.post_norm_w);
            let (fa_ffn_gate_w, fa_ffn_up_w, fa_ffn_down_w) = dq3(&layer.ffn_gate_w, &layer.ffn_up_w, &layer.ffn_down_w);

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &fa_attn_norm_w,
                wq: &fa_wq,
                wk: &fa_wk,
                wv: &fa_wv,
                wo: &fa_wo,
                q_norm_w: &fa_q_norm_w,
                k_norm_w: &fa_k_norm_w,
                post_norm_w: &fa_post_norm_w,
                ffn_gate_w: &fa_ffn_gate_w,
                ffn_up_w: &fa_ffn_up_w,
                ffn_down_w: &fa_ffn_down_w,
                moe_router_w: &moe_router_w,
                moe_gate_up_w: &moe_gate_up_w,
                moe_down_w: &moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w: &shexp_gate_w,
                shexp_up_w: &shexp_up_w,
                shexp_down_w: &shexp_down_w,
                shexp_gate_inp_w: &shexp_gate_inp_w,
                n_ff_shexp,
            };

            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;
            cache.ensure_capacity(nc + 1);

            hidden = crate::model::kernels::full_layer_forward(
                &hidden,
                layer_ref.attn_norm_w,
                layer_ref.wq,
                layer_ref.wk,
                layer_ref.wv,
                layer_ref.wo,
                layer_ref.q_norm_w,
                layer_ref.k_norm_w,
                pos,
                &rope_cfg,
                layer_ref.post_norm_w,
                layer_ref.ffn_gate_w,
                layer_ref.ffn_up_w,
                layer_ref.ffn_down_w,
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                1,
                eps,
                rope_sections,
                &moe_router_w,
                &moe_gate_up_w,
                &moe_down_w,
                n_expert,
                n_expert_used,
                &shexp_gate_w,
                &shexp_up_w,
                &shexp_down_w,
                &shexp_gate_inp_w,
                n_ff_shexp,
                Some(crate::model::kernels::KvCacheMut {
                    k: &mut cache.k,
                    v: &mut cache.v,
                    n_cached: nc,
                }),
            );
            cache.n_used = nc + 1;
        }
    }

    state.pos += 1;

    let output_norm_dq = dq(&model.output_norm_w);
    let output_weight_dq = dq(&model.output_weight);
    let logits = lm_head_logits(&hidden, &output_norm_dq, &output_weight_dq, n_embd, n_vocab, eps);
    Ok((hidden, logits))
}

// Helper to extract MoE fields from a delta-net layer
type MoeFields = (Vec<f32>, Vec<f32>, Vec<f32>, usize, usize,
      Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize);

fn moe_fields(
    layer: &LoadedDeltaNetLayer,
    cfg: &crate::model::config::Qwen3_5Config,
) -> MoeFields {
    if let Some(ref moe) = layer.moe_ffn {
        (dq(&moe.router_w), dq(&moe.gate_up_w), dq(&moe.down_w),
         cfg.expert_count as usize, cfg.expert_used_count as usize,
         dq(&moe.shexp_gate_w), dq(&moe.shexp_up_w), dq(&moe.shexp_down_w),
         dq(&moe.shexp_gate_inp_w), moe.n_ff_shexp)
    } else {
        (vec![], vec![], vec![], 0, 0,
         vec![], vec![], vec![], vec![], 0)
    }
}

// Helper to extract MoE fields from a full-attention layer
fn moe_fields_full(
    layer: &LoadedFullAttnLayer,
    cfg: &crate::model::config::Qwen3_5Config,
) -> MoeFields {
    if let Some(ref moe) = layer.moe_ffn {
        (dq(&moe.router_w), dq(&moe.gate_up_w), dq(&moe.down_w),
         cfg.expert_count as usize, cfg.expert_used_count as usize,
         dq(&moe.shexp_gate_w), dq(&moe.shexp_up_w), dq(&moe.shexp_down_w),
         dq(&moe.shexp_gate_inp_w), moe.n_ff_shexp)
    } else {
        (vec![], vec![], vec![], 0, 0,
         vec![], vec![], vec![], vec![], 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_names_format() {
        let t = TensorNames::new(3);
        assert_eq!(t.attn_norm(), "blk.3.attn_norm.weight");
        assert_eq!(t.attn_qkv(), "blk.3.attn_qkv.weight");
        assert_eq!(t.ssm_conv1d(), "blk.3.ssm_conv1d.weight");
        assert_eq!(t.ffn_gate(), "blk.3.ffn_gate.weight");
        assert_eq!(t.ffn_gate_inp(), "blk.3.ffn_gate_inp.weight");
    }

    #[test]
    fn forward_pass_smoke_with_zeros() {
        // Build a tiny model with all-zero weights.
        // This tests that the pipeline doesn't crash; outputs will be
        // degenerate but valid shapes are what we care about.

        let n_embd = 32;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_size = 8;
        let n_ff = 64;
        let n_vocab = 16;
        let eps = 1e-6;

        let embd = vec![0.1f32; n_vocab * n_embd];
        let embd_bytes: Vec<u8> = embd.iter().flat_map(|f| f.to_le_bytes()).collect();
        let out_norm = vec![1.0f32; n_embd];
        let out_norm_bytes: Vec<u8> = out_norm.iter().flat_map(|f| f.to_le_bytes()).collect();

        let (hidden, next_token) = forward_pass(0, &ModelWeights {
            cfg: crate::model::config::Qwen3_5Config {
                block_count: 0,
                embedding_length: n_embd as u32,
                attention_head_count: n_heads as u32,
                attention_head_count_kv: n_kv_heads as u32,
                attention_key_length: head_size as u32,
                attention_value_length: head_size as u32,
                attention_layer_norm_rms_epsilon: eps,
                expert_count: 0,
                expert_used_count: 0,
                expert_feed_forward_length: n_ff as u32,
                expert_shared_feed_forward_length: 0,
                rope_dimension_count: head_size as u32,
                rope_freq_base: 1e7,
                context_length: 128,
                ssm_state_size: 0,
                ssm_group_count: 0,
                ssm_time_step_rank: 0,
                ssm_conv_kernel: 0,
                ssm_inner_size: None,
                full_attention_interval: 4,
                rope_sections: [0; 4],
                key_dim: 0,
                value_dim: 0,
                conv_dim: 0,
                head_k_dim: 0,
                head_v_dim: 0,
                ba_dim: 0,
                full_attn_q_fused_dim: 0,
            },
            tok_embd: RawTensor::new(crate::gguf::GGmlType::F32, embd_bytes.clone(), n_vocab * n_embd),
            output_norm_w: RawTensor::new(crate::gguf::GGmlType::F32, out_norm_bytes, n_embd),
            output_weight: RawTensor::new(crate::gguf::GGmlType::F32, embd_bytes, n_vocab * n_embd),
            full_attn_layers: vec![],
            delta_net_layers: vec![],
        });

        // Shouldn't panic; token is in vocab range
        assert!((next_token as usize) < n_vocab, "next_token {next_token} >= n_vocab {n_vocab}");
        assert_eq!(hidden.len(), n_embd);
    }

    /// Bare model: all layers full-attention (`full_attention_interval = 1`),
    /// zero-filled global weights. Enough to exercise KV-cache state
    /// management without running layer forwards.
    fn bare_model(block_count: u32, context_length: u32) -> ModelWeights {
        let n_embd = 32usize;
        let embd = vec![0.1f32; 16 * n_embd];
        let embd_bytes: Vec<u8> = embd.iter().flat_map(|f| f.to_le_bytes()).collect();
        let ones: Vec<u8> = vec![1.0f32; n_embd].iter().flat_map(|f| f.to_le_bytes()).collect();
        ModelWeights {
            cfg: crate::model::config::Qwen3_5Config {
                block_count,
                embedding_length: n_embd as u32,
                attention_head_count: 4,
                attention_head_count_kv: 2,
                attention_key_length: 8,
                attention_value_length: 8,
                attention_layer_norm_rms_epsilon: 1e-6,
                expert_count: 0,
                expert_used_count: 0,
                expert_feed_forward_length: 64,
                expert_shared_feed_forward_length: 0,
                rope_dimension_count: 8,
                rope_freq_base: 1e7,
                context_length,
                ssm_state_size: 0,
                ssm_group_count: 0,
                ssm_time_step_rank: 0,
                ssm_conv_kernel: 0,
                ssm_inner_size: None,
                full_attention_interval: 1,
                rope_sections: [0; 4],
                key_dim: 0,
                value_dim: 0,
                conv_dim: 0,
                head_k_dim: 0,
                head_v_dim: 0,
                ba_dim: 0,
                full_attn_q_fused_dim: 0,
            },
            tok_embd: RawTensor::new(crate::gguf::GGmlType::F32, embd_bytes.clone(), 16 * n_embd),
            output_norm_w: RawTensor::new(crate::gguf::GGmlType::F32, ones.clone(), n_embd),
            output_weight: RawTensor::new(crate::gguf::GGmlType::F32, embd_bytes, 16 * n_embd),
            full_attn_layers: vec![],
            delta_net_layers: vec![],
        }
    }

    #[test]
    fn kv_cache_starts_small_and_grows_geometrically() {
        let mut cache = LayerKvCache::new(262_144, 2, 8);
        // Initial window: min(ctx, 1024) tokens, NOT the full 262K context.
        assert_eq!(cache.capacity_tokens(), 1024);
        assert_eq!(cache.k.len(), 1024 * 2 * 8);

        // Within capacity: no reallocation.
        cache.ensure_capacity(1000);
        assert_eq!(cache.capacity_tokens(), 1024);

        // Growth: geometric doubling to the next power-of-two multiple.
        cache.ensure_capacity(1025);
        assert_eq!(cache.capacity_tokens(), 2048);
        cache.ensure_capacity(3000);
        assert_eq!(cache.capacity_tokens(), 4096);

        // Data written before growth survives (absolute-position indexing).
        let mut fresh = LayerKvCache::with_capacity(4, 1, 4);
        fresh.k[..4].copy_from_slice(&[7.5, 8.5, 9.5, 10.5]);
        fresh.ensure_capacity(100);
        assert_eq!(&fresh.k[..4], &[7.5, 8.5, 9.5, 10.5]);
        assert_eq!(fresh.capacity_tokens(), 128);
    }

    #[test]
    fn generation_state_allocates_window_not_full_context() {
        let model = bare_model(2, 262_144);
        let state = GenerationState::new(&model);

        assert_eq!(state.kv_caches.len(), 2);
        for cache in &state.kv_caches {
            // 1024 tokens × kv_dim 16 × f32 ≈ 64 KB per tensor per layer,
            // vs ~16 MB if the full 262K context were preallocated.
            let bytes = cache.k.len() * 4;
            assert_eq!(cache.capacity_tokens(), 1024);
            assert!(bytes <= 128 * 1024, "initial allocation too large: {bytes} bytes");
        }
    }

    #[test]
    fn context_overflow_is_an_error_not_a_panic() {
        let model = bare_model(2, 32);

        // Prefill past the limit → Err from the guard.
        let mut state = GenerationState::new(&model);
        let tokens: Vec<u32> = (0..33).collect();
        let err = prefill(&mut state, &tokens, &model).unwrap_err();
        assert!(err.contains("context overflow"), "{err}");

        // Decode at a full context → Err from the guard, not an index panic.
        let mut state = GenerationState::new(&model);
        state.pos = 32;
        let err = generate_token_logits(&mut state, 0, &model).unwrap_err();
        assert!(err.contains("context overflow"), "{err}");
        let err = generate_token(&mut state, 0, &model).unwrap_err();
        assert!(err.contains("context overflow"), "{err}");
    }
}
