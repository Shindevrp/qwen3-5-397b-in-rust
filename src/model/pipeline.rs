//! End-to-end inference pipeline: GGUF weights → unified forward → token.
//!
//! Loads dequantised weight slices from a `ModelLoader`, wires them into the
//! kernel-level layer structs, and drives the token-by-token forward pass
//! through both recurrent (delta-net) and full-attention layers.

use crate::model::config::Qwen3_5Config;
use crate::model::kernels::{
    KvCacheMut, KvStoreMut, RopeConfig, embed_tokens, lm_head_argmax, lm_head_logits,
};
use crate::model::loader::ModelLoader;
use crate::model::quant::RawTensor;
use rayon::prelude::*;
use std::time::Instant;

/// Timing measurements for inference stages.
#[derive(Debug, Clone, Default)]
pub struct TimingInfo {
    /// Time to embed all prompt tokens (prefill only).
    pub embed_us: u64,
    /// Time spent in delta-net (recurrent) layers.
    pub delta_net_us: u64,
    /// Time spent in full-attention layers.
    pub full_attn_us: u64,
    /// Time for LM head (argmax or logits).
    pub lm_head_us: u64,
    /// Total wall time for the call.
    pub total_us: u64,
}

impl std::fmt::Display for TimingInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total={:.1}ms embed={:.1}ms delta_net={:.1}ms full_attn={:.1}ms lm_head={:.1}ms",
            self.total_us as f64 / 1000.0,
            self.embed_us as f64 / 1000.0,
            self.delta_net_us as f64 / 1000.0,
            self.full_attn_us as f64 / 1000.0,
            self.lm_head_us as f64 / 1000.0,
        )
    }
}

/// GGUF tensor name helpers per layer.
struct TensorNames {
    prefix: String,
}

impl TensorNames {
    fn new(layer: usize) -> Self {
        Self {
            prefix: format!("blk.{layer}."),
        }
    }

    fn name(&self, suffix: &str) -> String {
        format!("{}{}", self.prefix, suffix)
    }

    // --- shared ---
    fn attn_norm(&self) -> String {
        self.name("attn_norm.weight")
    }
    fn post_attn_norm(&self) -> String {
        self.name("post_attention_norm.weight")
    }

    // --- full attention ---
    fn attn_q(&self) -> String {
        self.name("attn_q.weight")
    }
    fn attn_k(&self) -> String {
        self.name("attn_k.weight")
    }
    fn attn_v(&self) -> String {
        self.name("attn_v.weight")
    }
    fn attn_o(&self) -> String {
        self.name("attn_output.weight")
    }
    fn attn_q_norm(&self) -> String {
        self.name("attn_q_norm.weight")
    }
    fn attn_k_norm(&self) -> String {
        self.name("attn_k_norm.weight")
    }

    // --- delta-net ---
    fn attn_qkv(&self) -> String {
        self.name("attn_qkv.weight")
    }
    fn attn_gate(&self) -> String {
        self.name("attn_gate.weight")
    }
    fn ssm_conv1d(&self) -> String {
        self.name("ssm_conv1d.weight")
    }
    fn ssm_dt(&self) -> String {
        self.name("ssm_dt.bias")
    }
    fn ssm_a(&self) -> String {
        self.name("ssm_a")
    }
    fn ssm_norm(&self) -> String {
        self.name("ssm_norm.weight")
    }
    fn ssm_out(&self) -> String {
        self.name("ssm_out.weight")
    }

    // --- dense FFN ---
    fn ffn_gate(&self) -> String {
        self.name("ffn_gate.weight")
    }
    fn ffn_up(&self) -> String {
        self.name("ffn_up.weight")
    }
    fn ffn_down(&self) -> String {
        self.name("ffn_down.weight")
    }

    // --- MoE FFN ---
    #[allow(dead_code)]
    fn ffn_gate_inp(&self) -> String {
        self.name("ffn_gate_inp.weight")
    }
    #[allow(dead_code)]
    fn ffn_gate_exps(&self) -> String {
        self.name("ffn_gate_exps.weight")
    }
    #[allow(dead_code)]
    fn ffn_up_exps(&self) -> String {
        self.name("ffn_up_exps.weight")
    }
    #[allow(dead_code)]
    fn ffn_down_exps(&self) -> String {
        self.name("ffn_down_exps.weight")
    }

    // --- shared expert ---
    #[allow(dead_code)]
    fn ffn_gate_inp_shexp(&self) -> String {
        self.name("ffn_gate_inp_shexp.weight")
    }
    #[allow(dead_code)]
    fn ffn_gate_shexp(&self) -> String {
        self.name("ffn_gate_shexp.weight")
    }
    #[allow(dead_code)]
    fn ffn_up_shexp(&self) -> String {
        self.name("ffn_up_shexp.weight")
    }
    #[allow(dead_code)]
    fn ffn_down_shexp(&self) -> String {
        self.name("ffn_down_shexp.weight")
    }
}

// ---------------------------------------------------------------------------
// MoE FFN weights (fused gate+up for the kernel)
// ---------------------------------------------------------------------------

pub struct LoadedMoeFfn {
    pub router_w: RawTensor,    // [n_expert, n_embd]
    pub gate_exps_q: RawTensor, // [n_expert, n_ff, n_embd] quantized
    pub up_exps_q: RawTensor,   // [n_expert, n_ff, n_embd] quantized
    pub down_w: RawTensor,      // [n_expert, n_embd, n_ff]
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
#[allow(dead_code)]
fn dq2(a: &RawTensor, b: &RawTensor) -> (Vec<f32>, Vec<f32>) {
    let mut ra = None;
    let mut rb = None;
    rayon::scope(|s| {
        s.spawn(|_| {
            ra = Some(dq(a));
        });
        s.spawn(|_| {
            rb = Some(dq(b));
        });
    });
    (ra.unwrap(), rb.unwrap())
}

/// Parallel dequantization of three `RawTensor`s.
fn dq3(a: &RawTensor, b: &RawTensor, c: &RawTensor) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut ra = None;
    let mut rb = None;
    let mut rc = None;
    rayon::scope(|s| {
        s.spawn(|_| {
            ra = Some(dq(a));
        });
        s.spawn(|_| {
            rb = Some(dq(b));
        });
        s.spawn(|_| {
            rc = Some(dq(c));
        });
    });
    (ra.unwrap(), rb.unwrap(), rc.unwrap())
}

impl ModelWeights {
    pub fn load(loader: &ModelLoader) -> Result<Self, String> {
        let cfg = loader.cfg.clone();
        let n_embd = cfg.embedding_length as usize;
        let n_vocab = {
            let meta = loader
                .tensor_meta("token_embd.weight")
                .ok_or("missing token_embd.weight")?;
            (meta.n_elements() / n_embd as u64) as usize
        };

        // Global weights — stored as raw quantized bytes
        let tok_embd = loader
            .raw_tensor("token_embd.weight")
            .map_err(|e| e.to_string())?;
        let output_norm_w = loader
            .raw_tensor("output_norm.weight")
            .map_err(|e| e.to_string())?;
        let output_weight = if loader.tensor_meta("output.weight").is_some() {
            loader
                .raw_tensor("output.weight")
                .map_err(|e| e.to_string())?
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
        // Keep gate/up quantized, dequant per expert on demand
        let load_moe = |t: &TensorNames| -> Result<Option<LoadedMoeFfn>, String> {
            if n_expert == 0 {
                return Ok(None);
            }
            let router_w = raw(&t.ffn_gate_inp())?;
            let gate_exps_q = raw(&t.ffn_gate_exps())?;
            let up_exps_q = raw(&t.ffn_up_exps())?;
            let down_w = raw(&t.ffn_down_exps())?;

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
                gate_exps_q,
                up_exps_q,
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
                    wq,
                    wk,
                    wv,
                    wo,
                    q_norm_w,
                    k_norm_w,
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
pub fn forward_pass(token_id: u32, model: &ModelWeights) -> (Vec<f32>, u32) {
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
                moe_gate_up_w_dq = vec![];
                moe_down_w_dq = vec![];
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

            let (dn_attn_norm_w, dn_conv_kernel, dn_alpha_bias) =
                dq3(&layer.attn_norm_w, &layer.conv_kernel, &layer.alpha_bias);
            let (dn_ssm_a, dn_ssm_norm_w, dn_post_norm_w) =
                dq3(&layer.ssm_a, &layer.ssm_norm_w, &layer.post_norm_w);

            hidden = crate::model::kernels::delta_net_layer_forward_q(
                &hidden,
                &dn_attn_norm_w,
                (layer.wqkv.data(), layer.wqkv.ty),
                (layer.wqkv_gate.data(), layer.wqkv_gate.ty),
                &dn_conv_kernel,
                &dn_alpha_bias,
                &dn_ssm_a,
                &dn_ssm_norm_w,
                (layer.ssm_out.data(), layer.ssm_out.ty),
                &dn_post_norm_w,
                (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
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
                layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.down_w),
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
                &mut conv_state,
                &mut ssm_state,
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
                moe_gate_up_w_dq = vec![];
                moe_down_w_dq = vec![];
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

            let (fa_attn_norm_w, fa_q_norm_w, fa_k_norm_w) =
                dq3(&layer.attn_norm_w, &layer.q_norm_w, &layer.k_norm_w);
            let fa_post_norm_w = dq(&layer.post_norm_w);

            let pos = [layer_idx as i32, 0, 0, 0];
            hidden = crate::model::kernels::full_layer_forward_q(
                &hidden,
                &fa_attn_norm_w,
                (layer.wq.data(), layer.wq.ty),
                (layer.wk.data(), layer.wk.ty),
                (layer.wv.data(), layer.wv.ty),
                (layer.wo.data(), layer.wo.ty),
                &fa_q_norm_w,
                &fa_k_norm_w,
                pos,
                &rope_cfg,
                &fa_post_norm_w,
                (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
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
                layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.down_w),
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
///
/// Phase 25: backing may be plain f32 rows or Q8_0-packed rows (~3.8x less
/// durable memory). Quantization is transparent to forward passes — kernels
/// pack on write and dequantize into a transient scratch on read.
pub struct LayerKvCache {
    backing: KvBacking,
    pub n_used: usize, // number of positions currently in use
    capacity_tokens: usize,
    kv_dim: usize,
}

enum KvBacking {
    F32 { k: Vec<f32>, v: Vec<f32> },
    Q8 { k: Vec<u8>, v: Vec<u8> },
}

/// Initial per-layer KV allocation (tokens). Grown on demand.
pub const KV_INITIAL_CAPACITY_TOKENS: usize = 1024;

impl LayerKvCache {
    pub fn new(n_ctx: usize, n_kv_heads: usize, head_size: usize) -> Self {
        Self::with_capacity(n_ctx.min(KV_INITIAL_CAPACITY_TOKENS), n_kv_heads, head_size)
    }

    /// Same as `new` but with Q8_0-packed K/V storage (Phase 25).
    pub fn new_quantized(n_ctx: usize, n_kv_heads: usize, head_size: usize) -> Self {
        Self::with_capacity_impl(
            n_ctx.min(KV_INITIAL_CAPACITY_TOKENS),
            n_kv_heads,
            head_size,
            true,
        )
    }

    pub fn with_capacity(capacity_tokens: usize, n_kv_heads: usize, head_size: usize) -> Self {
        Self::with_capacity_impl(capacity_tokens, n_kv_heads, head_size, false)
    }

    fn with_capacity_impl(
        capacity_tokens: usize,
        n_kv_heads: usize,
        head_size: usize,
        quantized: bool,
    ) -> Self {
        let kv_dim = n_kv_heads * head_size;
        let backing = if quantized {
            let row = KvStoreMut::q8_row_bytes(kv_dim);
            KvBacking::Q8 {
                k: vec![0u8; capacity_tokens * row],
                v: vec![0u8; capacity_tokens * row],
            }
        } else {
            KvBacking::F32 {
                k: vec![0.0; capacity_tokens * kv_dim],
                v: vec![0.0; capacity_tokens * kv_dim],
            }
        };
        Self {
            backing,
            n_used: 0,
            capacity_tokens,
            kv_dim,
        }
    }

    /// Whether this cache stores quantized (Q8_0) rows.
    pub fn is_quantized(&self) -> bool {
        matches!(self.backing, KvBacking::Q8 { .. })
    }

    /// Allocated token slots.
    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    /// Durable bytes currently allocated for this cache (K + V combined).
    pub fn allocated_bytes(&self) -> usize {
        match &self.backing {
            KvBacking::F32 { k, v } => (k.len() + v.len()) * 4,
            KvBacking::Q8 { k, v } => k.len() + v.len(),
        }
    }

    /// Grow allocations (geometric doubling) until `tokens` positions fit.
    /// Existing entries are preserved — indexing is by absolute position.
    pub fn ensure_capacity(&mut self, tokens: usize) {
        if tokens <= self.capacity_tokens {
            return;
        }
        let mut new_cap = self.capacity_tokens.max(1);
        while new_cap < tokens {
            new_cap *= 2;
        }
        match &mut self.backing {
            KvBacking::F32 { k, v } => {
                k.resize(new_cap * self.kv_dim, 0.0);
                v.resize(new_cap * self.kv_dim, 0.0);
            }
            KvBacking::Q8 { k, v } => {
                let row = KvStoreMut::q8_row_bytes(self.kv_dim);
                k.resize(new_cap * row, 0);
                v.resize(new_cap * row, 0);
            }
        }
        self.capacity_tokens = new_cap;
    }

    /// Borrow the backing store for a kernel call. The returned view borrows
    /// the whole cache; update `n_used` only after the borrow ends.
    pub fn kv_store_mut(&mut self) -> KvStoreMut<'_> {
        let nc = self.n_used;
        match &mut self.backing {
            KvBacking::F32 { k, v } => KvStoreMut::F32(KvCacheMut { k, v, n_cached: nc }),
            KvBacking::Q8 { k, v } => KvStoreMut::Q8 {
                k,
                v,
                n_cached: nc,
                kv_dim: self.kv_dim,
            },
        }
    }
}

/// Persistent generation state across calls.
pub struct GenerationState {
    pub kv_caches: Vec<LayerKvCache>,
    pub conv_states: Vec<Vec<f32>>,
    pub ssm_states: Vec<Vec<f32>>,
    pub pos: usize,
    /// Timing of the most recent prefill/generate call.
    pub last_timing: TimingInfo,
    /// Cached dequantized MoE weights per MoE layer.
    /// Indexed by MoE layer index (not global layer index).
    /// Populated lazily on first access; reused on subsequent tokens.
    moe_cache: Vec<Option<MoeWeightCacheEntry>>,
}

/// Dequantized *small* MoE weights for a single layer, cached across tokens.
/// Phase 30: expert gate_up/down are no longer cached at all — they stream
/// directly from mmap'd quantized bytes per selected expert.
struct MoeWeightCacheEntry {
    router_w: Vec<f32>,
    shexp_gate_w: Vec<f32>,
    shexp_up_w: Vec<f32>,
    shexp_down_w: Vec<f32>,
    shexp_gate_inp_w: Vec<f32>,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_shexp: usize,
}

impl MoeWeightCacheEntry {
    fn new(moe: &LoadedMoeFfn, n_expert: usize, n_expert_used: usize) -> Self {
        Self {
            router_w: dq(&moe.router_w),
            shexp_gate_w: dq(&moe.shexp_gate_w),
            shexp_up_w: dq(&moe.shexp_up_w),
            shexp_down_w: dq(&moe.shexp_down_w),
            shexp_gate_inp_w: dq(&moe.shexp_gate_inp_w),
            n_expert,
            n_expert_used,
            n_ff_shexp: moe.n_ff_shexp,
        }
    }

    fn as_fields(&self) -> MoeFieldsRef<'_> {
        static EMPTY: [f32; 0] = [];
        (
            &self.router_w[..],
            &EMPTY[..],
            &EMPTY[..],
            self.n_expert,
            self.n_expert_used,
            &self.shexp_gate_w[..],
            &self.shexp_up_w[..],
            &self.shexp_down_w[..],
            &self.shexp_gate_inp_w[..],
            self.n_ff_shexp,
        )
    }
}

impl GenerationState {
    pub fn new(model: &ModelWeights) -> Self {
        Self::new_impl(model, false, KV_INITIAL_CAPACITY_TOKENS)
    }

    /// Like `new` but full-attention layers get Q8_0-packed KV caches
    /// (~3.8x less durable cache memory, small dequantize-on-read cost).
    pub fn new_kv_q8(model: &ModelWeights) -> Self {
        Self::new_impl(model, true, KV_INITIAL_CAPACITY_TOKENS)
    }

    /// Memory-bounded construction with small KV cache.
    pub fn new_memory_bounded(model: &ModelWeights) -> Self {
        Self::new_impl(model, false, 64)
    }

    /// Memory-bounded construction with Q8 KV cache.
    pub fn new_memory_bounded_kv_q8(model: &ModelWeights) -> Self {
        Self::new_impl(model, true, 64)
    }

    fn new_impl(model: &ModelWeights, kv_quantized: bool, kv_init_tokens: usize) -> Self {
        let cfg = &model.cfg;
        let n_ctx = cfg.context_length as usize;
        let n_kv_heads = cfg.attention_head_count_kv as usize;
        let head_size = cfg.attention_key_length as usize;

        let mut kv_caches = Vec::new();
        let mut conv_states = Vec::new();
        let mut ssm_states = Vec::new();

        let init_cap = kv_init_tokens.min(n_ctx);
        for i in 0..cfg.block_count as usize {
            if cfg.is_recurrent(i) {
                let kernel_size = cfg.ssm_conv_kernel as usize;
                let channels = cfg.conv_dim as usize;
                conv_states.push(vec![0.0f32; channels * kernel_size.saturating_sub(1)]);
                let s_v = cfg.head_v_dim as usize;
                let n_heads_v = cfg.ssm_time_step_rank as usize;
                ssm_states.push(vec![0.0f32; s_v * s_v * n_heads_v]);
            } else {
                let cache =
                    LayerKvCache::with_capacity_impl(init_cap, n_kv_heads, head_size, kv_quantized);
                kv_caches.push(cache);
            }
        }

        // One cache slot per layer, indexed by absolute layer index. Dense
        // layers leave their slot None forever (an Option per layer is free);
        // MoE layers populate theirs on first forward and reuse it after.
        let moe_cache: Vec<Option<MoeWeightCacheEntry>> =
            (0..cfg.block_count as usize).map(|_| None).collect();

        Self {
            kv_caches,
            conv_states,
            ssm_states,
            pos: 0,
            last_timing: TimingInfo::default(),
            moe_cache,
        }
    }

    /// Number of MoE layers whose weights are currently resident in the cache.
    pub fn moe_cache_filled(&self) -> usize {
        self.moe_cache.iter().filter(|s| s.is_some()).count()
    }

    /// Total number of MoE cache slots (one per layer).
    pub fn moe_cache_slots(&self) -> usize {
        self.moe_cache.len()
    }

    /// Phase 24: eagerly dequantize every MoE layer's weights into the cache,
    /// parallelized across layers with rayon.
    ///
    /// Calling this once before a generation loop moves all first-touch
    /// dequantization off the per-token critical path: afterwards, every
    /// `generate_token` hits warm cache entries and pays zero dequant cost.
    /// Safe to call multiple times — already-filled slots are left untouched.
    pub fn warm_moe_cache(&mut self, model: &ModelWeights) {
        use rayon::prelude::*;

        let cfg = &model.cfg;
        let n_expert = cfg.expert_count as usize;
        let n_expert_used = cfg.expert_used_count as usize;

        // Map absolute layer index → MoE weights (if any) for that layer.
        // Layer arrays may be empty in bare-bones test models, so use .get().
        let mut delta_net_idx = 0usize;
        let mut full_attn_idx = 0usize;
        let moe_refs: Vec<Option<&LoadedMoeFfn>> = (0..cfg.block_count as usize)
            .map(|i| {
                if cfg.is_recurrent(i) {
                    let l = model.delta_net_layers.get(delta_net_idx);
                    delta_net_idx += 1;
                    l.and_then(|l| l.moe_ffn.as_ref())
                } else {
                    let l = model.full_attn_layers.get(full_attn_idx);
                    full_attn_idx += 1;
                    l.and_then(|l| l.moe_ffn.as_ref())
                }
            })
            .collect();

        // Dequantize in parallel (read-only access to mmap'd weights).
        let filled: Vec<Option<MoeWeightCacheEntry>> = moe_refs
            .par_iter()
            .map(|m| m.map(|m| MoeWeightCacheEntry::new(m, n_expert, n_expert_used)))
            .collect();

        // Merge: never overwrite an entry that was already resident.
        for (slot, entry) in self.moe_cache.iter_mut().zip(filled) {
            if slot.is_none() {
                *slot = entry;
            }
        }
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
    let t_start = Instant::now();
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
    let t_embed = Instant::now();
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
    let embed_us = t_embed.elapsed().as_micros() as u64;

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;
    let mut delta_net_us: u64 = 0;
    let mut full_attn_us: u64 = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
            ) = moe_fields_cached(
                &mut state.moe_cache[layer_idx],
                layer.moe_ffn.as_ref(),
                cfg.expert_count as usize,
                cfg.expert_used_count as usize,
            );

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let (dn_conv_kernel, dn_alpha_bias, dn_ssm_a) =
                dq3(&layer.conv_kernel, &layer.alpha_bias, &layer.ssm_a);
            let dn_ssm_norm_w = dq(&layer.ssm_norm_w);
            let dn_post_norm_w = dq(&layer.post_norm_w);

            // Process tokens one-by-one (delta-net is inherently sequential)
            let conv_dim = cfg.conv_dim as usize;
            let conv_kernel_size = cfg.ssm_conv_kernel as usize;
            let s_v = cfg.head_v_dim as usize;
            let n_heads_v = cfg.ssm_time_step_rank as usize;

            let t_dn = Instant::now();
            for t in 0..n_tokens {
                let token_hidden = &hidden[t * n_embd..(t + 1) * n_embd];
                let out = crate::model::kernels::delta_net_layer_forward_q(
                    token_hidden,
                    &dn_attn_norm_w,
                    (layer.wqkv.data(), layer.wqkv.ty),
                    (layer.wqkv_gate.data(), layer.wqkv_gate.ty),
                    &dn_conv_kernel,
                    &dn_alpha_bias,
                    &dn_ssm_a,
                    &dn_ssm_norm_w,
                    (layer.ssm_out.data(), layer.ssm_out.ty),
                    &dn_post_norm_w,
                    (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                    (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                    (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                    moe_router_w,
                    moe_gate_up_w,
                    moe_down_w,
                    n_expert,
                    n_expert_used,
                    shexp_gate_w,
                    shexp_up_w,
                    shexp_down_w,
                    shexp_gate_inp_w,
                    n_ff_shexp,
                    layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                    layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                    layer.moe_ffn.as_ref().map(|m| &m.down_w),
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
                    &mut state.conv_states[delta_net_idx - 1],
                    &mut state.ssm_states[delta_net_idx - 1],
                );
                hidden[t * n_embd..(t + 1) * n_embd].copy_from_slice(&out);
            }
            delta_net_us += t_dn.elapsed().as_micros() as u64;
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let (
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
            ) = moe_fields_cached(
                &mut state.moe_cache[layer_idx],
                layer.moe_ffn.as_ref(),
                cfg.expert_count as usize,
                cfg.expert_used_count as usize,
            );

            let (fa_attn_norm_w, fa_q_norm_w, fa_k_norm_w) =
                dq3(&layer.attn_norm_w, &layer.q_norm_w, &layer.k_norm_w);
            let fa_post_norm_w = dq(&layer.post_norm_w);

            let t_fa = Instant::now();
            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;
            cache.ensure_capacity(nc + n_tokens);

            hidden = crate::model::kernels::full_layer_forward_q(
                &hidden,
                &fa_attn_norm_w,
                (layer.wq.data(), layer.wq.ty),
                (layer.wk.data(), layer.wk.ty),
                (layer.wv.data(), layer.wv.ty),
                (layer.wo.data(), layer.wo.ty),
                &fa_q_norm_w,
                &fa_k_norm_w,
                pos,
                &rope_cfg,
                &fa_post_norm_w,
                (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                n_tokens,
                eps,
                rope_sections,
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
                layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.down_w),
                Some(cache.kv_store_mut()),
            );
            cache.n_used = nc + n_tokens;
            full_attn_us += t_fa.elapsed().as_micros() as u64;
        }
    }

    state.pos += n_tokens;
    state.last_timing = TimingInfo {
        embed_us,
        delta_net_us,
        full_attn_us,
        lm_head_us: 0,
        total_us: t_start.elapsed().as_micros() as u64,
    };
    Ok(hidden)
}

/// Chunked prefill: process prompt tokens in chunks of `chunk_size`,
/// populating KV cache and recurrent states incrementally.
///
/// Each chunk goes through ALL layers before the next chunk starts.
/// This bounds peak memory and allows interleaving with decode if needed.
/// Returns the hidden state after the last chunk.
pub fn prefill_chunked(
    state: &mut GenerationState,
    token_ids: &[u32],
    model: &ModelWeights,
    chunk_size: usize,
) -> Result<Vec<f32>, String> {
    let t_start = Instant::now();
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

    let tok_embd_dq = dq(&model.tok_embd);
    let n_tokens = token_ids.len();
    let mut all_hidden = vec![0.0f32; n_tokens * n_embd];

    let mut total_embed_us: u64 = 0;
    let mut total_delta_net_us: u64 = 0;
    let mut total_full_attn_us: u64 = 0;

    for chunk_start in (0..n_tokens).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_tokens);
        let chunk_n = chunk_end - chunk_start;

        // Embed this chunk (parallel across chunk tokens)
        let t_emb = Instant::now();
        let mut chunk_hidden = vec![0.0f32; chunk_n * n_embd];
        chunk_hidden
            .par_chunks_mut(n_embd)
            .zip(token_ids[chunk_start..chunk_end].par_iter())
            .for_each(|(chunk, &tid)| {
                let emb = embed_tokens(tid, &tok_embd_dq, n_embd);
                chunk.copy_from_slice(&emb);
            });
        total_embed_us += t_emb.elapsed().as_micros() as u64;

        let mut full_attn_idx = 0;
        let mut delta_net_idx = 0;

        for layer_idx in 0..cfg.block_count as usize {
            if cfg.is_recurrent(layer_idx) {
                let layer = &model.delta_net_layers[delta_net_idx];
                delta_net_idx += 1;

                let (
                    moe_router_w,
                    moe_gate_up_w,
                    moe_down_w,
                    n_expert,
                    n_expert_used,
                    shexp_gate_w,
                    shexp_up_w,
                    shexp_down_w,
                    shexp_gate_inp_w,
                    n_ff_shexp,
                ) = moe_fields_cached(
                    &mut state.moe_cache[layer_idx],
                    layer.moe_ffn.as_ref(),
                    cfg.expert_count as usize,
                    cfg.expert_used_count as usize,
                );

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let (dn_conv_kernel, dn_alpha_bias, dn_ssm_a) =
                dq3(&layer.conv_kernel, &layer.alpha_bias, &layer.ssm_a);
            let dn_ssm_norm_w = dq(&layer.ssm_norm_w);
            let dn_post_norm_w = dq(&layer.post_norm_w);

                let conv_dim = cfg.conv_dim as usize;
                let conv_kernel_size = cfg.ssm_conv_kernel as usize;
                let s_v = cfg.head_v_dim as usize;
                let n_heads_v = cfg.ssm_time_step_rank as usize;

                let t_dn = Instant::now();
                for t in 0..chunk_n {
                    let token_hidden = &chunk_hidden[t * n_embd..(t + 1) * n_embd];
                    let out = crate::model::kernels::delta_net_layer_forward_q(
                        token_hidden,
                        &dn_attn_norm_w,
                        (layer.wqkv.data(), layer.wqkv.ty),
                        (layer.wqkv_gate.data(), layer.wqkv_gate.ty),
                        &dn_conv_kernel,
                        &dn_alpha_bias,
                        &dn_ssm_a,
                        &dn_ssm_norm_w,
                        (layer.ssm_out.data(), layer.ssm_out.ty),
                        &dn_post_norm_w,
                        (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                        (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                        (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                        moe_router_w,
                        moe_gate_up_w,
                        moe_down_w,
                        n_expert,
                        n_expert_used,
                        shexp_gate_w,
                        shexp_up_w,
                        shexp_down_w,
                        shexp_gate_inp_w,
                        n_ff_shexp,
                        layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                        layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                        layer.moe_ffn.as_ref().map(|m| &m.down_w),
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
                        &mut state.conv_states[delta_net_idx - 1],
                        &mut state.ssm_states[delta_net_idx - 1],
                    );
                    chunk_hidden[t * n_embd..(t + 1) * n_embd].copy_from_slice(&out);
                }
                total_delta_net_us += t_dn.elapsed().as_micros() as u64;
            } else {
                let layer = &model.full_attn_layers[full_attn_idx];
                full_attn_idx += 1;

                let (
                    moe_router_w,
                    moe_gate_up_w,
                    moe_down_w,
                    n_expert,
                    n_expert_used,
                    shexp_gate_w,
                    shexp_up_w,
                    shexp_down_w,
                    shexp_gate_inp_w,
                    n_ff_shexp,
                ) = moe_fields_cached(
                    &mut state.moe_cache[layer_idx],
                    layer.moe_ffn.as_ref(),
                    cfg.expert_count as usize,
                    cfg.expert_used_count as usize,
                );

                let (fa_attn_norm_w, fa_q_norm_w, fa_k_norm_w) =
                    dq3(&layer.attn_norm_w, &layer.q_norm_w, &layer.k_norm_w);
                let fa_post_norm_w = dq(&layer.post_norm_w);

                let t_fa = Instant::now();
                let pos = [state.pos as i32, 0, 0, 0];
                let cache = &mut state.kv_caches[full_attn_idx - 1];
                let nc = cache.n_used;
                cache.ensure_capacity(nc + chunk_n);

                chunk_hidden = crate::model::kernels::full_layer_forward_q(
                    &chunk_hidden,
                    &fa_attn_norm_w,
                    (layer.wq.data(), layer.wq.ty),
                    (layer.wk.data(), layer.wk.ty),
                    (layer.wv.data(), layer.wv.ty),
                    (layer.wo.data(), layer.wo.ty),
                    &fa_q_norm_w,
                    &fa_k_norm_w,
                    pos,
                    &rope_cfg,
                    &fa_post_norm_w,
                    (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                    (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                    (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                    n_embd,
                    n_heads,
                    n_kv_heads,
                    head_size,
                    n_ff,
                    chunk_n,
                    eps,
                    rope_sections,
                    moe_router_w,
                    moe_gate_up_w,
                    moe_down_w,
                    n_expert,
                    n_expert_used,
                    shexp_gate_w,
                    shexp_up_w,
                    shexp_down_w,
                    shexp_gate_inp_w,
                    n_ff_shexp,
                    layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                    layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                    layer.moe_ffn.as_ref().map(|m| &m.down_w),
                    Some(cache.kv_store_mut()),
                );
                cache.n_used = nc + chunk_n;
                total_full_attn_us += t_fa.elapsed().as_micros() as u64;
            }
        }

        // Copy chunk output into the full hidden buffer
        all_hidden[chunk_start * n_embd..chunk_end * n_embd].copy_from_slice(&chunk_hidden);
        state.pos += chunk_n;
    }

    state.last_timing = TimingInfo {
        embed_us: total_embed_us,
        delta_net_us: total_delta_net_us,
        full_attn_us: total_full_attn_us,
        lm_head_us: 0,
        total_us: t_start.elapsed().as_micros() as u64,
    };
    Ok(all_hidden)
}

/// Decode a single token using KV cache and recurrent states.
/// Returns `(hidden_state, next_token_id)` or an error on context overflow.
pub fn generate_token(
    state: &mut GenerationState,
    token_id: u32,
    model: &ModelWeights,
) -> Result<(Vec<f32>, u32), String> {
    let (hidden, _logits) = generate_token_logits(state, token_id, model)?;
    let n_embd = model.cfg.embedding_length as usize;
    let n_vocab = model.tok_embd.n_elements / n_embd;
    let eps = model.cfg.attention_layer_norm_rms_epsilon;
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
    Ok((hidden, next_token))
}

/// Same as `generate_token` but returns raw logits instead of argmax token.
/// Caller can apply custom sampling (temperature, top-k, top-p, etc.).
pub fn generate_token_logits(
    state: &mut GenerationState,
    token_id: u32,
    model: &ModelWeights,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let t_start = Instant::now();
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

    let t_emb = Instant::now();
    let tok_embd_dq = dq(&model.tok_embd);
    let mut hidden = embed_tokens(token_id, &tok_embd_dq, n_embd);
    let embed_us = t_emb.elapsed().as_micros() as u64;

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;
    let mut delta_net_us: u64 = 0;
    let mut full_attn_us: u64 = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
            ) = moe_fields_cached(
                &mut state.moe_cache[layer_idx],
                layer.moe_ffn.as_ref(),
                cfg.expert_count as usize,
                cfg.expert_used_count as usize,
            );

            let dn_attn_norm_w = dq(&layer.attn_norm_w);
            let (dn_conv_kernel, dn_alpha_bias, dn_ssm_a) =
                dq3(&layer.conv_kernel, &layer.alpha_bias, &layer.ssm_a);
            let dn_ssm_norm_w = dq(&layer.ssm_norm_w);
            let dn_post_norm_w = dq(&layer.post_norm_w);

            let conv_dim = cfg.conv_dim as usize;
            let conv_kernel_size = cfg.ssm_conv_kernel as usize;
            let s_v = cfg.head_v_dim as usize;
            let n_heads_v = cfg.ssm_time_step_rank as usize;

            let t_dn = Instant::now();
            hidden = crate::model::kernels::delta_net_layer_forward_q(
                &hidden,
                &dn_attn_norm_w,
                (layer.wqkv.data(), layer.wqkv.ty),
                (layer.wqkv_gate.data(), layer.wqkv_gate.ty),
                &dn_conv_kernel,
                &dn_alpha_bias,
                &dn_ssm_a,
                &dn_ssm_norm_w,
                (layer.ssm_out.data(), layer.ssm_out.ty),
                &dn_post_norm_w,
                (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
                layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.down_w),
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
                &mut state.conv_states[delta_net_idx - 1],
                &mut state.ssm_states[delta_net_idx - 1],
            );
            delta_net_us += t_dn.elapsed().as_micros() as u64;
        } else {
            let layer = &model.full_attn_layers[full_attn_idx];
            full_attn_idx += 1;

            let (
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
            ) = moe_fields_cached(
                &mut state.moe_cache[layer_idx],
                layer.moe_ffn.as_ref(),
                cfg.expert_count as usize,
                cfg.expert_used_count as usize,
            );

            let (fa_attn_norm_w, fa_q_norm_w, fa_k_norm_w) =
                dq3(&layer.attn_norm_w, &layer.q_norm_w, &layer.k_norm_w);
            let fa_post_norm_w = dq(&layer.post_norm_w);

            let t_fa = Instant::now();
            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;
            cache.ensure_capacity(nc + 1);

            hidden = crate::model::kernels::full_layer_forward_q(
                &hidden,
                &fa_attn_norm_w,
                (layer.wq.data(), layer.wq.ty),
                (layer.wk.data(), layer.wk.ty),
                (layer.wv.data(), layer.wv.ty),
                (layer.wo.data(), layer.wo.ty),
                &fa_q_norm_w,
                &fa_k_norm_w,
                pos,
                &rope_cfg,
                &fa_post_norm_w,
                (layer.ffn_gate_w.data(), layer.ffn_gate_w.ty),
                (layer.ffn_up_w.data(), layer.ffn_up_w.ty),
                (layer.ffn_down_w.data(), layer.ffn_down_w.ty),
                n_embd,
                n_heads,
                n_kv_heads,
                head_size,
                n_ff,
                1,
                eps,
                rope_sections,
                moe_router_w,
                moe_gate_up_w,
                moe_down_w,
                n_expert,
                n_expert_used,
                shexp_gate_w,
                shexp_up_w,
                shexp_down_w,
                shexp_gate_inp_w,
                n_ff_shexp,
                layer.moe_ffn.as_ref().map(|m| &m.gate_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.up_exps_q),
                layer.moe_ffn.as_ref().map(|m| &m.down_w),
                Some(cache.kv_store_mut()),
            );
            cache.n_used = nc + 1;
            full_attn_us += t_fa.elapsed().as_micros() as u64;
        }
    }

    state.pos += 1;

    let t_lm = Instant::now();
    let output_norm_dq = dq(&model.output_norm_w);
    let output_weight_dq = dq(&model.output_weight);
    let logits = lm_head_logits(
        &hidden,
        &output_norm_dq,
        &output_weight_dq,
        n_embd,
        n_vocab,
        eps,
    );
    let lm_head_us = t_lm.elapsed().as_micros() as u64;

    state.last_timing = TimingInfo {
        embed_us,
        delta_net_us,
        full_attn_us,
        lm_head_us,
        total_us: t_start.elapsed().as_micros() as u64,
    };
    Ok((hidden, logits))
}

// ---------------------------------------------------------------------------
// Phase 23: cached MoE field extraction
// ---------------------------------------------------------------------------
//
// `moe_fields_cached` dequantizes a layer's MoE weights once, stores them in
// the per-state cache slot, and returns *references* into that cache. Callers
// rely on Rust's disjoint-field borrow rules to hold these refs while also
// mutably borrowing `state.conv_states` / `ssm_states` / `kv_caches`.

type MoeFieldsRef<'a> = (
    &'a [f32],
    &'a [f32],
    &'a [f32],
    usize,
    usize,
    &'a [f32],
    &'a [f32],
    &'a [f32],
    &'a [f32],
    usize,
);

/// Get (or populate) the MoE weight cache entry for one layer.
///
/// Works for both delta-net and full-attention layers since both store their
/// FFN in an identically-shaped `LoadedMoeFfn`.
fn moe_fields_cached<'a>(
    slot: &'a mut Option<MoeWeightCacheEntry>,
    moe: Option<&'a LoadedMoeFfn>,
    n_expert: usize,
    n_expert_used: usize,
) -> MoeFieldsRef<'a> {
    if let Some(m) = moe {
        if slot.is_none() {
            *slot = Some(MoeWeightCacheEntry::new(m, n_expert, n_expert_used));
        }
        slot.as_ref().unwrap().as_fields()
    } else {
        static EMPTY: [f32; 0] = [];
        (
            &EMPTY[..],
            &EMPTY[..],
            &EMPTY[..],
            0,
            0,
            &EMPTY[..],
            &EMPTY[..],
            &EMPTY[..],
            &EMPTY[..],
            0,
        )
    }
}

// ---------------------------------------------------------------------------
// Phase 27: speculative decoding
// ---------------------------------------------------------------------------

/// Rollback point for speculative verification: KV lengths plus copies of the
/// small recurrent states (conv/ssm). Cheap relative to weight traffic.
#[derive(Clone)]
struct StateSnapshot {
    kv_n_used: Vec<usize>,
    conv_states: Vec<Vec<f32>>,
    ssm_states: Vec<Vec<f32>>,
    pos: usize,
}

impl GenerationState {
    fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            kv_n_used: self.kv_caches.iter().map(|c| c.n_used).collect(),
            conv_states: self.conv_states.clone(),
            ssm_states: self.ssm_states.clone(),
            pos: self.pos,
        }
    }

    fn restore(&mut self, snap: &StateSnapshot) {
        for (cache, &n) in self.kv_caches.iter_mut().zip(&snap.kv_n_used) {
            cache.n_used = n;
        }
        self.conv_states.clone_from(&snap.conv_states);
        self.ssm_states.clone_from(&snap.ssm_states);
        self.pos = snap.pos;
    }
}

/// Result of verifying a draft continuation against the target model.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyResult {
    /// Draft tokens the target model agreed with, in order.
    pub accepted: Vec<u32>,
    /// Target model's own next token after `accepted` (always produced).
    pub bonus: u32,
}

/// Greedy speculative-decoding verification step.
///
/// Feeds `[context_token, draft...]` through the model in one multi-token
/// pass, computes per-position greedy predictions, and accepts the longest
/// prefix of `draft` the target reproduces exactly. The state is then rolled
/// back and re-driven through only the accepted tokens so recurrent (delta-net)
/// layers remain exact — correctness is unconditional; throughput gain depends
/// on draft quality.
///
/// `context_token` must be the last token already fed to the model. Returns
/// the accepted draft prefix plus the target's bonus token; feed `bonus` as
/// the next context token.
pub fn verify_draft(
    state: &mut GenerationState,
    context_token: u32,
    draft: &[u32],
    model: &ModelWeights,
) -> Result<VerifyResult, String> {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_vocab = model.tok_embd.n_elements / n_embd;
    let eps = cfg.attention_layer_norm_rms_epsilon;
    let norm_w = dq(&model.output_norm_w);
    let out_w = dq(&model.output_weight);

    if draft.is_empty() {
        // Degenerate case: plain greedy decode of one token.
        let (hidden, _logits) = generate_token_logits(state, context_token, model)?;
        let bonus = lm_head_argmax(&hidden, &norm_w, &out_w, n_embd, n_vocab, eps);
        return Ok(VerifyResult {
            accepted: vec![],
            bonus,
        });
    }

    let snap = state.snapshot();

    // One multi-token pass over [ctx, d0..dD]; per-row greedy predictions:
    // row i predicts feed[i+1], so row i judges draft i-1... concretely
    // preds[r] == draft[r] means the target reproduces draft[r].
    let mut feed = Vec::with_capacity(draft.len() + 1);
    feed.push(context_token);
    feed.extend_from_slice(draft);
    let hiddens = prefill(state, &feed, model)?;

    let mut preds: Vec<u32> = Vec::with_capacity(feed.len());
    for r in 0..feed.len() {
        let row = &hiddens[r * n_embd..(r + 1) * n_embd];
        preds.push(lm_head_argmax(row, &norm_w, &out_w, n_embd, n_vocab, eps));
    }

    let mut r = 0usize;
    while r < draft.len() && preds[r] == draft[r] {
        r += 1;
    }
    let accepted: Vec<u32> = draft[..r].to_vec();

    // Roll back everything and re-drive only [ctx] + accepted so the hybrid
    // delta-net/full-attn state stays exact.
    state.restore(&snap);
    let mut refeed = Vec::with_capacity(accepted.len() + 1);
    refeed.push(context_token);
    refeed.extend_from_slice(&accepted);
    let rows = prefill(state, &refeed, model)?;

    // Bonus = greedy from the final row (position of last re-fed token).
    let last_row = &rows[rows.len() - n_embd..];
    let bonus = lm_head_argmax(last_row, &norm_w, &out_w, n_embd, n_vocab, eps);

    Ok(VerifyResult { accepted, bonus })
}

// ---- Phase 15: batch inference ---------------------------------------------
//
// Sequences are fully independent (each owns its GenerationState), so the
// whole forward pass runs per-sequence under rayon. Nested parallelism is
// fine: rayon's work-stealing interleaves the per-layer expert/dequant jobs
// of different sequences on the same pool instead of oversubscribing.

/// Prefill multiple independent sequences in parallel.
///
/// `states[i]` is advanced by `prompts[i]`; returns the final hidden state
/// of each sequence. Errors if ANY sequence would overflow its context.
pub fn prefill_batch(
    states: &mut [GenerationState],
    prompts: &[&[u32]],
    model: &ModelWeights,
) -> Result<Vec<Vec<f32>>, String> {
    assert_eq!(states.len(), prompts.len(), "one state per prompt required");
    use rayon::prelude::*;
    states
        .par_iter_mut()
        .zip(prompts.par_iter())
        .map(|(state, tokens)| prefill(state, tokens, model))
        .collect()
}

/// One greedy decode step for every sequence, in parallel.
pub fn generate_token_batch(
    states: &mut [GenerationState],
    last_tokens: &[u32],
    model: &ModelWeights,
) -> Result<Vec<u32>, String> {
    assert_eq!(states.len(), last_tokens.len());
    use rayon::prelude::*;
    states
        .par_iter_mut()
        .zip(last_tokens.par_iter())
        .map(|(state, &tok)| generate_token(state, tok, model).map(|(_, next)| next))
        .collect()
}

/// One decode step returning full logits for every sequence (for sampling).
pub fn generate_token_logits_batch(
    states: &mut [GenerationState],
    last_tokens: &[u32],
    model: &ModelWeights,
) -> Result<Vec<Vec<f32>>, String> {
    assert_eq!(states.len(), last_tokens.len());
    use rayon::prelude::*;
    states
        .par_iter_mut()
        .zip(last_tokens.par_iter())
        .map(|(state, &tok)| generate_token_logits(state, tok, model).map(|(_, logits)| logits))
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 28: continuous batching scheduler
// ---------------------------------------------------------------------------

/// Opaque sequence handle returned by [`BatchScheduler::submit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqId(pub u64);

/// One scheduler tick's output per event.
#[derive(Debug, Clone, PartialEq)]
pub enum StepEvent {
    /// Prompt accepted into the active pool (prefill completed).
    Prefilled(SeqId),
    /// Greedy-decoded one token for a live sequence.
    Decoded(SeqId, u32),
    /// Sequence retired (max_new reached or EOS emitted).
    Finished(SeqId, Vec<u32>),
}

struct PendingSeq {
    id: SeqId,
    tokens: Vec<u32>,
    max_new: usize,
}

struct ActiveMeta {
    id: SeqId,
    last_token: u32,
    generated: Vec<u32>,
    max_new: usize,
}

/// Continuous-batching scheduler over independent sequences.
///
/// Unlike the Phase 15 lockstep batch API, sequences can **join mid-flight**:
/// submitted prompts are prefilled on the next tick (in parallel), then merged
/// into the decode pool. Retired sequences free their slots immediately, so
/// decode batch width tracks liveness instead of the initial cohort size.
///
/// Decode is greedy argmax; pass `eos_id` to stop sequences early.
pub struct BatchScheduler<'m> {
    model: &'m ModelWeights,
    eos_id: Option<u32>,
    next_id: u64,
    pending: Vec<PendingSeq>,
    /// Parallel arrays: metadata[i] pairs with states[i].
    meta: Vec<ActiveMeta>,
    states: Vec<GenerationState>,
}

impl<'m> BatchScheduler<'m> {
    pub fn new(model: &'m ModelWeights, eos_id: Option<u32>) -> Self {
        Self {
            model,
            eos_id,
            next_id: 0,
            pending: Vec::new(),
            meta: Vec::new(),
            states: Vec::new(),
        }
    }

    /// Queue a prompt for prefill on the next [`step`](Self::step) call.
    pub fn submit(&mut self, tokens: Vec<u32>, max_new: usize) -> SeqId {
        assert!(!tokens.is_empty(), "prompt must contain at least one token");
        let id = SeqId(self.next_id);
        self.next_id += 1;
        self.pending.push(PendingSeq {
            id,
            tokens,
            max_new,
        });
        id
    }

    /// Number of sequences currently in the decode pool.
    pub fn n_active(&self) -> usize {
        self.meta.len()
    }

    /// True when nothing is queued or decoding.
    pub fn is_idle(&self) -> bool {
        self.pending.is_empty() && self.meta.is_empty()
    }

    /// Advance the world by one tick:
    /// 1. drain the pending queue (parallel prefill),
    /// 2. decode one token for every active sequence (parallel),
    /// 3. retire finished sequences.
    ///
    /// Returns events in stable order: prefills first (by submit order),
    /// then decodes/finishes (by slot order).
    pub fn step(&mut self) -> Result<Vec<StepEvent>, String> {
        let mut events = Vec::new();

        // 1. Prefill everything queued since last tick.
        if !self.pending.is_empty() {
            let mut new_states: Vec<GenerationState> = self
                .pending
                .iter()
                .map(|_| GenerationState::new(self.model))
                .collect();
            let refs: Vec<&[u32]> = self.pending.iter().map(|p| p.tokens.as_slice()).collect();
            let _hiddens = prefill_batch(&mut new_states, &refs, self.model)?;

            for p in std::mem::take(&mut self.pending) {
                events.push(StepEvent::Prefilled(p.id));
                self.meta.push(ActiveMeta {
                    id: p.id,
                    last_token: *p.tokens.last().unwrap(),
                    generated: Vec::new(),
                    max_new: p.max_new,
                });
            }
            self.states.extend(new_states);
        }

        // 2. Lockstep greedy decode over the active pool.
        if self.meta.is_empty() {
            return Ok(events);
        }
        let last_tokens: Vec<u32> = self.meta.iter().map(|m| m.last_token).collect();
        let next_tokens = generate_token_batch(&mut self.states, &last_tokens, self.model)?;

        // 3. Append tokens; collect retirees.
        let mut retired_ids: Vec<SeqId> = Vec::new();
        for (slot, tok) in next_tokens.into_iter().enumerate() {
            let meta = &mut self.meta[slot];
            meta.generated.push(tok);
            meta.last_token = tok;
            events.push(StepEvent::Decoded(meta.id, tok));
            let done =
                meta.generated.len() >= meta.max_new || self.eos_id.is_some_and(|eos| tok == eos);
            if done {
                events.push(StepEvent::Finished(
                    meta.id,
                    std::mem::take(&mut meta.generated),
                ));
                retired_ids.push(meta.id);
            }
        }

        if !retired_ids.is_empty() {
            let mut keep_meta = Vec::with_capacity(self.meta.len());
            let mut keep_states = Vec::with_capacity(self.states.len());
            for (meta, state) in std::mem::take(&mut self.meta)
                .into_iter()
                .zip(std::mem::take(&mut self.states))
            {
                if retired_ids.contains(&meta.id) {
                    continue;
                }
                keep_meta.push(meta);
                keep_states.push(state);
            }
            self.meta = keep_meta;
            self.states = keep_states;
        }

        Ok(events)
    }

    /// Run until idle (bounded by total steps to avoid runaway loops).
    /// Returns every event observed.
    pub fn run_until_idle(&mut self, max_steps: usize) -> Result<Vec<StepEvent>, String> {
        let mut all = Vec::new();
        for _ in 0..max_steps {
            if self.is_idle() {
                break;
            }
            all.extend(self.step()?);
        }
        Ok(all)
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

        let (hidden, next_token) = forward_pass(
            0,
            &ModelWeights {
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
                tok_embd: RawTensor::new(
                    crate::gguf::GGmlType::F32,
                    embd_bytes.clone(),
                    n_vocab * n_embd,
                ),
                output_norm_w: RawTensor::new(crate::gguf::GGmlType::F32, out_norm_bytes, n_embd),
                output_weight: RawTensor::new(
                    crate::gguf::GGmlType::F32,
                    embd_bytes,
                    n_vocab * n_embd,
                ),
                full_attn_layers: vec![],
                delta_net_layers: vec![],
            },
        );

        // Shouldn't panic; token is in vocab range
        assert!(
            (next_token as usize) < n_vocab,
            "next_token {next_token} >= n_vocab {n_vocab}"
        );
        assert_eq!(hidden.len(), n_embd);
    }

    /// Bare model: all layers full-attention (`full_attention_interval = 1`),
    /// zero-filled global weights. Enough to exercise KV-cache state
    /// management without running layer forwards.
    fn bare_model(block_count: u32, context_length: u32) -> ModelWeights {
        let n_embd = 32usize;
        let embd = vec![0.1f32; 16 * n_embd];
        let embd_bytes: Vec<u8> = embd.iter().flat_map(|f| f.to_le_bytes()).collect();
        let ones: Vec<u8> = vec![1.0f32; n_embd]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
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
        assert_eq!(cache.allocated_bytes(), 1024 * 2 * 8 * 4 * 2); // K+V f32

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
        {
            match fresh.kv_store_mut() {
                crate::model::kernels::KvStoreMut::F32(c) => {
                    c.k[..4].copy_from_slice(&[7.5, 8.5, 9.5, 10.5]);
                }
                _ => panic!("expected F32 backing"),
            }
        }
        fresh.ensure_capacity(100);
        match fresh.kv_store_mut() {
            crate::model::kernels::KvStoreMut::F32(c) => {
                assert_eq!(&c.k[..4], &[7.5, 8.5, 9.5, 10.5]);
            }
            _ => panic!("expected F32 backing"),
        }
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
            let bytes = cache.allocated_bytes();
            assert_eq!(cache.capacity_tokens(), 1024);
            assert!(
                bytes <= 128 * 1024,
                "initial allocation too large: {bytes} bytes"
            );
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

    #[test]
    fn moe_cache_has_one_slot_per_layer_and_stays_empty_for_dense() {
        // bare_model is dense (no MoE weights at all): every slot stays None.
        // Forward passes aren't run here (bare weights are degenerate); the
        // synth-model crossval tests exercise the populated-cache path.
        let model = bare_model(4, 64);
        let mut state = GenerationState::new(&model);
        assert_eq!(state.moe_cache_slots(), 4);
        assert_eq!(state.moe_cache_filled(), 0);

        // Phase 24: warming a dense model is a safe no-op, twice over.
        state.warm_moe_cache(&model);
        state.warm_moe_cache(&model);
        assert_eq!(state.moe_cache_filled(), 0);
    }
}
