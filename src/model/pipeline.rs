//! End-to-end inference pipeline: GGUF weights → unified forward → token.
//!
//! Loads dequantised weight slices from a `ModelLoader`, wires them into the
//! kernel-level layer structs, and drives the token-by-token forward pass
//! through both recurrent (delta-net) and full-attention layers.

use crate::model::config::Qwen3_5Config;
use crate::model::kernels::{
    DeltaNetLayerWeights, FullAttnLayerWeights, RopeConfig,
    delta_net_layer_forward, embed_tokens,
    lm_head_argmax,
};
use crate::model::loader::ModelLoader;

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
    pub router_w: Vec<f32>,      // [n_expert, n_embd]
    pub gate_up_w: Vec<f32>,     // [n_expert, 2*n_ff, n_embd] (fused gate+up)
    pub down_w: Vec<f32>,        // [n_expert, n_embd, n_ff]
    // Shared expert (shexp)
    pub shexp_gate_w: Vec<f32>,     // [n_ff_shexp, n_embd]
    pub shexp_up_w: Vec<f32>,       // [n_ff_shexp, n_embd]
    pub shexp_down_w: Vec<f32>,     // [n_embd, n_ff_shexp]
    pub shexp_gate_inp_w: Vec<f32>, // [n_embd]
    pub n_ff_shexp: usize,
}

// ---------------------------------------------------------------------------
// Full-attention layer weights (loaded eagerly)
// ---------------------------------------------------------------------------

pub struct LoadedFullAttnLayer {
    pub attn_norm_w: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub q_norm_w: Vec<f32>,
    pub k_norm_w: Vec<f32>,
    pub post_norm_w: Vec<f32>,
    /// Dense FFN (when expert_count == 0)
    pub ffn_gate_w: Vec<f32>,
    pub ffn_up_w: Vec<f32>,
    pub ffn_down_w: Vec<f32>,
    /// MoE FFN (when expert_count > 0), None if dense
    pub moe_ffn: Option<LoadedMoeFfn>,
}

// ---------------------------------------------------------------------------
// Delta-net layer weights (loaded eagerly)
// ---------------------------------------------------------------------------

pub struct LoadedDeltaNetLayer {
    pub attn_norm_w: Vec<f32>,
    pub wqkv: Vec<f32>,
    pub wqkv_gate: Vec<f32>,
    pub conv_kernel: Vec<f32>,
    pub alpha_bias: Vec<f32>,
    pub ssm_a: Vec<f32>,
    pub ssm_norm_w: Vec<f32>,
    pub ssm_out: Vec<f32>,
    pub post_norm_w: Vec<f32>,
    /// Dense FFN (when expert_count == 0)
    pub ffn_gate_w: Vec<f32>,
    pub ffn_up_w: Vec<f32>,
    pub ffn_down_w: Vec<f32>,
    /// MoE FFN (when expert_count > 0), None if dense
    pub moe_ffn: Option<LoadedMoeFfn>,
}

// ---------------------------------------------------------------------------
// Full model weights
// ---------------------------------------------------------------------------

pub struct ModelWeights {
    pub cfg: Qwen3_5Config,
    pub tok_embd: Vec<f32>,
    pub output_norm_w: Vec<f32>,
    pub output_weight: Vec<f32>,
    pub full_attn_layers: Vec<LoadedFullAttnLayer>,
    pub delta_net_layers: Vec<LoadedDeltaNetLayer>,
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

        // Global weights
        let tok_embd = loader.dequant("token_embd.weight").map_err(|e| e.to_string())?;
        let output_norm_w = loader.dequant("output_norm.weight").map_err(|e| e.to_string())?;
        let output_weight = if loader.tensor_meta("output.weight").is_some() {
            loader.dequant("output.weight").map_err(|e| e.to_string())?
        } else {
            // Weight-tied: output shares token_embd
            tok_embd.clone()
        };

        // Verify dimensions
        assert_eq!(tok_embd.len(), n_vocab * n_embd,
            "token_embd shape mismatch: got {} elements, expected {n_vocab}x{n_embd}={}",
            tok_embd.len(), n_vocab * n_embd);
        assert_eq!(output_norm_w.len(), n_embd);
        assert_eq!(output_weight.len(), n_vocab * n_embd);

        let n_layers = cfg.block_count as usize;
        let n_expert = cfg.expert_count as usize;
        let n_ff = cfg.expert_feed_forward_length as usize;
        let n_ff_shexp = cfg.expert_shared_feed_forward_length as usize;
        let mut full_attn_layers = Vec::new();
        let mut delta_net_layers = Vec::new();

        let dequant = |name: &str| -> Result<Vec<f32>, String> {
            loader.dequant(name).map_err(|e| format!("{name}: {e}"))
        };

        // MoE FFN loader (shared by both layer types)
        let load_moe = |t: &TensorNames| -> Result<Option<LoadedMoeFfn>, String> {
            if n_expert == 0 {
                return Ok(None);
            }
            let router_w = dequant(&t.ffn_gate_inp())?;
            let gate_exps = dequant(&t.ffn_gate_exps())?;
            let up_exps = dequant(&t.ffn_up_exps())?;
            let down_w = dequant(&t.ffn_down_exps())?;

            // Fuse gate_exps [n_expert, n_ff, n_embd] + up_exps [n_expert, n_ff, n_embd]
            // into gate_up_w [n_expert, 2*n_ff, n_embd]
            let mut gate_up_w = vec![0.0f32; n_expert * 2 * n_ff * n_embd];
            for e in 0..n_expert {
                let gate_base = e * n_ff * n_embd;
                let up_base = e * n_ff * n_embd;
                let fused_base = e * 2 * n_ff * n_embd;
                // Copy gate weights (first n_ff rows)
                gate_up_w[fused_base..fused_base + n_ff * n_embd]
                    .copy_from_slice(&gate_exps[gate_base..gate_base + n_ff * n_embd]);
                // Copy up weights (next n_ff rows)
                gate_up_w[fused_base + n_ff * n_embd..fused_base + 2 * n_ff * n_embd]
                    .copy_from_slice(&up_exps[up_base..up_base + n_ff * n_embd]);
            }

            // Shared expert (shexp) — loaded when n_ff_shexp > 0
            let (shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w) = if n_ff_shexp > 0 {
                (
                    dequant(&t.ffn_gate_shexp())?,
                    dequant(&t.ffn_up_shexp())?,
                    dequant(&t.ffn_down_shexp())?,
                    dequant(&t.ffn_gate_inp_shexp())?,
                )
            } else {
                (vec![], vec![], vec![], vec![])
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
            let attn_norm_w = dequant(&t.attn_norm())?;
            let post_norm_w = dequant(&t.post_attn_norm())?;

            if cfg.is_recurrent(i) {
                // Delta-net layer
                let wqkv = dequant(&t.attn_qkv())?;
                let wqkv_gate = dequant(&t.attn_gate())?;
                let conv_kernel = dequant(&t.ssm_conv1d())?;
                let alpha_bias = dequant(&t.ssm_dt())?;
                let ssm_a = dequant(&t.ssm_a())?;
                let ssm_norm_w = dequant(&t.ssm_norm())?;
                let ssm_out = dequant(&t.ssm_out())?;

                // FFN: MoE or dense
                let moe_ffn = load_moe(&t)?;
                let (ffn_gate_w, ffn_up_w, ffn_down_w) = if moe_ffn.is_some() {
                    (vec![], vec![], vec![])
                } else {
                    (dequant(&t.ffn_gate())?, dequant(&t.ffn_up())?, dequant(&t.ffn_down())?)
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
                let wq = dequant(&t.attn_q())?;
                let wk = dequant(&t.attn_k())?;
                let wv = dequant(&t.attn_v())?;
                let wo = dequant(&t.attn_o())?;
                let q_norm_w = dequant(&t.attn_q_norm())?;
                let k_norm_w = dequant(&t.attn_k_norm())?;

                // FFN: MoE or dense
                let moe_ffn = load_moe(&t)?;
                let (ffn_gate_w, ffn_up_w, ffn_down_w) = if moe_ffn.is_some() {
                    (vec![], vec![], vec![])
                } else {
                    (dequant(&t.ffn_gate())?, dequant(&t.ffn_up())?, dequant(&t.ffn_down())?)
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
    let n_vocab = model.tok_embd.len() / n_embd;
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
    let mut hidden = embed_tokens(token_id, &model.tok_embd, n_embd);

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

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                if let Some(ref moe) = layer.moe_ffn {
                    (&moe.router_w[..], &moe.gate_up_w[..], &moe.down_w[..],
                     cfg.expert_count as usize, cfg.expert_used_count as usize,
                     &moe.shexp_gate_w[..], &moe.shexp_up_w[..],
                     &moe.shexp_down_w[..], &moe.shexp_gate_inp_w[..],
                     moe.n_ff_shexp)
                } else {
                    (&[][..], &[][..], &[][..], 0, 0,
                     &[][..], &[][..], &[][..], &[][..], 0)
                };

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wqkv: &layer.wqkv,
                wqkv_gate: &layer.wqkv_gate,
                conv_kernel: &layer.conv_kernel,
                alpha_bias: &layer.alpha_bias,
                ssm_a: &layer.ssm_a,
                ssm_norm_w: &layer.ssm_norm_w,
                ssm_out: &layer.ssm_out,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                if let Some(ref moe) = layer.moe_ffn {
                    (&moe.router_w[..], &moe.gate_up_w[..], &moe.down_w[..],
                     cfg.expert_count as usize, cfg.expert_used_count as usize,
                     &moe.shexp_gate_w[..], &moe.shexp_up_w[..],
                     &moe.shexp_down_w[..], &moe.shexp_gate_inp_w[..],
                     moe.n_ff_shexp)
                } else {
                    (&[][..], &[][..], &[][..], 0, 0,
                     &[][..], &[][..], &[][..], &[][..], 0)
                };

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wq: &layer.wq,
                wk: &layer.wk,
                wv: &layer.wv,
                wo: &layer.wo,
                q_norm_w: &layer.q_norm_w,
                k_norm_w: &layer.k_norm_w,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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
                None,
            );
        }
    }

    // LM head
    let next_token = lm_head_argmax(
        &hidden,
        &model.output_norm_w,
        &model.output_weight,
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
pub struct LayerKvCache {
    pub k: Vec<f32>,  // [n_ctx, n_kv_heads * head_size]
    pub v: Vec<f32>,  // [n_ctx, n_kv_heads * head_size]
    pub n_used: usize, // number of positions currently in use
}

impl LayerKvCache {
    pub fn new(n_ctx: usize, n_kv_heads: usize, head_size: usize) -> Self {
        let kv_dim = n_kv_heads * head_size;
        Self {
            k: vec![0.0; n_ctx * kv_dim],
            v: vec![0.0; n_ctx * kv_dim],
            n_used: 0,
        }
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
                let layer = &model.delta_net_layers[delta_net_count];
                delta_net_count += 1;
                conv_states.push(vec![0.0f32; layer.conv_kernel.len()]);
                ssm_states.push(vec![0.0f32; layer.ssm_out.len() / n_embd(cfg)]);
            } else {
                kv_caches.push(LayerKvCache::new(n_ctx, n_kv_heads, head_size));
            }
        }

        Self { kv_caches, conv_states, ssm_states, pos: 0 }
    }
}

fn n_embd(cfg: &crate::model::config::Qwen3_5Config) -> usize {
    cfg.embedding_length as usize
}

/// Prefill: process all prompt tokens at once, populating the KV cache
/// and recurrent states. Returns the hidden state after the last token.
pub fn prefill(
    state: &mut GenerationState,
    token_ids: &[u32],
    model: &ModelWeights,
) -> Vec<f32> {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
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

    // Embed all prompt tokens at once
    let n_tokens = token_ids.len();
    let mut hidden = vec![0.0f32; n_tokens * n_embd];
    for (t, &tid) in token_ids.iter().enumerate() {
        let emb = embed_tokens(tid, &model.tok_embd, n_embd);
        hidden[t * n_embd..(t + 1) * n_embd].copy_from_slice(&emb);
    }

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields(layer, cfg);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wqkv: &layer.wqkv,
                wqkv_gate: &layer.wqkv_gate,
                conv_kernel: &layer.conv_kernel,
                alpha_bias: &layer.alpha_bias,
                ssm_a: &layer.ssm_a,
                ssm_norm_w: &layer.ssm_norm_w,
                ssm_out: &layer.ssm_out,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wq: &layer.wq,
                wk: &layer.wk,
                wv: &layer.wv,
                wo: &layer.wo,
                q_norm_w: &layer.q_norm_w,
                k_norm_w: &layer.k_norm_w,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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
            };

            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;

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
    hidden
}

/// Decode a single token using KV cache and recurrent states.
/// Returns `(hidden_state, next_token_id)`.
pub fn generate_token(
    state: &mut GenerationState,
    token_id: u32,
    model: &ModelWeights,
) -> (Vec<f32>, u32) {
    let cfg = &model.cfg;
    let n_embd = cfg.embedding_length as usize;
    let n_heads = cfg.attention_head_count as usize;
    let n_kv_heads = cfg.attention_head_count_kv as usize;
    let head_size = cfg.attention_key_length as usize;
    let n_ff = cfg.expert_feed_forward_length as usize;
    let n_vocab = model.tok_embd.len() / n_embd;
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

    let mut hidden = embed_tokens(token_id, &model.tok_embd, n_embd);

    let mut full_attn_idx = 0;
    let mut delta_net_idx = 0;

    for layer_idx in 0..cfg.block_count as usize {
        if cfg.is_recurrent(layer_idx) {
            let layer = &model.delta_net_layers[delta_net_idx];
            delta_net_idx += 1;

            let (moe_router_w, moe_gate_up_w, moe_down_w, n_expert, n_expert_used,
                 shexp_gate_w, shexp_up_w, shexp_down_w, shexp_gate_inp_w, n_ff_shexp) =
                moe_fields(layer, cfg);

            let layer_ref = DeltaNetLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wqkv: &layer.wqkv,
                wqkv_gate: &layer.wqkv_gate,
                conv_kernel: &layer.conv_kernel,
                alpha_bias: &layer.alpha_bias,
                ssm_a: &layer.ssm_a,
                ssm_norm_w: &layer.ssm_norm_w,
                ssm_out: &layer.ssm_out,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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

            let layer_ref = FullAttnLayerWeights {
                attn_norm_w: &layer.attn_norm_w,
                wq: &layer.wq,
                wk: &layer.wk,
                wv: &layer.wv,
                wo: &layer.wo,
                q_norm_w: &layer.q_norm_w,
                k_norm_w: &layer.k_norm_w,
                post_norm_w: &layer.post_norm_w,
                ffn_gate_w: &layer.ffn_gate_w,
                ffn_up_w: &layer.ffn_up_w,
                ffn_down_w: &layer.ffn_down_w,
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
            };

            let pos = [state.pos as i32, 0, 0, 0];
            let cache = &mut state.kv_caches[full_attn_idx - 1];
            let nc = cache.n_used;

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

    let next_token = lm_head_argmax(&hidden, &model.output_norm_w, &model.output_weight, n_embd, n_vocab, eps);
    (hidden, next_token)
}

// Helper to extract MoE fields from a delta-net layer
type MoeFields<'a> = (&'a [f32], &'a [f32], &'a [f32], usize, usize,
      &'a [f32], &'a [f32], &'a [f32], &'a [f32], usize);

fn moe_fields<'a>(
    layer: &'a LoadedDeltaNetLayer,
    cfg: &crate::model::config::Qwen3_5Config,
) -> MoeFields<'a> {
    if let Some(ref moe) = layer.moe_ffn {
        (&moe.router_w, &moe.gate_up_w, &moe.down_w,
         cfg.expert_count as usize, cfg.expert_used_count as usize,
         &moe.shexp_gate_w, &moe.shexp_up_w, &moe.shexp_down_w,
         &moe.shexp_gate_inp_w, moe.n_ff_shexp)
    } else {
        (&[], &[], &[], 0, 0,
         &[], &[], &[], &[], 0)
    }
}

// Helper to extract MoE fields from a full-attention layer
fn moe_fields_full<'a>(
    layer: &'a LoadedFullAttnLayer,
    cfg: &crate::model::config::Qwen3_5Config,
) -> MoeFields<'a> {
    if let Some(ref moe) = layer.moe_ffn {
        (&moe.router_w, &moe.gate_up_w, &moe.down_w,
         cfg.expert_count as usize, cfg.expert_used_count as usize,
         &moe.shexp_gate_w, &moe.shexp_up_w, &moe.shexp_down_w,
         &moe.shexp_gate_inp_w, moe.n_ff_shexp)
    } else {
        (&[], &[], &[], 0, 0,
         &[], &[], &[], &[], 0)
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
        let out_norm = vec![1.0f32; n_embd];

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
            tok_embd: embd.clone(),
            output_norm_w: out_norm,
            output_weight: embd,
            full_attn_layers: vec![],
            delta_net_layers: vec![],
        });

        // Shouldn't panic; token is in vocab range
        assert!((next_token as usize) < n_vocab, "next_token {next_token} >= n_vocab {n_vocab}");
        assert_eq!(hidden.len(), n_embd);
    }
}
