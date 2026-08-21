//! Synthetic Qwen3.5 GGUF builder.
//!
//! Produces structurally valid models with Xavier-like random weights so the
//! full pipeline can be exercised without downloading a real checkpoint.
//! Used by the integration tests and the end-to-end benchmark mode.

use crate::gguf::value::ValueType;
use crate::gguf::writer::{GgufBuilder, TensorSpec};
use crate::gguf::{GGmlType, Value};

/// Dimensions for a synthetic model. Mirrors what the loader expects.
#[derive(Debug, Clone)]
pub struct SynthConfig {
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_size: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub n_layers: usize,
    pub full_attn_interval: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_conv_kernel: usize,
    pub eps: f32,
    pub rope_freq_base: f32,
    pub context_length: usize,
}

impl SynthConfig {
    /// Minimal model for integration tests.
    pub fn tiny() -> Self {
        Self {
            n_embd: 32,
            n_heads: 4,
            n_kv_heads: 2,
            head_size: 8,
            n_ff: 64,
            n_vocab: 16,
            n_layers: 4,
            full_attn_interval: 4,
            ssm_state_size: 8,
            ssm_group_count: 2,
            ssm_time_step_rank: 2,
            ssm_conv_kernel: 4,
            eps: 1e-6,
            rope_freq_base: 10_000.0,
            context_length: 128,
        }
    }

    /// ~25M-parameter dense model: heavy enough for stable timings.
    pub fn medium() -> Self {
        Self {
            n_embd: 512,
            n_heads: 16,
            n_kv_heads: 4,
            head_size: 64,
            n_ff: 1024,
            n_vocab: 2048,
            n_layers: 12,
            full_attn_interval: 4,
            context_length: 4096,
            ..Self::tiny()
        }
    }

    /// ~100M-parameter dense model for longer runs.
    pub fn large() -> Self {
        Self {
            n_embd: 1024,
            n_heads: 32,
            n_kv_heads: 8,
            head_size: 80,
            n_ff: 2816,
            n_vocab: 4096,
            n_layers: 16,
            full_attn_interval: 4,
            context_length: 8192,
            ..Self::tiny()
        }
    }

    pub fn preset(name: &str) -> Option<Self> {
        match name {
            "tiny" => Some(Self::tiny()),
            "medium" => Some(Self::medium()),
            "large" => Some(Self::large()),
            _ => None,
        }
    }

    pub fn conv_dim(&self) -> usize {
        let key_dim = self.ssm_state_size * self.ssm_group_count;
        let value_dim = self.ssm_state_size * self.ssm_time_step_rank;
        key_dim * 2 + value_dim
    }

    pub fn ba_dim(&self) -> usize {
        self.ssm_time_step_rank * 2
    }

    /// Delta-net layers are every layer except multiples of `full_attn_interval`.
    pub fn is_recurrent(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.full_attn_interval)
    }

    pub fn n_attn_layers(&self) -> usize {
        (0..self.n_layers).filter(|&l| !self.is_recurrent(l)).count()
    }

    fn rope_sections(&self) -> [i32; 4] {
        let half = (self.head_size / 4) as i32;
        [half, half, 0, 0]
    }

    /// Bytes of KV cache per token across all attention layers (f32 K+V).
    pub fn kv_bytes_per_token(&self) -> usize {
        2 * self.n_kv_heads * self.head_size * 4 * self.n_attn_layers()
    }
}

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn vec1d(name: &str, data: &[f32]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        ggml_type: GGmlType::F32,
        dims: vec![data.len() as u64],
        data: f32_bytes(data),
    }
}

fn vec2d(name: &str, rows: usize, cols: usize, data: &[f32]) -> TensorSpec {
    assert_eq!(data.len(), rows * cols);
    TensorSpec {
        name: name.to_string(),
        ggml_type: GGmlType::F32,
        dims: vec![rows as u64, cols as u64],
        data: f32_bytes(data),
    }
}

/// Build the complete GGUF byte stream for the configured model.
/// Xavier-like init (1/sqrt(fan_in)) keeps activations alive end-to-end.
pub fn build_gguf(cfg: &SynthConfig) -> Vec<u8> {
    let eps = cfg.eps;
    let n_embd = cfg.n_embd;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_size = cfg.head_size;
    let n_ff = cfg.n_ff;
    let n_vocab = cfg.n_vocab;
    let n_layers = cfg.n_layers;
    let conv_dim = cfg.conv_dim();
    let ba_dim = cfg.ba_dim();
    let ssm_state_size = cfg.ssm_state_size;
    let ssm_group_count = cfg.ssm_group_count;
    let ssm_time_step_rank = cfg.ssm_time_step_rank;
    let ssm_conv_kernel = cfg.ssm_conv_kernel;

    let scale = |fan_in: usize| -> f32 { (2.0 / fan_in as f32).sqrt() };

    let mut builder = GgufBuilder::new();

    builder = builder
        .metadata("general.architecture", Value::String("qwen35moe".into()))
        .metadata("qwen35moe.block_count", Value::U32(n_layers as u32))
        .metadata("qwen35moe.embedding_length", Value::U32(n_embd as u32))
        .metadata("qwen35moe.attention.head_count", Value::U32(n_heads as u32))
        .metadata("qwen35moe.attention.head_count_kv", Value::U32(n_kv_heads as u32))
        .metadata("qwen35moe.attention.key_length", Value::U32(head_size as u32))
        .metadata("qwen35moe.attention.value_length", Value::U32(head_size as u32))
        .metadata("qwen35moe.attention.layer_norm_rms_epsilon", Value::F32(eps))
        .metadata("qwen35moe.expert_count", Value::U32(0))
        .metadata("qwen35moe.expert_used_count", Value::U32(0))
        .metadata("qwen35moe.expert_feed_forward_length", Value::U32(n_ff as u32))
        .metadata("qwen35moe.expert_shared_feed_forward_length", Value::U32(0))
        .metadata("qwen35moe.rope.dimension_count", Value::U32(head_size as u32))
        .metadata("qwen35moe.rope.freq_base", Value::F32(cfg.rope_freq_base))
        .metadata("qwen35moe.context_length", Value::U32(cfg.context_length as u32))
        .metadata("qwen35moe.ssm.state_size", Value::U32(ssm_state_size as u32))
        .metadata("qwen35moe.ssm.group_count", Value::U32(ssm_group_count as u32))
        .metadata("qwen35moe.ssm.time_step_rank", Value::U32(ssm_time_step_rank as u32))
        .metadata("qwen35moe.ssm.conv_kernel", Value::U32(ssm_conv_kernel as u32))
        .metadata("qwen35moe.full_attention_interval", Value::U32(cfg.full_attn_interval as u32))
        .metadata(
            "qwen35moe.rope.dimension_sections",
            Value::Array {
                elem_type: ValueType::I32,
                items: cfg.rope_sections().iter().map(|&v| Value::I32(v)).collect(),
            },
        );

    let mut rng_state: u64 = 42;
    let mut rand_f32 = |fan_in: usize| -> Vec<f32> {
        let s = scale(fan_in);
        (0..fan_in)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0 * s
            })
            .collect()
    };

    // token_embd: identity-like rows for predictability in tests.
    let mut tok_embd = vec![0.0f32; n_vocab * n_embd];
    for v in 0..n_vocab {
        for i in 0..n_embd {
            tok_embd[v * n_embd + i] = if v == i % n_vocab {
                1.0
            } else {
                0.01 * ((v * 7 + i * 13) as f32 / (n_embd * n_vocab) as f32)
            };
        }
    }
    builder = builder.tensor(vec2d("token_embd.weight", n_vocab, n_embd, &tok_embd));
    builder = builder.tensor(vec1d("output_norm.weight", &vec![1.0; n_embd]));
    builder = builder.tensor(vec2d("output.weight", n_vocab, n_embd, &tok_embd));

    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        builder = builder.tensor(vec1d(&format!("{prefix}.attn_norm.weight"), &vec![1.0; n_embd]));
        builder = builder.tensor(vec1d(
            &format!("{prefix}.post_attention_norm.weight"),
            &vec![1.0; n_embd],
        ));

        if cfg.is_recurrent(i) {
            let wqkv: Vec<f32> = rand_f32(conv_dim * n_embd);
            builder =
                builder.tensor(vec2d(&format!("{prefix}.attn_qkv.weight"), conv_dim, n_embd, &wqkv));

            let wqkv_gate: Vec<f32> = rand_f32(ba_dim * n_embd);
            builder = builder
                .tensor(vec2d(&format!("{prefix}.attn_gate.weight"), ba_dim, n_embd, &wqkv_gate));

            let conv_kernel: Vec<f32> = rand_f32(conv_dim * ssm_conv_kernel);
            builder = builder.tensor(vec2d(
                &format!("{prefix}.ssm_conv1d.weight"),
                conv_dim,
                ssm_conv_kernel,
                &conv_kernel,
            ));

            let alpha_bias: Vec<f32> = rand_f32(ba_dim);
            builder =
                builder.tensor(vec1d(&format!("{prefix}.ssm_dt.bias"), &alpha_bias));

            let ssm_a: Vec<f32> = (0..ba_dim).map(|j| -((j + 1) as f32)).collect();
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_a"), &ssm_a));

            let v_size = ssm_state_size * ssm_time_step_rank;
            builder =
                builder.tensor(vec1d(&format!("{prefix}.ssm_alpha.weight"), &vec![1.0; v_size]));
            builder =
                builder.tensor(vec1d(&format!("{prefix}.ssm_beta.weight"), &vec![0.0; v_size]));
            builder =
                builder.tensor(vec1d(&format!("{prefix}.ssm_norm.weight"), &vec![1.0; v_size]));

            let ssm_out: Vec<f32> = rand_f32(v_size * n_embd);
            builder = builder
                .tensor(vec2d(&format!("{prefix}.ssm_out.weight"), v_size, n_embd, &ssm_out));
        } else {
            let q_out = 2 * n_heads * head_size; // fused QK rows in Q
            let kv_out = n_kv_heads * head_size;

            let wq: Vec<f32> = rand_f32(q_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_q.weight"), q_out, n_embd, &wq));
            let wk: Vec<f32> = rand_f32(kv_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_k.weight"), kv_out, n_embd, &wk));
            let wv: Vec<f32> = rand_f32(kv_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_v.weight"), kv_out, n_embd, &wv));
            let wo: Vec<f32> = rand_f32(n_embd * q_out);
            builder = builder
                .tensor(vec2d(&format!("{prefix}.attn_output.weight"), n_embd, q_out, &wo));

            builder = builder
                .tensor(vec1d(&format!("{prefix}.attn_q_norm.weight"), &vec![1.0; head_size]));
            builder = builder
                .tensor(vec1d(&format!("{prefix}.attn_k_norm.weight"), &vec![1.0; head_size]));
        }

        // Dense FFN (expert_count = 0)
        let ffn_gate: Vec<f32> = rand_f32(n_ff * n_embd);
        builder =
            builder.tensor(vec2d(&format!("{prefix}.ffn_gate.weight"), n_ff, n_embd, &ffn_gate));
        let ffn_up: Vec<f32> = rand_f32(n_ff * n_embd);
        builder = builder.tensor(vec2d(&format!("{prefix}.ffn_up.weight"), n_ff, n_embd, &ffn_up));
        let ffn_down: Vec<f32> = rand_f32(n_embd * n_ff);
        builder = builder.tensor(vec2d(&format!("{prefix}.ffn_down.weight"), n_embd, n_ff, &ffn_down));
    }

    builder.build()
}

/// Build into a temp file so it can be mmapped by the normal loader.
pub fn write_temp(cfg: &SynthConfig) -> std::io::Result<tempfile::NamedTempFile> {
    let bytes = build_gguf(cfg);
    let tmp = tempfile::NamedTempFile::new()?;
    tmp.as_file().write_all(&bytes)?;
    tmp.as_file().sync_all()?;
    Ok(tmp)
}

use std::io::Write;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_structurally_sane() {
        for name in ["tiny", "medium", "large"] {
            let cfg = SynthConfig::preset(name).unwrap();
            assert_eq!(cfg.n_heads % cfg.n_kv_heads, 0);
            assert!(cfg.head_size.is_multiple_of(4), "rope sections need quarters");
            assert!(cfg.n_attn_layers() > 0 && cfg.is_recurrent(0));
            assert!(cfg.kv_bytes_per_token() > 0);
        }
    }

    #[test]
    fn builds_and_loads_tiny() {
        let tmp = write_temp(&SynthConfig::tiny()).unwrap();
        let loader = crate::model::loader::ModelLoader::open(tmp.path()).unwrap();
        let model = crate::model::pipeline::ModelWeights::load(&loader).unwrap();
        assert_eq!(model.cfg.block_count, 4);
    }
}
