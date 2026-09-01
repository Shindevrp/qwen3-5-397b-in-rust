use crate::gguf::Metadata;
use crate::gguf::error::GgufError;

#[derive(Debug, Clone)]
pub struct Qwen3_5Config {
    pub block_count: u32,
    pub embedding_length: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub attention_key_length: u32,
    pub attention_value_length: u32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub expert_count: u32,
    pub expert_used_count: u32,
    pub expert_feed_forward_length: u32,
    pub expert_shared_feed_forward_length: u32,
    pub rope_dimension_count: u32,
    pub rope_freq_base: f32,
    pub context_length: u32,
    pub ssm_state_size: u32,
    pub ssm_group_count: u32,
    pub ssm_time_step_rank: u32,
    pub ssm_conv_kernel: u32,
    pub ssm_inner_size: Option<u32>,
    pub full_attention_interval: u32,
    /// Partial IMRoPE sections: Qwen3.5 applies RoPE to disjoint chunks of
    /// `rope_dimension_count` pairs, partitioned by these section sizes
    /// (e.g. [11, 11, 10, 0] for the 397B). The last entry is typically 0.
    pub rope_sections: [i32; 4],

    pub key_dim: u32,
    pub value_dim: u32,
    pub conv_dim: u32,
    pub head_k_dim: u32,
    pub head_v_dim: u32,
    pub ba_dim: u32,
    pub full_attn_q_fused_dim: u32,
}

impl Qwen3_5Config {
    pub fn is_recurrent(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.full_attention_interval as usize)
    }
}

impl Qwen3_5Config {
    pub fn from_metadata(metadata: &Metadata) -> Result<Self, GgufError> {
        let arch = metadata.get_str("general.architecture")?;
        if arch != "qwen35moe" && arch != "qwen3_5moe" {
            return Err(GgufError::TypeMismatch {
                key: "general.architecture".to_string(),
                actual: arch.to_string(),
                expected: "qwen35moe",
            });
        }

        // The merged GGUF uses "qwen35moe"; the original llama.cpp PR used
        // "qwen3_5moe". Accept the keys under whichever prefix the file declares.
        let prefix = if arch == "qwen3_5moe" {
            "qwen3_5moe"
        } else {
            "qwen35moe"
        };
        let key = |suffix: &str| format!("{prefix}.{suffix}");

        let block_count = metadata.get_u32(&key("block_count"))?;
        let embedding_length = metadata.get_u32(&key("embedding_length"))?;
        let attention_head_count = metadata.get_u32(&key("attention.head_count"))?;
        let attention_head_count_kv = metadata.get_u32(&key("attention.head_count_kv"))?;
        let attention_key_length = metadata.get_u32(&key("attention.key_length"))?;
        let attention_value_length = metadata
            .get_u32(&key("attention.value_length"))
            .unwrap_or(attention_key_length);
        let attention_layer_norm_rms_epsilon =
            metadata.get_f32(&key("attention.layer_norm_rms_epsilon"))?;
        let expert_count = metadata.get_u32(&key("expert_count"))?;
        let expert_used_count = metadata.get_u32(&key("expert_used_count"))?;
        let expert_feed_forward_length = metadata.get_u32(&key("expert_feed_forward_length"))?;
        let expert_shared_feed_forward_length =
            metadata.get_u32(&key("expert_shared_feed_forward_length"))?;
        let rope_dimension_count = metadata.get_u32(&key("rope.dimension_count")).unwrap_or(64);
        let rope_freq_base = metadata
            .get_f32(&key("rope.freq_base"))
            .unwrap_or(10_000_000.0);
        let context_length = metadata.get_u32(&key("context_length")).unwrap_or(262_144);
        let ssm_state_size = metadata.get_u32(&key("ssm.state_size"))?;
        let ssm_group_count = metadata.get_u32(&key("ssm.group_count"))?;
        let ssm_time_step_rank = metadata.get_u32(&key("ssm.time_step_rank"))?;
        let ssm_conv_kernel = metadata.get_u32(&key("ssm.conv_kernel"))?;
        let ssm_inner_size = metadata.get_u32(&key("ssm.inner_size")).ok();
        let full_attention_interval = metadata.get_u32(&key("full_attention_interval"))?;
        let rope_sections = metadata.get_i32_array(&key("rope.dimension_sections"))?;
        if rope_sections.len() != 4 {
            return Err(GgufError::Io(format!(
                "{prefix}.rope.dimension_sections has wrong array length; expected 4, got {}",
                rope_sections.len()
            )));
        }
        let mut rope_sections_arr = [0i32; 4];
        rope_sections_arr.copy_from_slice(&rope_sections);

        if attention_value_length != attention_key_length {
            return Err(GgufError::TypeMismatch {
                key: key("attention.value_length"),
                actual: attention_value_length.to_string(),
                expected: "equal to attention.key_length",
            });
        }

        let key_dim = ssm_state_size * ssm_group_count;
        let value_dim = ssm_state_size * ssm_time_step_rank;
        let conv_dim = key_dim * 2 + value_dim;
        let head_k_dim = ssm_state_size;
        let head_v_dim = ssm_inner_size
            .map(|d| d / ssm_time_step_rank)
            .unwrap_or(ssm_state_size);
        let ba_dim = ssm_state_size * ssm_time_step_rank;
        let full_attn_q_fused_dim = attention_key_length * attention_head_count * 2;

        let cfg = Self {
            block_count,
            embedding_length,
            attention_head_count,
            attention_head_count_kv,
            attention_key_length,
            attention_value_length,
            attention_layer_norm_rms_epsilon,
            expert_count,
            expert_used_count,
            expert_feed_forward_length,
            expert_shared_feed_forward_length,
            rope_dimension_count,
            rope_freq_base,
            context_length,
            ssm_state_size,
            ssm_group_count,
            ssm_time_step_rank,
            ssm_conv_kernel,
            ssm_inner_size,
            full_attention_interval,
            rope_sections: rope_sections_arr,
            key_dim,
            value_dim,
            conv_dim,
            head_k_dim,
            head_v_dim,
            ba_dim,
            full_attn_q_fused_dim,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Validate numeric invariants that must hold for correct inference.
    // Float comparisons here are validity bounds (>=, <=, == on config
    // constants), not ordering logic — silence the partial-ord lint.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn validate(&self) -> Result<(), GgufError> {
        macro_rules! check {
            ($cond:expr, $msg:expr) => {
                if !($cond) {
                    return Err(GgufError::Io($msg.to_string()));
                }
            };
        }

        check!(self.block_count > 0, "block_count must be > 0");
        check!(self.embedding_length > 0, "embedding_length must be > 0");
        check!(
            self.attention_head_count > 0,
            "attention.head_count must be > 0"
        );
        check!(
            self.attention_head_count_kv > 0
                && self.attention_head_count_kv <= self.attention_head_count,
            "attention.head_count_kv must be > 0 and <= head_count (GQA)"
        );
        check!(
            self.attention_key_length > 0
                && self.attention_key_length == self.attention_value_length,
            "attention.key_length must be > 0 and == value_length"
        );
        check!(
            self.attention_layer_norm_rms_epsilon > 0.0,
            "attention.layer_norm_rms_epsilon must be > 0"
        );
        check!(
            self.full_attention_interval > 0,
            "full_attention_interval must be > 0"
        );
        check!(self.ssm_state_size > 0, "ssm.state_size must be > 0");
        check!(self.ssm_group_count > 0, "ssm.group_count must be > 0");
        check!(
            self.ssm_time_step_rank > 0,
            "ssm.time_step_rank must be > 0"
        );
        check!(self.ssm_conv_kernel > 0, "ssm.conv_kernel must be > 0");
        check!(self.rope_freq_base > 0.0, "rope.freq_base must be > 0");
        check!(
            self.rope_dimension_count > 0 && self.rope_dimension_count.is_multiple_of(2),
            "rope.dimension_count must be > 0 and even"
        );
        check!(self.context_length > 0, "context_length must be > 0");

        // MoE invariants
        if self.expert_count > 0 {
            check!(
                self.expert_used_count > 0 && self.expert_used_count <= self.expert_count,
                "expert_used_count must be > 0 and <= expert_count when expert_count > 0"
            );
            check!(
                self.expert_feed_forward_length > 0,
                "expert_feed_forward_length must be > 0 when expert_count > 0"
            );
        }

        // Head dimension divisibility: ssm_state_size must be divisible by ssm_group_count
        check!(
            self.ssm_state_size.is_multiple_of(self.ssm_group_count),
            format!(
                "ssm.state_size ({}) must be divisible by ssm.group_count ({})",
                self.ssm_state_size, self.ssm_group_count
            )
        );

        // ba_dim = ssm_state_size * ssm_time_step_rank (z projection for the
        // gated norm: the attn_gate tensor maps n_embd -> [head_v_dim, num_v_heads])
        let expected_ba = self.ssm_state_size * self.ssm_time_step_rank;
        check!(
            self.ba_dim == expected_ba,
            format!(
                "ba_dim ({}) must equal ssm_state_size * ssm_time_step_rank ({})",
                self.ba_dim, expected_ba
            )
        );

        // conv_dim = 2 * key_dim + value_dim
        let expected_conv = self.key_dim * 2 + self.value_dim;
        check!(
            self.conv_dim == expected_conv,
            format!(
                "conv_dim ({}) must equal 2 * key_dim + value_dim ({})",
                self.conv_dim, expected_conv
            )
        );

        // Rope sections sum must not exceed rope_dimension_count
        let sections_sum: i32 = self.rope_sections.iter().sum();
        check!(
            sections_sum >= 0 && (sections_sum as u32) <= self.rope_dimension_count,
            format!(
                "rope_sections sum ({}) must be >= 0 and <= rope_dimension_count ({})",
                sections_sum, self.rope_dimension_count
            )
        );

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5ExpectedTensors {
    pub global: Vec<String>,
    pub per_layer: Vec<String>,
}

impl Qwen3_5ExpectedTensors {
    pub fn for_config(cfg: &Qwen3_5Config) -> Self {
        let global = vec![
            "token_embd.weight".to_string(),
            "output_norm.weight".to_string(),
            "output.weight".to_string(),
        ];

        let mut per_layer = Vec::new();
        let base_names = [
            "attn_norm.weight",
            "post_attention_norm.weight",
            "ffn_gate_inp.weight",
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
            "ffn_gate_inp_shexp.weight",
            "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight",
            "ffn_down_shexp.weight",
        ];

        let full_attn = [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
        ];

        let linear_attn = [
            "attn_qkv.weight",
            "attn_gate.weight",
            "ssm_conv1d.weight",
            "ssm_dt.bias",
            "ssm_a",
            "ssm_alpha.weight",
            "ssm_beta.weight",
            "ssm_norm.weight",
            "ssm_out.weight",
        ];

        for i in 0..cfg.block_count {
            for name in &base_names {
                per_layer.push(format!("blk.{i}.{}", name));
            }
            if !cfg.is_recurrent(i as usize) {
                for name in &full_attn {
                    per_layer.push(format!("blk.{i}.{}", name));
                }
            } else {
                for name in &linear_attn {
                    per_layer.push(format!("blk.{i}.{}", name));
                }
            }
        }

        Self { global, per_layer }
    }

    pub fn all_names(&self) -> Vec<String> {
        let mut all = self.global.clone();
        all.extend(self.per_layer.clone());
        all
    }
}

pub fn validate_tensors(
    metadata: &Metadata,
    tensors: &[crate::gguf::TensorMeta],
) -> Result<(), GgufError> {
    let cfg = Qwen3_5Config::from_metadata(metadata)?;
    let expected = Qwen3_5ExpectedTensors::for_config(&cfg);
    let expected_set: std::collections::HashSet<_> = expected.all_names().into_iter().collect();
    let actual_set: std::collections::HashSet<_> = tensors.iter().map(|t| t.name.clone()).collect();

    let mut missing: Vec<_> = expected_set.difference(&actual_set).cloned().collect();
    if !missing.is_empty() {
        missing.sort();
        // Split GGUF shards (general.split.count > 1) distribute tensors across
        // files and may even cut a layer mid-way, so a single shard is never
        // complete on its own. Only fail on a full (unsplit) model.
        let split = metadata
            .get("split.count")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(1);
        if split > 1 {
            eprintln!(
                "warning: missing expected tensors in this shard ({}){}, first {}: {}",
                missing.len(),
                if missing.len() > 10 {
                    " (shard boundary)".to_string()
                } else {
                    String::new()
                },
                missing.len().min(10),
                missing
                    .iter()
                    .take(10)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        } else {
            return Err(GgufError::Io(format!(
                "missing expected tensors ({}): {}",
                missing.len(),
                missing.join(", ")
            )));
        }
    }

    let mut extra: Vec<_> = actual_set.difference(&expected_set).cloned().collect();
    if !extra.is_empty() {
        extra.sort();
        eprintln!(
            "warning: unexpected tensors ({}), first {}: {}",
            extra.len(),
            extra.len().min(10),
            extra
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    Ok(())
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// Exact metadata probed remotely from
    /// lmstudio-community/Qwen3.5-397B-A17B-GGUF Q4_K_M shard 1 header.
    /// Guards against validate() rejecting the production model.
    #[test]
    fn validate_accepts_real_397b_metadata() {
        let cfg = Qwen3_5Config {
            block_count: 60,
            embedding_length: 4096,
            attention_head_count: 32,
            attention_head_count_kv: 2,
            attention_key_length: 256,
            attention_value_length: 256,
            attention_layer_norm_rms_epsilon: 1e-6,
            expert_count: 512,
            expert_used_count: 10,
            expert_feed_forward_length: 1024,
            expert_shared_feed_forward_length: 1024,
            rope_dimension_count: 64,
            rope_freq_base: 10_000_000.0,
            context_length: 262_144,
            ssm_state_size: 128,
            ssm_group_count: 16,
            ssm_time_step_rank: 64,
            ssm_conv_kernel: 4,
            ssm_inner_size: Some(8192),
            full_attention_interval: 4,
            rope_sections: [11, 11, 10, 0],
            // Derived by from_metadata; mirror the formulas here.
            key_dim: 128 * 16,
            value_dim: 128 * 64,
            conv_dim: 128 * 16 * 2 + 128 * 64,
            head_k_dim: 128,
            head_v_dim: 8192 / 64,
            ba_dim: 128 * 64,
            full_attn_q_fused_dim: 256 * 32 * 2,
        };
        cfg.validate()
            .expect("real 397B config must pass validation");
        assert_eq!(cfg.head_v_dim, 128);
        assert_eq!(cfg.conv_dim, 12288);
    }
}
