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
        let prefix = if arch == "qwen3_5moe" { "qwen3_5moe" } else { "qwen35moe" };
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
        let rope_dimension_count = metadata
            .get_u32(&key("rope.dimension_count"))
            .unwrap_or(64);
        let rope_freq_base = metadata
            .get_f32(&key("rope.freq_base"))
            .unwrap_or(10_000_000.0);
        let context_length = metadata
            .get_u32(&key("context_length"))
            .unwrap_or(262_144);
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
        let ba_dim = ssm_time_step_rank * 2;
        let full_attn_q_fused_dim = attention_key_length * attention_head_count * 2;

        Ok(Self {
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
        })
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
                if missing.len() > 10 { " (shard boundary)".to_string() } else { String::new() },
                missing.len().min(10),
                missing.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
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
            extra.iter().take(10).cloned().collect::<Vec<_>>().join(", ")
        );
    }

    Ok(())
}
