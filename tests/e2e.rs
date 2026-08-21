//! End-to-end integration test: build a tiny synthetic Qwen3.5 GGUF,
//! load it through the full pipeline, run prefill + generate, verify output.

use qwen3_5_397b_in_rust::gguf::writer::{GgufBuilder, TensorSpec};
use qwen3_5_397b_in_rust::gguf::{GGmlType, Value};
use qwen3_5_397b_in_rust::model::loader::ModelLoader;
use qwen3_5_397b_in_rust::model::pipeline::{
    generate_token, prefill, GenerationState, ModelWeights,
};
use qwen3_5_397b_in_rust::model::sampler::{sample, SamplerConfig};

use std::io::Write;

/// Tiny Qwen3.5 config: 4 layers (3 delta-net + 1 full-attn), dense FFN,
/// small dimensions. Just enough to exercise the full pipeline.
struct TinyConfig {
    n_embd: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_size: usize,
    n_ff: usize,
    n_vocab: usize,
    n_layers: usize,
    full_attn_interval: usize,
    // SSM (delta-net)
    ssm_state_size: usize,
    ssm_group_count: usize,
    ssm_time_step_rank: usize,
    ssm_conv_kernel: usize,
    eps: f32,
    rope_freq_base: f32,
    context_length: usize,
}

impl TinyConfig {
    fn small() -> Self {
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

    fn conv_dim(&self) -> usize {
        let key_dim = self.ssm_state_size * self.ssm_group_count;
        let value_dim = self.ssm_state_size * self.ssm_time_step_rank;
        key_dim * 2 + value_dim
    }

    fn ba_dim(&self) -> usize {
        self.ssm_time_step_rank * 2
    }

    fn is_recurrent(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.full_attn_interval)
    }

    fn rope_sections(&self) -> [i32; 4] {
        // Sections must sum to head_size / 2 (rotation pairs per head)
        let half = (self.head_size / 4) as i32; // split into 2 non-zero sections
        [half, half, 0, 0]
    }
}

/// Encode f32 slice to f32 bytes in LE.
fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Encode a 1-D f32 tensor.
fn vec1d(name: &str, data: &[f32]) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        ggml_type: GGmlType::F32,
        dims: vec![data.len() as u64],
        data: f32_bytes(data),
    }
}

/// Encode a 2-D f32 tensor (row-major).
fn vec2d(name: &str, rows: usize, cols: usize, data: &[f32]) -> TensorSpec {
    assert_eq!(data.len(), rows * cols);
    TensorSpec {
        name: name.to_string(),
        ggml_type: GGmlType::F32,
        dims: vec![rows as u64, cols as u64],
        data: f32_bytes(data),
    }
}

/// Build a synthetic GGUF file with all tensors needed for a tiny Qwen3.5 model.
/// Uses Xavier-like initialization (1/sqrt(fan_in)) so activations don't vanish.
fn build_tiny_gguf(cfg: &TinyConfig) -> Vec<u8> {
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

    // GGUF metadata (qwen35moe architecture)
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
                elem_type: qwen3_5_397b_in_rust::gguf::value::ValueType::I32,
                items: cfg.rope_sections().iter().map(|&v| Value::I32(v)).collect(),
            },
        );

    // Global tensors
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

    // token_embd: [n_vocab, n_embd] — use identity-like (one-hot scaled) for predictability
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

    // output_norm: all ones (identity norm)
    builder = builder.tensor(vec1d("output_norm.weight", &vec![1.0; n_embd]));

    // output.weight: same as token_embd (weight-tied)
    builder = builder.tensor(vec2d("output.weight", n_vocab, n_embd, &tok_embd));

    // Per-layer tensors
    for i in 0..n_layers {
        let prefix = format!("blk.{i}");

        // Norms: all ones
        builder = builder.tensor(vec1d(&format!("{prefix}.attn_norm.weight"), &vec![1.0; n_embd]));
        builder = builder.tensor(vec1d(&format!("{prefix}.post_attention_norm.weight"), &vec![1.0; n_embd]));

        if cfg.is_recurrent(i) {
            // Delta-net layer
            // wqkv: [conv_dim, n_embd]
            let wqkv: Vec<f32> = rand_f32(conv_dim * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_qkv.weight"), conv_dim, n_embd, &wqkv));

            // wqkv_gate: [ba_dim, n_embd]
            let wqkv_gate: Vec<f32> = rand_f32(ba_dim * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_gate.weight"), ba_dim, n_embd, &wqkv_gate));

            // conv_kernel: [conv_dim, ssm_conv_kernel]
            let conv_kernel: Vec<f32> = rand_f32(conv_dim * ssm_conv_kernel);
            builder = builder.tensor(vec2d(&format!("{prefix}.ssm_conv1d.weight"), conv_dim, ssm_conv_kernel, &conv_kernel));

            // alpha_bias: [ba_dim]
            let alpha_bias: Vec<f32> = rand_f32(ba_dim);
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_dt.bias"), &alpha_bias));

            // ssm_a: [ba_dim]
            let ssm_a: Vec<f32> = (0..ba_dim).map(|j| -((j + 1) as f32)).collect();
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_a"), &ssm_a));

            // ssm_norm_w: [ssm_time_step_rank * ssm_state_size] = [v_size]
            let v_size = cfg.ssm_state_size * cfg.ssm_time_step_rank;
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_alpha.weight"), &vec![1.0; v_size]));
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_beta.weight"), &vec![0.0; v_size]));
            builder = builder.tensor(vec1d(&format!("{prefix}.ssm_norm.weight"), &vec![1.0; v_size]));

            // ssm_out: [v_size, n_embd]
            let ssm_out: Vec<f32> = rand_f32(v_size * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.ssm_out.weight"), v_size, n_embd, &ssm_out));

            // Dense FFN (expert_count = 0)
            let ffn_gate: Vec<f32> = rand_f32(n_ff * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_gate.weight"), n_ff, n_embd, &ffn_gate));
            let ffn_up: Vec<f32> = rand_f32(n_ff * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_up.weight"), n_ff, n_embd, &ffn_up));
            let ffn_down: Vec<f32> = rand_f32(n_embd * n_ff);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_down.weight"), n_embd, n_ff, &ffn_down));
        } else {
            // Full-attention layer
            let q_out = 2 * n_heads * head_size; // fused QK in Q weight
            let kv_out = n_kv_heads * head_size;

            let wq: Vec<f32> = rand_f32(q_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_q.weight"), q_out, n_embd, &wq));

            let wk: Vec<f32> = rand_f32(kv_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_k.weight"), kv_out, n_embd, &wk));

            let wv: Vec<f32> = rand_f32(kv_out * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_v.weight"), kv_out, n_embd, &wv));

            let wo: Vec<f32> = rand_f32(n_embd * q_out);
            builder = builder.tensor(vec2d(&format!("{prefix}.attn_output.weight"), n_embd, q_out, &wo));

            // Q/K norms: all ones
            builder = builder.tensor(vec1d(&format!("{prefix}.attn_q_norm.weight"), &vec![1.0; head_size]));
            builder = builder.tensor(vec1d(&format!("{prefix}.attn_k_norm.weight"), &vec![1.0; head_size]));

            // Dense FFN
            let ffn_gate: Vec<f32> = rand_f32(n_ff * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_gate.weight"), n_ff, n_embd, &ffn_gate));
            let ffn_up: Vec<f32> = rand_f32(n_ff * n_embd);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_up.weight"), n_ff, n_embd, &ffn_up));
            let ffn_down: Vec<f32> = rand_f32(n_embd * n_ff);
            builder = builder.tensor(vec2d(&format!("{prefix}.ffn_down.weight"), n_embd, n_ff, &ffn_down));
        }
    }

    builder.build()
}

#[test]
fn e2e_prefill_and_generate() {
    let cfg = TinyConfig::small();
    let gguf_bytes = build_tiny_gguf(&cfg);

    // Write to temp file and load
    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    tmp.as_file().write_all(&gguf_bytes).expect("write gguf");
    tmp.as_file().sync_all().expect("sync");

    let loader = ModelLoader::open(tmp.path()).expect("open gguf");
    let model = ModelWeights::load(&loader).expect("load weights");

    // Verify config loaded correctly
    assert_eq!(model.cfg.block_count, cfg.n_layers as u32);
    assert_eq!(model.cfg.embedding_length, cfg.n_embd as u32);
    assert_eq!(model.cfg.attention_head_count, cfg.n_heads as u32);
    assert_eq!(model.cfg.expert_count, 0);

    // Build a short prompt (3 tokens, all in vocab range)
    let prompt_tokens: Vec<u32> = vec![0, 1, 2];
    assert!(prompt_tokens.iter().all(|&t| (t as usize) < cfg.n_vocab));

    let mut state = GenerationState::new(&model);

    // Prefill
    prefill(&mut state, &prompt_tokens, &model).expect("prefill");
    assert_eq!(state.pos, prompt_tokens.len());

    // Generate a few tokens
    let mut last_token = *prompt_tokens.last().unwrap();
    let mut generated = Vec::new();
    for _step in 0..10 {
        let (_hidden, next_token) =
            generate_token(&mut state, last_token, &model).expect("generate_token");
        assert!(
            (next_token as usize) < cfg.n_vocab,
            "generated token {next_token} out of vocab range {n_vocab}",
            n_vocab = cfg.n_vocab
        );
        generated.push(next_token);
        last_token = next_token;
    }

    assert_eq!(generated.len(), 10);
    assert_eq!(state.pos, prompt_tokens.len() + 10);

    println!("Generated tokens: {generated:?}");
}

#[test]
fn e2e_sampling() {
    let cfg = TinyConfig::small();
    let gguf_bytes = build_tiny_gguf(&cfg);

    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    tmp.as_file().write_all(&gguf_bytes).expect("write gguf");
    tmp.as_file().sync_all().expect("sync");

    let loader = ModelLoader::open(tmp.path()).expect("open gguf");
    let model = ModelWeights::load(&loader).expect("load weights");

    let prompt_tokens: Vec<u32> = vec![0];
    let mut state = GenerationState::new(&model);
    prefill(&mut state, &prompt_tokens, &model).expect("prefill");

    let cfg_sampler = SamplerConfig {
        temperature: 2.0,
        top_k: 5,
        top_p: 0.9,
        ..Default::default()
    };

    // Generate with sampling — should produce valid tokens
    let mut last_token = prompt_tokens[0];
    let mut generated = Vec::new();
    let mut history = prompt_tokens.clone();
    for _step in 0..20 {
        let (_hidden, logits) = qwen3_5_397b_in_rust::model::pipeline::generate_token_logits(
            &mut state, last_token, &model,
        )
        .expect("generate_token_logits");
        let token = sample(&logits, &cfg_sampler, &history);
        assert!((token as usize) < cfg.n_vocab);
        generated.push(token);
        history.push(token);
        last_token = token;
    }

    assert_eq!(generated.len(), 20);
    println!("Sampled tokens: {generated:?}");
}
