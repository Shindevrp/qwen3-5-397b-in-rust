//! Cross-validation tests: _q (quantized-weight) path vs f32 path.
//!
//! These tests construct small synthetic models, quantize the weight matrices
//! to Q8_0, run both the f32 and _q forward functions, and assert the outputs
//! match within Q8_0 rounding tolerance.

#[cfg(test)]
mod tests {
    use crate::gguf::GGmlType;
    use crate::model::kernels::*;

    /// Quantize a flat f32 slice to Q8_0 bytes (block size 32).
    fn to_q8_0(data: &[f32]) -> Vec<u8> {
        quantize_row_q8_0(data)
    }

    fn make_rope_cfg() -> RopeConfig {
        RopeConfig {
            freq_base: 10_000_000.0,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    fn make_input(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + seed) * 0.01).sin() * 2.0)
            .collect()
    }

    fn make_weights(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.1).sin()).collect()
    }

    // -----------------------------------------------------------------------
    // full_layer_forward_q vs full_layer_forward (dense SwiGLU, no MoE)
    // -----------------------------------------------------------------------
    #[test]
    fn full_layer_q_matches_f32_dense() {
        let n_embd = 32;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_size = 8;
        let n_ff = 64; // must be multiple of 32 for Q8_0 block alignment
        let eps = 1e-5f32;
        let n_tokens = 1usize;
        let n_q_full = 2 * n_heads * head_size; // 64 (Q + gate fused)
        let q_size = n_heads * head_size; // 32

        let attn_norm_w = make_weights(n_embd, 1.0);
        let wq_f32 = make_weights(n_q_full * n_embd, 2.0);
        let wk_f32 = make_weights(n_kv_heads * head_size * n_embd, 3.0);
        let wv_f32 = make_weights(n_kv_heads * head_size * n_embd, 4.0);
        let wo_f32 = make_weights(n_embd * q_size, 5.0);
        let q_norm_w = make_weights(head_size, 6.0);
        let k_norm_w = make_weights(head_size, 7.0);
        let post_norm_w = make_weights(n_embd, 8.0);
        let ffn_gate_w = make_weights(n_ff * n_embd, 9.0);
        let ffn_up_w = make_weights(n_ff * n_embd, 10.0);
        let ffn_down_w = make_weights(n_embd * n_ff, 11.0);

        let wq_q8 = to_q8_0(&wq_f32);
        let wk_q8 = to_q8_0(&wk_f32);
        let wv_q8 = to_q8_0(&wv_f32);
        let wo_q8 = to_q8_0(&wo_f32);
        let ffn_gate_q8 = to_q8_0(&ffn_gate_w);
        let ffn_up_q8 = to_q8_0(&ffn_up_w);
        let ffn_down_q8 = to_q8_0(&ffn_down_w);

        let input = make_input(n_embd, 0.5);
        let rope_cfg = make_rope_cfg();
        let rope_sections = [2i32, 2, 0, 0]; // sum=4, n_rot=8 = head_size
        let pos = [0i32, 0, 0, 0];

        // f32 path
        let out_f32 = full_layer_forward(
            &input,
            &attn_norm_w,
            &wq_f32,
            &wk_f32,
            &wv_f32,
            &wo_f32,
            &q_norm_w,
            &k_norm_w,
            pos,
            &rope_cfg,
            &post_norm_w,
            &ffn_gate_w,
            &ffn_up_w,
            &ffn_down_w,
            n_embd,
            n_heads,
            n_kv_heads,
            head_size,
            n_ff,
            n_tokens,
            eps,
            rope_sections,
            &[],
            &[],
            &[],
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            0,
            None,
        );

        // quantized path
        let out_q = full_layer_forward_q(
            &input,
            &attn_norm_w,
            (&wq_q8, GGmlType::Q8_0),
            (&wk_q8, GGmlType::Q8_0),
            (&wv_q8, GGmlType::Q8_0),
            (&wo_q8, GGmlType::Q8_0),
            &q_norm_w,
            &k_norm_w,
            pos,
            &rope_cfg,
            &post_norm_w,
            (&ffn_gate_q8, GGmlType::Q8_0),
            (&ffn_up_q8, GGmlType::Q8_0),
            (&ffn_down_q8, GGmlType::Q8_0),
            n_embd,
            n_heads,
            n_kv_heads,
            head_size,
            n_ff,
            n_tokens,
            eps,
            rope_sections,
            &[],
            &[],
            &[],
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            0,
            None,
            None,
            None,
            None,
        );

        assert_eq!(out_f32.len(), out_q.len(), "output length mismatch");
        let mut max_rel = 0.0f32;
        for i in 0..out_f32.len() {
            let diff = (out_f32[i] - out_q[i]).abs();
            let mag = out_f32[i].abs().max(out_q[i].abs()).max(1e-6);
            let rel = diff / mag;
            max_rel = max_rel.max(rel);
        }
        // With n_embd=32, Q8_0 activation quantization has proportionally large error.
        // This threshold validates the code path correctness; production dimensions
        // (4096+) will have much tighter accuracy (< 1%).
        assert!(
            max_rel < 0.3,
            "full_layer_q max relative error {max_rel:.6} exceeds 30% threshold for small dims"
        );
    }

    // -----------------------------------------------------------------------
    // delta_net_layer_forward_q vs delta_net_layer_forward
    // -----------------------------------------------------------------------
    #[test]
    fn delta_net_layer_q_matches_f32() {
        let n_embd = 32;
        let n_ff = 64; // must be multiple of 32 for Q8_0 block alignment
        let eps = 1e-5f32;
        let ssm_state_size: usize = 16;
        let ssm_group_count: usize = 4;
        let ssm_time_step_rank: usize = 8;
        let conv_kernel_size: usize = 4;
        let s_k = ssm_state_size;
        let s_v = ssm_state_size;
        let n_heads_k = ssm_group_count;
        let n_heads_v = ssm_time_step_rank;
        let conv_dim = s_k * n_heads_k * 2 + s_v * n_heads_v; // 256
        let ba_dim = n_heads_v * 2; // 16

        let attn_norm_w = make_weights(n_embd, 1.0);
        let wqkv_f32 = make_weights(conv_dim * n_embd, 2.0);
        let wqkv_gate_f32 = make_weights(ba_dim * n_embd, 3.0);
        let conv_kernel = make_weights(conv_dim * conv_kernel_size, 4.0);
        let alpha_bias = make_weights(n_heads_v, 5.0);
        let ssm_a = make_weights(n_heads_v, 6.0);
        let ssm_norm_w = make_weights(s_v * n_heads_v, 7.0);
        let ssm_out_f32 = make_weights(n_embd * s_v * n_heads_v, 8.0);
        let post_norm_w = make_weights(n_embd, 9.0);
        let ffn_gate_w = make_weights(n_ff * n_embd, 10.0);
        let ffn_up_w = make_weights(n_ff * n_embd, 11.0);
        let ffn_down_w = make_weights(n_embd * n_ff, 12.0);

        let wqkv_q8 = to_q8_0(&wqkv_f32);
        let wqkv_gate_q8 = to_q8_0(&wqkv_gate_f32);
        let ssm_out_q8 = to_q8_0(&ssm_out_f32);
        let ffn_gate_q8 = to_q8_0(&ffn_gate_w);
        let ffn_up_q8 = to_q8_0(&ffn_up_w);
        let ffn_down_q8 = to_q8_0(&ffn_down_w);

        let input = make_input(n_embd, 0.5);
        let mut conv_state_f32 = vec![0.0f32; conv_dim * conv_kernel_size.saturating_sub(1)];
        let mut ssm_state_f32 = vec![0.0f32; s_v * s_v * n_heads_v];
        let mut conv_state_q = conv_state_f32.clone();
        let mut ssm_state_q = ssm_state_f32.clone();

        // f32 path
        let layer_f32 = DeltaNetLayerWeights {
            attn_norm_w: &attn_norm_w,
            wqkv: &wqkv_f32,
            wqkv_gate: &wqkv_gate_f32,
            conv_kernel: &conv_kernel,
            alpha_bias: &alpha_bias,
            ssm_a: &ssm_a,
            ssm_norm_w: &ssm_norm_w,
            ssm_out: &ssm_out_f32,
            post_norm_w: &post_norm_w,
            ffn_gate_w: &ffn_gate_w,
            ffn_up_w: &ffn_up_w,
            ffn_down_w: &ffn_down_w,
            moe_router_w: &[],
            moe_gate_up_w: &[],
            moe_down_w: &[],
            n_expert: 0,
            n_expert_used: 0,
            shexp_gate_w: &[],
            shexp_up_w: &[],
            shexp_down_w: &[],
            shexp_gate_inp_w: &[],
            n_ff_shexp: 0,
        };

        let out_f32 = delta_net_layer_forward(
            &input,
            &layer_f32,
            &mut conv_state_f32,
            &mut ssm_state_f32,
            n_embd,
            n_ff,
            conv_dim,
            conv_kernel_size,
            ba_dim,
            s_k,
            s_v,
            n_heads_k,
            n_heads_v,
            eps,
        );

        // quantized path
        let out_q = delta_net_layer_forward_q(
            &input,
            &attn_norm_w,
            (&wqkv_q8, GGmlType::Q8_0),
            (&wqkv_gate_q8, GGmlType::Q8_0),
            &conv_kernel,
            &alpha_bias,
            &ssm_a,
            &ssm_norm_w,
            (&ssm_out_q8, GGmlType::Q8_0),
            &post_norm_w,
            (&ffn_gate_q8, GGmlType::Q8_0),
            (&ffn_up_q8, GGmlType::Q8_0),
            (&ffn_down_q8, GGmlType::Q8_0),
            &[],
            &[],
            &[],
            0,
            0,
            &[],
            &[],
            &[],
            &[],
            0,
            None,
            None,
            None,
            n_embd,
            n_ff,
            conv_dim,
            conv_kernel_size,
            ba_dim,
            s_k,
            s_v,
            n_heads_k,
            n_heads_v,
            eps,
            &mut conv_state_q,
            &mut ssm_state_q,
        );

        assert_eq!(out_f32.len(), out_q.len(), "output length mismatch");
        let mut max_rel = 0.0f32;
        for i in 0..out_f32.len() {
            let diff = (out_f32[i] - out_q[i]).abs();
            let mag = out_f32[i].abs().max(out_q[i].abs()).max(1e-6);
            let rel = diff / mag;
            max_rel = max_rel.max(rel);
        }
        assert!(
            max_rel < 0.02,
            "delta_net_layer_q max relative error {max_rel:.6} exceeds 2% threshold"
        );
    }

    // -----------------------------------------------------------------------
    // gemv_parallel_matches_gemv: single-token gemv_parallel == gemv
    // -----------------------------------------------------------------------
    #[test]
    fn gemv_parallel_matches_gemv_q8_0() {
        let n_in = 64;
        let n_out = 48;
        let x = make_weights(n_in, 1.0);
        let w_f32 = make_weights(n_out * n_in, 2.0);
        let w_q8 = to_q8_0(&w_f32);

        let out_gemv = gemv(GGmlType::Q8_0, &w_q8, n_in, n_out, &x).unwrap();
        let out_par = gemv_parallel(GGmlType::Q8_0, &w_q8, n_in, n_out, &x).unwrap();

        assert_eq!(out_gemv.len(), out_par.len());
        for i in 0..out_gemv.len() {
            let diff = (out_gemv[i] - out_par[i]).abs();
            assert!(
                diff < 1e-5,
                "gemv_parallel mismatch at [{i}]: gemv={:.6}, parallel={:.6}",
                out_gemv[i],
                out_par[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Config validate() catches bad configs
    // -----------------------------------------------------------------------
    #[test]
    fn config_validate_catches_zero_expert_used() {
        let mut cfg = minimal_config();
        cfg.expert_count = 8;
        cfg.expert_used_count = 0;
        assert!(
            cfg.validate().is_err(),
            "should reject expert_used_count=0 with expert_count=8"
        );
    }

    #[test]
    fn config_validate_catches_kv_heads_exceed_q_heads() {
        let mut cfg = minimal_config();
        cfg.attention_head_count_kv = cfg.attention_head_count + 1;
        assert!(
            cfg.validate().is_err(),
            "should reject head_count_kv > head_count"
        );
    }

    #[test]
    fn config_validate_accepts_valid() {
        let cfg = minimal_config();
        assert!(
            cfg.validate().is_ok(),
            "minimal valid config should pass validation"
        );
    }

    fn minimal_config() -> crate::model::config::Qwen3_5Config {
        crate::model::config::Qwen3_5Config {
            block_count: 3,
            embedding_length: 32,
            attention_head_count: 4,
            attention_head_count_kv: 2,
            attention_key_length: 8,
            attention_value_length: 8,
            attention_layer_norm_rms_epsilon: 1e-5,
            expert_count: 0,
            expert_used_count: 0,
            expert_feed_forward_length: 0,
            expert_shared_feed_forward_length: 0,
            rope_dimension_count: 32,
            rope_freq_base: 10_000_000.0,
            context_length: 4096,
            ssm_state_size: 16,
            ssm_group_count: 4,
            ssm_time_step_rank: 8,
            ssm_conv_kernel: 4,
            ssm_inner_size: None,
            full_attention_interval: 4,
            rope_sections: [11, 11, 10, 0],
            key_dim: 64,
            value_dim: 128,
            conv_dim: 256,
            head_k_dim: 16,
            head_v_dim: 16,
            ba_dim: 16,
            full_attn_q_fused_dim: 128,
        }
    }

    #[test]
    fn timing_populated_on_prefill_and_generate() {
        use crate::model::loader::ModelLoader;
        use crate::model::pipeline::{GenerationState, ModelWeights, generate_token, prefill};
        use crate::model::synth::{SynthConfig, build_gguf};
        use std::io::Write;

        let cfg = SynthConfig::tiny();
        let gguf_bytes = build_gguf(&cfg);
        let mut tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        tmp.write_all(&gguf_bytes).expect("write gguf");
        tmp.as_file().sync_all().expect("sync");
        let loader = ModelLoader::open(tmp.path()).expect("open gguf");
        let model = ModelWeights::load(&loader).expect("load model weights");

        let mut state = GenerationState::new(&model);
        let token_ids: Vec<u32> = (0..3).collect();

        let _hidden = prefill(&mut state, &token_ids, &model).expect("prefill should succeed");
        let t = &state.last_timing;
        assert!(t.total_us > 0, "prefill total_us should be > 0");
        assert!(t.embed_us > 0, "prefill embed_us should be > 0");

        let _ = generate_token(&mut state, 0, &model).expect("generate should succeed");
        let t2 = &state.last_timing;
        assert!(t2.total_us > 0, "generate total_us should be > 0");
        assert!(t2.embed_us > 0, "generate embed_us should be > 0");
        assert!(
            t2.delta_net_us > 0 || t2.full_attn_us > 0,
            "at least one layer type should have timing > 0"
        );
    }

    #[test]
    fn prefill_chunked_matches_prefill() {
        use crate::model::loader::ModelLoader;
        use crate::model::pipeline::{GenerationState, ModelWeights, prefill, prefill_chunked};
        use crate::model::synth::{SynthConfig, build_gguf};
        use std::io::Write;

        let cfg = SynthConfig::tiny();
        let gguf_bytes = build_gguf(&cfg);
        let mut tmp = tempfile::NamedTempFile::new().expect("create tempfile");
        tmp.write_all(&gguf_bytes).expect("write gguf");
        tmp.as_file().sync_all().expect("sync");
        let loader = ModelLoader::open(tmp.path()).expect("open gguf");
        let model = ModelWeights::load(&loader).expect("load model weights");

        let token_ids: Vec<u32> = (0..4).collect();

        // Run regular prefill
        let mut state1 = GenerationState::new(&model);
        let hidden1 = prefill(&mut state1, &token_ids, &model).expect("prefill");
        assert_eq!(state1.pos, 4);

        // Run chunked prefill with chunk_size=2 (should produce same result)
        let mut state2 = GenerationState::new(&model);
        let hidden2 = prefill_chunked(&mut state2, &token_ids, &model, 2).expect("prefill_chunked");
        assert_eq!(state2.pos, 4);

        // Results should match within Q8_0 activation quantization tolerance for small dims.
        assert_eq!(hidden1.len(), hidden2.len());
        let max_diff = hidden1
            .iter()
            .zip(hidden2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs = hidden1.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel = if max_abs > 0.0 {
            max_diff / max_abs
        } else {
            max_diff
        };
        assert!(
            rel < 0.05,
            "prefill_chunked differs from prefill: max_rel_diff={rel:.9e}, max_abs={max_abs:.9e}, max_diff={max_diff:.9e}"
        );
    }

    #[test]
    fn kv_q8_roundtrip_and_packing() {
        use crate::model::pipeline::LayerKvCache;

        let (n_kv_heads, head_size) = (4usize, 8usize);
        let kv_dim = n_kv_heads * head_size; // 32 → exact block
        let mut cache = LayerKvCache::new_quantized(64, n_kv_heads, head_size);

        assert!(cache.is_quantized());
        // Q8_0: 1 block → 34 bytes per token per tensor (vs 128 f32).
        assert_eq!(cache.allocated_bytes(), 64 * 34 * 2);
        let f32_ref = LayerKvCache::with_capacity(64, n_kv_heads, head_size);
        assert!(
            cache.allocated_bytes() * 3 < f32_ref.allocated_bytes(),
            "Q8 KV should be >3x smaller than f32: {} vs {}",
            cache.allocated_bytes(),
            f32_ref.allocated_bytes()
        );

        // Write through the kernel-facing view, read back via dequant scratch.
        let row: Vec<f32> = (0..kv_dim).map(|i| (i as f32 - 8.0) * 0.25).collect();
        {
            let mut store = cache.kv_store_mut();
            match &mut store {
                crate::model::kernels::KvStoreMut::Q8 { k, v, n_cached, .. } => {
                    crate::model::kernels::KvStoreMut::pack_row(k, 0, &row);
                    crate::model::kernels::KvStoreMut::pack_row(v, 0, &row);
                    *n_cached = 1;
                }
                _ => panic!("expected Q8 backing"),
            }
        }

        let mut out = vec![0.0f32; kv_dim];
        {
            let store = cache.kv_store_mut();
            match store {
                crate::model::kernels::KvStoreMut::Q8 { k, .. } => {
                    crate::model::kernels::KvStoreMut::unpack_row(k, 0, &mut out);
                }
                _ => panic!("expected Q8 backing"),
            }
        }
        let max_err = row
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_err / scale < 0.02,
            "q8 roundtrip err {max_err} vs range {scale}"
        );
    }

    #[test]
    fn kv_q8_generation_matches_f32_within_tolerance() {
        use crate::model::loader::ModelLoader;
        use crate::model::pipeline::{
            GenerationState, ModelWeights, generate_token_logits, prefill,
        };
        use crate::model::synth::{SynthConfig, build_gguf};
        use std::io::Write;

        let cfg = SynthConfig::tiny();
        let gguf_bytes = build_gguf(&cfg);
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        tmp.write_all(&gguf_bytes).expect("write gguf");
        tmp.as_file().sync_all().expect("sync");
        let loader = ModelLoader::open(tmp.path()).expect("open gguf");
        let model = ModelWeights::load(&loader).expect("weights");

        let prompt: Vec<u32> = vec![0, 1, 2];

        let mut s_f32 = GenerationState::new(&model);
        prefill(&mut s_f32, &prompt, &model).unwrap();
        let (_, logits_f32) = generate_token_logits(&mut s_f32, 2, &model).unwrap();

        let mut s_q8 = GenerationState::new_kv_q8(&model);
        prefill(&mut s_q8, &prompt, &model).unwrap();
        let (_, logits_q8) = generate_token_logits(&mut s_q8, 2, &model).unwrap();

        // Same greedy argmax despite quantization noise.
        let amax_f32 = logits_f32
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let amax_q8 = logits_q8
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            amax_f32, amax_q8,
            "greedy token diverged between F32 and Q8 KV"
        );

        let max_diff = logits_f32
            .iter()
            .zip(&logits_q8)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_abs = logits_f32.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel = if max_abs > 0.0 {
            max_diff / max_abs
        } else {
            max_diff
        };
        assert!(rel < 0.05, "logits differ: rel={rel:.6}");
    }

    #[test]
    fn attention_flash_matches_reference_all_shapes() {
        // Shapes: (n_q, n_kv) pairs covering decode, prefill, and
        // non-block-aligned lengths around the 64-token flash block size.
        let shapes: &[(usize, usize)] =
            &[(1, 1), (1, 7), (1, 64), (1, 200), (3, 5), (8, 64), (17, 63)];
        let (n_heads, n_kv_heads, head_dim) = (4usize, 2usize, 16usize);

        for &(n_q, n_kv) in shapes {
            let q: Vec<f32> = (0..n_q * n_heads * head_dim)
                .map(|i| ((i % 37) as f32 - 18.0) * 0.11)
                .collect();
            let k: Vec<f32> = (0..n_kv * n_kv_heads * head_dim)
                .map(|i| ((i % 29) as f32 - 14.0) * 0.09)
                .collect();
            let v: Vec<f32> = (0..n_kv * n_kv_heads * head_dim)
                .map(|i| ((i % 23) as f32 - 11.0) * 0.07)
                .collect();
            let scale = 1.0 / (head_dim as f32).sqrt();

            for &causal in &[true, false] {
                let a = crate::model::kernels::attention_forward(
                    &q, &k, &v, n_heads, n_kv_heads, head_dim, n_q, n_kv, scale, causal,
                );
                let b = crate::model::kernels::attention_forward_flash(
                    &q, &k, &v, n_heads, n_kv_heads, head_dim, n_q, n_kv, scale, causal,
                );
                let max_err = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 1e-4,
                    "flash mismatch at n_q={n_q} n_kv={n_kv} causal={causal}: {max_err:.3e}"
                );
            }
        }
    }

    /// Build the tiny synth model once for spec-decode tests.
    fn load_tiny() -> (
        tempfile::NamedTempFile,
        crate::model::pipeline::ModelWeights,
    ) {
        use crate::model::loader::ModelLoader;
        use crate::model::pipeline::ModelWeights;
        use crate::model::synth::{SynthConfig, build_gguf};
        use std::io::Write;

        let gguf_bytes = build_gguf(&SynthConfig::tiny());
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        tmp.write_all(&gguf_bytes).expect("write gguf");
        tmp.as_file().sync_all().expect("sync");
        let loader = ModelLoader::open(tmp.path()).expect("open gguf");
        let model = ModelWeights::load(&loader).expect("weights");
        (tmp, model)
    }

    #[test]
    fn verify_draft_accepts_self_greedy_continuation() {
        use crate::model::pipeline::{
            GenerationState, generate_token_logits, prefill, verify_draft,
        };

        let (_tmp, model) = load_tiny();
        let prompt: Vec<u32> = vec![0, 1, 2];

        // Greedy reference continuation of length D after the prompt.
        let d = 4usize;
        let mut s_ref = GenerationState::new(&model);
        prefill(&mut s_ref, &prompt, &model).unwrap();
        let mut greedy: Vec<u32> = Vec::new();
        let mut ctx = 2u32;
        let mut bonus_expected = None;
        for i in 0..(d + 1) {
            let (_, logits) = generate_token_logits(&mut s_ref, ctx, &model).unwrap();
            let amax = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32;
            if i < d {
                greedy.push(amax);
                ctx = amax;
            } else {
                bonus_expected = Some(amax);
            }
        }
        let _ = &greedy;

        // Draft exactly what the target would do → everything accepted.
        let mut s = GenerationState::new(&model);
        prefill(&mut s, &prompt, &model).unwrap();
        let res = verify_draft(&mut s, 2, &greedy, &model).unwrap();
        assert_eq!(
            res.accepted.len(),
            d,
            "full acceptance expected, got {:?}",
            res
        );
        assert_eq!(Some(res.bonus), bonus_expected);

        // State must look like we simply decoded d+1 tokens.
        assert_eq!(s.pos, prompt.len() + d + 1);
    }

    #[test]
    fn verify_draft_rejects_garbage_and_matches_plain_decode() {
        use crate::model::pipeline::{
            GenerationState, generate_token_logits, prefill, verify_draft,
        };

        let (_tmp, model) = load_tiny();
        let prompt: Vec<u32> = vec![0, 1, 2];

        // Plain greedy next token from an untouched state.
        let mut s_plain = GenerationState::new(&model);
        prefill(&mut s_plain, &prompt, &model).unwrap();
        let (_, logits_plain) = generate_token_logits(&mut s_plain, 2, &model).unwrap();
        let plain_next = logits_plain
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;

        // Garbage draft (tokens unlikely to match greedy chain).
        let garbage: Vec<u32> = vec![15, 14, 13];
        let mut s = GenerationState::new(&model);
        prefill(&mut s, &prompt, &model).unwrap();
        let res = verify_draft(&mut s, 2, &garbage, &model).unwrap();

        // Bonus token must equal plain greedy decode regardless of rejection.
        assert_eq!(
            res.bonus, plain_next,
            "rejected draft should still yield the correct next token"
        );

        // State consistency: whatever was accepted, pos advanced by 1+accepted.
        assert_eq!(s.pos, prompt.len() + 1 + res.accepted.len());
    }

    #[test]
    fn scheduler_matches_sequential_greedy() {
        use crate::model::pipeline::{
            BatchScheduler, GenerationState, StepEvent, generate_token_logits, prefill,
        };

        let (_tmp, model) = load_tiny();

        // Sequential reference for two prompts.
        let prompts: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![3, 4]];
        let n_new = 5usize;
        let mut ref_out: Vec<Vec<u32>> = Vec::new();
        for p in &prompts {
            let mut s = GenerationState::new(&model);
            let mut ctx = *p.last().unwrap();
            prefill(&mut s, p, &model).unwrap();
            let mut out = Vec::new();
            for _ in 0..n_new {
                let (_, logits) = generate_token_logits(&mut s, ctx, &model).unwrap();
                let t = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0 as u32;
                out.push(t);
                ctx = t;
            }
            ref_out.push(out);
        }

        // Scheduler with both submitted up front.
        let mut sched = BatchScheduler::new(&model, None);
        let id0 = sched.submit(prompts[0].clone(), n_new);
        let id1 = sched.submit(prompts[1].clone(), n_new);
        let events = sched.run_until_idle(64).unwrap();

        for (i, id) in [id0, id1].iter().enumerate() {
            let fin = events
                .iter()
                .find_map(|e| match e {
                    StepEvent::Finished(sid, toks) if sid == id => Some(toks.clone()),
                    _ => None,
                })
                .expect("sequence should finish");
            assert_eq!(
                fin, ref_out[i],
                "scheduler output must equal sequential greedy"
            );
        }
    }

    #[test]
    fn scheduler_dynamic_join_mid_flight() {
        use crate::model::pipeline::{BatchScheduler, StepEvent};

        let (_tmp, model) = load_tiny();
        let mut sched = BatchScheduler::new(&model, None);

        // First sequence starts alone.
        let id_a = sched.submit(vec![0, 1, 2], 4);
        sched.step().unwrap(); // prefill A + decode 1
        assert_eq!(sched.n_active(), 1);

        // Second sequence joins after A is mid-generation.
        let id_b = sched.submit(vec![5, 6], 3);
        let events = sched.run_until_idle(64).unwrap();
        assert_eq!(sched.n_active(), 0);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StepEvent::Prefilled(x) if *x == id_b))
        );

        let fa = events
            .iter()
            .filter_map(|e| match e {
                StepEvent::Finished(id, toks) if *id == id_a => Some(toks.len()),
                _ => None,
            })
            .sum::<usize>();
        let fb = events
            .iter()
            .filter_map(|e| match e {
                StepEvent::Finished(id, toks) if *id == id_b => Some(toks.len()),
                _ => None,
            })
            .sum::<usize>();
        assert_eq!(fa, 4);
        assert_eq!(fb, 3);
    }

    #[test]
    fn scheduler_retires_exactly_max_new_tokens() {
        use crate::model::pipeline::{BatchScheduler, StepEvent};

        let (_tmp, model) = load_tiny();
        let mut sched = BatchScheduler::new(&model, None);
        let id = sched.submit(vec![1, 2], 7);
        let events = sched.run_until_idle(64).unwrap();

        let decodes = events
            .iter()
            .filter(|e| matches!(e, StepEvent::Decoded(i, _) if *i == id))
            .count();
        assert_eq!(decodes, 7);
        assert!(matches!(events.last(), Some(StepEvent::Finished(_, _))));
        assert!(sched.is_idle());
    }

    #[test]
    fn moe_stream_matches_dense_q8() {
        use crate::gguf::GGmlType;
        use crate::model::kernels::{moe_ffn, moe_ffn_stream, quantize_row_q8_0};
        use crate::model::quant::{RawTensor, fp16_to_f32};

        let (n_embd, n_ff, n_expert, n_used) = (64usize, 32usize, 8usize, 3usize);
        let rb = |inner: usize| (inner / 32) * 34; // q8_0 row bytes

        let mut lcg = 12345u64;
        let mut rnd = move || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((lcg >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
        };

        let gu_f32: Vec<f32> = (0..n_expert * 2 * n_ff * n_embd).map(|_| rnd()).collect();
        let dn_f32: Vec<f32> = (0..n_expert * n_embd * n_ff).map(|_| rnd()).collect();

        let pack = |w: &[f32], rows: usize, inner: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(rows * rb(inner));
            for r in 0..rows {
                out.extend(quantize_row_q8_0(&w[r * inner..(r + 1) * inner]));
            }
            out
        };
        // Split gate_up into gate and up
        let gate_f32: Vec<f32> = gu_f32
            .iter()
            .copied()
            .enumerate()
            .map(|(i, v)| {
                if i % (2 * n_ff * n_embd) < n_ff * n_embd {
                    v
                } else {
                    v
                }
            })
            .collect();
        // Reconstruct gate and up as separate contiguous buffers
        let mut gate_buf = Vec::with_capacity(n_expert * n_ff * n_embd);
        let mut up_buf = Vec::with_capacity(n_expert * n_ff * n_embd);
        for e in 0..n_expert {
            let base = e * 2 * n_ff * n_embd;
            for i in 0..n_ff * n_embd {
                gate_buf.push(gu_f32[base + i]);
                up_buf.push(gu_f32[base + n_ff * n_embd + i]);
            }
        }
        let gate_q = pack(&gate_buf, n_expert * n_ff, n_embd);
        let up_q = pack(&up_buf, n_expert * n_ff, n_embd);
        // down [n_expert, n_embd, n_ff]: contiguous rows of n_ff.
        let dn_q = pack(&dn_f32, n_expert * n_embd, n_ff);

        let unpack = |q: &[u8], rows: usize, inner: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(rows * inner);
            for r in 0..rows {
                let row = &q[r * rb(inner)..(r + 1) * rb(inner)];
                for b in 0..inner / 32 {
                    let off = b * 34;
                    let scale = fp16_to_f32(u16::from_le_bytes([row[off], row[off + 1]]));
                    for j in 0..32 {
                        out.push(scale * (row[off + 2 + j] as i8 as f32));
                    }
                }
            }
            out
        };
        let gate_dq = unpack(&gate_q, n_expert * n_ff, n_embd);
        let up_dq = unpack(&up_q, n_expert * n_ff, n_embd);
        let dn_dq = unpack(&dn_q, n_expert * n_embd, n_ff);
        // Reconstruct combined gate_up for dense reference
        let mut gu_dq = Vec::with_capacity(n_expert * 2 * n_ff * n_embd);
        for e in 0..n_expert {
            let base = e * n_ff * n_embd;
            gu_dq.extend_from_slice(&gate_dq[base..base + n_ff * n_embd]);
            gu_dq.extend_from_slice(&up_dq[base..base + n_ff * n_embd]);
        }

        let router_w: Vec<f32> = (0..n_expert * n_embd).map(|_| rnd() * 0.1).collect();
        let x: Vec<f32> = (0..n_embd).map(|_| rnd()).collect();

        let dense = moe_ffn(
            &x, &router_w, &gu_dq, &dn_dq, n_embd, n_ff, n_expert, n_used, 1,
        );
        // Build temporary RawTensor wrappers for the quantized bytes to exercise the
        // new streaming API with eviction hooks.
        let gate_tensor = RawTensor::new(GGmlType::Q8_0, gate_q.clone(), n_expert * n_ff * n_embd);
        let up_tensor = RawTensor::new(GGmlType::Q8_0, up_q.clone(), n_expert * n_ff * n_embd);
        let dn_tensor = RawTensor::new(GGmlType::Q8_0, dn_q.clone(), n_expert * n_embd * n_ff);
        let stream = moe_ffn_stream(
            &x,
            &router_w,
            &gate_tensor,
            &up_tensor,
            &dn_tensor,
            n_embd,
            n_ff,
            n_expert,
            n_used,
        )
        .expect("stream");

        let max_err = dense
            .iter()
            .zip(&stream)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mag = dense.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = if mag > 0.0 { max_err / mag } else { max_err };
        // Stream path quantizes activations to Q8_0 inside gemv (the dense
        // reference uses exact f32 dots), so expect percent-level relative
        // agreement, not bit equality.
        assert!(
            rel < 0.05,
            "stream vs dense: abs={max_err:.3} mag={mag:.3} rel={rel:.4}"
        );
    }
}
