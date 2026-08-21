//! End-to-end integration tests: synthetic Qwen3.5 GGUF through the full
//! pipeline (see `src/model/synth.rs` for the model factory).

use qwen3_5_397b_in_rust::model::pipeline::{
    generate_token, generate_token_batch, prefill, prefill_batch, GenerationState, ModelWeights,
};
use qwen3_5_397b_in_rust::model::sampler::{sample, SamplerConfig};
use qwen3_5_397b_in_rust::model::synth::{build_gguf, SynthConfig};

use std::io::Write;

/// Load a synthetic model from an in-memory GGUF.
fn load_synth(cfg: &SynthConfig) -> (tempfile::NamedTempFile, ModelWeights) {
    let gguf_bytes = build_gguf(cfg);
    let tmp = tempfile::NamedTempFile::new().expect("create tempfile");
    tmp.as_file().write_all(&gguf_bytes).expect("write gguf");
    tmp.as_file().sync_all().expect("sync");
    let loader =
        qwen3_5_397b_in_rust::model::loader::ModelLoader::open(tmp.path()).expect("open gguf");
    let model = ModelWeights::load(&loader).expect("load weights");
    (tmp, model)
}

#[test]
fn e2e_prefill_and_generate() {
    let cfg = SynthConfig::tiny();

    // Write to temp file and load
    let (_tmp, model) = load_synth(&cfg);

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
    let cfg = SynthConfig::tiny();

    let (_tmp, model) = load_synth(&cfg);

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

#[test]
fn e2e_batch_matches_sequential() {
    let (_tmp, model) = load_synth(&SynthConfig::tiny());

    // Distinct prompts of different lengths.
    let prompts: Vec<Vec<u32>> = vec![vec![0], vec![1, 2], vec![3, 4, 5, 6]];

    // Sequential reference: greedy decode each sequence one at a time.
    let mut sequential = Vec::new();
    for p in &prompts {
        let mut state = GenerationState::new(&model);
        prefill(&mut state, p, &model).expect("prefill");
        let mut last = *p.last().unwrap();
        let mut seq_gen = Vec::new();
        for _ in 0..8 {
            let (_h, next) = generate_token(&mut state, last, &model).expect("generate");
            seq_gen.push(next);
            last = next;
        }
        sequential.push(seq_gen);
    }

    // Batch: prefill + lockstep greedy decode.
    let mut states: Vec<GenerationState> =
        prompts.iter().map(|_| GenerationState::new(&model)).collect();
    let refs: Vec<&[u32]> = prompts.iter().map(|v| v.as_slice()).collect();
    prefill_batch(&mut states, &refs, &model).expect("prefill_batch");

    let mut last_tokens: Vec<u32> = prompts.iter().map(|v| *v.last().unwrap()).collect();
    let mut batched: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    for _step in 0..8 {
        let next =
            generate_token_batch(&mut states, &last_tokens, &model).expect("generate_batch");
        for (j, &t) in next.iter().enumerate() {
            batched[j].push(t);
        }
        last_tokens = next;
    }

    assert_eq!(batched, sequential, "batch output must match sequential");

    // Positions advanced identically too.
    for (state, p) in states.iter().zip(&prompts) {
        assert_eq!(state.pos, p.len() + 8);
    }
}

#[test]
fn e2e_batch_overflow_is_an_error() {
    // Tiny context so a long prompt overflows.
    let mut cfg = SynthConfig::tiny();
    cfg.context_length = 8;
    let (_tmp, model) = load_synth(&cfg);

    let prompts: Vec<Vec<u32>> = vec![vec![0, 1], vec![2, 3, 4, 5, 6, 7, 8, 9, 10]];
    let mut states: Vec<GenerationState> =
        prompts.iter().map(|_| GenerationState::new(&model)).collect();
    let refs: Vec<&[u32]> = prompts.iter().map(|v| v.as_slice()).collect();
    let err = prefill_batch(&mut states, &refs, &model).unwrap_err();
    assert!(err.contains("context overflow"), "{err}");
}
