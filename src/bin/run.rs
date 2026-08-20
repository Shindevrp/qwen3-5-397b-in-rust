//! End-to-end inference: load GGUF + tokenizer, encode prompt, generate tokens with sampling.

use std::env;
use std::path::PathBuf;

use qwen3_5_397b_in_rust::model::loader::ModelLoader;
use qwen3_5_397b_in_rust::model::pipeline::{
    generate_token, generate_token_logits, prefill, GenerationState, ModelWeights,
};
use qwen3_5_397b_in_rust::model::sampler::{sample, SamplerConfig};
use qwen3_5_397b_in_rust::tokenizer::QwenTokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: run <model.gguf> <tokenizer.json> [prompt] [options]");
        eprintln!("Options:");
        eprintln!("  --n-predict N      Max tokens to generate (default: 128)");
        eprintln!("  --temperature T    Sampling temperature (default: 1.0)");
        eprintln!("  --top-k K          Top-k sampling, 0=disabled (default: 40)");
        eprintln!("  --top-p P          Top-p nucleus sampling, 1.0=disabled (default: 0.9)");
        eprintln!("  --repeat-penalty R Repetition penalty (default: 1.0)");
        eprintln!("  --argmax           Use greedy decoding (no sampling)");
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let tokenizer_path = PathBuf::from(&args[2]);

    let mut prompt = String::new();
    let mut n_predict: usize = 128;
    let mut cfg = SamplerConfig::default();
    let mut use_argmax = false;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--n-predict" => {
                i += 1;
                if i < args.len() {
                    n_predict = args[i].parse().unwrap_or(128);
                }
            }
            "--temperature" => {
                i += 1;
                if i < args.len() {
                    cfg.temperature = args[i].parse().unwrap_or(1.0);
                }
            }
            "--top-k" => {
                i += 1;
                if i < args.len() {
                    cfg.top_k = args[i].parse().unwrap_or(40);
                }
            }
            "--top-p" => {
                i += 1;
                if i < args.len() {
                    cfg.top_p = args[i].parse().unwrap_or(0.9);
                }
            }
            "--repeat-penalty" => {
                i += 1;
                if i < args.len() {
                    cfg.repeat_penalty = args[i].parse().unwrap_or(1.0);
                }
            }
            "--argmax" => {
                use_argmax = true;
            }
            other if prompt.is_empty() => {
                prompt = other.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    if prompt.is_empty() {
        prompt = "Hello, world!".to_string();
    }

    eprintln!("Loading model from {} ...", model_path.display());
    let loader = ModelLoader::open(&model_path)?;

    eprintln!("Loading tokenizer from {} ...", tokenizer_path.display());
    let tokenizer = QwenTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading weights ...");
    let model = ModelWeights::load(&loader).map_err(|e| anyhow::anyhow!("{e}"))?;

    eprintln!("Prompt: {prompt}");
    let tokens = tokenizer.encode(&prompt, false)?;
    eprintln!("Input tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);

    let mut state = GenerationState::new(&model);

    eprintln!("Prefilling {} tokens ...", tokens.len());
    prefill(&mut state, &tokens, &model);

    let eot_id: Option<u32> = tokenizer
        .token_to_id("\u{2023}")
        .or_else(|| tokenizer.token_to_id("\u{2018}"));

    if use_argmax {
        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();

        for _step in 0..n_predict {
            let (_hidden, next_token) = generate_token(&mut state, last_token_id, &model);
            gen_tokens.push(next_token);
            last_token_id = next_token;

            if Some(next_token) == eot_id {
                break;
            }
        }

        eprintln!("Generated {} tokens (greedy)", gen_tokens.len());
        let text = tokenizer.decode(&gen_tokens, true)?;
        print!("{text}");
    } else {
        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();
        let mut history: Vec<u32> = tokens.clone();

        eprintln!(
            "Sampling: temp={:.2} top_k={} top_p={:.2} repeat_penalty={:.1}",
            cfg.temperature, cfg.top_k, cfg.top_p, cfg.repeat_penalty
        );

        for _step in 0..n_predict {
            let (_hidden, logits) = generate_token_logits(&mut state, last_token_id, &model);

            // Apply repetition penalty with prompt + generated so far
            let token = sample(&logits, &cfg, &history);

            gen_tokens.push(token);
            history.push(token);
            last_token_id = token;

            if Some(token) == eot_id {
                break;
            }
        }

        eprintln!("Generated {} tokens", gen_tokens.len());
        let text = tokenizer.decode(&gen_tokens, true)?;
        print!("{text}");
    }

    Ok(())
}
