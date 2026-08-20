//! End-to-end inference: load GGUF + tokenizer, encode prompt, generate tokens.

use std::env;
use std::path::PathBuf;

use qwen3_5_397b_in_rust::model::loader::ModelLoader;
use qwen3_5_397b_in_rust::model::pipeline::{generate_token, prefill, GenerationState, ModelWeights};
use qwen3_5_397b_in_rust::tokenizer::QwenTokenizer;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: run <model.gguf> <tokenizer.json> [prompt] [--n-predict N]");
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let tokenizer_path = PathBuf::from(&args[2]);

    let mut prompt = "Hello, world!".to_string();
    let mut n_predict: usize = 128;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--n-predict" => {
                i += 1;
                if i < args.len() {
                    n_predict = args[i].parse().unwrap_or(128);
                }
            }
            other => {
                prompt = other.to_string();
            }
        }
        i += 1;
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

    // Prefill: process all prompt tokens at once
    eprintln!("Prefilling {} tokens ...", tokens.len());
    prefill(&mut state, &tokens, &model);

    // Autoregressive generation with KV cache
    let eot_id: Option<u32> = tokenizer
        .token_to_id("‣")
        .or_else(|| tokenizer.token_to_id("‘"));

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

    eprintln!("Generated {} tokens", gen_tokens.len());

    let text = tokenizer.decode(&gen_tokens, true)?;
    print!("{text}");

    Ok(())
}
