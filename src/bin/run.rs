//! End-to-end inference: load GGUF + tokenizer, encode prompt, generate tokens.

use std::env;
use std::path::PathBuf;

use qwen3_5_397b_in_rust::model::loader::ModelLoader;
use qwen3_5_397b_in_rust::model::pipeline::{forward_pass, ModelWeights};
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

    let mut all_tokens = tokens.clone();
    let eot_id: Option<u32> = tokenizer
        .token_to_id("<|im_end|>")
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"));

    for _step in 0..n_predict {
        let token_id = *all_tokens.last().unwrap();
        let (_hidden, next_token) = forward_pass(token_id, &model);

        all_tokens.push(next_token);

        if Some(next_token) == eot_id {
            break;
        }
    }

    let gen_len = all_tokens.len() - tokens.len();
    eprintln!("Generated {gen_len} tokens");

    let new_tokens = &all_tokens[tokens.len()..];
    let text = tokenizer.decode(new_tokens, true)?;
    print!("{text}");

    Ok(())
}
