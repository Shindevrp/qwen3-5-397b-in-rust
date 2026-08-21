//! End-to-end inference: load GGUF + tokenizer, encode prompt, generate tokens with sampling.
//!
//! Two modes:
//! - raw completion (default): encode the prompt as-is
//! - interactive chat (`--chat`): Qwen3.5 ChatML template with multi-turn history

use std::env;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use qwen3_5_397b_in_rust::chat::{render_chat, ChatRenderOptions, Message, Role};
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
        eprintln!("Chat mode:");
        eprintln!("  --chat             Interactive multi-turn chat (Qwen3.5 template)");
        eprintln!("  --system TEXT      System prompt (chat mode)");
        eprintln!("  --no-think         Disable thinking mode (chat mode)");
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let tokenizer_path = PathBuf::from(&args[2]);

    let mut prompt = String::new();
    let mut n_predict: usize = 128;
    let mut cfg = SamplerConfig::default();
    let mut use_argmax = false;
    let mut chat_mode = false;
    let mut system_prompt: Option<String> = None;
    let mut enable_thinking = true;
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
            "--chat" => {
                chat_mode = true;
            }
            "--system" => {
                i += 1;
                if i < args.len() {
                    system_prompt = Some(args[i].clone());
                }
            }
            "--no-think" => {
                enable_thinking = false;
            }
            other if prompt.is_empty() && !chat_mode => {
                prompt = other.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    eprintln!("Loading model from {} ...", model_path.display());
    let loader = ModelLoader::open(&model_path)?;

    eprintln!("Loading tokenizer from {} ...", tokenizer_path.display());
    let tokenizer = QwenTokenizer::from_file(&tokenizer_path)?;

    eprintln!("Loading weights ...");
    let model = ModelWeights::load(&loader).map_err(|e| anyhow::anyhow!("{e}"))?;

    // End-of-turn ids: <|im_end|> ends an assistant turn in chat mode;
    // <|endoftext|> is the generic EOS fallback.
    let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    if chat_mode {
        chat_loop(
            &model,
            &tokenizer,
            &cfg,
            n_predict,
            use_argmax,
            system_prompt,
            enable_thinking,
            &stop_ids,
        )
    } else {
        if prompt.is_empty() {
            prompt = "Hello, world!".to_string();
        }
        run_completion(&model, &tokenizer, &prompt, &cfg, n_predict, use_argmax, &stop_ids)
    }
}

/// Encode a rendered prompt, dropping the oldest non-system messages until it
/// fits in the context window with room left for generation.
fn encode_with_truncation(
    history: &[Message],
    opts: &ChatRenderOptions,
    tokenizer: &QwenTokenizer,
    max_ctx: usize,
    reserve: usize,
) -> anyhow::Result<(Vec<Message>, Vec<u32>)> {
    let mut msgs: Vec<Message> = history.to_vec();
    loop {
        let rendered = render_chat(&msgs, opts).map_err(|e| anyhow::anyhow!("{e}"))?;
        let tokens = tokenizer.encode(&rendered, false)?;
        // Keep at least the (optional) system message + last user message.
        let system_count =
            if !msgs.is_empty() && msgs[0].role == Role::System { 1 } else { 0 };
        let droppable = msgs.len() - system_count - 1;
        if tokens.len() + reserve <= max_ctx || droppable == 0 {
            return Ok((msgs, tokens));
        }
        // Drop the oldest non-system message.
        msgs.remove(system_count);
        eprintln!(
            "(context full: dropped oldest message, prompt now ~{} tokens)",
            tokens.len()
        );
    }
}

/// Interactive multi-turn chat using the Qwen3.5 template.
#[allow(clippy::too_many_arguments)]
fn chat_loop(
    model: &ModelWeights,
    tokenizer: &QwenTokenizer,
    cfg: &SamplerConfig,
    n_predict: usize,
    use_argmax: bool,
    system_prompt: Option<String>,
    enable_thinking: bool,
    stop_ids: &[u32],
) -> anyhow::Result<()> {
    let max_ctx = model.cfg.context_length as usize;

    let mut history: Vec<Message> = Vec::new();
    if let Some(s) = &system_prompt {
        history.push(Message::system(s.clone()));
    }

    eprintln!(
        "Interactive chat ({}thinking mode). Commands: /reset /quit",
        if enable_thinking { "" } else { "no-" }
    );

    let stdin = std::io::stdin();
    loop {
        eprint!("\nuser> ");
        std::io::stderr().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/quit" | "/exit" => break,
            "/reset" => {
                history.clear();
                if let Some(s) = &system_prompt {
                    history.push(Message::system(s.clone()));
                }
                eprintln!("(conversation cleared)");
                continue;
            }
            _ => {}
        }

        history.push(Message::user(input.to_string()));

        let opts = ChatRenderOptions { add_generation_prompt: true, enable_thinking };
        let (history_fit, tokens) =
            encode_with_truncation(&history, &opts, tokenizer, max_ctx, n_predict)?;
        history = history_fit;

        eprintln!("[prompt: {} tokens]", tokens.len());

        // Fresh state each turn: re-prefill the whole conversation.
        let mut state = GenerationState::new(model);
        prefill(&mut state, &tokens, model).map_err(|e| anyhow::anyhow!(e))?;

        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();
        let mut sample_history = tokens.clone();

        for _step in 0..n_predict {
            let token = if use_argmax {
                let (_hidden, next) = generate_token(&mut state, last_token_id, model).map_err(|e| anyhow::anyhow!(e))?;
                next
            } else {
                let (_hidden, logits) = generate_token_logits(&mut state, last_token_id, model).map_err(|e| anyhow::anyhow!(e))?;
                sample(&logits, cfg, &sample_history)
            };

            gen_tokens.push(token);
            sample_history.push(token);
            last_token_id = token;

            if stop_ids.contains(&token) {
                break;
            }
        }

        // skip_special_tokens strips <|im_end|> but keeps literal <think> text.
        let text = tokenizer.decode(&gen_tokens, true)?;
        println!("{text}");
        history.push(Message::assistant(text));
    }

    Ok(())
}

/// Single-shot raw completion (pre-chat behavior).
#[allow(clippy::too_many_arguments)]
fn run_completion(
    model: &ModelWeights,
    tokenizer: &QwenTokenizer,
    prompt: &str,
    cfg: &SamplerConfig,
    n_predict: usize,
    use_argmax: bool,
    stop_ids: &[u32],
) -> anyhow::Result<()> {
    eprintln!("Prompt: {prompt}");
    let tokens = tokenizer.encode(prompt, false)?;
    eprintln!("Input tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(20)]);

    let mut state = GenerationState::new(model);

    eprintln!("Prefilling {} tokens ...", tokens.len());
    prefill(&mut state, &tokens, model).map_err(|e| anyhow::anyhow!(e))?;

    if use_argmax {
        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();

        for _step in 0..n_predict {
            let (_hidden, next_token) = generate_token(&mut state, last_token_id, model).map_err(|e| anyhow::anyhow!(e))?;
            gen_tokens.push(next_token);
            last_token_id = next_token;

            if stop_ids.contains(&next_token) {
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
            let (_hidden, logits) = generate_token_logits(&mut state, last_token_id, model).map_err(|e| anyhow::anyhow!(e))?;

            // Apply repetition penalty with prompt + generated so far
            let token = sample(&logits, cfg, &history);

            gen_tokens.push(token);
            history.push(token);
            last_token_id = token;

            if stop_ids.contains(&token) {
                break;
            }
        }

        eprintln!("Generated {} tokens", gen_tokens.len());
        let text = tokenizer.decode(&gen_tokens, true)?;
        print!("{text}");
    }

    Ok(())
}
