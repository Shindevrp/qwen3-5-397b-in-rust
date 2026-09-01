//! End-to-end inference: load GGUF + tokenizer, encode prompt, generate tokens with sampling.
//!
//! Three modes:
//! - raw completion (default): encode the prompt as-is
//! - interactive chat (`--chat`): Qwen3.5 ChatML template with multi-turn history
//! - batch (`--batch FILE`): one prompt per line, sequences run in parallel

use std::env;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use qwen3_5_397b_in_rust::chat::{ChatRenderOptions, Message, Role, render_chat};
use qwen3_5_397b_in_rust::model::loader::ModelLoader;
use qwen3_5_397b_in_rust::model::memory::MemoryStats;
use qwen3_5_397b_in_rust::model::pipeline::{
    GenerationState, ModelWeights, generate_token, generate_token_batch, generate_token_logits,
    generate_token_logits_batch, prefill, prefill_batch,
};
use qwen3_5_397b_in_rust::model::sampler::{SamplerConfig, sample};
use qwen3_5_397b_in_rust::tokenizer::QwenTokenizer;
use qwen3_5_397b_in_rust::tokenizer::StreamingDecoder;

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
        eprintln!("  --kv-q8            Store KV cache as Q8_0 (~3.8x less memory)");
        eprintln!("  --memory-bounded   Enable memory-bounded inference mode");
        eprintln!("Chat mode:");
        eprintln!("  --chat             Interactive multi-turn chat (Qwen3.5 template)");
        eprintln!("  --system TEXT      System prompt (chat mode)");
        eprintln!("  --no-think         Disable thinking mode (chat mode)");
        eprintln!("Batch mode:");
        eprintln!("  --batch FILE       One prompt per line; sequences run in parallel");
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
    let mut batch_file: Option<PathBuf> = None;
    let mut kv_q8 = false;
    let mut memory_bounded = false;
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
            "--kv-q8" => {
                kv_q8 = true;
            }
            "--memory-bounded" => {
                memory_bounded = true;
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
            "--batch" => {
                i += 1;
                if i < args.len() {
                    batch_file = Some(PathBuf::from(&args[i]));
                }
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
    MemoryStats::log("after-model-load");

    // End-of-turn ids: <|im_end|> ends an assistant turn in chat mode;
    // <|endoftext|> is the generic EOS fallback.
    let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();

    if let Some(path) = batch_file {
        let content = std::fs::read_to_string(&path)
            .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
        let prompts: Vec<String> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect();
        if prompts.is_empty() {
            anyhow::bail!("batch file {} has no prompts", path.display());
        }
        return run_batch(
            &model,
            &tokenizer,
            &prompts,
            &cfg,
            n_predict,
            use_argmax,
            kv_q8,
            memory_bounded,
            &stop_ids,
        );
    }

    if chat_mode {
        chat_loop(
            &model,
            &tokenizer,
            &cfg,
            n_predict,
            use_argmax,
            kv_q8,
            memory_bounded,
            system_prompt,
            enable_thinking,
            &stop_ids,
        )
    } else {
        if prompt.is_empty() {
            prompt = "Hello, world!".to_string();
        }
        run_completion(
            &model,
            &tokenizer,
            &prompt,
            &cfg,
            n_predict,
            use_argmax,
            kv_q8,
            memory_bounded,
            &stop_ids,
        )
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
        let system_count = if !msgs.is_empty() && msgs[0].role == Role::System {
            1
        } else {
            0
        };
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
    kv_q8: bool,
    memory_bounded: bool,
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

        let opts = ChatRenderOptions {
            add_generation_prompt: true,
            enable_thinking,
        };
        let (history_fit, tokens) =
            encode_with_truncation(&history, &opts, tokenizer, max_ctx, n_predict)?;
        history = history_fit;

        eprintln!("[prompt: {} tokens]", tokens.len());

        // Fresh state each turn: re-prefill the whole conversation.
        let mut state = if memory_bounded {
            GenerationState::new_memory_bounded_kv_q8(model)
        } else if kv_q8 {
            GenerationState::new_kv_q8(model)
        } else {
            GenerationState::new(model)
        };
        prefill(&mut state, &tokens, model).map_err(|e| anyhow::anyhow!(e))?;
        // Phase 24: dequantize MoE experts up front (parallel) so decode
        // never pays first-touch dequantization cost.
        if !memory_bounded {
            state.warm_moe_cache(model);
        }

        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();
        let mut sample_history = tokens.clone();
        let mut stream = StreamingDecoder::new(tokenizer);
        let mut stdout = std::io::stdout();

        for _step in 0..n_predict {
            let token = if use_argmax {
                let (_hidden, next) = generate_token(&mut state, last_token_id, model)
                    .map_err(|e| anyhow::anyhow!(e))?;
                next
            } else {
                let (_hidden, logits) = generate_token_logits(&mut state, last_token_id, model)
                    .map_err(|e| anyhow::anyhow!(e))?;
                sample(&logits, cfg, &sample_history)
            };

            gen_tokens.push(token);
            sample_history.push(token);
            last_token_id = token;

            // Stream the response as it is generated.
            {
                let chunk = stream.push(token)?;
                print!("{chunk}");
                stdout.flush()?;
            }

            if stop_ids.contains(&token) {
                break;
            }
        }
        println!();

        // Full text for history (stream may have withheld a partial char).
        let text = tokenizer.decode(&gen_tokens, true)?;
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
    kv_q8: bool,
    memory_bounded: bool,
    stop_ids: &[u32],
) -> anyhow::Result<()> {
    eprintln!("Prompt: {prompt}");
    let tokens = tokenizer.encode(prompt, false)?;
    eprintln!(
        "Input tokens ({}): {:?}",
        tokens.len(),
        &tokens[..tokens.len().min(20)]
    );

    let mut state = if memory_bounded {
        GenerationState::new_memory_bounded_kv_q8(model)
    } else if kv_q8 {
        GenerationState::new_kv_q8(model)
    } else {
        GenerationState::new(model)
    };

    eprintln!("Prefilling {} tokens ...", tokens.len());
    prefill(&mut state, &tokens, model).map_err(|e| anyhow::anyhow!(e))?;
    MemoryStats::log("after-prefill");
    // Phase 24: dequantize MoE experts up front (parallel) so decode
    // never pays first-touch dequantization cost.
    if !memory_bounded {
        state.warm_moe_cache(model);
    }
    MemoryStats::log("after-warmup");

    if use_argmax {
        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();
        let mut stream = StreamingDecoder::new(tokenizer);
        let mut stdout = std::io::stdout();
        let mut first = true;

        for _step in 0..n_predict {
            let (_hidden, next_token) =
                generate_token(&mut state, last_token_id, model).map_err(|e| anyhow::anyhow!(e))?;
            if first {
                MemoryStats::log("after-first-decode");
                first = false;
            }
            gen_tokens.push(next_token);
            last_token_id = next_token;

            let chunk = stream.push(next_token)?;
            print!("{chunk}");
            stdout.flush()?;

            if stop_ids.contains(&next_token) {
                break;
            }
        }
        println!();
    } else {
        let mut last_token_id = *tokens.last().unwrap();
        let mut gen_tokens = Vec::new();
        let mut history: Vec<u32> = tokens.clone();
        let mut stream = StreamingDecoder::new(tokenizer);
        let mut stdout = std::io::stdout();

        eprintln!(
            "Sampling: temp={:.2} top_k={} top_p={:.2} repeat_penalty={:.1}",
            cfg.temperature, cfg.top_k, cfg.top_p, cfg.repeat_penalty
        );

        let mut first = true;
        for _step in 0..n_predict {
            let (_hidden, logits) = generate_token_logits(&mut state, last_token_id, model)
                .map_err(|e| anyhow::anyhow!(e))?;
            if first {
                MemoryStats::log("after-first-decode");
                first = false;
            }

            // Apply repetition penalty with prompt + generated so far
            let token = sample(&logits, cfg, &history);

            gen_tokens.push(token);
            history.push(token);
            last_token_id = token;

            let chunk = stream.push(token)?;
            print!("{chunk}");
            stdout.flush()?;

            if stop_ids.contains(&token) {
                break;
            }
        }
        println!();
    }

    Ok(())
}

/// Phase 15: batch inference — prefill and decode multiple independent
/// sequences in parallel (rayon over per-sequence `GenerationState`s).
///
/// Sequences that hit a stop token leave the batch immediately (lockstep
/// steps shrink as sequences finish). Output is printed once per sequence.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    model: &ModelWeights,
    tokenizer: &QwenTokenizer,
    prompts: &[String],
    cfg: &SamplerConfig,
    n_predict: usize,
    use_argmax: bool,
    kv_q8: bool,
    memory_bounded: bool,
    stop_ids: &[u32],
) -> anyhow::Result<()> {
    let max_ctx = model.cfg.context_length as usize;

    // Encode all prompts up-front; skip ones that cannot fit.
    let mut token_prompts: Vec<Vec<u32>> = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let tokens = tokenizer.encode(p, false)?;
        if tokens.len() + n_predict > max_ctx {
            eprintln!(
                "[seq {i}] skipped: {} tokens + {n_predict} > context {max_ctx}",
                tokens.len()
            );
            continue;
        }
        token_prompts.push(tokens);
    }
    if token_prompts.is_empty() {
        anyhow::bail!("no prompts fit the context window");
    }
    let n_total = token_prompts.len();

    // One state per sequence; prefill all of them in parallel.
    let t0 = Instant::now();
    let mut states: Vec<GenerationState> = (0..n_total)
        .map(|_| {
            if memory_bounded {
                GenerationState::new_memory_bounded_kv_q8(model)
            } else if kv_q8 {
                GenerationState::new_kv_q8(model)
            } else {
                GenerationState::new(model)
            }
        })
        .collect();
    let prompt_refs: Vec<&[u32]> = token_prompts.iter().map(|v| v.as_slice()).collect();
    prefill_batch(&mut states, &prompt_refs, model).map_err(|e| anyhow::anyhow!(e))?;
    // Phase 24: parallel warm-up of every sequence's MoE weight cache.
    if !memory_bounded {
        use rayon::prelude::*;
        states.par_iter_mut().for_each(|s| s.warm_moe_cache(model));
    }
    let prefill_ms = t0.elapsed().as_millis();
    eprintln!("[batch] prefilled {n_total} sequences in {prefill_ms} ms");

    // Parallel slot arrays; `order[j]` is the original sequence index of slot j.
    let mut order: Vec<usize> = (0..n_total).collect();
    let mut last_tokens: Vec<u32> = token_prompts.iter().map(|v| *v.last().unwrap()).collect();
    let mut histories: Vec<Vec<u32>> = token_prompts.clone();
    let mut generated: Vec<Vec<u32>> = vec![Vec::new(); n_total];

    let t1 = Instant::now();
    let mut total_tokens = 0usize;
    for _step in 0..n_predict {
        if states.is_empty() {
            break;
        }

        // One lockstep decode step across every active sequence.
        let step_tokens: Vec<u32> = if use_argmax {
            generate_token_batch(&mut states, &last_tokens, model)
                .map_err(|e| anyhow::anyhow!(e))?
        } else {
            let logits = generate_token_logits_batch(&mut states, &last_tokens, model)
                .map_err(|e| anyhow::anyhow!(e))?;
            logits
                .iter()
                .zip(&histories)
                .map(|(lg, h)| sample(lg, cfg, h))
                .collect()
        };

        // Record outputs; retire finished sequences via swap_remove so the
        // next step only pays for the survivors. A swapped-in element is
        // re-examined because `j` does not advance on removal.
        let mut j = 0;
        while j < states.len() {
            let tok = step_tokens[j];
            histories[j].push(tok);
            generated[order[j]].push(tok);
            total_tokens += 1;

            if stop_ids.contains(&tok) {
                states.swap_remove(j);
                last_tokens.swap_remove(j);
                histories.swap_remove(j);
                order.swap_remove(j);
            } else {
                last_tokens[j] = tok;
                j += 1;
            }
        }
    }
    let decode_ms = t1.elapsed().as_millis();
    let secs = decode_ms as f64 / 1000.0;

    println!();
    for (i, toks) in generated.iter().enumerate() {
        let text = tokenizer.decode(toks, true)?;
        println!("[seq {i}] {}", text.trim());
    }
    eprintln!(
        "[batch] {} sequences, {} tokens decoded in {} ms ({:.1} tok/s aggregate)",
        n_total,
        total_tokens,
        decode_ms,
        if secs > 0.0 {
            total_tokens as f64 / secs
        } else {
            f64::INFINITY
        }
    );

    Ok(())
}
