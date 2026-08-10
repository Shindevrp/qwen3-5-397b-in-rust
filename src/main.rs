mod gguf;
mod model;

use gguf::{Gguf, Value};
use model::config::{Qwen3_5Config, validate_tensors};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "usage: qwen3-5-397b-in-rust <model.gguf> [--metadata] [--tensor <name>] [--config]"
        );
        std::process::exit(2);
    }
    let path = std::path::Path::new(&args[1]);
    let dump_metadata = args.iter().any(|a| a == "--metadata");
    let dump_config = args.iter().any(|a| a == "--config");
    let lookup = args
        .iter()
        .position(|a| a == "--tensor")
        .map(|i| args[i + 1].clone());

    let gguf = Gguf::open(path)?;

    if dump_config {
        let cfg = Qwen3_5Config::from_metadata(&gguf.metadata)?;
        println!("Qwen3.5-MoE Config:");
        println!("  block_count:                      {}", cfg.block_count);
        println!(
            "  embedding_length:                 {}",
            cfg.embedding_length
        );
        println!(
            "  attention_head_count:               {}",
            cfg.attention_head_count
        );
        println!(
            "  attention_head_count_kv:            {}",
            cfg.attention_head_count_kv
        );
        println!(
            "  attention_key_length:               {}",
            cfg.attention_key_length
        );
        println!(
            "  attention_value_length:             {}",
            cfg.attention_value_length
        );
        println!(
            "  attention_layer_norm_rms_epsilon:   {}",
            cfg.attention_layer_norm_rms_epsilon
        );
        println!("  expert_count:                       {}", cfg.expert_count);
        println!(
            "  expert_used_count:                  {}",
            cfg.expert_used_count
        );
        println!(
            "  expert_feed_forward_length:         {}",
            cfg.expert_feed_forward_length
        );
        println!(
            "  expert_shared_feed_forward_length:  {}",
            cfg.expert_shared_feed_forward_length
        );
        println!(
            "  rope_dimension_count:               {}",
            cfg.rope_dimension_count
        );
        println!(
            "  rope_freq_base:                     {}",
            cfg.rope_freq_base
        );
        println!(
            "  context_length:                     {}",
            cfg.context_length
        );
        println!(
            "  ssm_state_size:                     {}",
            cfg.ssm_state_size
        );
        println!(
            "  ssm_group_count:                    {}",
            cfg.ssm_group_count
        );
        println!(
            "  ssm_time_step_rank:                 {}",
            cfg.ssm_time_step_rank
        );
        println!(
            "  ssm_conv_kernel:                    {}",
            cfg.ssm_conv_kernel
        );
        println!(
            "  ssm_inner_size:                     {:?}",
            cfg.ssm_inner_size
        );
        println!();
        println!("Derived dims:");
        println!("  key_dim (state * group):            {}", cfg.key_dim);
        println!("  value_dim (state * time_step):      {}", cfg.value_dim);
        println!("  conv_dim (2*key + value):           {}", cfg.conv_dim);
        println!("  ba_dim (2 * time_step):             {}", cfg.ba_dim);
        println!(
            "  full_attn_q_fused_dim:              {}",
            cfg.full_attn_q_fused_dim
        );
        println!();

        validate_tensors(&gguf.metadata, &gguf.tensors)?;
        println!("Tensor schema validation: OK");
    }

    if !dump_config {
        println!("file: {}", path.display());
        println!("size: {} bytes", gguf.len());
        println!("gguf version: {}", gguf.header.version);
        println!("alignment: {}", gguf.alignment);
        println!("metadata entries: {}", gguf.metadata.len());
        println!("tensor count: {}", gguf.header.tensor_count);
        println!("data offset: {}", gguf.data_offset);

        if dump_metadata {
            println!();
            println!("metadata:");
            for (key, value) in gguf.metadata.iter() {
                println!("  {key}: {}", format_value(value));
            }
        }

        println!();
        println!("tensors:");
        for t in &gguf.tensors {
            let dims = t
                .dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("x");
            let quant = if t.is_quantized() { " q" } else { "  " };
            println!(
                "{quant} {:<62} {} [{}] elems={} @ {}",
                t.name,
                t.ggml_type.name(),
                dims,
                t.n_elements(),
                t.offset
            );
        }

        if let Some(name) = lookup {
            println!();
            match gguf.tensor(&name) {
                Some(t) => {
                    let data = gguf.data_slice(t);
                    println!(
                        "tensor \"{name}\": type={} dims={:?} elems={} offset={} data_bytes_available={}",
                        t.ggml_type.name(),
                        t.dims,
                        t.n_elements(),
                        t.offset,
                        data.len()
                    );
                }
                None => {
                    eprintln!("tensor \"{name}\" not found");
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}

fn format_value(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{s:?}"),
        Value::Array { items, .. } => {
            let parts: Vec<String> = items.iter().map(format_value).collect();
            format!("[{}]", parts.join(", "))
        }
        other => format!("{other:?}"),
    }
}
