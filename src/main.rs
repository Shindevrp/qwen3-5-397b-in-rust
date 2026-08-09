mod gguf;

use gguf::{Gguf, Value};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: qwen3-5-397b-in-rust <model.gguf> [--metadata] [--tensor <name>]");
        std::process::exit(2);
    }
    let path = std::path::Path::new(&args[1]);
    let dump_metadata = args.iter().any(|a| a == "--metadata");
    let lookup = args
        .iter()
        .position(|a| a == "--tensor")
        .map(|i| args[i + 1].clone());

    let gguf = Gguf::open(path)?;

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
