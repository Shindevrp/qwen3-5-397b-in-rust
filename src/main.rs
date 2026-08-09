mod gguf;

use anyhow::Result;

fn main() -> Result<()> {
    // Minimal demo that uses the gguf module
    let v = gguf::value::Value::Int(42);
    println!("created value: {}", v);

    let header = gguf::types::Header { version: 1 };
    println!("gguf header version: {}", header.version);

    Ok(())
}
