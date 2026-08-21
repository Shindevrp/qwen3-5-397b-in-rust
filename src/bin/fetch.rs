//! Fetch a Qwen3.5 GGUF (plus tokenizer.json) from the HuggingFace Hub.
//!
//! Usage:
//!   fetch <repo_id> [--file NAME] [--out DIR] [--token TOKEN] [--no-tokenizer]
//!   fetch <repo_id> --list
//!
//! - Downloads resume automatically from `<dest>.part` on re-run.
//! - LFS SHA-256 is verified when the repo publishes a pointer for the file.
//! - `HF_TOKEN` is honored when `--token` is absent.

use std::path::PathBuf;
use std::time::Instant;

use qwen3_5_397b_in_rust::hf::{
    download_file, expected_sha256, list_gguf_files, resolve_url, sha256_file,
};

fn usage() {
    eprintln!(
        "Usage: fetch <repo_id> [options]\n\
         \n\
         Arguments:\n\
         \x20 <repo_id>       HuggingFace repo, e.g. Qwen/Qwen3.5-35B-A3B\n\
         \n\
         Options:\n\
         \x20 --file NAME     GGUF file inside the repo (default: auto-detect)\n\
         \x20 --out DIR       Output directory (default: models/<repo name>)\n\
         \x20 --token TOKEN   HuggingFace token (falls back to $HF_TOKEN)\n\
         \x20 --list          List .gguf files in the repo and exit\n\
         \x20 --no-tokenizer  Skip fetching tokenizer.json\n\
         \x20 -h, --help      This help"
    );
}

fn human_bytes(n: u64) -> String {
    let mut v = n as f64;
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"] {
        if v < 1024.0 {
            return format!("{v:.1} {unit}");
        }
        v /= 1024.0;
    }
    format!("{v:.1} PiB")
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") || args.is_empty() {
        usage();
        if args.is_empty() {
            std::process::exit(2);
        }
        return Ok(());
    }

    let repo = args[0].clone();
    let mut file: Option<String> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut token = std::env::var("HF_TOKEN").ok();
    let mut list_only = false;
    let mut want_tokenizer = true;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--file" => {
                i += 1;
                file = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--file needs a value"))?
                        .clone(),
                );
            }
            "--out" => {
                i += 1;
                out_dir = Some(PathBuf::from(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--out needs a value"))?,
                ));
            }
            "--token" => {
                i += 1;
                token = Some(
                    args.get(i)
                        .ok_or_else(|| anyhow::anyhow!("--token needs a value"))?
                        .clone(),
                );
            }
            "--list" => list_only = true,
            "--no-tokenizer" => want_tokenizer = false,
            other => anyhow::bail!("unknown option {other:?}"),
        }
        i += 1;
    }

    // Discover which GGUF to grab.
    let gguf = match file {
        Some(f) => f,
        None => {
            eprintln!("listing GGUF files in {repo} ...");
            let files = list_gguf_files(&repo, token.as_deref())?;
            if files.is_empty() {
                anyhow::bail!("{repo} publishes no .gguf files; pass --file manually");
            }
            if files.len() > 1 && !list_only {
                eprintln!("{repo} has {} GGUF files:", files.len());
                for f in &files {
                    eprintln!("  {f}");
                }
                anyhow::bail!("pick one with --file <name>");
            }
            match files.first() {
                Some(f) => f.clone(),
                None => anyhow::bail!("no GGUF found"),
            }
        }
    };

    if list_only {
        for f in list_gguf_files(&repo, token.as_deref())? {
            println!("{f}");
        }
        return Ok(());
    }

    let default_out = PathBuf::from("models").join(repo.rsplit('/').next().unwrap_or(&repo));
    let out_dir = out_dir.unwrap_or(default_out);
    std::fs::create_dir_all(&out_dir)?;

    // --- model weights ---
    let dest = out_dir.join(&gguf);
    let url = resolve_url(&repo, &gguf)?;
    let agent = ureq::AgentBuilder::new().build();
    let t0 = Instant::now();
    eprintln!("downloading {repo}/{gguf}");
    let transferred = download_file(&agent, &url, token.as_deref(), &dest, |done, total| {
        if total > 0 {
            let pct = done as f64 / total as f64 * 100.0;
            let secs = t0.elapsed().as_secs_f64().max(1e-6);
            let mibs = done as f64 / 1024.0 / 1024.0 / secs;
            eprint!("\r  {}/{} ({pct:5.1}%) {:6.1} MiB/s ", human_bytes(done), human_bytes(total), mibs);
        } else {
            eprint!("\r  {}", human_bytes(done));
        }
        let _ = std::io::Write::flush(&mut std::io::stderr());
    })?;

    // --- integrity ---
    match expected_sha256(&agent, &repo, &gguf, token.as_deref())? {
        Some(expected) => {
            eprintln!("\nverifying SHA-256 ...");
            let actual = sha256_file(&dest)?;
            if actual != expected {
                let _ = std::fs::remove_file(&dest);
                anyhow::bail!("SHA-256 mismatch: expected {expected}, got {actual} (file removed)");
            }
            eprintln!("  ok ({actual})");
        }
        None => eprintln!("\nno LFS pointer published; skipping hash verification"),
    }

    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "saved {} in {:.1}s ({} this run)",
        dest.display(),
        secs,
        human_bytes(transferred)
    );

    // --- tokenizer ---
    if want_tokenizer {
        let tok_dest = out_dir.join("tokenizer.json");
        if tok_dest.exists() {
            eprintln!("tokenizer.json already present at {}", tok_dest.display());
        } else {
            eprintln!("downloading tokenizer.json");
            let tok_url = resolve_url(&repo, "tokenizer.json")?;
            let res = download_file(&agent, &tok_url, token.as_deref(), &tok_dest, |done, _total| {
                eprint!("\r  {}", human_bytes(done));
                let _ = std::io::Write::flush(&mut std::io::stderr());
            });
            match res {
                Ok(_) => eprintln!(),
                Err(e) => eprintln!(
                    "\nwarning: could not fetch tokenizer.json ({e}); grab it manually if needed"
                ),
            }
        }
    }

    eprintln!("\ndone. try:\n  cargo run --release --bin run -- {} {} --chat", dest.display(), out_dir.join("tokenizer.json").display());
    Ok(())
}
