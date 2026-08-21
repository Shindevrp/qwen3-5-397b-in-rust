//! HuggingFace Hub helpers: URL building, GGUF discovery, integrity hashing,
//! and resumable downloads.
//!
//! Deliberately dependency-light: HTTP via `ureq` (blocking, rustls), JSON
//! handled by targeted scanning of the well-defined Hub API shapes rather
//! than a full serde stack, and SHA-256 implemented here (FIPS 180-4) so
//! multi-gigabyte downloads can be verified against the LFS object id.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context};

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

const K256: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Streaming SHA-256.
pub struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    pub fn new() -> Self {
        Sha256 {
            state: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
                0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total: 0,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        self.absorb(data);
    }

    /// Buffer + compress without touching the message-length counter.
    fn absorb(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let (block, rest) = data.split_at(64);
            self.compress(block.try_into().expect("64-byte block"));
            data = rest;
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(chunk.try_into().expect("4-byte chunk"));
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K256[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        let round = [a, b, c, d, e, f, g, h];
        for (s, v) in self.state.iter_mut().zip(round) {
            *s = s.wrapping_add(v);
        }
    }

    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.total.wrapping_mul(8);
        self.absorb(&[0x80]);
        while self.buf_len != 56 {
            if self.buf_len == 64 {
                let block = self.buf;
                self.compress(&block);
                self.buf_len = 0;
            } else {
                self.absorb(&[0]);
            }
        }
        self.absorb(&bits.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0, "length must complete the final block");
        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }
}

/// One-shot digest as lowercase hex.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    to_hex(&h.finalize())
}

/// Stream a file through SHA-256, returning the lowercase hex digest.
pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(to_hex(&h.finalize()))
}

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ---------------------------------------------------------------------------
// Hub URLs and API responses
// ---------------------------------------------------------------------------

/// Validate a repo id ("org/name") or file path segment against traversal.
fn validate_segment(seg: &str, what: &str) -> anyhow::Result<()> {
    if seg.is_empty() || seg.starts_with('/') || seg.ends_with('/') || seg.contains("..") {
        bail!("invalid {what}: {seg:?}");
    }
    Ok(())
}

/// CDN-resolving download URL for a file on the main branch.
pub fn resolve_url(repo: &str, file: &str) -> anyhow::Result<String> {
    validate_segment(repo, "repo id")?;
    validate_segment(file, "file name")?;
    Ok(format!("https://huggingface.co/{repo}/resolve/main/{file}"))
}

/// Raw (pointer-text) URL — for LFS files this returns the pointer, not data.
pub fn raw_url(repo: &str, file: &str) -> anyhow::Result<String> {
    validate_segment(repo, "repo id")?;
    validate_segment(file, "file name")?;
    Ok(format!("https://huggingface.co/{repo}/raw/main/{file}"))
}

pub fn api_models_url(repo: &str) -> anyhow::Result<String> {
    validate_segment(repo, "repo id")?;
    Ok(format!("https://huggingface.co/api/models/{repo}"))
}

/// Extract every `"rfilename":"..."` value from a Hub API model response.
///
/// The Hub emits a flat siblings array; targeted scanning avoids pulling in a
/// JSON crate. Handles `\/` and `\"` escapes; filenames in practice are ASCII.
pub fn scan_rfilenames(api_json: &str) -> Vec<String> {
    const NEEDLE: &str = "\"rfilename\":\"";
    let bytes = api_json.as_bytes();
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = api_json[cursor..].find(NEEDLE) {
        let start = cursor + rel + NEEDLE.len();
        let mut j = start;
        while j < bytes.len() && bytes[j] != b'"' {
            j = if bytes[j] == b'\\' { j + 2 } else { j + 1 };
        }
        if j > bytes.len() {
            break; // truncated input
        }
        out.push(api_json[start..j].replace("\\/", "/").replace("\\\"", "\""));
        cursor = j + 1;
    }
    out
}

/// Parse `oid sha256:<hex>` out of an LFS pointer document.
pub fn parse_lfs_sha256(pointer_text: &str) -> Option<String> {
    pointer_text
        .lines()
        .find_map(|l| l.strip_prefix("oid sha256:"))
        .map(str::trim)
        .map(str::to_string)
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------

fn http_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(30))
        .build()
}

fn auth_header(token: Option<&str>) -> Option<String> {
    token.map(|t| format!("Bearer {t}"))
}

/// List the `.gguf` files published by a repo.
pub fn list_gguf_files(repo: &str, token: Option<&str>) -> anyhow::Result<Vec<String>> {
    let url = api_models_url(repo)?;
    let mut req = http_agent().get(&url);
    if let Some(h) = auth_header(token) {
        req = req.set("Authorization", &h);
    }
    let body = req.call().context("Hub API request failed")?.into_string()?;
    let mut files: Vec<String> = scan_rfilenames(&body)
        .into_iter()
        .filter(|f| f.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    files.sort();
    Ok(files)
}

/// Fetch the expected SHA-256 for a file from its LFS pointer, if it has one.
pub fn expected_sha256(
    agent: &ureq::Agent,
    repo: &str,
    file: &str,
    token: Option<&str>,
) -> anyhow::Result<Option<String>> {
    let url = raw_url(repo, file)?;
    let mut req = agent.get(&url);
    if let Some(h) = auth_header(token) {
        req = req.set("Authorization", &h);
    }
    let body = match req.call() {
        Ok(resp) => resp.into_string()?,
        Err(ureq::Error::Status(code, _)) if code == 404 || code == 401 => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    Ok(parse_lfs_sha256(&body))
}

/// Download `url` to `dest`, resuming from an existing `<dest>.part`.
///
/// Returns the number of bytes transferred **this call**. Verifies nothing;
/// callers combine with [`expected_sha256`] for integrity checking.
pub fn download_file(
    agent: &ureq::Agent,
    url: &str,
    token: Option<&str>,
    dest: &Path,
    mut progress: impl FnMut(u64, u64),
) -> anyhow::Result<u64> {
    const RANGE_REJECTED: &str = "server ignored Range request";
    let part_path = part_path_for(dest);

    // Issue a (possibly resuming) GET; returns total size, bytes already on
    // disk, and the response whose body streams the remainder.
    let issue =
        |resume_from: u64| -> anyhow::Result<(u64, u64, ureq::Response)> {
            let mut req = agent.get(url);
            if let Some(h) = auth_header(token) {
                req = req.set("Authorization", &h);
            }
            if resume_from > 0 {
                req = req.set("Range", &format!("bytes={resume_from}-"));
            }
            let resp = req.call().map_err(|e| anyhow!("{e}"))?;
            let status = resp.status();
            if resume_from > 0 && status != 206 {
                bail!("{RANGE_REJECTED} (status {status})");
            }
            let total = resp
                .header("Content-Length")
                .and_then(|v| v.parse::<u64>().ok())
                .map(|len| len + resume_from)
                .unwrap_or(0);
            Ok((total, if status == 206 { resume_from } else { 0 }, resp))
        };

    let existing = part_len(&part_path).unwrap_or(0);
    if let Some(parent) = part_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    let (total, mut done, resp) = match existing {
        0 => issue(0)?,
        n => match issue(n) {
            Ok(r) => r,
            // Stale/unsatisfiable partial: drop it and restart cleanly.
            Err(e) if e.to_string().contains(RANGE_REJECTED) => {
                let _ = std::fs::remove_file(&part_path);
                issue(0)?
            }
            Err(e) => return Err(e),
        },
    };
    let mut reader = resp.into_reader();

    let raw_file = if done > 0 {
        OpenOptions::new().append(true).open(&part_path)?
    } else {
        File::create(&part_path)?
    };
    let mut file = std::io::BufWriter::new(raw_file);

    let started = Instant::now();
    let mut last_report = Instant::now();
    let mut buf = vec![0u8; 1 << 17]; // 128 KiB chunks
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if last_report.elapsed() >= Duration::from_millis(400) {
            progress(done, total);
            last_report = Instant::now();
        }
    }

    if total != 0 && done != total {
        bail!(
            "incomplete download: got {done} of {total} bytes (partial kept at {})",
            part_path.display()
        );
    }
    file.flush()?;
    file.into_inner()?.sync_all()?;
    std::fs::rename(&part_path, dest)
        .with_context(|| format!("finalize {}", dest.display()))?;
    progress(done, total);

    let _ = started;
    Ok(done - existing)
}

fn part_path_for(dest: &Path) -> std::path::PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

fn part_len(part: &Path) -> Option<u64> {
    std::fs::metadata(part).ok().map(|m| m.len()).filter(|&n| n > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        // Multi-block (> 64 bytes) plus exact-block boundary cases.
        let long = vec![b'a'; 1_000_000];
        assert_eq!(
            sha256_hex(&long),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        let exact = vec![0x42u8; 64];
        assert_eq!(
            sha256_hex(&exact),
            "c422e7070cb1cb455b5de9afee0d975e303d0239c72030cd7414ab5c382d3ae8"
        );
    }

    #[test]
    fn sha256_chunked_matches_oneshot() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        for chunk in [1usize, 7, 63, 64, 65, 4096] {
            let mut h = Sha256::new();
            for piece in data.chunks(chunk) {
                h.update(piece);
            }
            assert_eq!(to_hex(&h.finalize()), sha256_hex(&data), "chunk {chunk}");
        }
    }

    #[test]
    fn scanner_finds_gguf_siblings() {
        let json = r#"{"id":"Qwen/Tiny","siblings":[
            {"rfilename":".gitattributes"},
            {"rfilename":"config.json"},
            {"rfilename":"model-Q4_K_M.gguf"},
            {"rfilename":"sub\/dir\/other.gguf"},
            {"rfilename":"weird\"name.gguf"}
        ]}"#;
        let names = scan_rfilenames(json);
        assert_eq!(
            names,
            vec![
                ".gitattributes".to_string(),
                "config.json".to_string(),
                "model-Q4_K_M.gguf".to_string(),
                "sub/dir/other.gguf".to_string(),
                "weird\"name.gguf".to_string(),
            ]
        );
    }

    #[test]
    fn lfs_pointer_parses() {
        let ptr = "version https://git-lfs.github.com/spec/v1\n\
                   oid sha256:4d7a214614ab2935c943f9e0be69a7d60989df2f\n\
                   size 12345\n";
        assert_eq!(
            parse_lfs_sha256(ptr).as_deref(),
            Some("4d7a214614ab2935c943f9e0be69a7d60989df2f")
        );
        assert_eq!(parse_lfs_sha256("not a pointer"), None);
    }

    #[test]
    fn urls_reject_traversal() {
        assert!(resolve_url("../etc/passwd", "x.gguf").is_err());
        assert!(resolve_url("org/repo", "../secret").is_err());
        assert!(resolve_url("", "x.gguf").is_err());
        assert_eq!(
            resolve_url("Qwen/Qwen3.5", "m.gguf").unwrap(),
            "https://huggingface.co/Qwen/Qwen3.5/resolve/main/m.gguf"
        );
    }
}
