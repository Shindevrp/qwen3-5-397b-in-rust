//! Shard-aware tensor loader for split GGUF models.
//!
//! Multi-file GGUFs produced by gguf-py / llama.cpp conversion tooling follow
//! this layout (confirmed against `gguf/gguf_writer.py`):
//!
//!   * Each shard is a complete, self-contained GGUF. Tensor offsets stored in
//!     a shard's tensor-info section are RELATIVE to that shard's own data
//!     section (each file restarts `offset_tensor` at 0), so reading tensor
//!     `T` means finding the one shard that lists it and slicing its data at
//!     `data_offset + T.offset`.
//!   * Tensors never span shards; splitting is done at whole-tensor
//!     granularity.
//!   * Only the first shard carries the model hyperparameters; later shards
//!     carry just the `split.*` keys. Tensor data may start in shard 2
//!     (the `small_first_shard` layout).
//!
//! The config is therefore always taken from whichever shard exposes
//! `general.architecture` (i.e. `split.no == 0`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::gguf::{Gguf, GgufError, TensorMeta};
use crate::model::config::Qwen3_5Config;
use crate::model::quant;

/// One open GGUF shard file.
pub struct Shard {
    pub path: PathBuf,
    pub split_no: u16,
    pub gguf: Gguf,
}

/// All shards of a split model plus a merged, shard-aware tensor index.
pub struct ModelLoader {
    pub cfg: Qwen3_5Config,
    pub shards: Vec<Shard>,
    by_name: HashMap<String, (usize, TensorMeta)>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("failed to open GGUF shard {path}: {source}")]
    ShardOpen { path: PathBuf, source: GgufError },
    #[error("no shard declares general.architecture; cannot build model config")]
    NoConfigShard,
    #[error(
        "cannot discover split.* shards from {0}: \
         expected a name of the form <stem>-NNNNN-of-NNNNN.gguf, got no matching suffix"
    )]
    ShardDiscovery(PathBuf),
    #[error("tensor \"{name}\" not found in any shard")]
    TensorNotFound { name: String },
    #[error("tensor \"{name}\": {source}")]
    Quant {
        name: String,
        #[source]
        source: quant::QuantError,
    },
    #[error(
        "tensor \"{name}\" data out of bounds: needs {expected} bytes at offset {off}, \
         but shard {shard} only has {len} bytes available"
    )]
    OutOfBounds { name: String, expected: u64, off: u64, shard: usize, len: usize },
}

impl ModelLoader {
    /// Open the shard pointed to by `first_path` and any siblings found via
    /// the `-NNNNN-of-NNNNN.gguf` naming convention (or `split.count`).
    pub fn open<P: AsRef<Path>>(first_path: P) -> Result<Self, LoaderError> {
        let first_path = first_path.as_ref();

        let probe = Gguf::open(first_path)
            .map_err(|source| LoaderError::ShardOpen { path: first_path.to_path_buf(), source })?;

        // How many shards does this model claim to have?
        let split_count = probe
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u32().ok())
            .unwrap_or(1) as usize;

        let mut paths: Vec<PathBuf> = Vec::new();
        if split_count > 1 {
            let stem = first_path
                .file_stem()
                .and_then(|s| s.to_str())
                .ok_or_else(|| LoaderError::ShardDiscovery(first_path.to_path_buf()))?;
            let (idx, n) = split_suffix(stem)
                .ok_or_else(|| LoaderError::ShardDiscovery(first_path.to_path_buf()))?;
            if n != split_count {
                return Err(LoaderError::ShardDiscovery(first_path.to_path_buf()));
            }
            let suffix = format!("-{idx:05}-of-{n:05}");
            let stem_base = &stem[..stem.len() - suffix.len()];
            let ext = first_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("gguf");
            for i in 1..=split_count {
                paths.push(
                    first_path
                        .with_file_name(format!("{stem_base}-{i:05}-of-{n:05}.{ext}")),
                );
            }
        } else {
            paths.push(first_path.to_path_buf());
        }

        let mut shards = Vec::with_capacity(paths.len());
        for path in &paths {
            let gguf = Gguf::open(path)
                .map_err(|source| LoaderError::ShardOpen { path: path.clone(), source })?;
            let split_no = gguf
                .metadata
                .get("split.no")
                .and_then(|v| v.as_u32().ok())
                .unwrap_or(0) as u16;
            shards.push(Shard { path: path.clone(), split_no, gguf });
        }

        // The config lives in the shard that declares the architecture.
        let cfg_shard = shards
            .iter()
            .find(|s| s.gguf.metadata.get_str("general.architecture").is_ok())
            .ok_or(LoaderError::NoConfigShard)?;
        let cfg = Qwen3_5Config::from_metadata(&cfg_shard.gguf.metadata).map_err(|source| {
            LoaderError::ShardOpen { path: cfg_shard.path.clone(), source }
        })?;

        let mut by_name: HashMap<String, (usize, TensorMeta)> = HashMap::new();
        for (si, shard) in shards.iter().enumerate() {
            for meta in &shard.gguf.tensors {
                match by_name.get(&meta.name) {
                    None => {
                        by_name.insert(meta.name.clone(), (si, meta.clone()));
                    }
                    // A tensor name may be listed in several files (e.g. tools
                    // that mirror the whole tensor-info section into every
                    // shard). Keep the entry whose data actually fits inside
                    // the shard file.
                    Some((existing_si, _)) => {
                        if !fits(shard, meta) && fits(&shards[*existing_si], &by_name[&meta.name].1)
                        {
                            continue;
                        }
                        by_name.insert(meta.name.clone(), (si, meta.clone()));
                    }
                }
            }
        }

        Ok(Self { cfg, shards, by_name })
    }

    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|(_, meta)| meta)
    }

    /// Index of the shard that holds `name`'s data.
    pub fn shard_of(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).map(|(si, _)| *si)
    }

    pub fn tensor_count(&self) -> usize {
        self.by_name.len()
    }

    /// Raw (still quantized) payload bytes for a named tensor.
    pub fn data_slice(&self, name: &str) -> Result<&[u8], LoaderError> {
        let (si, meta) = self
            .by_name
            .get(name)
            .ok_or_else(|| LoaderError::TensorNotFound { name: name.to_string() })?;
        let shard = &self.shards[*si];
        let expected = quant::tensor_size(meta.ggml_type, meta.n_elements())
            .map_err(|source| LoaderError::Quant { name: name.to_string(), source })?;
        let start = shard.gguf.data_offset + meta.offset as usize;
        let slice = shard.gguf.data_slice(meta);
        let got = (slice.len() as u64).min((shard.gguf.len() as u64).saturating_sub(start as u64));
        if got < expected {
            return Err(LoaderError::OutOfBounds {
                name: name.to_string(),
                expected,
                off: start as u64,
                shard: *si,
                len: got as usize,
            });
        }
        Ok(&slice[..expected as usize])
    }

    /// Dequantize a named tensor into f32 (row-major, ggml order).
    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, LoaderError> {
        let (_, meta) = self
            .by_name
            .get(name)
            .ok_or_else(|| LoaderError::TensorNotFound { name: name.to_string() })?;
        let data = self.data_slice(name)?;
        quant::dequantize(meta.ggml_type, data, meta.n_elements())
            .map_err(|source| LoaderError::Quant { name: name.to_string(), source })
    }
}

/// Does this tensor's relative data range fit inside its shard file?
fn fits(shard: &Shard, meta: &TensorMeta) -> bool {
    quant::tensor_size(meta.ggml_type, meta.n_elements())
        .map(|size| {
            shard.gguf.data_offset as u64 + meta.offset + size <= shard.gguf.len() as u64
        })
        .unwrap_or(false)
}

/// Extract `(index, total)` from a shard filename stem like
/// `Model-00003-of-00006`. Both fields must be exactly 5-digit zero-padded
/// (gguf-py `SHARD_NAME_FORMAT = "{:s}-{:05d}-of-{:05d}.gguf"`).
fn split_suffix(stem: &str) -> Option<(usize, usize)> {
    let marker = "-of-";
    let of_pos = stem.rfind(marker)?;
    let tail = &stem[of_pos + marker.len()..];
    if tail.len() == 5 && tail.chars().all(|c| c.is_ascii_digit()) {
        let n: usize = tail.parse().ok()?;
        let head = &stem[..of_pos];
        let dash = head.rfind('-')?;
        let idx = &head[dash + 1..];
        if idx.len() == 5 && idx.chars().all(|c| c.is_ascii_digit()) {
            return Some((idx.parse().ok()?, n));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::split_suffix;

    #[test]
    fn split_suffix_parses() {
        assert_eq!(split_suffix("Model-00003-of-00006"), Some((3, 6)));
        assert_eq!(split_suffix("foo"), None);
        assert_eq!(split_suffix("a-1-of-2"), None, "fields must be zero-padded");
    }
}
