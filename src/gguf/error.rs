use thiserror::Error;

#[derive(Error, Debug)]
pub enum GgufError {
    #[error("bad magic: expected b\"GGUF\", got {0:?}")]
    BadMagic([u8; 4]),

    #[error("unsupported gguf version {0} (supported: 2, 3)")]
    UnsupportedVersion(u32),

    #[error("unknown gguf value type {0}")]
    UnknownValueType(u32),

    #[error("truncated file: needed {needed} bytes at offset {offset}, file length {len}")]
    Truncated {
        offset: usize,
        needed: usize,
        len: usize,
    },

    #[error("invalid utf8 string at offset {0}: {1}")]
    InvalidUtf8(usize, std::str::Utf8Error),

    #[error("missing metadata key \"{0}\"")]
    MissingKey(String),

    #[error("metadata key \"{key}\" has type {actual}, expected {expected}")]
    TypeMismatch {
        key: String,
        actual: String,
        expected: &'static str,
    },

    #[error("value has type {actual}, expected {expected}")]
    ValueTypeMismatch {
        actual: &'static str,
        expected: &'static str,
    },

    #[error("tensor \"{name}\" has {dims} dimensions (maximum is 4)")]
    TooManyDims { name: String, dims: u32 },

    #[error("alignment must be a power of two, got {0}")]
    BadAlignment(u64),

    #[error("{0}")]
    Io(String),
}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e.to_string())
    }
}
