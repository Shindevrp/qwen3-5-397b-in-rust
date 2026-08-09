use std::fs;

use tempfile::NamedTempFile;

use crate::gguf::error::GgufError;
use crate::gguf::value::{Value, ValueType};
use crate::gguf::writer::{GgufBuilder, TensorSpec};
use crate::gguf::{GGmlType, Gguf};

fn write_temp(bytes: Vec<u8>) -> NamedTempFile {
    let f = NamedTempFile::new().unwrap();
    fs::write(f.path(), bytes).unwrap();
    f
}

fn sample_bytes(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i % 251) as u8).collect()
}

#[test]
fn roundtrip_all_value_types() {
    let f32_data = sample_bytes(4096 * 4);
    let f16_data = sample_bytes(4096 * 2);
    let q4k_data = sample_bytes(256 * 144);

    let gguf_bytes = GgufBuilder::new()
        .metadata("general.name", Value::String("fixture".into()))
        .metadata("test.u8", Value::U8(0x12))
        .metadata("test.i8", Value::I8(-7))
        .metadata("test.u16", Value::U16(0x1234))
        .metadata("test.i16", Value::I16(-1000))
        .metadata("test.u32", Value::U32(42))
        .metadata("test.i32", Value::I32(-42))
        .metadata("test.f32", Value::F32(1.5))
        .metadata("test.bool", Value::Bool(true))
        .metadata("test.u64", Value::U64(1 << 40))
        .metadata("test.i64", Value::I64(-(1 << 40)))
        .metadata("test.f64", Value::F64(2.25))
        .metadata(
            "test.strs",
            Value::Array {
                elem_type: ValueType::String,
                items: vec![Value::String("alpha".into()), Value::String("beta".into())],
            },
        )
        .metadata(
            "test.nums",
            Value::Array {
                elem_type: ValueType::I64,
                items: vec![Value::I64(1), Value::I64(2), Value::I64(3)],
            },
        )
        .tensor(TensorSpec {
            name: "token_embd.weight".into(),
            ggml_type: GGmlType::F32,
            dims: vec![4096, 1],
            data: f32_data,
        })
        .tensor(TensorSpec {
            name: "blk.0.attn_norm.weight".into(),
            ggml_type: GGmlType::F16,
            dims: vec![4096],
            data: f16_data,
        })
        .tensor(TensorSpec {
            name: "blk.0.ffn_down_exps.weight".into(),
            ggml_type: GGmlType::Q4_K,
            dims: vec![1024, 64],
            data: q4k_data,
        })
        .build();

    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();

    assert_eq!(gguf.header.version, 3);
    assert_eq!(gguf.header.tensor_count, 3);
    assert_eq!(gguf.header.metadata_kv_count, 14);
    assert_eq!(gguf.alignment, 32);
    assert_eq!(gguf.metadata.len(), 14);

    assert_eq!(gguf.metadata.get_str("general.name").unwrap(), "fixture");
    assert_eq!(gguf.metadata.get_u32("test.u32").unwrap(), 42);
    assert_eq!(gguf.metadata.get_i32("test.i32").unwrap(), -42);
    assert_eq!(gguf.metadata.get_f32("test.f32").unwrap(), 1.5);
    assert!(gguf.metadata.get_bool("test.bool").unwrap());
    assert_eq!(gguf.metadata.get_u64("test.u64").unwrap(), 1 << 40);
    assert_eq!(gguf.metadata.get_i64("test.i64").unwrap(), -(1 << 40));
    assert_eq!(
        gguf.metadata.get_str_array("test.strs").unwrap(),
        vec!["alpha", "beta"]
    );

    let t0 = gguf.tensor("token_embd.weight").unwrap();
    assert_eq!(t0.name, "token_embd.weight");
    assert_eq!(t0.ggml_type, GGmlType::F32);
    assert_eq!(t0.dims, vec![4096, 1]);
    assert_eq!(t0.n_elements(), 4096);
    assert!(!t0.is_quantized());

    let t1 = gguf.tensor("blk.0.attn_norm.weight").unwrap();
    assert_eq!(t1.ggml_type, GGmlType::F16);
    assert_eq!(t1.dims, vec![4096]);

    let t2 = gguf.tensor("blk.0.ffn_down_exps.weight").unwrap();
    assert_eq!(t2.ggml_type, GGmlType::Q4_K);
    assert!(t2.is_quantized());
    assert_eq!(gguf.data_slice(t2)[..256 * 144], sample_bytes(256 * 144));

    assert!(gguf.data_slice(t0).len() >= 4096 * 4);
    assert!(gguf.tensor("does.not.exist").is_none());
}

#[test]
fn roundtrip_custom_alignment() {
    let gguf_bytes = GgufBuilder::new()
        .with_alignment(64)
        .metadata("general.alignment", Value::U32(64))
        .tensor(TensorSpec {
            name: "a".into(),
            ggml_type: GGmlType::F32,
            dims: vec![8],
            data: sample_bytes(32),
        })
        .tensor(TensorSpec {
            name: "b".into(),
            ggml_type: GGmlType::F32,
            dims: vec![8],
            data: sample_bytes(32),
        })
        .build();

    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();

    assert_eq!(gguf.alignment, 64);
    for t in &gguf.tensors {
        assert_eq!(t.offset % 64, 0);
    }
    assert!((gguf.data_offset as u64) <= gguf.tensors[0].offset);
}

#[test]
fn rejects_bad_magic() {
    let f = write_temp(b"GUFG".to_vec());
    let err = Gguf::open(f.path()).unwrap_err();
    assert!(matches!(err, GgufError::BadMagic(_)));
}

#[test]
fn rejects_unsupported_version() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&99u32.to_le_bytes());
    let f = write_temp(bytes);
    let err = Gguf::open(f.path()).unwrap_err();
    assert!(matches!(err, GgufError::UnsupportedVersion(99)));
}

#[test]
fn rejects_truncated_header() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let f = write_temp(bytes);
    let err = Gguf::open(f.path()).unwrap_err();
    assert!(matches!(err, GgufError::Truncated { .. }));
}

#[test]
fn rejects_unknown_value_type() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&12u64.to_le_bytes());
    bytes.extend_from_slice(b"general.test");
    bytes.extend_from_slice(&999u32.to_le_bytes());
    let f = write_temp(bytes);
    let err = Gguf::open(f.path()).unwrap_err();
    assert!(matches!(err, GgufError::UnknownValueType(999)));
}

#[test]
fn rejects_unknown_ggml_type_value() {
    let gguf_bytes = GgufBuilder::new()
        .tensor(TensorSpec {
            name: "weird".into(),
            ggml_type: GGmlType::Unknown(999),
            dims: vec![4],
            data: vec![0; 4],
        })
        .build();
    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();
    let t = gguf.tensor("weird").unwrap();
    assert_eq!(t.ggml_type, GGmlType::Unknown(999));
    assert_eq!(t.ggml_type.name(), "Unknown(999)");
}

#[test]
fn metadata_type_mismatch_is_reported() {
    let gguf_bytes = GgufBuilder::new()
        .metadata("general.name", Value::U32(7))
        .build();
    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();
    let err = gguf.metadata.get_str("general.name").unwrap_err();
    match err {
        GgufError::TypeMismatch {
            key,
            actual,
            expected,
        } => {
            assert_eq!(key, "general.name");
            assert_eq!(actual, "uint32");
            assert_eq!(expected, "string");
        }
        other => panic!("unexpected error: {other}"),
    }
}
