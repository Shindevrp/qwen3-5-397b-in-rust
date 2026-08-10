use std::fs;

use tempfile::NamedTempFile;

use crate::gguf::error::GgufError;
use crate::gguf::value::{Value, ValueType};
use crate::gguf::writer::{GgufBuilder, TensorSpec};
use crate::gguf::{GGmlType, Gguf};
use crate::model::config::Qwen3_5Config;

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
    assert_eq!(gguf.tensors[0].offset, 0);
    for t in &gguf.tensors {
        assert_eq!(t.offset % 64, 0);
    }
    assert!(gguf.data_offset.is_multiple_of(64));
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

#[test]
fn qwen35_config_loads_and_validates() {
    let f32_data = sample_bytes(4096 * 4);

    let gguf_bytes = GgufBuilder::new()
        .metadata("general.architecture", Value::String("qwen3_5moe".into()))
        .metadata("general.name", Value::String("Qwen3.5-397B-A17B".into()))
        .metadata("qwen3_5moe.block_count", Value::U32(60))
        .metadata("qwen3_5moe.embedding_length", Value::U32(4096))
        .metadata("qwen3_5moe.attention.head_count", Value::U32(32))
        .metadata("qwen3_5moe.attention.head_count_kv", Value::U32(2))
        .metadata("qwen3_5moe.attention.key_length", Value::U32(256))
        .metadata(
            "qwen3_5moe.attention.layer_norm_rms_epsilon",
            Value::F32(1e-6),
        )
        .metadata("qwen3_5moe.expert_count", Value::U32(512))
        .metadata("qwen3_5moe.expert_used_count", Value::U32(10))
        .metadata("qwen3_5moe.expert_feed_forward_length", Value::U32(1024))
        .metadata(
            "qwen3_5moe.expert_shared_feed_forward_length",
            Value::U32(1024),
        )
        .metadata("qwen3_5moe.rope.dimension_count", Value::U32(64))
        .metadata("qwen3_5moe.rope.freq_base", Value::F32(10_000_000.0))
        .metadata("qwen3_5moe.context_length", Value::U32(262_144))
        .metadata("qwen3_5moe.ssm.state_size", Value::U32(128))
        .metadata("qwen3_5moe.ssm.group_count", Value::U32(16))
        .metadata("qwen3_5moe.ssm.time_step_rank", Value::U32(64))
        .metadata("qwen3_5moe.ssm.conv_kernel", Value::U32(4))
        .metadata("qwen3_5moe.ssm.inner_size", Value::U32(8192))
        .metadata("qwen3_5moe.full_attention_interval", Value::U32(4))
        .metadata(
            "qwen3_5moe.rope.dimension_sections",
            Value::Array {
                elem_type: ValueType::I32,
                items: vec![
                    Value::I32(11),
                    Value::I32(11),
                    Value::I32(10),
                    Value::I32(0),
                ],
            },
        )
        .tensor(TensorSpec {
            name: "token_embd.weight".into(),
            ggml_type: GGmlType::F32,
            dims: vec![4096, 248320],
            data: sample_bytes(4096 * 248320),
        })
        .tensor(TensorSpec {
            name: "output_norm.weight".into(),
            ggml_type: GGmlType::F32,
            dims: vec![4096],
            data: f32_data,
        })
        .tensor(TensorSpec {
            name: "output.weight".into(),
            ggml_type: GGmlType::F32,
            dims: vec![4096, 248320],
            data: sample_bytes(4096 * 248320),
        })
        .build();

    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();

    let cfg = Qwen3_5Config::from_metadata(&gguf.metadata).unwrap();
    assert_eq!(cfg.block_count, 60);
    assert_eq!(cfg.embedding_length, 4096);
    assert_eq!(cfg.attention_head_count, 32);
    assert_eq!(cfg.attention_head_count_kv, 2);
    assert_eq!(cfg.attention_key_length, 256);
    assert_eq!(cfg.expert_count, 512);
    assert_eq!(cfg.expert_used_count, 10);
    assert_eq!(cfg.key_dim, 128 * 16);
    assert_eq!(cfg.value_dim, 128 * 64);
    assert_eq!(cfg.conv_dim, (128 * 16) * 2 + 128 * 64);
    assert_eq!(cfg.ba_dim, 64 * 2);
    assert_eq!(cfg.full_attn_q_fused_dim, 256 * 32 * 2);
}

#[test]
fn qwen35_config_rejects_wrong_arch() {
    let gguf_bytes = GgufBuilder::new()
        .metadata("general.architecture", Value::String("llama".into()))
        .metadata("qwen3_5moe.block_count", Value::U32(60))
        .build();
    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();
    let err = Qwen3_5Config::from_metadata(&gguf.metadata).unwrap_err();
    assert!(matches!(err, GgufError::TypeMismatch { key, .. } if key == "general.architecture"));
}

#[test]
fn qwen35_config_rejects_missing_key() {
    let gguf_bytes = GgufBuilder::new()
        .metadata("general.architecture", Value::String("qwen3_5moe".into()))
        .build();
    let f = write_temp(gguf_bytes);
    let gguf = Gguf::open(f.path()).unwrap();
    let err = Qwen3_5Config::from_metadata(&gguf.metadata).unwrap_err();
    assert!(matches!(err, GgufError::MissingKey { .. }));
}
