use byteorder::{LittleEndian, WriteBytesExt};

use crate::gguf::tensor::GGmlType;
use crate::gguf::value::{Value, ValueType};

#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub ggml_type: GGmlType,
    pub dims: Vec<u64>,
    pub data: Vec<u8>,
}

impl TensorSpec {
    fn index_entry_len(&self) -> u64 {
        8 + self.name.len() as u64 + 4 + 8 * self.dims.len() as u64 + 4 + 8
    }
}

pub struct GgufBuilder {
    metadata: Vec<(String, Value)>,
    tensors: Vec<TensorSpec>,
    alignment: u64,
}

impl Default for GgufBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufBuilder {
    pub fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            alignment: 32,
        }
    }

    pub fn with_alignment(mut self, alignment: u64) -> Self {
        self.alignment = alignment;
        self
    }

    pub fn metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    pub fn tensor(mut self, spec: TensorSpec) -> Self {
        self.tensors.push(spec);
        self
    }

    pub fn build(self) -> Vec<u8> {
        debug_assert!(self.alignment.is_power_of_two());
        let mut out: Vec<u8> = Vec::new();

        out.extend_from_slice(b"GGUF");
        out.write_u32::<LittleEndian>(3).unwrap();
        out.write_u64::<LittleEndian>(self.tensors.len() as u64)
            .unwrap();
        out.write_u64::<LittleEndian>(self.metadata.len() as u64)
            .unwrap();

        for (key, value) in &self.metadata {
            write_str(&mut out, key);
            out.write_u32::<LittleEndian>(value_type_of(value).as_raw())
                .unwrap();
            write_value(&mut out, value);
        }

        let index_end = out.len() + self.tensors.iter().map(|t| t.index_entry_len() as usize).sum::<usize>();
        let data_start = align_up(index_end as u64, self.alignment);

        let mut data_pos = 0u64;
        let mut offsets = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            data_pos = align_up(data_pos, self.alignment);
            offsets.push(data_pos);
            data_pos += t.data.len() as u64;
        }

        for (t, offset) in self.tensors.iter().zip(&offsets) {
            write_str(&mut out, &t.name);
            out.write_u32::<LittleEndian>(t.dims.len() as u32).unwrap();
            for &d in &t.dims {
                out.write_u64::<LittleEndian>(d).unwrap();
            }
            out.write_u32::<LittleEndian>(t.ggml_type.as_raw()).unwrap();
            out.write_u64::<LittleEndian>(*offset).unwrap();
        }

        for (t, offset) in self.tensors.iter().zip(&offsets) {
            while (out.len() as u64) < data_start + *offset {
                out.push(0);
            }
            out.extend_from_slice(&t.data);
        }

        out
    }
}

fn value_type_of(v: &Value) -> ValueType {
    match v {
        Value::U8(_) => ValueType::U8,
        Value::I8(_) => ValueType::I8,
        Value::U16(_) => ValueType::U16,
        Value::I16(_) => ValueType::I16,
        Value::U32(_) => ValueType::U32,
        Value::I32(_) => ValueType::I32,
        Value::F32(_) => ValueType::F32,
        Value::Bool(_) => ValueType::Bool,
        Value::String(_) => ValueType::String,
        Value::Array { .. } => ValueType::Array,
        Value::U64(_) => ValueType::U64,
        Value::I64(_) => ValueType::I64,
        Value::F64(_) => ValueType::F64,
    }
}

fn write_str(out: &mut Vec<u8>, s: &str) {
    out.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::U8(x) => out.write_u8(*x).unwrap(),
        Value::I8(x) => out.write_i8(*x).unwrap(),
        Value::U16(x) => out.write_u16::<LittleEndian>(*x).unwrap(),
        Value::I16(x) => out.write_i16::<LittleEndian>(*x).unwrap(),
        Value::U32(x) => out.write_u32::<LittleEndian>(*x).unwrap(),
        Value::I32(x) => out.write_i32::<LittleEndian>(*x).unwrap(),
        Value::F32(x) => out.write_f32::<LittleEndian>(*x).unwrap(),
        Value::Bool(b) => out.write_u8(u8::from(*b)).unwrap(),
        Value::String(s) => write_str(out, s),
        Value::Array { elem_type, items } => {
            out.write_u32::<LittleEndian>(elem_type.as_raw()).unwrap();
            out.write_u64::<LittleEndian>(items.len() as u64).unwrap();
            for item in items {
                write_value(out, item);
            }
        }
        Value::U64(x) => out.write_u64::<LittleEndian>(*x).unwrap(),
        Value::I64(x) => out.write_i64::<LittleEndian>(*x).unwrap(),
        Value::F64(x) => out.write_f64::<LittleEndian>(*x).unwrap(),
    }
}

fn align_up(x: u64, alignment: u64) -> u64 {
    (x + alignment - 1) & !(alignment - 1)
}
