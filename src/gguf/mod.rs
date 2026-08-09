pub mod error;
pub mod header;
pub mod metadata;
pub mod tensor;
pub mod value;
#[cfg(test)]
pub mod writer;

pub use error::GgufError;
pub use header::Header;
pub use metadata::Metadata;
pub use tensor::{GGmlType, TensorMeta};
pub use value::{Value, ValueType};

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;

pub const GGUF_MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
const MAX_TENSOR_DIMS: u32 = 4;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct Gguf {
    mmap: Mmap,
    pub header: Header,
    pub metadata: Metadata,
    pub tensors: Vec<TensorMeta>,
    by_name: HashMap<String, usize>,
    pub data_offset: usize,
    pub alignment: u64,
}

impl Gguf {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Gguf, GgufError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let mut r = Reader::new(&mmap);

        let header = r.read_header()?;
        let (metadata, alignment) = r.read_metadata(header.metadata_kv_count)?;
        r.align(alignment);
        let tensors = r.read_tensor_index(header.tensor_count)?;
        let data_offset = r.pos;

        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();

        Ok(Gguf {
            mmap,
            header,
            metadata,
            tensors,
            by_name,
            data_offset,
            alignment,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorMeta> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    pub fn data_slice(&self, tensor: &TensorMeta) -> &[u8] {
        &self.mmap[tensor.offset as usize..]
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn ensure(&self, n: usize) -> Result<(), GgufError> {
        if self.pos + n > self.buf.len() {
            Err(GgufError::Truncated {
                offset: self.pos,
                needed: n,
                len: self.buf.len(),
            })
        } else {
            Ok(())
        }
    }

    fn read_u8(&mut self) -> Result<u8, GgufError> {
        self.ensure(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, GgufError> {
        self.ensure(1)?;
        let v = self.buf[self.pos] as i8;
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, GgufError> {
        self.ensure(2)?;
        let v = LittleEndian::read_u16(&self.buf[self.pos..]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, GgufError> {
        self.ensure(2)?;
        let v = LittleEndian::read_i16(&self.buf[self.pos..]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        self.ensure(4)?;
        let v = LittleEndian::read_u32(&self.buf[self.pos..]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, GgufError> {
        self.ensure(4)?;
        let v = LittleEndian::read_i32(&self.buf[self.pos..]);
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        self.ensure(8)?;
        let v = LittleEndian::read_u64(&self.buf[self.pos..]);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, GgufError> {
        self.ensure(8)?;
        let v = LittleEndian::read_i64(&self.buf[self.pos..]);
        self.pos += 8;
        Ok(v)
    }

    fn read_f32(&mut self) -> Result<f32, GgufError> {
        self.ensure(4)?;
        let v = LittleEndian::read_f32(&self.buf[self.pos..]);
        self.pos += 4;
        Ok(v)
    }

    fn read_f64(&mut self) -> Result<f64, GgufError> {
        self.ensure(8)?;
        let v = LittleEndian::read_f64(&self.buf[self.pos..]);
        self.pos += 8;
        Ok(v)
    }

    fn read_str(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()? as usize;
        self.ensure(len)?;
        let s = std::str::from_utf8(&self.buf[self.pos..self.pos + len])
            .map_err(|e| GgufError::InvalidUtf8(self.pos, e))?;
        self.pos += len;
        Ok(s.to_string())
    }

    fn align(&mut self, alignment: u64) {
        let rem = self.pos % alignment as usize;
        if rem != 0 {
            self.pos += alignment as usize - rem;
        }
    }

    fn read_header(&mut self) -> Result<Header, GgufError> {
        self.ensure(4)?;
        let magic: [u8; 4] = [self.buf[0], self.buf[1], self.buf[2], self.buf[3]];
        self.pos += 4;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }
        let version = self.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = self.read_u64()?;
        let metadata_kv_count = self.read_u64()?;
        Ok(Header {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_metadata(&mut self, kv_count: u64) -> Result<(Metadata, u64), GgufError> {
        let mut metadata = Metadata::new();
        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..kv_count {
            let key = self.read_str()?;
            let type_raw = self.read_u32()?;
            let value_type = ValueType::from_raw(type_raw)?;
            let value = self.read_value(value_type)?;
            if key == "general.alignment" {
                match value {
                    Value::U32(a) => {
                        let a = u64::from(a);
                        if a == 0 || !a.is_power_of_two() {
                            return Err(GgufError::BadAlignment(a));
                        }
                        alignment = a;
                    }
                    _ => {
                        return Err(GgufError::TypeMismatch {
                            key: "general.alignment".to_string(),
                            actual: value.type_name().to_string(),
                            expected: "uint32",
                        });
                    }
                }
            }
            metadata.insert(key, value);
        }
        Ok((metadata, alignment))
    }

    fn read_value(&mut self, value_type: ValueType) -> Result<Value, GgufError> {
        Ok(match value_type {
            ValueType::U8 => Value::U8(self.read_u8()?),
            ValueType::I8 => Value::I8(self.read_i8()?),
            ValueType::U16 => Value::U16(self.read_u16()?),
            ValueType::I16 => Value::I16(self.read_i16()?),
            ValueType::U32 => Value::U32(self.read_u32()?),
            ValueType::I32 => Value::I32(self.read_i32()?),
            ValueType::F32 => Value::F32(self.read_f32()?),
            ValueType::Bool => Value::Bool(self.read_u8()? != 0),
            ValueType::String => Value::String(self.read_str()?),
            ValueType::Array => {
                let elem_raw = self.read_u32()?;
                let elem_type = ValueType::from_raw(elem_raw)?;
                let count = self.read_u64()?;
                let mut items = Vec::with_capacity(count.min(1_000_000) as usize);
                for _ in 0..count {
                    items.push(self.read_value(elem_type)?);
                }
                Value::Array { elem_type, items }
            }
            ValueType::U64 => Value::U64(self.read_u64()?),
            ValueType::I64 => Value::I64(self.read_i64()?),
            ValueType::F64 => Value::F64(self.read_f64()?),
        })
    }

    fn read_tensor_index(&mut self, tensor_count: u64) -> Result<Vec<TensorMeta>, GgufError> {
        let mut out = Vec::with_capacity(tensor_count.min(1_000_000) as usize);
        for _ in 0..tensor_count {
            let name = self.read_str()?;
            let n_dims = self.read_u32()?;
            if n_dims > MAX_TENSOR_DIMS {
                return Err(GgufError::TooManyDims { name, dims: n_dims });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(self.read_u32()?);
            }
            let type_raw = self.read_u32()?;
            let offset = self.read_u64()?;
            out.push(TensorMeta {
                name,
                ggml_type: GGmlType::from_raw(type_raw),
                dims,
                offset,
            });
        }
        Ok(out)
    }
}
