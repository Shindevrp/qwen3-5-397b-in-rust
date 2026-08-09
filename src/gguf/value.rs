use crate::gguf::error::GgufError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    pub fn from_raw(raw: u32) -> Result<Self, GgufError> {
        Ok(match raw {
            0 => ValueType::U8,
            1 => ValueType::I8,
            2 => ValueType::U16,
            3 => ValueType::I16,
            4 => ValueType::U32,
            5 => ValueType::I32,
            6 => ValueType::F32,
            7 => ValueType::Bool,
            8 => ValueType::String,
            9 => ValueType::Array,
            10 => ValueType::U64,
            11 => ValueType::I64,
            12 => ValueType::F64,
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }

    #[allow(dead_code)]
    pub fn as_raw(self) -> u32 {
        match self {
            ValueType::U8 => 0,
            ValueType::I8 => 1,
            ValueType::U16 => 2,
            ValueType::I16 => 3,
            ValueType::U32 => 4,
            ValueType::I32 => 5,
            ValueType::F32 => 6,
            ValueType::Bool => 7,
            ValueType::String => 8,
            ValueType::Array => 9,
            ValueType::U64 => 10,
            ValueType::I64 => 11,
            ValueType::F64 => 12,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ValueType::U8 => "uint8",
            ValueType::I8 => "int8",
            ValueType::U16 => "uint16",
            ValueType::I16 => "int16",
            ValueType::U32 => "uint32",
            ValueType::I32 => "int32",
            ValueType::F32 => "float32",
            ValueType::Bool => "bool",
            ValueType::String => "string",
            ValueType::Array => "array",
            ValueType::U64 => "uint64",
            ValueType::I64 => "int64",
            ValueType::F64 => "float64",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        elem_type: ValueType,
        items: Vec<Value>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::U8(_) => ValueType::U8.name(),
            Value::I8(_) => ValueType::I8.name(),
            Value::U16(_) => ValueType::U16.name(),
            Value::I16(_) => ValueType::I16.name(),
            Value::U32(_) => ValueType::U32.name(),
            Value::I32(_) => ValueType::I32.name(),
            Value::F32(_) => ValueType::F32.name(),
            Value::Bool(_) => ValueType::Bool.name(),
            Value::String(_) => ValueType::String.name(),
            Value::Array { .. } => ValueType::Array.name(),
            Value::U64(_) => ValueType::U64.name(),
            Value::I64(_) => ValueType::I64.name(),
            Value::F64(_) => ValueType::F64.name(),
        }
    }

    pub fn as_str(&self) -> Result<&str, GgufError> {
        match self {
            Value::String(s) => Ok(s),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "string",
            }),
        }
    }

    pub fn as_u32(&self) -> Result<u32, GgufError> {
        match self {
            Value::U32(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "uint32",
            }),
        }
    }

    pub fn as_i32(&self) -> Result<i32, GgufError> {
        match self {
            Value::I32(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "int32",
            }),
        }
    }

    pub fn as_u64(&self) -> Result<u64, GgufError> {
        match self {
            Value::U64(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "uint64",
            }),
        }
    }

    pub fn as_i64(&self) -> Result<i64, GgufError> {
        match self {
            Value::I64(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "int64",
            }),
        }
    }

    pub fn as_f32(&self) -> Result<f32, GgufError> {
        match self {
            Value::F32(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "float32",
            }),
        }
    }

    pub fn as_bool(&self) -> Result<bool, GgufError> {
        match self {
            Value::Bool(v) => Ok(*v),
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "bool",
            }),
        }
    }

    pub fn as_str_array(&self) -> Result<Vec<&str>, GgufError> {
        match self {
            Value::Array { items, .. } => {
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(item.as_str()?);
                }
                Ok(out)
            }
            other => Err(GgufError::ValueTypeMismatch {
                actual: other.type_name(),
                expected: "array",
            }),
        }
    }
}
