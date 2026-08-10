use std::collections::HashMap;

use crate::gguf::error::GgufError;
use crate::gguf::value::Value;

#[derive(Debug, Clone, Default)]
pub struct Metadata {
    map: HashMap<String, Value>,
}

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: String, value: Value) {
        self.map.insert(key, value);
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.get(key)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.map.iter()
    }
}

#[allow(dead_code)]
impl Metadata {
    pub fn get_str(&self, key: &str) -> Result<&str, GgufError> {
        let v = self.require(key)?;
        v.as_str().map_err(|_| self.mismatch(key, v, "string"))
    }

    pub fn get_u32(&self, key: &str) -> Result<u32, GgufError> {
        let v = self.require(key)?;
        v.as_u32().map_err(|_| self.mismatch(key, v, "uint32"))
    }

    pub fn get_i32(&self, key: &str) -> Result<i32, GgufError> {
        let v = self.require(key)?;
        v.as_i32().map_err(|_| self.mismatch(key, v, "int32"))
    }

    pub fn get_u64(&self, key: &str) -> Result<u64, GgufError> {
        let v = self.require(key)?;
        v.as_u64().map_err(|_| self.mismatch(key, v, "uint64"))
    }

    pub fn get_i64(&self, key: &str) -> Result<i64, GgufError> {
        let v = self.require(key)?;
        v.as_i64().map_err(|_| self.mismatch(key, v, "int64"))
    }

    pub fn get_f32(&self, key: &str) -> Result<f32, GgufError> {
        let v = self.require(key)?;
        v.as_f32().map_err(|_| self.mismatch(key, v, "float32"))
    }

    pub fn get_bool(&self, key: &str) -> Result<bool, GgufError> {
        let v = self.require(key)?;
        v.as_bool().map_err(|_| self.mismatch(key, v, "bool"))
    }

    pub fn get_str_array(&self, key: &str) -> Result<Vec<&str>, GgufError> {
        let v = self.require(key)?;
        v.as_str_array().map_err(|_| self.mismatch(key, v, "array"))
    }

    pub fn get_i32_array(&self, key: &str) -> Result<Vec<i32>, GgufError> {
        let v = self.require(key)?;
        v.as_i32_array()
            .map_err(|_| self.mismatch(key, v, "int32 array"))
    }

    fn require(&self, key: &str) -> Result<&Value, GgufError> {
        self.get(key)
            .ok_or_else(|| GgufError::MissingKey(key.to_string()))
    }

    fn mismatch(&self, key: &str, v: &Value, expected: &'static str) -> GgufError {
        GgufError::TypeMismatch {
            key: key.to_string(),
            actual: v.type_name().to_string(),
            expected,
        }
    }
}
