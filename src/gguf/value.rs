use std::fmt;

use thiserror::Error;

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Text(String),
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("unsupported value type")]
    Unsupported,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(i) => write!(f, "Int({})", i),
            Value::Float(x) => write!(f, "Float({})", x),
            Value::Text(s) => write!(f, "Text({})", s),
        }
    }
}

impl Value {
    pub fn as_i64(&self) -> Result<i64, Error> {
        match self {
            Value::Int(i) => Ok(*i),
            _ => Err(Error::Unsupported),
        }
    }
}
