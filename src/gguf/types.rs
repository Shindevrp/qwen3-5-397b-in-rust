#[derive(Debug, Clone)]
pub struct Header {
    pub version: u32,
}

impl Header {
    pub fn new(version: u32) -> Self {
        Self { version }
    }
}
