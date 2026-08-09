#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ1_S,
    IQ4_NL,
    IQ3_S,
    IQ2_S,
    IQ4_XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1_M,
    BF16,
    TQ1_0,
    TQ2_0,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_0,
    Unknown(u32),
}

impl GGmlType {
    pub fn from_raw(raw: u32) -> Self {
        match raw {
            0 => GGmlType::F32,
            1 => GGmlType::F16,
            2 => GGmlType::Q4_0,
            3 => GGmlType::Q4_1,
            6 => GGmlType::Q5_0,
            7 => GGmlType::Q5_1,
            8 => GGmlType::Q8_0,
            9 => GGmlType::Q8_1,
            10 => GGmlType::Q2_K,
            11 => GGmlType::Q3_K,
            12 => GGmlType::Q4_K,
            13 => GGmlType::Q5_K,
            14 => GGmlType::Q6_K,
            15 => GGmlType::Q8_K,
            16 => GGmlType::IQ2_XXS,
            17 => GGmlType::IQ2_XS,
            18 => GGmlType::IQ3_XXS,
            19 => GGmlType::IQ1_S,
            20 => GGmlType::IQ4_NL,
            21 => GGmlType::IQ3_S,
            22 => GGmlType::IQ2_S,
            23 => GGmlType::IQ4_XS,
            24 => GGmlType::I8,
            25 => GGmlType::I16,
            26 => GGmlType::I32,
            27 => GGmlType::I64,
            28 => GGmlType::F64,
            29 => GGmlType::IQ1_M,
            30 => GGmlType::BF16,
            34 => GGmlType::TQ1_0,
            35 => GGmlType::TQ2_0,
            39 => GGmlType::MXFP4,
            40 => GGmlType::NVFP4,
            41 => GGmlType::Q1_0,
            42 => GGmlType::Q2_0,
            other => GGmlType::Unknown(other),
        }
    }

    #[allow(dead_code)]
    pub fn as_raw(self) -> u32 {
        match self {
            GGmlType::F32 => 0,
            GGmlType::F16 => 1,
            GGmlType::Q4_0 => 2,
            GGmlType::Q4_1 => 3,
            GGmlType::Q5_0 => 6,
            GGmlType::Q5_1 => 7,
            GGmlType::Q8_0 => 8,
            GGmlType::Q8_1 => 9,
            GGmlType::Q2_K => 10,
            GGmlType::Q3_K => 11,
            GGmlType::Q4_K => 12,
            GGmlType::Q5_K => 13,
            GGmlType::Q6_K => 14,
            GGmlType::Q8_K => 15,
            GGmlType::IQ2_XXS => 16,
            GGmlType::IQ2_XS => 17,
            GGmlType::IQ3_XXS => 18,
            GGmlType::IQ1_S => 19,
            GGmlType::IQ4_NL => 20,
            GGmlType::IQ3_S => 21,
            GGmlType::IQ2_S => 22,
            GGmlType::IQ4_XS => 23,
            GGmlType::I8 => 24,
            GGmlType::I16 => 25,
            GGmlType::I32 => 26,
            GGmlType::I64 => 27,
            GGmlType::F64 => 28,
            GGmlType::IQ1_M => 29,
            GGmlType::BF16 => 30,
            GGmlType::TQ1_0 => 34,
            GGmlType::TQ2_0 => 35,
            GGmlType::MXFP4 => 39,
            GGmlType::NVFP4 => 40,
            GGmlType::Q1_0 => 41,
            GGmlType::Q2_0 => 42,
            GGmlType::Unknown(raw) => raw,
        }
    }

    pub fn name(self) -> String {
        match self {
            GGmlType::F32 => "F32".to_string(),
            GGmlType::F16 => "F16".to_string(),
            GGmlType::Q4_0 => "Q4_0".to_string(),
            GGmlType::Q4_1 => "Q4_1".to_string(),
            GGmlType::Q5_0 => "Q5_0".to_string(),
            GGmlType::Q5_1 => "Q5_1".to_string(),
            GGmlType::Q8_0 => "Q8_0".to_string(),
            GGmlType::Q8_1 => "Q8_1".to_string(),
            GGmlType::Q2_K => "Q2_K".to_string(),
            GGmlType::Q3_K => "Q3_K".to_string(),
            GGmlType::Q4_K => "Q4_K".to_string(),
            GGmlType::Q5_K => "Q5_K".to_string(),
            GGmlType::Q6_K => "Q6_K".to_string(),
            GGmlType::Q8_K => "Q8_K".to_string(),
            GGmlType::IQ2_XXS => "IQ2_XXS".to_string(),
            GGmlType::IQ2_XS => "IQ2_XS".to_string(),
            GGmlType::IQ3_XXS => "IQ3_XXS".to_string(),
            GGmlType::IQ1_S => "IQ1_S".to_string(),
            GGmlType::IQ4_NL => "IQ4_NL".to_string(),
            GGmlType::IQ3_S => "IQ3_S".to_string(),
            GGmlType::IQ2_S => "IQ2_S".to_string(),
            GGmlType::IQ4_XS => "IQ4_XS".to_string(),
            GGmlType::I8 => "I8".to_string(),
            GGmlType::I16 => "I16".to_string(),
            GGmlType::I32 => "I32".to_string(),
            GGmlType::I64 => "I64".to_string(),
            GGmlType::F64 => "F64".to_string(),
            GGmlType::IQ1_M => "IQ1_M".to_string(),
            GGmlType::BF16 => "BF16".to_string(),
            GGmlType::TQ1_0 => "TQ1_0".to_string(),
            GGmlType::TQ2_0 => "TQ2_0".to_string(),
            GGmlType::MXFP4 => "MXFP4".to_string(),
            GGmlType::NVFP4 => "NVFP4".to_string(),
            GGmlType::Q1_0 => "Q1_0".to_string(),
            GGmlType::Q2_0 => "Q2_0".to_string(),
            GGmlType::Unknown(raw) => format!("Unknown({raw})"),
        }
    }

    pub fn is_quantized(self) -> bool {
        !matches!(
            self,
            GGmlType::F32
                | GGmlType::F16
                | GGmlType::I8
                | GGmlType::I16
                | GGmlType::I32
                | GGmlType::I64
                | GGmlType::F64
                | GGmlType::BF16
        )
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub ggml_type: GGmlType,
    pub dims: Vec<u32>,
    pub offset: u64,
}

impl TensorMeta {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().map(|&d| u64::from(d)).product()
    }

    pub fn is_quantized(&self) -> bool {
        self.ggml_type.is_quantized()
    }
}
