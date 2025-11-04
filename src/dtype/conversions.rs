//! Primitive type conversions for DType

use super::{types::{BFloat16, QuantizedI4, QuantizedU8}, DType};

// Conversion traits
impl From<f32> for DType {
    fn from(_: f32) -> Self {
        DType::F32
    }
}

impl From<f64> for DType {
    fn from(_: f64) -> Self {
        DType::F64
    }
}

impl From<i8> for DType {
    fn from(_: i8) -> Self {
        DType::I8
    }
}

impl From<i16> for DType {
    fn from(_: i16) -> Self {
        DType::I16
    }
}

impl From<i32> for DType {
    fn from(_: i32) -> Self {
        DType::I32
    }
}

impl From<i64> for DType {
    fn from(_: i64) -> Self {
        DType::I64
    }
}

impl From<u8> for DType {
    fn from(_: u8) -> Self {
        DType::U8
    }
}

impl From<u16> for DType {
    fn from(_: u16) -> Self {
        DType::U16
    }
}

impl From<u32> for DType {
    fn from(_: u32) -> Self {
        DType::U32
    }
}

impl From<u64> for DType {
    fn from(_: u64) -> Self {
        DType::U64
    }
}

impl From<bool> for DType {
    fn from(_: bool) -> Self {
        DType::Bool
    }
}

impl From<BFloat16> for DType {
    fn from(_: BFloat16) -> Self {
        DType::BF16
    }
}

impl From<QuantizedI4> for DType {
    fn from(_: QuantizedI4) -> Self {
        DType::QI4
    }
}

impl From<QuantizedU8> for DType {
    fn from(_: QuantizedU8) -> Self {
        DType::QU8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_from_primitives() {
        assert_eq!(DType::from(1.0f32), DType::F32);
        assert_eq!(DType::from(1.0f64), DType::F64);
        assert_eq!(DType::from(1i32), DType::I32);
        assert_eq!(DType::from(1u64), DType::U64);
        assert_eq!(DType::from(true), DType::Bool);
    }

    #[test]
    fn dtype_from_custom_types() {
        let bf16 = BFloat16::from_f32(1.0);
        assert_eq!(DType::from(bf16), DType::BF16);

        let qi4 = QuantizedI4::from_i8(5, 1.0);
        assert_eq!(DType::from(qi4), DType::QI4);

        let qu8 = QuantizedU8::quantize(2.5, 0.01);
        assert_eq!(DType::from(qu8), DType::QU8);
    }
}
