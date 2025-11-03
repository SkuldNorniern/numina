//! Primitive type conversions for DType

use super::DType;

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
}
