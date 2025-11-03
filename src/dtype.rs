//! Data type definitions for tensors

use std::fmt;

/// Data type enumeration for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 16-bit floating point
    F16,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// Boolean
    Bool,
}

impl DType {
    /// Returns the size in bytes of this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Bool => 1,
        }
    }

    /// Returns true if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F16 | DType::F32 | DType::F64)
    }

    /// Returns true if this is an integer type
    pub fn is_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64 |
                       DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Returns true if this is a signed integer type
    pub fn is_signed_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// Returns true if this is an unsigned integer type
    pub fn is_unsigned_int(&self) -> bool {
        matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Returns true if this is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::F16 => write!(f, "f16"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::U16 => write!(f, "u16"),
            DType::U32 => write!(f, "u32"),
            DType::U64 => write!(f, "u64"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

// Convenience constants
pub const F32: DType = DType::F32;
pub const F64: DType = DType::F64;
pub const F16: DType = DType::F16;
pub const I8: DType = DType::I8;
pub const I16: DType = DType::I16;
pub const I32: DType = DType::I32;
pub const I64: DType = DType::I64;
pub const U8: DType = DType::U8;
pub const U16: DType = DType::U16;
pub const U32: DType = DType::U32;
pub const U64: DType = DType::U64;
pub const BOOL: DType = DType::Bool;

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
    fn dtype_sizes() {
        assert_eq!(F32.size_bytes(), 4);
        assert_eq!(F64.size_bytes(), 8);
        assert_eq!(I32.size_bytes(), 4);
        assert_eq!(U8.size_bytes(), 1);
        assert_eq!(BOOL.size_bytes(), 1);
    }

    #[test]
    fn dtype_classification() {
        assert!(F32.is_float());
        assert!(!F32.is_int());

        assert!(I32.is_int());
        assert!(I32.is_signed_int());
        assert!(!I32.is_unsigned_int());
        assert!(!I32.is_float());

        assert!(U32.is_int());
        assert!(!U32.is_signed_int());
        assert!(U32.is_unsigned_int());

        assert!(BOOL.is_bool());
        assert!(!BOOL.is_float());
        assert!(!BOOL.is_int());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(format!("{}", F32), "f32");
        assert_eq!(format!("{}", I64), "i64");
        assert_eq!(format!("{}", BOOL), "bool");
    }

    #[test]
    fn dtype_from_primitives() {
        assert_eq!(DType::from(1.0f32), F32);
        assert_eq!(DType::from(1.0f64), F64);
        assert_eq!(DType::from(1i32), I32);
        assert_eq!(DType::from(1u64), U64);
        assert_eq!(DType::from(true), BOOL);
    }
}
