//! Data type definitions and implementations for tensors

use std::fmt;

// Core modules
pub mod conversions;
pub mod types;

// Re-exports for convenience
pub use types::{BFloat16, QuantizedU8};

/// Trait for mapping concrete Rust types to Numina dtypes
pub trait DTypeLike: Copy {
    /// Static dtype descriptor for this Rust type
    const DTYPE: DType;
}

/// Trait for types that can be used as tensor data types
pub trait DTypeCandidate: Copy + Clone + PartialEq + Eq + std::hash::Hash {
    /// Returns the size in bytes of this data type
    fn size_bytes(&self) -> usize;

    /// Returns true if this is a floating point type
    fn is_float(&self) -> bool;

    /// Returns true if this is an integer type
    fn is_int(&self) -> bool;

    /// Returns true if this is a signed integer type
    fn is_signed_int(&self) -> bool {
        self.is_int() && self.is_signed()
    }

    /// Returns true if this is an unsigned integer type
    fn is_unsigned_int(&self) -> bool {
        self.is_int() && !self.is_signed()
    }

    /// Returns true if this is a signed type (for integers)
    fn is_signed(&self) -> bool;

    /// Returns true if this is a boolean type
    fn is_bool(&self) -> bool;

    /// Returns a string representation of the type
    fn type_name(&self) -> &'static str;

    /// Convert from raw bytes (used internally)
    /// # Safety
    /// The caller must ensure the bytes are valid for this type
    unsafe fn from_bytes(bytes: &[u8]) -> Self;

    /// Convert to raw bytes (used internally)
    fn to_bytes(&self) -> Vec<u8>;
}

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

impl DTypeCandidate for DType {
    fn size_bytes(&self) -> usize {
        self.dtype_size_bytes()
    }

    fn is_float(&self) -> bool {
        self.is_float()
    }

    fn is_int(&self) -> bool {
        self.is_int()
    }

    fn is_signed(&self) -> bool {
        self.is_signed()
    }

    fn is_bool(&self) -> bool {
        self.is_bool()
    }

    fn type_name(&self) -> &'static str {
        self.type_name()
    }

    unsafe fn from_bytes(_bytes: &[u8]) -> Self {
        panic!("Cannot convert bytes to DType enum directly - use concrete types instead")
    }

    fn to_bytes(&self) -> Vec<u8> {
        // Convert the discriminant to bytes
        vec![*self as u8]
    }
}

// Instance methods that delegate to the enum variants
impl DType {
    /// Returns the size in bytes of this data type
    pub fn dtype_size_bytes(&self) -> usize {
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
        matches!(
            self,
            DType::I8
                | DType::I16
                | DType::I32
                | DType::I64
                | DType::U8
                | DType::U16
                | DType::U32
                | DType::U64
        )
    }

    /// Returns true if this is a signed integer type
    pub fn is_signed_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64)
    }

    /// Returns true if this is an unsigned integer type
    pub fn is_unsigned_int(&self) -> bool {
        matches!(self, DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Returns true if this is a signed type (for integers)
    pub fn is_signed(&self) -> bool {
        self.is_signed_int()
    }

    /// Returns true if this is a boolean type
    pub fn is_bool(&self) -> bool {
        matches!(self, DType::Bool)
    }

    /// Returns a string representation of the type
    pub fn type_name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::F16 => "f16",
            DType::I8 => "i8",
            DType::I16 => "i16",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::U8 => "u8",
            DType::U16 => "u16",
            DType::U32 => "u32",
            DType::U64 => "u64",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.type_name())
    }
}

// DTypeLike implementations for primitive Rust types
impl DTypeLike for f32 {
    const DTYPE: DType = DType::F32;
}

impl DTypeLike for f64 {
    const DTYPE: DType = DType::F64;
}

impl DTypeLike for i8 {
    const DTYPE: DType = DType::I8;
}

impl DTypeLike for i16 {
    const DTYPE: DType = DType::I16;
}

impl DTypeLike for i32 {
    const DTYPE: DType = DType::I32;
}

impl DTypeLike for i64 {
    const DTYPE: DType = DType::I64;
}

impl DTypeLike for u8 {
    const DTYPE: DType = DType::U8;
}

impl DTypeLike for u16 {
    const DTYPE: DType = DType::U16;
}

impl DTypeLike for u32 {
    const DTYPE: DType = DType::U32;
}

impl DTypeLike for u64 {
    const DTYPE: DType = DType::U64;
}

impl DTypeLike for bool {
    const DTYPE: DType = DType::Bool;
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

// Constants are already defined above, no need to re-export

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
}
