//! Data type definitions and implementations for tensors

use std::fmt;

// Core modules
pub mod conversions;
pub mod types;

// Re-exports for convenience
pub use types::{
    BFloat8, BFloat16, Complex32, Complex64, Complex128, Float8E4M3Fn, Float8E5M2, Float16,
    QuantizedI4, QuantizedU8,
};

/// Stable DType identifier for Lamina/Laminax/Cetana serialization
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeId(pub u8);

impl DTypeId {
    pub const F16: DTypeId = DTypeId(1);
    pub const F32: DTypeId = DTypeId(2);
    pub const F64: DTypeId = DTypeId(3);
    pub const BF16: DTypeId = DTypeId(4);
    pub const BF8: DTypeId = DTypeId(5);
    pub const F8E4M3FN: DTypeId = DTypeId(6);
    pub const F8E5M2: DTypeId = DTypeId(7);
    pub const COMPLEX32: DTypeId = DTypeId(50);
    pub const COMPLEX64: DTypeId = DTypeId(51);
    pub const COMPLEX128: DTypeId = DTypeId(52);
    pub const I8: DTypeId = DTypeId(10);
    pub const I16: DTypeId = DTypeId(11);
    pub const I32: DTypeId = DTypeId(12);
    pub const I64: DTypeId = DTypeId(13);
    pub const U8: DTypeId = DTypeId(20);
    pub const U16: DTypeId = DTypeId(21);
    pub const U32: DTypeId = DTypeId(22);
    pub const U64: DTypeId = DTypeId(23);
    pub const BOOL: DTypeId = DTypeId(30);
    pub const QI4: DTypeId = DTypeId(40);
    pub const QU8: DTypeId = DTypeId(41);
}

/// Static descriptor for dtype metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DTypeInfo {
    pub id: DTypeId,
    pub name: &'static str,
    pub byte_size: usize,
    pub storage_bits: u16,
    pub align: usize,
    pub is_float: bool,
    pub is_int: bool,
    pub is_bool: bool,
}

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
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit floating point
    F16 = 1,
    /// 32-bit floating point
    F32 = 2,
    /// 64-bit floating point
    F64 = 3,
    /// Brain Float 16-bit
    BF16 = 4,
    /// Brain Float 8-bit
    BF8 = 5,
    /// Float8 E4M3FN
    F8E4M3FN = 6,
    /// Float8 E5M2
    F8E5M2 = 7,
    /// Complex with float16 components
    Complex32 = 50,
    /// Complex with float32 components
    Complex64 = 51,
    /// Complex with float64 components
    Complex128 = 52,
    /// 8-bit signed integer
    I8 = 10,
    /// 16-bit signed integer
    I16 = 11,
    /// 32-bit signed integer
    I32 = 12,
    /// 64-bit signed integer
    I64 = 13,
    /// 8-bit unsigned integer
    U8 = 20,
    /// 16-bit unsigned integer
    U16 = 21,
    /// 32-bit unsigned integer
    U32 = 22,
    /// 64-bit unsigned integer
    U64 = 23,
    /// Boolean
    Bool = 30,
    /// Quantized 4-bit signed integer
    QI4 = 40,
    /// Quantized 8-bit unsigned integer
    QU8 = 41,
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
        vec![self.id().0]
    }
}

// Instance methods that delegate to the enum variants
impl DType {
    /// Returns the size in bytes of this data type
    pub fn dtype_size_bytes(&self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::BF16 => 2,
            DType::BF8 => 1,
            DType::F8E4M3FN => 1,
            DType::F8E5M2 => 1,
            DType::Complex32 => 4,
            DType::Complex64 => 8,
            DType::Complex128 => 16,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Bool => 1,
            DType::QI4 => 1, // 4 bits per value, but allocated per byte
            DType::QU8 => 1,
        }
    }

    /// Returns the storage size in bits for this dtype
    pub fn storage_bits(&self) -> u16 {
        match self {
            DType::QI4 => 4,
            _ => (self.dtype_size_bytes() * 8) as u16,
        }
    }

    /// Returns the stable dtype id
    pub fn id(&self) -> DTypeId {
        DTypeId(*self as u8)
    }

    /// Convert from a stable dtype id
    pub fn from_id(id: DTypeId) -> Option<Self> {
        match id.0 {
            1 => Some(DType::F16),
            2 => Some(DType::F32),
            3 => Some(DType::F64),
            4 => Some(DType::BF16),
            5 => Some(DType::BF8),
            6 => Some(DType::F8E4M3FN),
            7 => Some(DType::F8E5M2),
            50 => Some(DType::Complex32),
            51 => Some(DType::Complex64),
            52 => Some(DType::Complex128),
            10 => Some(DType::I8),
            11 => Some(DType::I16),
            12 => Some(DType::I32),
            13 => Some(DType::I64),
            20 => Some(DType::U8),
            21 => Some(DType::U16),
            22 => Some(DType::U32),
            23 => Some(DType::U64),
            30 => Some(DType::Bool),
            40 => Some(DType::QI4),
            41 => Some(DType::QU8),
            _ => None,
        }
    }

    /// Returns a static descriptor for this dtype
    pub fn info(&self) -> DTypeInfo {
        let (name, align) = match self {
            DType::F16 => ("float16", 2),
            DType::F32 => ("float32", 4),
            DType::F64 => ("float64", 8),
            DType::BF16 => ("bfloat16", 2),
            DType::BF8 => ("bfloat8", 1),
            DType::F8E4M3FN => ("float8_e4m3fn", 1),
            DType::F8E5M2 => ("float8_e5m2", 1),
            DType::Complex32 => ("complex32", 2),
            DType::Complex64 => ("complex64", 4),
            DType::Complex128 => ("complex128", 8),
            DType::I8 => ("int8", 1),
            DType::I16 => ("int16", 2),
            DType::I32 => ("int32", 4),
            DType::I64 => ("int64", 8),
            DType::U8 => ("uint8", 1),
            DType::U16 => ("uint16", 2),
            DType::U32 => ("uint32", 4),
            DType::U64 => ("uint64", 8),
            DType::Bool => ("bool", 1),
            DType::QI4 => ("quantized_i4", 1),
            DType::QU8 => ("quantized_u8", 1),
        };

        DTypeInfo {
            id: self.id(),
            name,
            byte_size: self.dtype_size_bytes(),
            storage_bits: self.storage_bits(),
            align,
            is_float: self.is_float(),
            is_int: self.is_int(),
            is_bool: self.is_bool(),
        }
    }

    /// Returns true if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DType::F16
                | DType::F32
                | DType::F64
                | DType::BF16
                | DType::BF8
                | DType::F8E4M3FN
                | DType::F8E5M2
                | DType::Complex32
                | DType::Complex64
                | DType::Complex128
        )
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
                | DType::QI4
                | DType::QU8
        )
    }

    /// Returns true if this is a signed integer type
    pub fn is_signed_int(&self) -> bool {
        matches!(
            self,
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::QI4
        )
    }

    /// Returns true if this is an unsigned integer type
    pub fn is_unsigned_int(&self) -> bool {
        matches!(
            self,
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::QU8
        )
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
            DType::F16 => "float16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::BF16 => "bfloat16",
            DType::BF8 => "bfloat8",
            DType::F8E4M3FN => "float8_e4m3fn",
            DType::F8E5M2 => "float8_e5m2",
            DType::Complex32 => "complex32",
            DType::Complex64 => "complex64",
            DType::Complex128 => "complex128",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::Bool => "bool",
            DType::QI4 => "quantized_i4",
            DType::QU8 => "quantized_u8",
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

impl DTypeLike for BFloat16 {
    const DTYPE: DType = DType::BF16;
}

impl DTypeLike for BFloat8 {
    const DTYPE: DType = DType::BF8;
}

impl DTypeLike for Float16 {
    const DTYPE: DType = DType::F16;
}

impl DTypeLike for Float8E4M3Fn {
    const DTYPE: DType = DType::F8E4M3FN;
}

impl DTypeLike for Float8E5M2 {
    const DTYPE: DType = DType::F8E5M2;
}

impl DTypeLike for Complex32 {
    const DTYPE: DType = DType::Complex32;
}

impl DTypeLike for Complex64 {
    const DTYPE: DType = DType::Complex64;
}

impl DTypeLike for Complex128 {
    const DTYPE: DType = DType::Complex128;
}

impl DTypeLike for QuantizedI4 {
    const DTYPE: DType = DType::QI4;
}

impl DTypeLike for QuantizedU8 {
    const DTYPE: DType = DType::QU8;
}

// Convenience constants
pub const F32: DType = DType::F32;
pub const F64: DType = DType::F64;
pub const F16: DType = DType::F16;
pub const F8E4M3FN: DType = DType::F8E4M3FN;
pub const F8E5M2: DType = DType::F8E5M2;
pub const FLOAT16: DType = DType::F16;
pub const FLOAT32: DType = DType::F32;
pub const FLOAT64: DType = DType::F64;
pub const FLOAT8_E4M3FN: DType = DType::F8E4M3FN;
pub const FLOAT8_E5M2: DType = DType::F8E5M2;
pub const I8: DType = DType::I8;
pub const INT8: DType = DType::I8;
pub const I16: DType = DType::I16;
pub const INT16: DType = DType::I16;
pub const I32: DType = DType::I32;
pub const INT32: DType = DType::I32;
pub const I64: DType = DType::I64;
pub const INT64: DType = DType::I64;
pub const U8: DType = DType::U8;
pub const UINT8: DType = DType::U8;
pub const U16: DType = DType::U16;
pub const UINT16: DType = DType::U16;
pub const U32: DType = DType::U32;
pub const UINT32: DType = DType::U32;
pub const U64: DType = DType::U64;
pub const UINT64: DType = DType::U64;
pub const BOOL: DType = DType::Bool;
pub const BF16: DType = DType::BF16;
pub const BFLOAT16: DType = DType::BF16;
pub const BF8: DType = DType::BF8;
pub const BFLOAT8: DType = DType::BF8;
pub const COMPLEX32: DType = DType::Complex32;
pub const COMPLEX64: DType = DType::Complex64;
pub const COMPLEX128: DType = DType::Complex128;
pub const QI4: DType = DType::QI4;
pub const QU8: DType = DType::QU8;

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
        assert_eq!(format!("{}", F32), "float32");
        assert_eq!(format!("{}", I64), "int64");
        assert_eq!(format!("{}", BOOL), "bool");
    }

    #[test]
    fn dtype_info_table() {
        let table = [
            (DType::F16, 1, "float16", 2usize, 16u16, 2usize),
            (DType::F32, 2, "float32", 4, 32, 4),
            (DType::F64, 3, "float64", 8, 64, 8),
            (DType::BF16, 4, "bfloat16", 2, 16, 2),
            (DType::BF8, 5, "bfloat8", 1, 8, 1),
            (DType::F8E4M3FN, 6, "float8_e4m3fn", 1, 8, 1),
            (DType::F8E5M2, 7, "float8_e5m2", 1, 8, 1),
            (DType::Complex32, 50, "complex32", 4, 32, 2),
            (DType::Complex64, 51, "complex64", 8, 64, 4),
            (DType::Complex128, 52, "complex128", 16, 128, 8),
            (DType::I8, 10, "int8", 1, 8, 1),
            (DType::I16, 11, "int16", 2, 16, 2),
            (DType::I32, 12, "int32", 4, 32, 4),
            (DType::I64, 13, "int64", 8, 64, 8),
            (DType::U8, 20, "uint8", 1, 8, 1),
            (DType::U16, 21, "uint16", 2, 16, 2),
            (DType::U32, 22, "uint32", 4, 32, 4),
            (DType::U64, 23, "uint64", 8, 64, 8),
            (DType::Bool, 30, "bool", 1, 8, 1),
            (DType::QI4, 40, "quantized_i4", 1, 4, 1),
            (DType::QU8, 41, "quantized_u8", 1, 8, 1),
        ];

        for (dtype, id, name, bytes, bits, align) in table {
            let info = dtype.info();
            assert_eq!(info.id.0, id);
            assert_eq!(info.name, name);
            assert_eq!(info.byte_size, bytes);
            assert_eq!(info.storage_bits, bits);
            assert_eq!(info.align, align);
            assert_eq!(DType::from_id(info.id), Some(dtype));
        }
    }
}
