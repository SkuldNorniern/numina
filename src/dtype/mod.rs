//! Data type definitions and implementations for arrays.
//!
//! Numina's dtype system has two layers:
//! - [`DType`]/[`DTypeId`]: small, stable identifiers for dispatch and serialization
//! - "concrete" types (e.g. [`Float16`], [`BFloat16`], [`Float8E4M3Fn`]) that define byte layout and
//!   conversion behavior
//!
//! ## Serialization
//! When converting values to/from bytes, Numina uses **little-endian** encoding.

use std::fmt;

// Core modules
pub mod conversions;
pub mod types;

// Re-exports for convenience
pub use types::{
    BFloat8, BFloat16, Complex32, Complex64, Complex128, Float8E4M3Fn, Float8E5M2, Float16,
    Float32, QuantizedI4, QuantizedU8,
};

/// Stable dtype identifier for Lamina/Laminax/Cetana serialization.
///
/// This is the value that should be stored in IR / runtime formats. The numeric values are part of
/// Numina's compatibility contract and must not be renumbered.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeId(pub u8);

impl DTypeId {
    /// Stable ID for [`DType::F16`] (`float16`).
    pub const F16: DTypeId = DTypeId(1);
    /// Stable ID for [`DType::F32`] (`float32`).
    pub const F32: DTypeId = DTypeId(2);
    /// Stable ID for [`DType::F64`] (`float64`).
    pub const F64: DTypeId = DTypeId(3);
    /// Stable ID for [`DType::BF16`] (`bfloat16`).
    pub const BF16: DTypeId = DTypeId(4);
    /// Stable ID for [`DType::BF8`] (`bfloat8`).
    pub const BF8: DTypeId = DTypeId(5);
    /// Stable ID for [`DType::F8E4M3FN`] (`float8_e4m3fn`).
    pub const F8E4M3FN: DTypeId = DTypeId(6);
    /// Stable ID for [`DType::F8E5M2`] (`float8_e5m2`).
    pub const F8E5M2: DTypeId = DTypeId(7);

    /// Stable ID for [`DType::Complex32`] (`complex32`).
    pub const COMPLEX32: DTypeId = DTypeId(50);
    /// Stable ID for [`DType::Complex64`] (`complex64`).
    pub const COMPLEX64: DTypeId = DTypeId(51);
    /// Stable ID for [`DType::Complex128`] (`complex128`).
    pub const COMPLEX128: DTypeId = DTypeId(52);

    /// Stable ID for [`DType::I8`] (`int8`).
    pub const I8: DTypeId = DTypeId(10);
    /// Stable ID for [`DType::I16`] (`int16`).
    pub const I16: DTypeId = DTypeId(11);
    /// Stable ID for [`DType::I32`] (`int32`).
    pub const I32: DTypeId = DTypeId(12);
    /// Stable ID for [`DType::I64`] (`int64`).
    pub const I64: DTypeId = DTypeId(13);

    /// Stable ID for [`DType::U8`] (`uint8`).
    pub const U8: DTypeId = DTypeId(20);
    /// Stable ID for [`DType::U16`] (`uint16`).
    pub const U16: DTypeId = DTypeId(21);
    /// Stable ID for [`DType::U32`] (`uint32`).
    pub const U32: DTypeId = DTypeId(22);
    /// Stable ID for [`DType::U64`] (`uint64`).
    pub const U64: DTypeId = DTypeId(23);

    /// Stable ID for [`DType::Bool`] (`bool`).
    pub const BOOL: DTypeId = DTypeId(30);

    /// Stable ID for [`DType::QI4`] (`qi4`).
    pub const QI4: DTypeId = DTypeId(40);
    /// Stable ID for [`DType::QU8`] (`qu8`).
    pub const QU8: DTypeId = DTypeId(41);
}

/// Static descriptor for dtype metadata.
///
/// Values are intended to be ABI-relevant (byte size, alignment, storage bits) and stable across
/// Numina versions for a given [`DTypeId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DTypeInfo {
    /// Stable dtype ID used for serialization.
    pub id: DTypeId,
    /// Human-readable name.
    pub name: &'static str,
    /// Logical size in bytes for one element.
    pub byte_size: usize,
    /// Number of storage bits used (e.g. `QI4` uses 4 bits per value but is stored in a byte).
    pub storage_bits: u16,
    /// Required alignment in bytes.
    pub align: usize,
    /// Whether the dtype should be treated as a "float-like" dtype for certain conversions.
    pub is_float: bool,
    /// Whether the dtype is an integer-like dtype.
    pub is_int: bool,
    /// Whether the dtype is boolean.
    pub is_bool: bool,
}

/// Trait for mapping concrete Rust types to Numina dtypes.
///
/// # Safety
/// Implementors must guarantee their in-memory representation matches the canonical layout for
/// `Self::DTYPE` (size, alignment, and byte encoding), since Numina may reinterpret raw bytes as the
/// corresponding primitive/type for that dtype.
pub unsafe trait DTypeLike: Copy {
    /// Static dtype descriptor for this Rust type
    const DTYPE: DType;
}

/// Trait for dtype-backed value serialization.
///
/// Implementations must write values using little-endian encoding where applicable.
pub trait DTypeValue: Copy {
    /// Static dtype descriptor for this Rust type.
    const DTYPE: DType;
    /// Append the canonical little-endian encoding of this value to `out`.
    fn write_bytes(self, out: &mut Vec<u8>);
}

/// Marker trait for types that can be used as `Array<T>` elements.
///
/// This is mainly a convenience bound for "Numina element" types used in generic APIs.
pub trait DTypeElement: DTypeLike + DTypeValue + Copy + Default + Send + Sync + 'static {}

impl<T> DTypeElement for T where T: DTypeLike + DTypeValue + Copy + Default + Send + Sync + 'static {}

/// Trait for types that can be used as dtype candidates.
///
/// This trait is implemented by both the enum [`DType`] and the concrete dtype wrapper types.
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

/// Trait for float-like dtype conversions.
///
/// This is used to implement `encode_float_bytes` / `decode_float_bytes` for custom float formats.
pub trait FloatDType: DTypeCandidate {
    /// Convert from an `f32` (possibly lossy).
    fn from_f32(value: f32) -> Self;
    /// Convert to `f32` (possibly lossy).
    fn to_f32(self) -> f32;
}

/// Data type enumeration for array elements.
///
/// Discriminants are explicit and match [`DTypeId`] values.
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

/// Returns `true` if `dtype` can be lossily converted to/from `f32` via
/// [`encode_float_bytes`] / [`decode_float_bytes`].
pub fn is_float_convertible(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F16
            | DType::F32
            | DType::F64
            | DType::BF16
            | DType::BF8
            | DType::F8E4M3FN
            | DType::F8E5M2
    )
}

fn decode_with<T: FloatDType>(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let element_size = std::mem::size_of::<T>();
    if element_size == 0 || !bytes.len().is_multiple_of(element_size) {
        return Err("invalid byte length for float dtype".to_string());
    }

    Ok(bytes
        .chunks_exact(element_size)
        .map(|chunk| unsafe { T::from_bytes(chunk) }.to_f32())
        .collect())
}

fn encode_with<T: FloatDType>(values: &[f32]) -> Vec<u8> {
    let element_size = std::mem::size_of::<T>();
    let mut bytes = Vec::with_capacity(values.len() * element_size);
    for value in values {
        bytes.extend_from_slice(&T::from_f32(*value).to_bytes());
    }
    bytes
}

/// Decode a byte buffer of `dtype`-encoded floating point values into `Vec<f32>`.
///
/// Bytes are interpreted as little-endian.
///
/// # Errors
/// Returns `Err` if the byte length is invalid for the dtype or if the dtype is unsupported.
pub fn decode_float_bytes(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        DType::F32 => {
            if !bytes.len().is_multiple_of(4) {
                return Err("invalid f32 byte length".to_string());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        }
        DType::F64 => {
            if !bytes.len().is_multiple_of(8) {
                return Err("invalid f64 byte length".to_string());
            }
            Ok(bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()) as f32)
                .collect())
        }
        DType::F16 => decode_with::<Float16>(bytes),
        DType::BF16 => decode_with::<BFloat16>(bytes),
        DType::BF8 => decode_with::<BFloat8>(bytes),
        DType::F8E4M3FN => decode_with::<Float8E4M3Fn>(bytes),
        DType::F8E5M2 => decode_with::<Float8E5M2>(bytes),
        _ => Err(format!("dtype {} is not supported", dtype)),
    }
}

/// Encode `values` into a byte buffer for a given float-like dtype.
///
/// Bytes are emitted in little-endian order.
///
/// # Errors
/// Returns `Err` if the dtype is unsupported.
pub fn encode_float_bytes(dtype: DType, values: &[f32]) -> Result<Vec<u8>, String> {
    match dtype {
        DType::F32 => {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(bytes)
        }
        DType::F64 => {
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for value in values {
                bytes.extend_from_slice(&(*value as f64).to_le_bytes());
            }
            Ok(bytes)
        }
        DType::F16 => Ok(encode_with::<Float16>(values)),
        DType::BF16 => Ok(encode_with::<BFloat16>(values)),
        DType::BF8 => Ok(encode_with::<BFloat8>(values)),
        DType::F8E4M3FN => Ok(encode_with::<Float8E4M3Fn>(values)),
        DType::F8E5M2 => Ok(encode_with::<Float8E5M2>(values)),
        _ => Err(format!("dtype {} is not supported", dtype)),
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.type_name())
    }
}

// DTypeLike implementations for primitive Rust types
unsafe impl DTypeLike for f32 {
    const DTYPE: DType = DType::F32;
}

unsafe impl DTypeLike for f64 {
    const DTYPE: DType = DType::F64;
}

impl DTypeValue for f32 {
    const DTYPE: DType = DType::F32;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for f64 {
    const DTYPE: DType = DType::F64;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for i8 {
    const DTYPE: DType = DType::I8;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.push(self as u8);
    }
}

impl DTypeValue for i16 {
    const DTYPE: DType = DType::I16;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for i32 {
    const DTYPE: DType = DType::I32;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for i64 {
    const DTYPE: DType = DType::I64;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for u8 {
    const DTYPE: DType = DType::U8;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.push(self);
    }
}

impl DTypeValue for u16 {
    const DTYPE: DType = DType::U16;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for u32 {
    const DTYPE: DType = DType::U32;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for u64 {
    const DTYPE: DType = DType::U64;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

impl DTypeValue for bool {
    const DTYPE: DType = DType::Bool;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
}

impl<T> DTypeValue for T
where
    T: DTypeCandidate + DTypeLike,
{
    const DTYPE: DType = T::DTYPE;

    fn write_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bytes());
    }
}

unsafe impl DTypeLike for i8 {
    const DTYPE: DType = DType::I8;
}

unsafe impl DTypeLike for i16 {
    const DTYPE: DType = DType::I16;
}

unsafe impl DTypeLike for i32 {
    const DTYPE: DType = DType::I32;
}

unsafe impl DTypeLike for i64 {
    const DTYPE: DType = DType::I64;
}

unsafe impl DTypeLike for u8 {
    const DTYPE: DType = DType::U8;
}

unsafe impl DTypeLike for u16 {
    const DTYPE: DType = DType::U16;
}

unsafe impl DTypeLike for u32 {
    const DTYPE: DType = DType::U32;
}

unsafe impl DTypeLike for u64 {
    const DTYPE: DType = DType::U64;
}

unsafe impl DTypeLike for bool {
    const DTYPE: DType = DType::Bool;
}

unsafe impl DTypeLike for BFloat16 {
    const DTYPE: DType = DType::BF16;
}

unsafe impl DTypeLike for BFloat8 {
    const DTYPE: DType = DType::BF8;
}

unsafe impl DTypeLike for Float16 {
    const DTYPE: DType = DType::F16;
}

unsafe impl DTypeLike for Float32 {
    const DTYPE: DType = DType::F32;
}

unsafe impl DTypeLike for Float8E4M3Fn {
    const DTYPE: DType = DType::F8E4M3FN;
}

unsafe impl DTypeLike for Float8E5M2 {
    const DTYPE: DType = DType::F8E5M2;
}

unsafe impl DTypeLike for Complex32 {
    const DTYPE: DType = DType::Complex32;
}

unsafe impl DTypeLike for Complex64 {
    const DTYPE: DType = DType::Complex64;
}

unsafe impl DTypeLike for Complex128 {
    const DTYPE: DType = DType::Complex128;
}

unsafe impl DTypeLike for QuantizedI4 {
    const DTYPE: DType = DType::QI4;
}

unsafe impl DTypeLike for QuantizedU8 {
    const DTYPE: DType = DType::QU8;
}

// Convenience constants
/// Alias for [`DType::F32`].
pub const F32: DType = DType::F32;
/// Alias for [`DType::F64`].
pub const F64: DType = DType::F64;
/// Alias for [`DType::F16`].
pub const F16: DType = DType::F16;
/// Alias for [`DType::F8E4M3FN`].
pub const F8E4M3FN: DType = DType::F8E4M3FN;
/// Alias for [`DType::F8E5M2`].
pub const F8E5M2: DType = DType::F8E5M2;
/// Alias for [`DType::F16`].
pub const FLOAT16: DType = DType::F16;
/// Alias for [`DType::F32`].
pub const FLOAT32: DType = DType::F32;
/// Alias for [`DType::F64`].
pub const FLOAT64: DType = DType::F64;
/// Alias for [`DType::F8E4M3FN`].
pub const FLOAT8_E4M3FN: DType = DType::F8E4M3FN;
/// Alias for [`DType::F8E5M2`].
pub const FLOAT8_E5M2: DType = DType::F8E5M2;
/// Alias for [`DType::I8`].
pub const I8: DType = DType::I8;
/// Alias for [`DType::I8`].
pub const INT8: DType = DType::I8;
/// Alias for [`DType::I16`].
pub const I16: DType = DType::I16;
/// Alias for [`DType::I16`].
pub const INT16: DType = DType::I16;
/// Alias for [`DType::I32`].
pub const I32: DType = DType::I32;
/// Alias for [`DType::I32`].
pub const INT32: DType = DType::I32;
/// Alias for [`DType::I64`].
pub const I64: DType = DType::I64;
/// Alias for [`DType::I64`].
pub const INT64: DType = DType::I64;
/// Alias for [`DType::U8`].
pub const U8: DType = DType::U8;
/// Alias for [`DType::U8`].
pub const UINT8: DType = DType::U8;
/// Alias for [`DType::U16`].
pub const U16: DType = DType::U16;
/// Alias for [`DType::U16`].
pub const UINT16: DType = DType::U16;
/// Alias for [`DType::U32`].
pub const U32: DType = DType::U32;
/// Alias for [`DType::U32`].
pub const UINT32: DType = DType::U32;
/// Alias for [`DType::U64`].
pub const U64: DType = DType::U64;
/// Alias for [`DType::U64`].
pub const UINT64: DType = DType::U64;
/// Alias for [`DType::Bool`].
pub const BOOL: DType = DType::Bool;
/// Alias for [`DType::BF16`].
pub const BF16: DType = DType::BF16;
/// Alias for [`DType::BF16`].
pub const BFLOAT16: DType = DType::BF16;
/// Alias for [`DType::BF8`].
pub const BF8: DType = DType::BF8;
/// Alias for [`DType::BF8`].
pub const BFLOAT8: DType = DType::BF8;
/// Alias for [`DType::Complex32`].
pub const COMPLEX32: DType = DType::Complex32;
/// Alias for [`DType::Complex64`].
pub const COMPLEX64: DType = DType::Complex64;
/// Alias for [`DType::Complex128`].
pub const COMPLEX128: DType = DType::Complex128;
/// Alias for [`DType::QI4`].
pub const QI4: DType = DType::QI4;
/// Alias for [`DType::QU8`].
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
