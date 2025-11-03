//! Data type definitions for tensors

use std::fmt;

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

// Example implementation: BFloat16 (Brain Float 16-bit)
/// Brain Float 16-bit floating point type
/// Uses the same 16 bits as IEEE half-precision but with different exponent/manitssa split
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BFloat16(u16);

impl BFloat16 {
    /// Create a BFloat16 from f32, truncating the mantissa
    pub fn from_f32(value: f32) -> Self {
        // BFloat16: 1 sign bit, 8 exponent bits, 7 mantissa bits
        // Take the top 16 bits of f32 (sign + 8 exp + 7 mantissa)
        let bits = value.to_bits();
        BFloat16((bits >> 16) as u16)
    }

    /// Convert to f32 (lossless for the precision we store)
    pub fn to_f32(self) -> f32 {
        // Reconstruct f32 from BFloat16 bits
        let bits = (self.0 as u32) << 16;
        f32::from_bits(bits)
    }
}

impl DTypeCandidate for BFloat16 {
    fn size_bytes(&self) -> usize {
        2
    }

    fn is_float(&self) -> bool {
        true
    }

    fn is_int(&self) -> bool {
        false
    }

    fn is_signed(&self) -> bool {
        true // Floating point types are signed
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "bf16"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 2, "BFloat16 requires exactly 2 bytes");
        let value = u16::from_le_bytes([bytes[0], bytes[1]]);
        BFloat16(value)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl fmt::Display for BFloat16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

// Example implementation: Custom quantized type (8-bit with offset)
/// Quantized 8-bit type with offset for asymmetric quantization
/// Note: This is a simplified example. Real quantization would need proper metadata handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizedU8 {
    value: u8,
    // For simplicity, we'll store scale as bits that can be compared
    scale_bits: u32, // IEEE 754 f32 bits for scale
}

impl QuantizedU8 {
    pub fn new(value: u8, zero_point: u8, scale: f32) -> Self {
        QuantizedU8 {
            value,
            scale_bits: scale.to_bits(),
        }
    }

    pub fn dequantize(&self) -> f32 {
        // For this example, we'll assume zero_point = 0
        let scale = f32::from_bits(self.scale_bits);
        (self.value as f32) * scale
    }

    pub fn quantize(value: f32, scale: f32) -> Self {
        let quantized = (value / scale).round() as u8;
        QuantizedU8::new(quantized, 0, scale)
    }

    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

impl DTypeCandidate for QuantizedU8 {
    fn size_bytes(&self) -> usize {
        1 // Just the u8 value, zero_point and scale are metadata
    }

    fn is_float(&self) -> bool {
        false
    }

    fn is_int(&self) -> bool {
        true
    }

    fn is_signed(&self) -> bool {
        false
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "qu8"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "QuantizedU8 requires exactly 1 byte");
        // Note: This loses zero_point and scale information
        // In practice, you'd need a way to store/retrieve quantization parameters
        QuantizedU8::new(bytes[0], 0, 1.0)
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.value]
    }
}

impl fmt::Display for QuantizedU8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.type_name(), self.dequantize())
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

    #[test]
    fn bfloat16_conversion() {
        let original = 3.14159f32;
        let bf16 = BFloat16::from_f32(original);
        let back_to_f32 = bf16.to_f32();

        // BFloat16 has less precision, so it won't be exactly equal
        assert!((back_to_f32 - original).abs() < 0.01); // Should be reasonably close

        // Test byte conversion
        let bytes = bf16.to_bytes();
        let reconstructed = unsafe { BFloat16::from_bytes(&bytes) };
        assert_eq!(bf16, reconstructed);
    }

    #[test]
    fn bfloat16_dtype_candidate() {
        let bf16 = BFloat16::from_f32(1.0);
        assert_eq!(bf16.size_bytes(), 2);
        assert!(bf16.is_float());
        assert!(!bf16.is_int());
        assert!(bf16.is_signed());
        assert!(!bf16.is_bool());
        assert_eq!(bf16.type_name(), "bf16");
    }

    #[test]
    fn quantized_u8_conversion() {
        let original = 2.5f32;
        let quantized = QuantizedU8::quantize(original, 0.01);
        let dequantized = quantized.dequantize();

        // Should be close to original (within quantization error)
        assert!((dequantized - original).abs() < 0.1);

        // Test byte conversion
        let bytes = quantized.to_bytes();
        let reconstructed = unsafe { QuantizedU8::from_bytes(&bytes) };
        // Note: reconstructed will have default scale=1.0
        // In practice, you'd need a way to store/retrieve quantization parameters
        assert_eq!(quantized.value, reconstructed.value);
    }

    #[test]
    fn quantized_u8_dtype_candidate() {
        let qu8 = QuantizedU8::quantize(1.0, 0.1);
        assert_eq!(qu8.size_bytes(), 1);
        assert!(!qu8.is_float());
        assert!(qu8.is_int());
        assert!(!qu8.is_signed());
        assert!(!qu8.is_bool());
        assert_eq!(qu8.type_name(), "qu8");
    }
}
