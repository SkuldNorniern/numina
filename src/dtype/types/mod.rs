//! Custom data type implementations

use std::fmt;
use crate::dtype::DTypeCandidate;

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
