//! Quantized data type implementations

use crate::dtype::DTypeCandidate;
use std::fmt;

/// Quantized 8-bit type with offset for asymmetric quantization
/// Note: This is a simplified example. Real quantization would need proper metadata handling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QuantizedU8 {
    value: u8,
    // For simplicity, we'll store scale as bits that can be compared
    scale_bits: u32, // IEEE 754 f32 bits for scale
}

impl QuantizedU8 {
    pub fn new(value: u8, _zero_point: u8, scale: f32) -> Self {
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
