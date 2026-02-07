//! Quantized u8 data type implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

/// Quantized 8-bit unsigned integer.
///
/// This type is a 1-byte wrapper to match Numina's canonical QU8 byte layout.
/// Quantization parameters (scale/zero-point) are treated as external metadata.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QuantizedU8(pub u8);

impl QuantizedU8 {
    /// Wrap a raw quantized `u8` storage value.
    pub fn from_raw(value: u8) -> Self {
        Self(value)
    }

    /// Return the underlying raw `u8` storage value.
    pub fn raw(self) -> u8 {
        self.0
    }

    /// Quantize a `f32` value using `scale` (no zero-point).
    ///
    /// Values are rounded to the nearest integer and clamped to `[0, 255]`.
    pub fn quantize(value: f32, scale: f32) -> Self {
        // Simplified example (no zero-point).
        let q = (value / scale).round().clamp(0.0, u8::MAX as f32) as u8;
        Self(q)
    }

    /// Dequantize this value using `scale` (no zero-point).
    pub fn dequantize(self, scale: f32) -> f32 {
        (self.0 as f32) * scale
    }
}

impl DTypeCandidate for QuantizedU8 {
    fn size_bytes(&self) -> usize {
        1
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
        "quantized_u8"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "QuantizedU8 requires exactly 1 byte");
        Self(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
}

impl fmt::Display for QuantizedU8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.type_name(), self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_u8_conversion() {
        let original = 2.5f32;
        let scale = 0.01f32;
        let quantized = QuantizedU8::quantize(original, scale);
        let dequantized = quantized.dequantize(scale);

        // Should be close to original (within quantization error)
        assert!((dequantized - original).abs() < 0.1);

        // Test byte conversion
        let bytes = quantized.to_bytes();
        let reconstructed = unsafe { QuantizedU8::from_bytes(&bytes) };
        assert_eq!(quantized, reconstructed);
    }

    #[test]
    fn quantized_u8_dtype_candidate() {
        let qu8 = QuantizedU8::quantize(1.0, 0.1);
        assert_eq!(qu8.size_bytes(), 1);
        assert!(!qu8.is_float());
        assert!(qu8.is_int());
        assert!(!qu8.is_signed());
        assert!(!qu8.is_bool());
        assert_eq!(qu8.type_name(), "quantized_u8");
    }
}
