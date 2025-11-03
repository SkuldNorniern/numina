//! INT4 quantized data type implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

/// Quantized 4-bit signed integer type
/// Stores two INT4 values per byte for memory efficiency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QuantizedI4 {
    // Two 4-bit values packed into one byte
    // High 4 bits: first value, Low 4 bits: second value
    packed_values: u8,
    // Store scale as IEEE 754 bits for Hash/Eq compatibility
    scale_bits: u32,
    // Index to determine which 4-bit value to use (0 or 1)
    index: u8,
}

impl QuantizedI4 {
    /// Create a new QuantizedI4 from a single i8 value
    pub fn from_i8(value: i8, scale: f32) -> Self {
        // Clamp to valid INT4 range (-8 to 7)
        let clamped = value.clamp(-8, 7);
        // Convert to unsigned representation (0-15) and store in high bits
        let unsigned = (clamped + 8) as u8;
        QuantizedI4 {
            packed_values: unsigned << 4, // Store in high bits, low bits unused for single value
            scale_bits: scale.to_bits(),
            index: 0,
        }
    }

    /// Pack two INT4 values into one byte
    pub fn pack(val1: i8, val2: i8, scale: f32) -> Self {
        let u1 = ((val1.clamp(-8, 7) + 8) as u8) & 0xF;
        let u2 = ((val2.clamp(-8, 7) + 8) as u8) & 0xF;
        QuantizedI4 {
            packed_values: (u1 << 4) | u2,
            scale_bits: scale.to_bits(),
            index: 0, // Not used when packed
        }
    }

    /// Dequantize back to i8
    pub fn dequantize(&self) -> i8 {
        let unsigned = if self.index == 0 {
            self.packed_values >> 4 // High 4 bits
        } else {
            self.packed_values & 0xF // Low 4 bits
        };
        // Convert back to signed (-8 to 7)
        let signed = (unsigned as i8) - 8;
        let scale = f32::from_bits(self.scale_bits);
        (signed as f32 * scale) as i8
    }

    /// Get the scale factor
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }

    /// Get the raw packed byte value
    pub fn packed_byte(&self) -> u8 {
        self.packed_values
    }
}

impl DTypeCandidate for QuantizedI4 {
    fn size_bytes(&self) -> usize {
        1 // 4 bits per value, but we allocate per byte for simplicity
    }

    fn is_float(&self) -> bool {
        false
    }

    fn is_int(&self) -> bool {
        true
    }

    fn is_signed(&self) -> bool {
        true
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "qi4"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "QuantizedI4 requires exactly 1 byte");
        QuantizedI4 {
            packed_values: bytes[0],
            scale_bits: 1.0f32.to_bits(), // Default scale
            index: 0,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.packed_values]
    }
}

impl fmt::Display for QuantizedI4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.type_name(), self.dequantize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_i4_single_value() {
        let original: i8 = 5;
        let quantized = QuantizedI4::from_i8(original, 1.0);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized, original);
    }

    #[test]
    fn quantized_i4_packed_values() {
        let val1: i8 = 3;
        let val2: i8 = -2;
        let packed = QuantizedI4::pack(val1, val2, 1.0);

        // Test first value
        let q1 = QuantizedI4 {
            packed_values: packed.packed_values,
            scale_bits: packed.scale_bits,
            index: 0,
        };
        assert_eq!(q1.dequantize(), val1);

        // Test second value
        let q2 = QuantizedI4 {
            packed_values: packed.packed_values,
            scale_bits: packed.scale_bits,
            index: 1,
        };
        assert_eq!(q2.dequantize(), val2);
    }

    #[test]
    fn quantized_i4_dtype_candidate() {
        let qi4 = QuantizedI4::from_i8(2, 0.5);
        assert_eq!(qi4.size_bytes(), 1);
        assert!(!qi4.is_float());
        assert!(qi4.is_int());
        assert!(qi4.is_signed());
        assert!(!qi4.is_bool());
        assert_eq!(qi4.type_name(), "qi4");
    }

    #[test]
    fn quantized_i4_clamping() {
        // Test clamping to valid INT4 range
        let too_large = QuantizedI4::from_i8(10, 1.0);
        assert_eq!(too_large.dequantize(), 7); // Clamped to max

        let too_small = QuantizedI4::from_i8(-10, 1.0);
        assert_eq!(too_small.dequantize(), -8); // Clamped to min
    }
}
