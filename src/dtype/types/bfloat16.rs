//! BFloat16 (Brain Float 16-bit) implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

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
}
