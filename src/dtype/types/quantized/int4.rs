//! INT4 quantized data type implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

/// Quantized 4-bit signed integer stored in a single byte.
///
/// This type uses the low nibble (4 bits) to store a signed value in the range [-8, 7]
/// using a bias of +8. The high nibble is unused for single-value APIs, but `pack()` can
/// store two values (hi/lo) in one byte.
///
/// Quantization parameters (scale/zero-point) are treated as external metadata.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QuantizedI4(pub u8);

impl QuantizedI4 {
    #[inline]
    fn encode_nibble(value: i8) -> u8 {
        // Clamp to valid INT4 range (-8 to 7) and bias to unsigned [0, 15].
        ((value.clamp(-8, 7) + 8) as u8) & 0xF
    }

    #[inline]
    fn decode_nibble(nibble: u8) -> i8 {
        (nibble & 0xF) as i8 - 8
    }

    /// Create a single INT4 value encoded in the low nibble.
    pub fn from_i8(value: i8) -> Self {
        Self(Self::encode_nibble(value))
    }

    /// Decode the low-nibble value.
    pub fn to_i8(self) -> i8 {
        Self::decode_nibble(self.0)
    }

    /// Pack two INT4 values into one byte (hi nibble = `hi`, lo nibble = `lo`).
    pub fn pack(hi: i8, lo: i8) -> Self {
        let u_hi = Self::encode_nibble(hi);
        let u_lo = Self::encode_nibble(lo);
        Self((u_hi << 4) | u_lo)
    }

    /// Unpack the high nibble and decode it to a signed INT4 value in `[-8, 7]`.
    pub fn unpack_hi(self) -> i8 {
        Self::decode_nibble(self.0 >> 4)
    }

    /// Unpack the low nibble and decode it to a signed INT4 value in `[-8, 7]`.
    pub fn unpack_lo(self) -> i8 {
        Self::decode_nibble(self.0)
    }

    /// Return the raw packed storage byte (hi nibble + lo nibble).
    pub fn packed_byte(self) -> u8 {
        self.0
    }
}

impl DTypeCandidate for QuantizedI4 {
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
        true
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "quantized_i4"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "QuantizedI4 requires exactly 1 byte");
        Self(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
}

impl fmt::Display for QuantizedI4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.type_name(), self.to_i8())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_i4_single_value() {
        let original: i8 = 5;
        let quantized = QuantizedI4::from_i8(original);
        let decoded = quantized.to_i8();
        assert_eq!(decoded, original);
    }

    #[test]
    fn quantized_i4_packed_values() {
        let hi: i8 = 3;
        let lo: i8 = -2;
        let packed = QuantizedI4::pack(hi, lo);

        assert_eq!(packed.unpack_hi(), hi);
        assert_eq!(packed.unpack_lo(), lo);
    }

    #[test]
    fn quantized_i4_clamps_range() {
        let too_large = QuantizedI4::from_i8(10);
        assert_eq!(too_large.to_i8(), 7);

        let too_small = QuantizedI4::from_i8(-10);
        assert_eq!(too_small.to_i8(), -8);
    }
}
