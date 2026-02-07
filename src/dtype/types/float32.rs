//! Float32 (IEEE 754 single-precision) implementation.
//!
//! This is a thin wrapper around the raw IEEE-754 bit pattern. It is byte-for-byte equivalent to
//! `f32` and uses little-endian encoding for serialization.

use crate::dtype::{DTypeCandidate, FloatDType};
use std::fmt;

/// IEEE-754 single-precision floating point stored as raw bits.
///
/// Layout: `#[repr(transparent)]` over `u32` holding the raw IEEE-754 bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Float32(u32);

impl Float32 {
    /// Create a Float32 from f32
    pub fn from_f32(value: f32) -> Self {
        Float32(value.to_bits())
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        f32::from_bits(self.0)
    }

    /// Create a Float32 from raw IEEE-754 bits.
    pub fn from_bits(bits: u32) -> Self {
        Float32(bits)
    }

    /// Return the raw IEEE-754 bits.
    pub fn to_bits(self) -> u32 {
        self.0
    }
}

impl From<f32> for Float32 {
    fn from(value: f32) -> Self {
        Float32::from_f32(value)
    }
}

impl From<Float32> for f32 {
    fn from(value: Float32) -> Self {
        value.to_f32()
    }
}

impl DTypeCandidate for Float32 {
    fn size_bytes(&self) -> usize {
        4
    }

    fn is_float(&self) -> bool {
        true
    }

    fn is_int(&self) -> bool {
        false
    }

    fn is_signed(&self) -> bool {
        true
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "float32"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 4, "Float32 requires exactly 4 bytes");
        let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Float32(value)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl FloatDType for Float32 {
    fn from_f32(value: f32) -> Self {
        Float32::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl fmt::Display for Float32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float32_conversions() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        for &v in &values {
            let f32w = Float32::from_f32(v);
            let back = f32::from(f32w);
            assert_eq!(back, v);
        }
    }

    #[test]
    fn float32_special_values() {
        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let nan = f32::NAN;

        assert!(Float32::from_f32(inf).to_f32().is_infinite());
        assert!(Float32::from_f32(neg_inf).to_f32().is_infinite());
        assert!(Float32::from_f32(nan).to_f32().is_nan());
    }

    #[test]
    fn float32_dtype_candidate() {
        let val = Float32::from_f32(1.0);
        assert_eq!(val.size_bytes(), 4);
        assert!(val.is_float());
        assert!(!val.is_int());
        assert!(val.is_signed());
        assert!(!val.is_bool());
        assert_eq!(val.type_name(), "float32");
    }
}
