//! Complex number types.
//!
//! These are lightweight complex representations used for dtype identity and byte layout.
//! Canonical encoding is little-endian and matches the field order in the structs.

use super::float16::Float16;
use crate::dtype::DTypeCandidate;
use std::fmt;

/// Complex number with two `Float16` components (`re`, `im`).
///
/// Layout is `re` then `im` (each `Float16` is 16-bit).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Complex32 {
    /// Real component.
    pub re: Float16,
    /// Imaginary component.
    pub im: Float16,
}

/// Complex number with two `f32` components stored as raw IEEE-754 bit patterns.
///
/// Storing raw bits allows `Eq`/`Hash` without floating-point semantic edge cases.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Complex64 {
    /// Real component bits (`f32::to_bits()`).
    pub re_bits: u32,
    /// Imaginary component bits (`f32::to_bits()`).
    pub im_bits: u32,
}

/// Complex number with two `f64` components stored as raw IEEE-754 bit patterns.
///
/// Storing raw bits allows `Eq`/`Hash` without floating-point semantic edge cases.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Complex128 {
    /// Real component bits (`f64::to_bits()`).
    pub re_bits: u64,
    /// Imaginary component bits (`f64::to_bits()`).
    pub im_bits: u64,
}

impl Complex32 {
    /// Construct a complex number from `f32` real/imag parts (lossily converted to `Float16`).
    pub fn new(re: f32, im: f32) -> Self {
        Self {
            re: Float16::from(re),
            im: Float16::from(im),
        }
    }

    /// Convert to `(re, im)` as `f32` values.
    pub fn to_f32_tuple(self) -> (f32, f32) {
        (f32::from(self.re), f32::from(self.im))
    }
}

impl Complex64 {
    /// Construct a complex number from `f32` real/imag parts.
    pub fn new(re: f32, im: f32) -> Self {
        Self {
            re_bits: re.to_bits(),
            im_bits: im.to_bits(),
        }
    }

    /// Convert to `(re, im)` as `f32` values.
    pub fn to_f32_tuple(self) -> (f32, f32) {
        (f32::from_bits(self.re_bits), f32::from_bits(self.im_bits))
    }
}

impl Complex128 {
    /// Construct a complex number from `f64` real/imag parts.
    pub fn new(re: f64, im: f64) -> Self {
        Self {
            re_bits: re.to_bits(),
            im_bits: im.to_bits(),
        }
    }

    /// Convert to `(re, im)` as `f64` values.
    pub fn to_f64_tuple(self) -> (f64, f64) {
        (f64::from_bits(self.re_bits), f64::from_bits(self.im_bits))
    }
}

impl DTypeCandidate for Complex32 {
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
        "complex32"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 4, "Complex32 requires exactly 4 bytes");
        let re_bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let im_bits = u16::from_le_bytes([bytes[2], bytes[3]]);
        Self {
            re: Float16::from_bits(re_bits),
            im: Float16::from_bits(im_bits),
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let re = self.re.to_bits().to_le_bytes();
        let im = self.im.to_bits().to_le_bytes();
        vec![re[0], re[1], im[0], im[1]]
    }
}

impl DTypeCandidate for Complex64 {
    fn size_bytes(&self) -> usize {
        8
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
        "complex64"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 8, "Complex64 requires exactly 8 bytes");
        let re = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let im = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        Self {
            re_bits: re,
            im_bits: im,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let re = self.re_bits.to_le_bytes();
        let im = self.im_bits.to_le_bytes();
        vec![re[0], re[1], re[2], re[3], im[0], im[1], im[2], im[3]]
    }
}

impl DTypeCandidate for Complex128 {
    fn size_bytes(&self) -> usize {
        16
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
        "complex128"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 16, "Complex128 requires exactly 16 bytes");
        let re = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        let im = u64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self {
            re_bits: re,
            im_bits: im,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let re = self.re_bits.to_le_bytes();
        let im = self.im_bits.to_le_bytes();
        vec![
            re[0], re[1], re[2], re[3], re[4], re[5], re[6], re[7], im[0], im[1], im[2], im[3],
            im[4], im[5], im[6], im[7],
        ]
    }
}

impl fmt::Display for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.to_f32_tuple();
        if im >= 0.0 {
            write!(f, "{}+{}i", re, im)
        } else {
            write!(f, "{}{}i", re, im)
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.to_f32_tuple();
        if im >= 0.0 {
            write!(f, "{}+{}i", re, im)
        } else {
            write!(f, "{}{}i", re, im)
        }
    }
}

impl fmt::Display for Complex128 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (re, im) = self.to_f64_tuple();
        if im >= 0.0 {
            write!(f, "{}+{}i", re, im)
        } else {
            write!(f, "{}{}i", re, im)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex32_bytes_roundtrip() {
        let value = Complex32::new(1.5, -2.0);
        let bytes = value.to_bytes();
        let decoded = unsafe { Complex32::from_bytes(&bytes) };
        let (re, im) = decoded.to_f32_tuple();
        assert!((re - 1.5).abs() < 0.01);
        assert!((im + 2.0).abs() < 0.01);
        assert_eq!(decoded.type_name(), "complex32");
    }

    #[test]
    fn complex64_bytes_roundtrip() {
        let value = Complex64::new(1.5, -2.0);
        let bytes = value.to_bytes();
        let decoded = unsafe { Complex64::from_bytes(&bytes) };
        let (re, im) = decoded.to_f32_tuple();
        assert_eq!(re, 1.5);
        assert_eq!(im, -2.0);
        assert_eq!(decoded.type_name(), "complex64");
    }

    #[test]
    fn complex128_bytes_roundtrip() {
        let value = Complex128::new(1.5, -2.0);
        let bytes = value.to_bytes();
        let decoded = unsafe { Complex128::from_bytes(&bytes) };
        let (re, im) = decoded.to_f64_tuple();
        assert_eq!(re, 1.5);
        assert_eq!(im, -2.0);
        assert_eq!(decoded.type_name(), "complex128");
    }
}
