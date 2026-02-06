//! Float16 (IEEE 754 half-precision) implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

/// IEEE 754 half-precision floating point type
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Float16(u16);

impl Float16 {
    /// Create a Float16 from f32
    pub fn from_f32(value: f32) -> Self {
        value.into()
    }

    /// Create a Float16 from raw bits
    pub(crate) fn from_bits(bits: u16) -> Self {
        Float16(bits)
    }

    /// Return the raw bits
    pub(crate) fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        self.into()
    }
}

impl From<f32> for Float16 {
    fn from(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = bits >> 31;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = bits & 0x007fffff;

        if exp == 128 {
            if mantissa != 0 {
                return Self(0x7e00); // NaN
            }
            return Self(((sign << 15) | 0x7c00) as u16); // Infinity
        }

        if exp < -24 {
            return Self(0);
        }

        if exp > 15 {
            return Self(((sign << 15) | 0x7c00) as u16); // Infinity
        }

        let new_exp = if exp < -14 {
            0
        } else {
            ((exp + 15) as u32) & 0x1f
        };
        let new_mantissa = (mantissa + 0x1000) >> 13;
        Self(((sign << 15) as u16) | ((new_exp as u16) << 10) | (new_mantissa as u16))
    }
}

impl From<Float16> for f32 {
    fn from(x: Float16) -> Self {
        let bits = x.0;
        let sign = (bits >> 15) as u32;
        let exp = ((bits >> 10) & 0x1f) as i32;
        let mantissa = (bits & 0x3ff) as u32;

        if exp == 0 {
            if mantissa == 0 {
                return f32::from_bits(sign << 31);
            }
            let mut e = -14;
            let mut m = mantissa;
            while (m & 0x400) == 0 {
                e -= 1;
                m <<= 1;
            }
            let new_mantissa = (m & 0x3ff) << 13;
            let final_exp = e + 127;
            if final_exp <= 0 {
                return 0.0;
            }
            let new_bits = (sign << 31) | ((final_exp as u32) << 23) | new_mantissa;
            f32::from_bits(new_bits)
        } else if exp == 0x1f {
            if mantissa == 0 {
                f32::from_bits((sign << 31) | 0x7f800000)
            } else {
                f32::from_bits(0x7fc00000)
            }
        } else {
            let exp_adj: i32 = exp - 15 + 127;
            let new_exp = ((exp_adj as u32) & 0xff) << 23;
            let new_mantissa = mantissa << 13;
            f32::from_bits((sign << 31) | new_exp | new_mantissa)
        }
    }
}

impl DTypeCandidate for Float16 {
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
        true
    }

    fn is_bool(&self) -> bool {
        false
    }

    fn type_name(&self) -> &'static str {
        "float16"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 2, "Float16 requires exactly 2 bytes");
        let value = u16::from_le_bytes([bytes[0], bytes[1]]);
        Float16(value)
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
}

impl fmt::Display for Float16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float16_conversions() {
        let values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0];
        for &v in &values {
            let f16 = Float16::from(v);
            let back = f32::from(f16);
            assert!(
                (back - v).abs() < 0.001,
                "Conversion failed for {}: got {}",
                v,
                back
            );
        }

        let inf = f32::INFINITY;
        let neg_inf = f32::NEG_INFINITY;
        let nan = f32::NAN;

        assert_eq!(Float16::from(inf).0, 0x7c00);
        assert_eq!(Float16::from(neg_inf).0, 0xfc00);
        assert_eq!(Float16::from(nan).0, 0x7e00);
    }

    #[test]
    fn float16_dtype_candidate() {
        let val = Float16::from_f32(1.0);
        assert_eq!(val.size_bytes(), 2);
        assert!(val.is_float());
        assert!(!val.is_int());
        assert!(val.is_signed());
        assert!(!val.is_bool());
        assert_eq!(val.type_name(), "float16");
    }
}
