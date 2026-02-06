//! BFloat8 (Brain Float 8-bit) implementation

use crate::dtype::DTypeCandidate;
use std::fmt;

/// Brain Float 8-bit floating point type
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BFloat8(u8);

impl BFloat8 {
    /// Create a BFloat8 from f32
    pub fn from_f32(value: f32) -> Self {
        value.into()
    }

    /// Convert to f32
    pub fn to_f32(self) -> f32 {
        self.into()
    }
}

impl From<f32> for BFloat8 {
    fn from(x: f32) -> Self {
        let bits = x.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = ((bits >> 23) & 0xff) as i32 - 127;
        let mantissa = bits & 0x007fffff;

        if exp == 128 {
            if mantissa != 0 {
                return Self(0x7C); // NaN
            }
            return Self((sign << 7) | 0x70); // Infinity
        }

        if x == 0.0 {
            return Self(if sign == 1 { 0x80 } else { 0x00 });
        }

        let new_exp = exp + 3;

        if new_exp >= 7 {
            return Self((sign << 7) | 0x70);
        }

        if new_exp < -3 {
            return Self(if sign == 1 { 0x80 } else { 0x00 });
        }

        let mut frac = (mantissa >> 19) as u8;
        if new_exp <= 0 {
            frac = (frac | 0x10) >> (1 - new_exp);
            let final_frac = frac & 0xF;
            return Self((sign << 7) | final_frac);
        }

        frac = ((frac as u16 + 0x8) >> 4) as u8;
        let final_frac = frac & 0xF;

        Self((sign << 7) | ((new_exp as u8) << 4) | final_frac)
    }
}

impl From<BFloat8> for f32 {
    fn from(x: BFloat8) -> Self {
        let bits = x.0;
        let sign = ((bits >> 7) & 1) as u32;
        let exp = ((bits >> 4) & 0x7) as i32;
        let frac = (bits & 0xF) as u32;

        if exp == 0x7 {
            if (frac & 0xC) == 0xC {
                return f32::NAN;
            }
            return if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            };
        }

        if exp == 0 && frac == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        if exp == 0 {
            let mut mantissa = frac;
            let mut e = -2;
            while mantissa != 0 && (mantissa & 0x10) == 0 {
                mantissa <<= 1;
                e -= 1;
            }
            let new_exp = ((e + 127) as u32) << 23;
            let new_mantissa = (mantissa & 0xF) << 19;
            return f32::from_bits((sign << 31) | new_exp | new_mantissa);
        }

        let new_exp = ((exp - 3 + 127) as u32) << 23;
        let new_mantissa = frac << 19;
        f32::from_bits((sign << 31) | new_exp | new_mantissa)
    }
}

impl DTypeCandidate for BFloat8 {
    fn size_bytes(&self) -> usize {
        1
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
        "bfloat8"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "BFloat8 requires exactly 1 byte");
        BFloat8(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
}

impl fmt::Display for BFloat8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bfloat8_conversions() {
        let test_values = [
            0.0f32,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];

        for &v in &test_values {
            let bf8 = BFloat8::from(v);
            let back = f32::from(bf8);

            if v.is_nan() {
                assert!(back.is_nan(), "NaN conversion failed");
            } else if v.is_infinite() {
                assert!(back.is_infinite(), "Infinity conversion failed");
                assert_eq!(back.is_sign_negative(), v.is_sign_negative());
            } else if v == 0.0 {
                assert_eq!(back, v);
                assert_eq!(back.is_sign_negative(), v.is_sign_negative());
            } else {
                let rel_error = ((back - v) / v).abs();
                assert!(
                    rel_error < 0.5,
                    "Conversion failed for {}: got {}, relative error: {}",
                    v,
                    back,
                    rel_error
                );
            }
        }
    }

    #[test]
    fn bfloat8_dtype_candidate() {
        let val = BFloat8::from_f32(1.0);
        assert_eq!(val.size_bytes(), 1);
        assert!(val.is_float());
        assert!(!val.is_int());
        assert!(val.is_signed());
        assert!(!val.is_bool());
        assert_eq!(val.type_name(), "bfloat8");
    }
}
