//! Float8 types (E4M3FN, E5M2).
//!
//! These are compact 8-bit float-like formats used by Numina. The canonical byte representation is
//! the raw 8-bit payload.

use crate::dtype::{DTypeCandidate, FloatDType};
use std::fmt;

/// Float8 in E4M3FN (finite-only) format.
///
/// Layout: `#[repr(transparent)]` over `u8` holding the raw bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Float8E4M3Fn(u8);

/// Float8 in E5M2 (Inf/NaN capable) format.
///
/// Layout: `#[repr(transparent)]` over `u8` holding the raw bits.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Float8E5M2(u8);

impl Float8E4M3Fn {
    /// Convert an `f32` into Float8 E4M3FN (lossy).
    pub fn from_f32(value: f32) -> Self {
        Float8E4M3Fn(encode_e4m3fn(value))
    }

    /// Convert this value to `f32`.
    pub fn to_f32(self) -> f32 {
        decode_e4m3fn(self.0)
    }

    #[cfg(test)]
    pub(crate) fn from_bits(bits: u8) -> Self {
        Float8E4M3Fn(bits)
    }

    #[cfg(test)]
    pub(crate) fn to_bits(self) -> u8 {
        self.0
    }
}

impl Float8E5M2 {
    /// Convert an `f32` into Float8 E5M2 (lossy).
    pub fn from_f32(value: f32) -> Self {
        Float8E5M2(encode_e5m2(value))
    }

    /// Convert this value to `f32`.
    pub fn to_f32(self) -> f32 {
        decode_e5m2(self.0)
    }

    #[cfg(test)]
    pub(crate) fn from_bits(bits: u8) -> Self {
        Float8E5M2(bits)
    }

    #[cfg(test)]
    pub(crate) fn to_bits(self) -> u8 {
        self.0
    }
}

impl From<f32> for Float8E4M3Fn {
    fn from(value: f32) -> Self {
        Float8E4M3Fn::from_f32(value)
    }
}

impl From<f32> for Float8E5M2 {
    fn from(value: f32) -> Self {
        Float8E5M2::from_f32(value)
    }
}

impl From<Float8E4M3Fn> for f32 {
    fn from(value: Float8E4M3Fn) -> Self {
        value.to_f32()
    }
}

impl From<Float8E5M2> for f32 {
    fn from(value: Float8E5M2) -> Self {
        value.to_f32()
    }
}

impl DTypeCandidate for Float8E4M3Fn {
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
        "float8_e4m3fn"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "Float8E4M3Fn requires exactly 1 byte");
        Float8E4M3Fn(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
}

impl FloatDType for Float8E4M3Fn {
    fn from_f32(value: f32) -> Self {
        Float8E4M3Fn::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl DTypeCandidate for Float8E5M2 {
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
        "float8_e5m2"
    }

    unsafe fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 1, "Float8E5M2 requires exactly 1 byte");
        Float8E5M2(bytes[0])
    }

    fn to_bytes(&self) -> Vec<u8> {
        vec![self.0]
    }
}

impl FloatDType for Float8E5M2 {
    fn from_f32(value: f32) -> Self {
        Float8E5M2::from_f32(value)
    }

    fn to_f32(self) -> f32 {
        self.to_f32()
    }
}

impl fmt::Display for Float8E4M3Fn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

impl fmt::Display for Float8E5M2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f32())
    }
}

fn encode_e4m3fn(value: f32) -> u8 {
    const EXP_BITS: i32 = 4;
    const MANT_BITS: i32 = 3;
    const BIAS: i32 = 7;
    let sign = if value.is_sign_negative() { 1u8 } else { 0u8 };
    let exp_mask = (1u8 << (EXP_BITS as u32)) - 1;
    let mant_mask = (1u8 << (MANT_BITS as u32)) - 1;

    if value.is_nan() {
        return (sign << 7) | (exp_mask << (MANT_BITS as u32)) | mant_mask;
    }

    if value == 0.0 {
        return sign << 7;
    }

    if value.is_infinite() {
        return e4m3fn_max_finite_bits(sign);
    }

    let abs = value.abs() as f64;
    let min_normal_exp = 1 - BIAS;
    let min_normal = 2f64.powi(min_normal_exp);
    let max_exp_field = exp_mask as i32;
    let max_normal_exp = max_exp_field - BIAS;
    let exp_unbiased = abs.log2().floor() as i32;

    if exp_unbiased > max_normal_exp {
        return e4m3fn_max_finite_bits(sign);
    }

    if exp_unbiased < min_normal_exp {
        let scaled = abs / min_normal * (1u32 << (MANT_BITS as u32)) as f64;
        let mant = round_to_even(scaled);
        if mant <= 0 {
            return sign << 7;
        }
        if mant >= (1 << MANT_BITS) {
            return (sign << 7) | (1u8 << (MANT_BITS as u32));
        }
        return (sign << 7) | (mant as u8);
    }

    let mut exp_unbiased = exp_unbiased;
    let base = 2f64.powi(exp_unbiased);
    let frac = abs / base - 1.0;
    let mut mant = round_to_even(frac * (1u32 << (MANT_BITS as u32)) as f64);

    if mant == (1 << MANT_BITS) {
        mant = 0;
        exp_unbiased += 1;
        if exp_unbiased > max_normal_exp {
            return e4m3fn_max_finite_bits(sign);
        }
    }

    let exp_field = (exp_unbiased + BIAS) as u8;
    let mant_field = mant as u8;
    if exp_field == exp_mask && mant_field == mant_mask {
        return e4m3fn_max_finite_bits(sign);
    }

    (sign << 7) | (exp_field << (MANT_BITS as u32)) | mant_field
}

fn decode_e4m3fn(bits: u8) -> f32 {
    const EXP_BITS: i32 = 4;
    const MANT_BITS: i32 = 3;
    const BIAS: i32 = 7;
    let sign = (bits >> 7) & 0x1;
    let exp_mask = (1u8 << (EXP_BITS as u32)) - 1;
    let mant_mask = (1u8 << (MANT_BITS as u32)) - 1;
    let exp = ((bits >> MANT_BITS) & exp_mask) as i32;
    let mant = (bits & mant_mask) as i32;

    let mut value = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            let frac = mant as f32 / (1 << MANT_BITS) as f32;
            frac * 2f32.powi(1 - BIAS)
        }
    } else if exp == exp_mask as i32 {
        if mant == mant_mask as i32 {
            f32::NAN
        } else {
            let frac = 1.0 + mant as f32 / (1 << MANT_BITS) as f32;
            frac * 2f32.powi(exp - BIAS)
        }
    } else {
        let frac = 1.0 + mant as f32 / (1 << MANT_BITS) as f32;
        frac * 2f32.powi(exp - BIAS)
    };

    if sign == 1 {
        value = -value;
    }

    value
}

fn encode_e5m2(value: f32) -> u8 {
    const EXP_BITS: i32 = 5;
    const MANT_BITS: i32 = 2;
    const BIAS: i32 = 15;
    let sign = if value.is_sign_negative() { 1u8 } else { 0u8 };
    let exp_mask = (1u8 << (EXP_BITS as u32)) - 1;
    let mant_mask = (1u8 << (MANT_BITS as u32)) - 1;

    if value.is_nan() {
        return (sign << 7) | (exp_mask << (MANT_BITS as u32)) | 1;
    }

    if value == 0.0 {
        return sign << 7;
    }

    if value.is_infinite() {
        return (sign << 7) | (exp_mask << (MANT_BITS as u32));
    }

    let abs = value.abs() as f64;
    let min_normal_exp = 1 - BIAS;
    let min_normal = 2f64.powi(min_normal_exp);
    let max_exp_field = (exp_mask - 1) as i32;
    let max_normal_exp = max_exp_field - BIAS;
    let max_finite = (2.0 - 2f64.powi(-MANT_BITS)) * 2f64.powi(max_normal_exp);
    let exp_unbiased = abs.log2().floor() as i32;

    if abs > max_finite {
        return (sign << 7) | (exp_mask << (MANT_BITS as u32));
    }

    if exp_unbiased > max_normal_exp {
        return (sign << 7) | (exp_mask << (MANT_BITS as u32));
    }

    if exp_unbiased < min_normal_exp {
        let scaled = abs / min_normal * (1u32 << (MANT_BITS as u32)) as f64;
        let mant = round_to_even(scaled);
        if mant <= 0 {
            return sign << 7;
        }
        if mant >= (1 << MANT_BITS) {
            return (sign << 7) | (1u8 << (MANT_BITS as u32));
        }
        return (sign << 7) | (mant as u8);
    }

    let mut exp_unbiased = exp_unbiased;
    let base = 2f64.powi(exp_unbiased);
    let frac = abs / base - 1.0;
    let mut mant = round_to_even(frac * (1u32 << (MANT_BITS as u32)) as f64);

    if mant == (1 << MANT_BITS) {
        mant = 0;
        exp_unbiased += 1;
        if exp_unbiased > max_normal_exp {
            return (sign << 7) | (exp_mask << (MANT_BITS as u32));
        }
    }

    let exp_field = (exp_unbiased + BIAS) as u8;
    let mant_field = mant as u8 & mant_mask;
    (sign << 7) | (exp_field << (MANT_BITS as u32)) | mant_field
}

fn decode_e5m2(bits: u8) -> f32 {
    const EXP_BITS: i32 = 5;
    const MANT_BITS: i32 = 2;
    const BIAS: i32 = 15;
    let sign = (bits >> 7) & 0x1;
    let exp_mask = (1u8 << (EXP_BITS as u32)) - 1;
    let mant_mask = (1u8 << (MANT_BITS as u32)) - 1;
    let exp = ((bits >> MANT_BITS) & exp_mask) as i32;
    let mant = (bits & mant_mask) as i32;

    let mut value = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            let frac = mant as f32 / (1 << MANT_BITS) as f32;
            frac * 2f32.powi(1 - BIAS)
        }
    } else if exp == exp_mask as i32 {
        if mant == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        let frac = 1.0 + mant as f32 / (1 << MANT_BITS) as f32;
        frac * 2f32.powi(exp - BIAS)
    };

    if sign == 1 {
        value = -value;
    }

    value
}

fn round_to_even(value: f64) -> i32 {
    let floor = value.floor();
    let frac = value - floor;
    if frac > 0.5 {
        return (floor + 1.0) as i32;
    }
    if frac < 0.5 {
        return floor as i32;
    }
    let floor_i = floor as i64;
    if floor_i % 2 == 0 {
        floor_i as i32
    } else {
        (floor_i + 1) as i32
    }
}

fn e4m3fn_max_finite_bits(sign: u8) -> u8 {
    let exp_mask = (1u8 << 4) - 1;
    let mant_max_finite = (1u8 << 3) - 2;
    (sign << 7) | (exp_mask << 3) | mant_max_finite
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float8_e4m3fn_limits() {
        let max = Float8E4M3Fn::from_f32(448.0);
        assert_eq!(max.to_bits(), 0x7E);
        assert_eq!(max.to_f32(), 448.0);
        let min_normal = Float8E4M3Fn::from_bits(0x08);
        assert_eq!(min_normal.to_f32(), 0.015625);
        assert_eq!(Float8E4M3Fn::from_f32(0.015625).to_bits(), 0x08);
        let min_sub = Float8E4M3Fn::from_bits(0x01);
        assert_eq!(min_sub.to_f32(), 0.001953125);
        assert_eq!(Float8E4M3Fn::from_f32(0.001953125).to_bits(), 0x01);
        let overflow = Float8E4M3Fn::from_f32(480.0);
        assert_eq!(overflow.to_bits(), 0x7E);
        let inf = Float8E4M3Fn::from_f32(f32::INFINITY);
        assert_eq!(inf.to_bits(), 0x7E);
        let nan = Float8E4M3Fn::from_f32(f32::NAN);
        assert_eq!(nan.to_bits() & 0x7F, 0x7F);
        assert!(f32::from(Float8E4M3Fn::from_bits(0x7F)).is_nan());
    }

    #[test]
    fn float8_e5m2_limits() {
        let max = Float8E5M2::from_f32(57344.0);
        assert_eq!(max.to_bits(), 0x7B);
        assert_eq!(max.to_f32(), 57344.0);
        let min_normal = Float8E5M2::from_bits(0x04);
        assert_eq!(min_normal.to_f32(), 6.103515625e-05);
        assert_eq!(Float8E5M2::from_f32(6.103515625e-05).to_bits(), 0x04);
        let min_sub = Float8E5M2::from_bits(0x01);
        assert_eq!(min_sub.to_f32(), 1.52587890625e-05);
        assert_eq!(Float8E5M2::from_f32(1.52587890625e-05).to_bits(), 0x01);
        let overflow = Float8E5M2::from_f32(60000.0);
        assert_eq!(overflow.to_bits(), 0x7C);
        let inf = Float8E5M2::from_f32(f32::INFINITY);
        assert_eq!(inf.to_bits(), 0x7C);
        let nan = Float8E5M2::from_f32(f32::NAN);
        assert_eq!(nan.to_bits() & 0x7F, 0x7D);
        assert!(f32::from(Float8E5M2::from_bits(0x7D)).is_nan());
    }

    #[test]
    fn float8_rtne_rounding() {
        let e4m3_low = Float8E4M3Fn::from_f32(1.0625);
        assert_eq!(e4m3_low.to_f32(), 1.0);
        assert_eq!(e4m3_low.to_bits(), 0x38);
        let e4m3_high = Float8E4M3Fn::from_f32(1.1875);
        assert_eq!(e4m3_high.to_f32(), 1.25);
        assert_eq!(e4m3_high.to_bits(), 0x3A);
        let e5m2_low = Float8E5M2::from_f32(1.125);
        assert_eq!(e5m2_low.to_f32(), 1.0);
        assert_eq!(e5m2_low.to_bits(), 0x3C);
        let e5m2_high = Float8E5M2::from_f32(1.375);
        assert_eq!(e5m2_high.to_f32(), 1.5);
        assert_eq!(e5m2_high.to_bits(), 0x3E);
    }

    #[test]
    fn float8_dtype_candidate() {
        let v1 = Float8E4M3Fn::from_f32(1.0);
        let v2 = Float8E5M2::from_f32(1.0);
        assert_eq!(v1.size_bytes(), 1);
        assert_eq!(v2.size_bytes(), 1);
        assert!(v1.is_float());
        assert!(v2.is_float());
    }
}
