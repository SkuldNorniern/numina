//! Element-wise array operations built on the `NdArray` abstraction

use crate::array::{
    data_as_slice, data_as_slice_mut, ensure_binary_compat, ensure_host_accessible, NdArray,
};
use crate::{DType, Shape};

/// Element-wise addition
pub fn add<A, B>(a: &A, b: &B) -> Result<Box<dyn NdArray>, String>
where
    A: NdArray,
    B: NdArray,
{
    ensure_binary_compat(a, b, "add")?;

    let mut result = a.zeros(a.shape().clone())?;

    match a.dtype() {
        DType::F16 => {
            let lhs = unsafe { data_as_slice::<crate::Float16>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) + f32::from(rhs[i]);
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] + rhs[i];
            }
        }
        DType::F64 => {
            let lhs = unsafe { data_as_slice::<f64>(a) };
            let rhs = unsafe { data_as_slice::<f64>(b) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] + rhs[i];
            }
        }
        DType::BF16 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat16>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = lhs[i].to_f32() + rhs[i].to_f32();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat8>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat8>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) + f32::from(rhs[i]);
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let lhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) + f32::from(rhs[i]);
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let lhs = unsafe { data_as_slice::<crate::Float8E5M2>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E5M2>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) + f32::from(rhs[i]);
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Addition not implemented for {}", a.dtype()));
        }
        DType::I8 => {
            let lhs = unsafe { data_as_slice::<i8>(a) };
            let rhs = unsafe { data_as_slice::<i8>(b) };
            let dst = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::I16 => {
            let lhs = unsafe { data_as_slice::<i16>(a) };
            let rhs = unsafe { data_as_slice::<i16>(b) };
            let dst = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::I32 => {
            let lhs = unsafe { data_as_slice::<i32>(a) };
            let rhs = unsafe { data_as_slice::<i32>(b) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::I64 => {
            let lhs = unsafe { data_as_slice::<i64>(a) };
            let rhs = unsafe { data_as_slice::<i64>(b) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::U8 => {
            let lhs = unsafe { data_as_slice::<u8>(a) };
            let rhs = unsafe { data_as_slice::<u8>(b) };
            let dst = unsafe { data_as_slice_mut::<u8>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::U16 => {
            let lhs = unsafe { data_as_slice::<u16>(a) };
            let rhs = unsafe { data_as_slice::<u16>(b) };
            let dst = unsafe { data_as_slice_mut::<u16>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::U32 => {
            let lhs = unsafe { data_as_slice::<u32>(a) };
            let rhs = unsafe { data_as_slice::<u32>(b) };
            let dst = unsafe { data_as_slice_mut::<u32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::U64 => {
            let lhs = unsafe { data_as_slice::<u64>(a) };
            let rhs = unsafe { data_as_slice::<u64>(b) };
            let dst = unsafe { data_as_slice_mut::<u64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_add(rhs[i]);
            }
        }
        DType::Bool => {
            return Err("Addition not supported for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Addition not implemented for quantized types {}",
                a.dtype()
            ));
        }
    }

    Ok(result)
}

/// Element-wise multiplication
pub fn mul<A, B>(a: &A, b: &B) -> Result<Box<dyn NdArray>, String>
where
    A: NdArray,
    B: NdArray,
{
    ensure_binary_compat(a, b, "mul")?;

    let mut result = a.zeros(a.shape().clone())?;

    match a.dtype() {
        DType::F16 => {
            let lhs = unsafe { data_as_slice::<crate::Float16>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) * f32::from(rhs[i]);
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] * rhs[i];
            }
        }
        DType::F64 => {
            let lhs = unsafe { data_as_slice::<f64>(a) };
            let rhs = unsafe { data_as_slice::<f64>(b) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] * rhs[i];
            }
        }
        DType::BF16 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat16>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = lhs[i].to_f32() * rhs[i].to_f32();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat8>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat8>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) * f32::from(rhs[i]);
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let lhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) * f32::from(rhs[i]);
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let lhs = unsafe { data_as_slice::<crate::Float8E5M2>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E5M2>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..a.len() {
                let result_f32 = f32::from(lhs[i]) * f32::from(rhs[i]);
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Multiplication not implemented for {}", a.dtype()));
        }
        DType::I8 => {
            let lhs = unsafe { data_as_slice::<i8>(a) };
            let rhs = unsafe { data_as_slice::<i8>(b) };
            let dst = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::I16 => {
            let lhs = unsafe { data_as_slice::<i16>(a) };
            let rhs = unsafe { data_as_slice::<i16>(b) };
            let dst = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::I32 => {
            let lhs = unsafe { data_as_slice::<i32>(a) };
            let rhs = unsafe { data_as_slice::<i32>(b) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::I64 => {
            let lhs = unsafe { data_as_slice::<i64>(a) };
            let rhs = unsafe { data_as_slice::<i64>(b) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::U8 => {
            let lhs = unsafe { data_as_slice::<u8>(a) };
            let rhs = unsafe { data_as_slice::<u8>(b) };
            let dst = unsafe { data_as_slice_mut::<u8>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::U16 => {
            let lhs = unsafe { data_as_slice::<u16>(a) };
            let rhs = unsafe { data_as_slice::<u16>(b) };
            let dst = unsafe { data_as_slice_mut::<u16>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::U32 => {
            let lhs = unsafe { data_as_slice::<u32>(a) };
            let rhs = unsafe { data_as_slice::<u32>(b) };
            let dst = unsafe { data_as_slice_mut::<u32>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::U64 => {
            let lhs = unsafe { data_as_slice::<u64>(a) };
            let rhs = unsafe { data_as_slice::<u64>(b) };
            let dst = unsafe { data_as_slice_mut::<u64>(&mut *result) };

            for i in 0..a.len() {
                dst[i] = lhs[i].wrapping_mul(rhs[i]);
            }
        }
        DType::Bool => {
            return Err("Multiplication not supported for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Multiplication not implemented for quantized types {}",
                a.dtype()
            ));
        }
    }

    Ok(result)
}

/// Scalar addition
pub fn add_scalar<A: NdArray>(array: &A, scalar: f64) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "add_scalar")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let value = f32::from(src[i]) + scalar as f32;
                dst[i] = crate::Float16::from(value);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i] + scalar as f32;
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i] + scalar;
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let value = src[i].to_f32() + scalar as f32;
                dst[i] = crate::BFloat16::from_f32(value);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let value = f32::from(src[i]) + scalar as f32;
                dst[i] = crate::BFloat8::from(value);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let value = f32::from(src[i]) + scalar as f32;
                dst[i] = crate::Float8E4M3Fn::from(value);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let value = f32::from(src[i]) + scalar as f32;
                dst[i] = crate::Float8E5M2::from(value);
            }
        }
        _ => {
            return Err(format!(
                "Scalar addition not implemented for {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
}

/// Element-wise exponential
pub fn exp<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "exp")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).exp();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].exp();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].exp();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().exp();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).exp();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).exp();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).exp();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => {
            return Err(format!(
                "Exp only supported for floating point types, got {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
}

/// Element-wise logarithm
pub fn log<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "log")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).ln();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].ln();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].ln();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().ln();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).ln();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).ln();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).ln();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Log not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise square root
pub fn sqrt<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sqrt")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sqrt();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].sqrt();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].sqrt();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().sqrt();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sqrt();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sqrt();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sqrt();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => {
            return Err(format!(
                "Sqrt only supported for floating point types, got {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
}

/// Element-wise sine
pub fn sin<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sin")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sin();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].sin();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].sin();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().sin();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sin();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sin();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).sin();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Sin not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise cosine
pub fn cos<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "cos")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).cos();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].cos();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].cos();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().cos();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).cos();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).cos();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).cos();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Cos not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise tangent
pub fn tan<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "tan")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).tan();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].tan();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].tan();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().tan();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).tan();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).tan();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).tan();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Tan not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise arcsine
pub fn asin<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "asin")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).asin();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].asin();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].asin();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().asin();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).asin();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).asin();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).asin();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Asin not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise arccosine
pub fn acos<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "acos")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).acos();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].acos();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].acos();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().acos();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).acos();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).acos();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).acos();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Acos not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise arctangent
pub fn atan<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "atan")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).atan();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].atan();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].atan();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().atan();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).atan();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).atan();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).atan();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Atan not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise power (array^exponent)
pub fn pow<A: NdArray>(array: &A, exponent: f64) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "pow")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).powf(exponent as f32);
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].powf(exponent as f32);
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].powf(exponent);
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().powf(exponent as f32);
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).powf(exponent as f32);
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).powf(exponent as f32);
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).powf(exponent as f32);
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        _ => return Err(format!("Pow not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Element-wise absolute value
pub fn abs<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "abs")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).abs();
                dst[i] = crate::Float16::from(result_f32);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = src[i].to_f32().abs();
                dst[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).abs();
                dst[i] = crate::BFloat8::from(result_f32);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).abs();
                dst[i] = crate::Float8E4M3Fn::from(result_f32);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let result_f32 = f32::from(src[i]).abs();
                dst[i] = crate::Float8E5M2::from(result_f32);
            }
        }
        DType::I8 => {
            let src = unsafe { data_as_slice::<i8>(array) };
            let dst = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::I16 => {
            let src = unsafe { data_as_slice::<i16>(array) };
            let dst = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::I32 => {
            let src = unsafe { data_as_slice::<i32>(array) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::I64 => {
            let src = unsafe { data_as_slice::<i64>(array) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
            // For unsigned types, abs is identity (no-op)
            return Err(format!(
                "Abs not needed for unsigned type {}",
                array.dtype()
            ));
        }
        DType::Bool => {
            return Err("Abs not applicable for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Abs not implemented for quantized types {}",
                array.dtype()
            ));
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Abs not implemented for {}", array.dtype()));
        }
    }

    Ok(result)
}

/// Element-wise sign function
pub fn sign<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sign")?;

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let src = unsafe { data_as_slice::<crate::Float16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..array.len() {
                let value = f32::from(src[i]);
                let sign = if value > 0.0 {
                    1.0
                } else if value < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dst[i] = crate::Float16::from(sign);
            }
        }
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = if src[i] > 0.0 {
                    1.0
                } else if src[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = if src[i] > 0.0 {
                    1.0
                } else if src[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        DType::BF16 => {
            let src = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..array.len() {
                let val = src[i].to_f32();
                let sign_val = if val > 0.0 {
                    1.0
                } else if val < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dst[i] = crate::BFloat16::from_f32(sign_val);
            }
        }
        DType::BF8 => {
            let src = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..array.len() {
                let val = f32::from(src[i]);
                let sign_val = if val > 0.0 {
                    1.0
                } else if val < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dst[i] = crate::BFloat8::from(sign_val);
            }
        }
        DType::F8E4M3FN => {
            let src = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..array.len() {
                let val = f32::from(src[i]);
                let sign_val = if val > 0.0 {
                    1.0
                } else if val < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dst[i] = crate::Float8E4M3Fn::from(sign_val);
            }
        }
        DType::F8E5M2 => {
            let src = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..array.len() {
                let val = f32::from(src[i]);
                let sign_val = if val > 0.0 {
                    1.0
                } else if val < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                dst[i] = crate::Float8E5M2::from(sign_val);
            }
        }
        DType::I8 => {
            let src = unsafe { data_as_slice::<i8>(array) };
            let dst = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        DType::I16 => {
            let src = unsafe { data_as_slice::<i16>(array) };
            let dst = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        DType::I32 => {
            let src = unsafe { data_as_slice::<i32>(array) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        DType::I64 => {
            let src = unsafe { data_as_slice::<i64>(array) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        DType::U8 | DType::U16 | DType::U32 | DType::U64 => {
            // For unsigned types, sign is always 1 or 0
            return Err(format!(
                "Sign not applicable for unsigned type {}",
                array.dtype()
            ));
        }
        DType::Bool => {
            return Err("Sign not applicable for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Sign not implemented for quantized types {}",
                array.dtype()
            ));
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Sign not implemented for {}", array.dtype()));
        }
    }

    Ok(result)
}

/// Matrix multiplication (2D arrays)
pub fn matmul<A, B>(a: &A, b: &B) -> Result<Box<dyn NdArray>, String>
where
    A: NdArray,
    B: NdArray,
{
    ensure_host_accessible(a, "matmul")?;
    ensure_host_accessible(b, "matmul")?;

    if a.dtype() != b.dtype() {
        return Err(format!(
            "matmul dtype mismatch: {} vs {}",
            a.dtype(),
            b.dtype()
        ));
    }

    if a.shape().ndim() != 2 || b.shape().ndim() != 2 {
        return Err("matmul requires 2D arrays".to_string());
    }

    let (m, k) = (a.shape().dim(0), a.shape().dim(1));
    let (k2, n) = (b.shape().dim(0), b.shape().dim(1));

    if k != k2 {
        return Err(format!("matmul dimension mismatch: {} vs {}", k, k2));
    }

    let result_shape = Shape::from([m, n]);
    let mut result = a.zeros(result_shape)?;

    match a.dtype() {
        DType::F16 => {
            let lhs = unsafe { data_as_slice::<crate::Float16>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let left = f32::from(lhs[i * k + l]);
                        let right = f32::from(rhs[l * n + j]);
                        sum += left * right;
                    }
                    dst[i * n + j] = crate::Float16::from(sum);
                }
            }
        }
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += lhs[i * k + l] * rhs[l * n + j];
                    }
                    dst[i * n + j] = sum;
                }
            }
        }
        DType::F64 => {
            let lhs = unsafe { data_as_slice::<f64>(a) };
            let rhs = unsafe { data_as_slice::<f64>(b) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for l in 0..k {
                        sum += lhs[i * k + l] * rhs[l * n + j];
                    }
                    dst[i * n + j] = sum;
                }
            }
        }
        DType::BF16 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat16>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat16>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += lhs[i * k + l].to_f32() * rhs[l * n + j].to_f32();
                    }
                    dst[i * n + j] = crate::BFloat16::from_f32(sum);
                }
            }
        }
        DType::BF8 => {
            let lhs = unsafe { data_as_slice::<crate::BFloat8>(a) };
            let rhs = unsafe { data_as_slice::<crate::BFloat8>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let left = f32::from(lhs[i * k + l]);
                        let right = f32::from(rhs[l * n + j]);
                        sum += left * right;
                    }
                    dst[i * n + j] = crate::BFloat8::from(sum);
                }
            }
        }
        DType::F8E4M3FN => {
            let lhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E4M3Fn>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let left = f32::from(lhs[i * k + l]);
                        let right = f32::from(rhs[l * n + j]);
                        sum += left * right;
                    }
                    dst[i * n + j] = crate::Float8E4M3Fn::from(sum);
                }
            }
        }
        DType::F8E5M2 => {
            let lhs = unsafe { data_as_slice::<crate::Float8E5M2>(a) };
            let rhs = unsafe { data_as_slice::<crate::Float8E5M2>(b) };
            let dst = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        let left = f32::from(lhs[i * k + l]);
                        let right = f32::from(rhs[l * n + j]);
                        sum += left * right;
                    }
                    dst[i * n + j] = crate::Float8E5M2::from(sum);
                }
            }
        }
        _ => return Err(format!("Matmul not implemented for {}", a.dtype())),
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::Shape;

    #[test]
    fn test_add_array() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], Shape::from([3])).unwrap();
        let b = Array::from_slice(&[4.0f32, 5.0, 6.0], Shape::from([3])).unwrap();

        let result = add(&a, &b).unwrap();

        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), crate::dtype::F32);

        // Verify the actual values: [1+4, 2+5, 3+6] = [5, 7, 9]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 5.0);
        assert_eq!(result_data[1], 7.0);
        assert_eq!(result_data[2], 9.0);
    }

    #[test]
    fn test_matmul() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2])).unwrap();
        let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2])).unwrap();

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &Shape::from([2, 2]));
        assert_eq!(result.dtype(), crate::dtype::F32);

        // Verify matrix multiplication result: [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 19.0); // 1*5 + 2*7
        assert_eq!(result_data[1], 22.0); // 1*6 + 2*8
        assert_eq!(result_data[2], 43.0); // 3*5 + 4*7
        assert_eq!(result_data[3], 50.0); // 3*6 + 4*8
    }

    #[test]
    fn test_exp() {
        let a = Array::from_slice(&[0.0f32, 1.0, 2.0], Shape::from([3])).unwrap();
        let result = exp(&a).unwrap();

        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), crate::dtype::F32);

        // Verify exp results: exp(0) ≈ 1.0, exp(1) ≈ 2.718, exp(2) ≈ 7.389
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert!((result_data[0] - 1.0).abs() < 0.001); // exp(0) ≈ 1
        assert!((result_data[1] - std::f32::consts::E).abs() < 0.001); // exp(1) ≈ e
        assert!((result_data[2] - (std::f32::consts::E * std::f32::consts::E)).abs() < 0.001);
        // exp(2) ≈ e²
    }

    #[test]
    fn test_sqrt() {
        let a = Array::from_slice(&[4.0f32, 9.0, 16.0], Shape::from([3])).unwrap();
        let result = sqrt(&a).unwrap();

        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), crate::dtype::F32);

        // Verify sqrt results: sqrt([4,9,16]) = [2,3,4]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 2.0);
        assert_eq!(result_data[1], 3.0);
        assert_eq!(result_data[2], 4.0);
    }

    #[test]
    fn test_abs() {
        let a = Array::from_slice(&[-1.0f32, 2.0, -3.0], Shape::from([3])).unwrap();
        let result = abs(&a).unwrap();

        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), crate::dtype::F32);

        // Verify abs results: abs([-1,2,-3]) = [1,2,3]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 1.0);
        assert_eq!(result_data[1], 2.0);
        assert_eq!(result_data[2], 3.0);
    }
}
