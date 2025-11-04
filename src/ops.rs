//! Element-wise array operations built on the `NdArray` abstraction

use crate::array::{NdArray, CpuBytesArray, data_as_slice, data_as_slice_mut, ensure_host_accessible, ensure_binary_compat};
use crate::{DType, Shape};

/// Element-wise addition
pub fn add<A, B>(a: &A, b: &B) -> Result<Box<dyn NdArray>, String>
where
    A: NdArray,
    B: NdArray,
{
    ensure_binary_compat(a, b, "add")?;

    let mut result = CpuBytesArray::zeros(a.dtype(), a.shape().clone());

    match a.dtype() {
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] + rhs[i];
            }
        }
        DType::F64 => {
            let lhs = unsafe { data_as_slice::<f64>(a) };
            let rhs = unsafe { data_as_slice::<f64>(b) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] + rhs[i];
            }
        }
        _ => return Err(format!("Addition not implemented for {}", a.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise multiplication
pub fn mul<A, B>(a: &A, b: &B) -> Result<Box<dyn NdArray>, String>
where
    A: NdArray,
    B: NdArray,
{
    ensure_binary_compat(a, b, "mul")?;

    let mut result = CpuBytesArray::zeros(a.dtype(), a.shape().clone());

    match a.dtype() {
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] * rhs[i];
            }
        }
        DType::F64 => {
            let lhs = unsafe { data_as_slice::<f64>(a) };
            let rhs = unsafe { data_as_slice::<f64>(b) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..a.len() {
                dst[i] = lhs[i] * rhs[i];
            }
        }
        _ => return Err(format!("Multiplication not implemented for {}", a.dtype())),
    }

    Ok(result.into_boxed())
}

/// Scalar addition
pub fn add_scalar<A: NdArray>(array: &A, scalar: f64) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "add_scalar")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i] + scalar as f32;
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i] + scalar;
            }
        }
        _ => {
            return Err(format!(
                "Scalar addition not implemented for {}",
                array.dtype()
            ));
        }
    }

    Ok(result.into_boxed())
}

/// Element-wise exponential
pub fn exp<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "exp")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].exp();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].exp();
            }
        }
        _ => return Err(format!("Exp not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise logarithm
pub fn log<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "log")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].ln();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].ln();
            }
        }
        _ => return Err(format!("Log not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise square root
pub fn sqrt<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sqrt")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].sqrt();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].sqrt();
            }
        }
        _ => return Err(format!("Sqrt not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise sine
pub fn sin<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sin")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].sin();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].sin();
            }
        }
        _ => return Err(format!("Sin not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise cosine
pub fn cos<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "cos")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].cos();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].cos();
            }
        }
        _ => return Err(format!("Cos not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise tangent
pub fn tan<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "tan")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].tan();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].tan();
            }
        }
        _ => return Err(format!("Tan not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise arcsine
pub fn asin<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "asin")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].asin();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].asin();
            }
        }
        _ => return Err(format!("Asin not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise arccosine
pub fn acos<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "acos")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].acos();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].acos();
            }
        }
        _ => return Err(format!("Acos not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise arctangent
pub fn atan<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "atan")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].atan();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].atan();
            }
        }
        _ => return Err(format!("Atan not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise power (array^exponent)
pub fn pow<A: NdArray>(array: &A, exponent: f64) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "pow")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].powf(exponent as f32);
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].powf(exponent);
            }
        }
        _ => return Err(format!("Pow not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise absolute value
pub fn abs<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "abs")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::F64 => {
            let src = unsafe { data_as_slice::<f64>(array) };
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::I32 => {
            let src = unsafe { data_as_slice::<i32>(array) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        DType::I64 => {
            let src = unsafe { data_as_slice::<i64>(array) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].abs();
            }
        }
        _ => return Err(format!("Abs not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Element-wise sign function
pub fn sign<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sign")?;

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let src = unsafe { data_as_slice::<f32>(array) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

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
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

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
        DType::I32 => {
            let src = unsafe { data_as_slice::<i32>(array) };
            let dst = unsafe { data_as_slice_mut::<i32>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        DType::I64 => {
            let src = unsafe { data_as_slice::<i64>(array) };
            let dst = unsafe { data_as_slice_mut::<i64>(&mut result) };

            for i in 0..array.len() {
                dst[i] = src[i].signum();
            }
        }
        _ => return Err(format!("Sign not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
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
    let mut result = CpuBytesArray::zeros(a.dtype(), result_shape);

    match a.dtype() {
        DType::F32 => {
            let lhs = unsafe { data_as_slice::<f32>(a) };
            let rhs = unsafe { data_as_slice::<f32>(b) };
            let dst = unsafe { data_as_slice_mut::<f32>(&mut result) };

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
            let dst = unsafe { data_as_slice_mut::<f64>(&mut result) };

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
        _ => return Err(format!("Matmul not implemented for {}", a.dtype())),
    }

    Ok(result.into_boxed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::{Shape, Tensor};

    #[test]
    fn test_add_tensor() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], Shape::from([3]));
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], Shape::from([3]));

        let result = add(&a, &b).unwrap();
        let tensor_result = Tensor::from_ndarray(result);

        assert_eq!(tensor_result.shape(), &Shape::from([3]));
        assert_eq!(tensor_result.dtype(), crate::dtype::F32);
    }

    #[test]
    fn test_add_array() {
        let a = Array::from_slice(&[1.0f32, 2.0, 3.0], Shape::from([3])).unwrap();
        let b = Array::from_slice(&[4.0f32, 5.0, 6.0], Shape::from([3])).unwrap();

        let result = add(&a, &b).unwrap();
        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), crate::dtype::F32);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2]));
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2]));

        let result = matmul(&a, &b).unwrap();
        let tensor_result = Tensor::from_ndarray(result);
        assert_eq!(tensor_result.shape(), &Shape::from([2, 2]));
        assert_eq!(tensor_result.dtype(), crate::dtype::F32);
    }

    #[test]
    fn test_exp() {
        let a = Tensor::from_slice(&[0.0f32], Shape::from([1]));
        let result = exp(&a).unwrap();
        let tensor_result = Tensor::from_ndarray(result);

        assert_eq!(tensor_result.shape(), &Shape::from([1]));
        assert_eq!(tensor_result.dtype(), crate::dtype::F32);
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::from_slice(&[4.0f32, 9.0, 16.0], Shape::from([3]));
        let result = sqrt(&a).unwrap();
        let tensor_result = Tensor::from_ndarray(result);

        assert_eq!(tensor_result.shape(), &Shape::from([3]));
        assert_eq!(tensor_result.dtype(), crate::dtype::F32);
    }

    #[test]
    fn test_abs() {
        let a = Tensor::from_slice(&[-1.0f32, 2.0, -3.0], Shape::from([3]));
        let result = abs(&a).unwrap();
        let tensor_result = Tensor::from_ndarray(result);

        assert_eq!(tensor_result.shape(), &Shape::from([3]));
        assert_eq!(tensor_result.dtype(), crate::dtype::F32);
    }
}
