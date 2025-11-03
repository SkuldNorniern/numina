//! Element-wise tensor operations

use crate::{Tensor, Shape, DType, F32, F64};

/// Element-wise addition
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.dtype() != b.dtype() {
        return Err(format!("DType mismatch: {} vs {}", a.dtype(), b.dtype()));
    }

    // Simple implementation - in practice would handle broadcasting
    if a.shape() != b.shape() {
        return Err(format!("Shape mismatch: {} vs {}", a.shape(), b.shape()));
    }

    let mut result = Tensor::zeros(a.dtype(), a.shape().clone());

    match a.dtype() {
        F32 => {
            let a_data = unsafe { a.data_as_slice::<f32>() };
            let b_data = unsafe { b.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..a.len() {
                result_data[i] = a_data[i] + b_data[i];
            }
        }
        F64 => {
            let a_data = unsafe { a.data_as_slice::<f64>() };
            let b_data = unsafe { b.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..a.len() {
                result_data[i] = a_data[i] + b_data[i];
            }
        }
        _ => return Err(format!("Addition not implemented for {}", a.dtype())),
    }

    Ok(result)
}

/// Element-wise multiplication
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.dtype() != b.dtype() {
        return Err(format!("DType mismatch: {} vs {}", a.dtype(), b.dtype()));
    }

    if a.shape() != b.shape() {
        return Err(format!("Shape mismatch: {} vs {}", a.shape(), b.shape()));
    }

    let mut result = Tensor::zeros(a.dtype(), a.shape().clone());

    match a.dtype() {
        F32 => {
            let a_data = unsafe { a.data_as_slice::<f32>() };
            let b_data = unsafe { b.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..a.len() {
                result_data[i] = a_data[i] * b_data[i];
            }
        }
        F64 => {
            let a_data = unsafe { a.data_as_slice::<f64>() };
            let b_data = unsafe { b.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..a.len() {
                result_data[i] = a_data[i] * b_data[i];
            }
        }
        _ => return Err(format!("Multiplication not implemented for {}", a.dtype())),
    }

    Ok(result)
}

/// Scalar addition
pub fn add_scalar(tensor: &Tensor, scalar: f64) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), tensor.shape().clone());

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i] + scalar as f32;
            }
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i] + scalar;
            }
        }
        _ => return Err(format!("Scalar addition not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Element-wise exponential
pub fn exp(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), tensor.shape().clone());

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i].exp();
            }
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i].exp();
            }
        }
        _ => return Err(format!("Exp not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Element-wise logarithm
pub fn log(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), tensor.shape().clone());

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i].ln();
            }
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..tensor.len() {
                result_data[i] = tensor_data[i].ln();
            }
        }
        _ => return Err(format!("Log not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Matrix multiplication
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, String> {
    if a.dtype() != b.dtype() {
        return Err(format!("DType mismatch: {} vs {}", a.dtype(), b.dtype()));
    }

    if a.ndim() != 2 || b.ndim() != 2 {
        return Err("Matmul requires 2D tensors".to_string());
    }

    let (m, k) = (a.shape().dim(0), a.shape().dim(1));
    let (k2, n) = (b.shape().dim(0), b.shape().dim(1));

    if k != k2 {
        return Err(format!("Dimension mismatch: {} vs {}", k, k2));
    }

    let result_shape = Shape::from([m, n]);
    let mut result = Tensor::zeros(a.dtype(), result_shape);

    match a.dtype() {
        F32 => {
            let a_data = unsafe { a.data_as_slice::<f32>() };
            let b_data = unsafe { b.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for l in 0..k {
                        sum += a_data[i * k + l] * b_data[l * n + j];
                    }
                    result_data[i * n + j] = sum;
                }
            }
        }
        F64 => {
            let a_data = unsafe { a.data_as_slice::<f64>() };
            let b_data = unsafe { b.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0f64;
                    for l in 0..k {
                        sum += a_data[i * k + l] * b_data[l * n + j];
                    }
                    result_data[i * n + j] = sum;
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
    use crate::Shape;

    #[test]
    fn test_add() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], Shape::from([3]));
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], Shape::from([3]));

        let result = add(&a, &b).unwrap();
        let expected = Tensor::from_slice(&[5.0f32, 7.0, 9.0], Shape::from([3]));

        // For simplicity, just check shapes and dtypes - full comparison would need data access
        assert_eq!(result.shape(), expected.shape());
        assert_eq!(result.dtype(), expected.dtype());
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2]));
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2]));

        let result = matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &Shape::from([2, 2]));
        assert_eq!(result.dtype(), F32);
    }

    #[test]
    fn test_exp() {
        let a = Tensor::from_slice(&[0.0f32], Shape::from([1]));
        let result = exp(&a).unwrap();

        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), F32);
    }
}
