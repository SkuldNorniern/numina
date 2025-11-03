//! Reduction operations (sum, mean, max, min, etc.)

use crate::{Tensor, Shape, DType, F32, F64};

/// Sum reduction along specified axis (or all axes if None)
pub fn sum(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
    match axis {
        Some(axis) => sum_axis(tensor, axis),
        None => sum_all(tensor),
    }
}

/// Sum along a specific axis
fn sum_axis(tensor: &Tensor, axis: usize) -> Result<Tensor, String> {
    if axis >= tensor.ndim() {
        return Err(format!("Axis {} out of bounds for {}D tensor", axis, tensor.ndim()));
    }

    // Calculate output shape (remove the specified axis)
    let mut output_dims = tensor.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = Tensor::zeros(tensor.dtype(), output_shape);

    // Simple implementation - in practice would be much more optimized
    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            // This is a simplified implementation for 2D tensors
            if tensor.ndim() == 2 {
                let (rows, cols) = (tensor.shape().dim(0), tensor.shape().dim(1));

                if axis == 0 {
                    // Sum along rows (output: [cols])
                    for j in 0..cols {
                        let mut sum = 0.0f32;
                        for i in 0..rows {
                            sum += tensor_data[i * cols + j];
                        }
                        result_data[j] = sum;
                    }
                } else if axis == 1 {
                    // Sum along columns (output: [rows])
                    for i in 0..rows {
                        let mut sum = 0.0f32;
                        for j in 0..cols {
                            sum += tensor_data[i * cols + j];
                        }
                        result_data[i] = sum;
                    }
                }
            } else {
                return Err("Multi-dimensional sum_axis not implemented".to_string());
            }
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            // Simplified 2D implementation
            if tensor.ndim() == 2 {
                let (rows, cols) = (tensor.shape().dim(0), tensor.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut sum = 0.0f64;
                        for i in 0..rows {
                            sum += tensor_data[i * cols + j];
                        }
                        result_data[j] = sum;
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut sum = 0.0f64;
                        for j in 0..cols {
                            sum += tensor_data[i * cols + j];
                        }
                        result_data[i] = sum;
                    }
                }
            } else {
                return Err("Multi-dimensional sum_axis not implemented".to_string());
            }
        }
        _ => return Err(format!("Sum not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Sum all elements
fn sum_all(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), Shape::from([1]));

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            let mut sum = 0.0f32;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            let mut sum = 0.0f64;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        _ => return Err(format!("Sum not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Mean reduction
pub fn mean(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
    let mut sum_result = sum(tensor, axis)?;
    let count = match axis {
        Some(axis) => tensor.shape().dim(axis) as f64,
        None => tensor.len() as f64,
    };

    // Divide sum by count
    match tensor.dtype() {
        F32 => {
            let data = unsafe { sum_result.data_as_slice_mut::<f32>() };
            for val in data.iter_mut() {
                *val /= count as f32;
            }
        }
        F64 => {
            let data = unsafe { sum_result.data_as_slice_mut::<f64>() };
            for val in data.iter_mut() {
                *val /= count;
            }
        }
        _ => return Err(format!("Mean not implemented for {}", tensor.dtype())),
    }

    Ok(sum_result)
}

/// Maximum value
pub fn max(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
    match axis {
        Some(axis) => max_axis(tensor, axis),
        None => max_all(tensor),
    }
}

/// Max along a specific axis
fn max_axis(tensor: &Tensor, axis: usize) -> Result<Tensor, String> {
    if axis >= tensor.ndim() {
        return Err(format!("Axis {} out of bounds for {}D tensor", axis, tensor.ndim()));
    }

    // Calculate output shape (remove the specified axis)
    let mut output_dims = tensor.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = Tensor::zeros(tensor.dtype(), output_shape);

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            if tensor.ndim() == 2 {
                let (rows, cols) = (tensor.shape().dim(0), tensor.shape().dim(1));

                if axis == 0 {
                    // Max along rows
                    for j in 0..cols {
                        let mut max_val = f32::NEG_INFINITY;
                        for i in 0..rows {
                            max_val = max_val.max(tensor_data[i * cols + j]);
                        }
                        result_data[j] = max_val;
                    }
                } else if axis == 1 {
                    // Max along columns
                    for i in 0..rows {
                        let mut max_val = f32::NEG_INFINITY;
                        for j in 0..cols {
                            max_val = max_val.max(tensor_data[i * cols + j]);
                        }
                        result_data[i] = max_val;
                    }
                }
            }
        }
        _ => return Err(format!("Max not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Max of all elements
fn max_all(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), Shape::from([1]));

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            let mut max_val = f32::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            let mut max_val = f64::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        _ => return Err(format!("Max not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Minimum value
pub fn min(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
    match axis {
        Some(axis) => min_axis(tensor, axis),
        None => min_all(tensor),
    }
}

/// Min along a specific axis
fn min_axis(tensor: &Tensor, axis: usize) -> Result<Tensor, String> {
    if axis >= tensor.ndim() {
        return Err(format!("Axis {} out of bounds for {}D tensor", axis, tensor.ndim()));
    }

    let mut output_dims = tensor.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = Tensor::zeros(tensor.dtype(), output_shape);

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            if tensor.ndim() == 2 {
                let (rows, cols) = (tensor.shape().dim(0), tensor.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut min_val = f32::INFINITY;
                        for i in 0..rows {
                            min_val = min_val.min(tensor_data[i * cols + j]);
                        }
                        result_data[j] = min_val;
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut min_val = f32::INFINITY;
                        for j in 0..cols {
                            min_val = min_val.min(tensor_data[i * cols + j]);
                        }
                        result_data[i] = min_val;
                    }
                }
            }
        }
        _ => return Err(format!("Min not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Min of all elements
fn min_all(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::zeros(tensor.dtype(), Shape::from([1]));

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            let mut min_val = f32::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            let mut min_val = f64::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        _ => return Err(format!("Min not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Product of all elements
pub fn prod(tensor: &Tensor, axis: Option<usize>) -> Result<Tensor, String> {
    match axis {
        Some(axis) => prod_axis(tensor, axis),
        None => prod_all(tensor),
    }
}

/// Product along a specific axis
fn prod_axis(tensor: &Tensor, axis: usize) -> Result<Tensor, String> {
    if axis >= tensor.ndim() {
        return Err(format!("Axis {} out of bounds for {}D tensor", axis, tensor.ndim()));
    }

    let mut output_dims = tensor.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = Tensor::ones(tensor.dtype(), output_shape);

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            if tensor.ndim() == 2 {
                let (rows, cols) = (tensor.shape().dim(0), tensor.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut prod = 1.0f32;
                        for i in 0..rows {
                            prod *= tensor_data[i * cols + j];
                        }
                        result_data[j] = prod;
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut prod = 1.0f32;
                        for j in 0..cols {
                            prod *= tensor_data[i * cols + j];
                        }
                        result_data[i] = prod;
                    }
                }
            }
        }
        _ => return Err(format!("Prod not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

/// Product of all elements
fn prod_all(tensor: &Tensor) -> Result<Tensor, String> {
    let mut result = Tensor::ones(tensor.dtype(), Shape::from([1]));

    match tensor.dtype() {
        F32 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f32>() };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            let mut prod = 1.0f32;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        F64 => {
            let tensor_data = unsafe { tensor.data_as_slice::<f64>() };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            let mut prod = 1.0f64;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        _ => return Err(format!("Prod not implemented for {}", tensor.dtype())),
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::F32;

    #[test]
    fn test_sum_all() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, Shape::from([2, 2]));

        let result = sum(&tensor, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), F32);
    }

    #[test]
    fn test_sum_axis() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, Shape::from([2, 2]));

        let result = sum(&tensor, Some(0)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Sum along axis 0
        assert_eq!(result.dtype(), F32);
    }

    #[test]
    fn test_mean() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor = Tensor::from_slice(&data, Shape::from([2, 2]));

        let result = mean(&tensor, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), F32);
        // Mean of [1,2,3,4] = 2.5
    }

    #[test]
    fn test_max_all() {
        let data = [1.0f32, 5.0, 3.0, 2.0];
        let tensor = Tensor::from_slice(&data, Shape::from([2, 2]));

        let result = max(&tensor, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), F32);
    }

    #[test]
    fn test_min_axis() {
        let data = [3.0f32, 1.0, 4.0, 2.0];
        let tensor = Tensor::from_slice(&data, Shape::from([2, 2]));

        let result = min(&tensor, Some(1)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Min along axis 1
        assert_eq!(result.dtype(), F32);
    }
}
