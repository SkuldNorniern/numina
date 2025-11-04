//! Reduction operations (sum, mean, max, min, etc.)

use crate::array::{NdArray, CpuBytesArray, data_as_slice, data_as_slice_mut, ensure_host_accessible};
use crate::{DType, Shape};

/// Sum reduction along specified axis (or all axes if None)
pub fn sum<A: NdArray>(array: &A, axis: Option<usize>) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sum")?;

    match axis {
        Some(axis) => sum_axis(array, axis),
        None => sum_all(array),
    }
}

/// Sum along a specific axis
fn sum_axis<A: NdArray>(array: &A, axis: usize) -> Result<Box<dyn NdArray>, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D tensor",
            axis,
            array.shape().ndim()
        ));
    }

    // Calculate output shape (remove the specified axis)
    let mut output_dims = array.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = CpuBytesArray::zeros(array.dtype(), output_shape);

    // Simple implementation - in practice would be much more optimized
    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            // This is a simplified implementation for 2D tensors
            if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

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
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };

            // Simplified 2D implementation
            if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

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
        _ => return Err(format!("Sum not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Sum all elements
fn sum_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = CpuBytesArray::zeros(array.dtype(), Shape::from([1]));

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            let mut sum = 0.0f32;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };

            let mut sum = 0.0f64;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        _ => return Err(format!("Sum not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Mean reduction
pub fn mean<A: NdArray>(array: &A, axis: Option<usize>) -> Result<Box<dyn NdArray>, String> {
    let sum_result = sum(array, axis)?;
    let count = match axis {
        Some(axis) => array.shape().dim(axis) as f64,
        None => array.len() as f64,
    };

    // Create new result and divide sum by count
    let result_shape = sum_result.shape().clone();
    let mut result = CpuBytesArray::zeros(array.dtype(), result_shape);

    match array.dtype() {
        DType::F32 => {
            let sum_data = unsafe { data_as_slice::<f32>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };
            for i in 0..sum_data.len() {
                result_data[i] = sum_data[i] / count as f32;
            }
        }
        DType::F64 => {
            let sum_data = unsafe { data_as_slice::<f64>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };
            for i in 0..sum_data.len() {
                result_data[i] = sum_data[i] / count;
            }
        }
        _ => return Err(format!("Mean not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Maximum value
pub fn max<A: NdArray>(array: &A, axis: Option<usize>) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "max")?;

    match axis {
        Some(axis) => max_axis(array, axis),
        None => max_all(array),
    }
}

/// Max along a specific axis
fn max_axis<A: NdArray>(array: &A, axis: usize) -> Result<Box<dyn NdArray>, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D tensor",
            axis,
            array.shape().ndim()
        ));
    }

    // Calculate output shape (remove the specified axis)
    let mut output_dims = array.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = CpuBytesArray::zeros(array.dtype(), output_shape);

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

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
        _ => return Err(format!("Max not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Max of all elements
fn max_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = CpuBytesArray::zeros(array.dtype(), Shape::from([1]));

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            let mut max_val = f32::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };

            let mut max_val = f64::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        _ => return Err(format!("Max not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Minimum value
pub fn min<A: NdArray>(array: &A, axis: Option<usize>) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "min")?;

    match axis {
        Some(axis) => min_axis(array, axis),
        None => min_all(array),
    }
}

/// Min along a specific axis
fn min_axis<A: NdArray>(array: &A, axis: usize) -> Result<Box<dyn NdArray>, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D tensor",
            axis,
            array.shape().ndim()
        ));
    }

    let mut output_dims = array.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = CpuBytesArray::zeros(array.dtype(), output_shape);

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

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
        _ => return Err(format!("Min not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Min of all elements
fn min_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = CpuBytesArray::zeros(array.dtype(), Shape::from([1]));

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            let mut min_val = f32::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };

            let mut min_val = f64::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        _ => return Err(format!("Min not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Product of all elements
pub fn prod<A: NdArray>(array: &A, axis: Option<usize>) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "prod")?;

    match axis {
        Some(axis) => prod_axis(array, axis),
        None => prod_all(array),
    }
}

/// Product along a specific axis
fn prod_axis<A: NdArray>(array: &A, axis: usize) -> Result<Box<dyn NdArray>, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D tensor",
            axis,
            array.shape().ndim()
        ));
    }

    let mut output_dims = array.shape().dims().to_vec();
    output_dims.remove(axis);
    let output_shape = if output_dims.is_empty() {
        Shape::from([1])
    } else {
        Shape::from(output_dims)
    };

    let mut result = CpuBytesArray::ones(array.dtype(), output_shape);

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

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
        _ => return Err(format!("Prod not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

/// Product of all elements
fn prod_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = CpuBytesArray::ones(array.dtype(), Shape::from([1]));

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut result) };

            let mut prod = 1.0f32;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut result) };

            let mut prod = 1.0f64;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        _ => return Err(format!("Prod not implemented for {}", array.dtype())),
    }

    Ok(result.into_boxed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;
    use crate::dtype;

    #[test]
    fn test_sum_all() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = sum(&array, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), dtype::F32);
    }

    #[test]
    fn test_sum_axis() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = sum(&array, Some(0)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Sum along axis 0
        assert_eq!(result.dtype(), dtype::F32);
    }

    #[test]
    fn test_sum_array_backend() {
        use crate::array::Array;

        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2])).unwrap();
        let result = sum(&array, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), dtype::F32);
    }

    #[test]
    fn test_mean() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = mean(&array, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), dtype::F32);
        // Mean of [1,2,3,4] = 2.5
    }

    #[test]
    fn test_max_all() {
        let data = [1.0f32, 5.0, 3.0, 2.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = max(&array, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), dtype::F32);
    }

    #[test]
    fn test_min_axis() {
        let data = [3.0f32, 1.0, 4.0, 2.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = min(&array, Some(1)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Min along axis 1
        assert_eq!(result.dtype(), dtype::F32);
    }
}
