//! Reduction operations (sum, mean, max, min, etc.)

<<<<<<< HEAD
use crate::array::{NdArray, data_as_slice, data_as_slice_mut, ensure_host_accessible};
=======
use crate::array::{
    CpuBytesArray, NdArray, data_as_slice, data_as_slice_mut, ensure_host_accessible,
};
>>>>>>> b996c862c8d52a59d50b3035ebc36183885da337
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

    let mut result = array.zeros(output_shape)?;

    // Simple implementation - in practice would be much more optimized
    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

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
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

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

    Ok(result)
}

/// Sum all elements
fn sum_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = array.zeros(Shape::from([1]))?;

    match array.dtype() {
        DType::F16 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut sum = 0.0f32;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut sum = 0.0f32;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            let mut sum = 0.0f64;
            for &val in tensor_data {
                sum += val;
            }
            result_data[0] = sum;
        }
        DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            let mut sum = 0.0f32;
            for &val in tensor_data {
                sum += val.to_f32();
            }
            result_data[0] = crate::BFloat16::from_f32(sum);
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };
            let result_data = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            let mut sum = 0i8;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::I16 => {
            let tensor_data = unsafe { data_as_slice::<i16>(array) };
            let result_data = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            let mut sum = 0i16;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::I32 => {
            let tensor_data = unsafe { data_as_slice::<i32>(array) };
            let result_data = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            let mut sum = 0i32;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::I64 => {
            let tensor_data = unsafe { data_as_slice::<i64>(array) };
            let result_data = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            let mut sum = 0i64;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::U8 => {
            let tensor_data = unsafe { data_as_slice::<u8>(array) };
            let result_data = unsafe { data_as_slice_mut::<u8>(&mut *result) };

            let mut sum = 0u8;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::U16 => {
            let tensor_data = unsafe { data_as_slice::<u16>(array) };
            let result_data = unsafe { data_as_slice_mut::<u16>(&mut *result) };

            let mut sum = 0u16;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::U32 => {
            let tensor_data = unsafe { data_as_slice::<u32>(array) };
            let result_data = unsafe { data_as_slice_mut::<u32>(&mut *result) };

            let mut sum = 0u32;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::U64 => {
            let tensor_data = unsafe { data_as_slice::<u64>(array) };
            let result_data = unsafe { data_as_slice_mut::<u64>(&mut *result) };

            let mut sum = 0u64;
            for &val in tensor_data {
                sum = sum.wrapping_add(val);
            }
            result_data[0] = sum;
        }
        DType::Bool => {
            return Err("Sum not supported for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Sum not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
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
    let mut result = array.zeros(result_shape)?;

    match array.dtype() {
        DType::F16 => {
            let sum_data = unsafe { data_as_slice::<f32>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = sum_data[i] / count as f32;
            }
        }
        DType::F32 => {
            let sum_data = unsafe { data_as_slice::<f32>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = sum_data[i] / count as f32;
            }
        }
        DType::F64 => {
            let sum_data = unsafe { data_as_slice::<f64>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = sum_data[i] / count;
            }
        }
        DType::BF16 => {
            let sum_data = unsafe { data_as_slice::<crate::BFloat16>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };
            for i in 0..sum_data.len() {
                let result_f32 = sum_data[i].to_f32() / count as f32;
                result_data[i] = crate::BFloat16::from_f32(result_f32);
            }
        }
        DType::I8 => {
            let sum_data = unsafe { data_as_slice::<i8>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<i8>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f32 / count as f32) as i8;
            }
        }
        DType::I16 => {
            let sum_data = unsafe { data_as_slice::<i16>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<i16>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f32 / count as f32) as i16;
            }
        }
        DType::I32 => {
            let sum_data = unsafe { data_as_slice::<i32>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<i32>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f64 / count) as i32;
            }
        }
        DType::I64 => {
            let sum_data = unsafe { data_as_slice::<i64>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<i64>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f64 / count) as i64;
            }
        }
        DType::U8 => {
            let sum_data = unsafe { data_as_slice::<u8>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<u8>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f32 / count as f32) as u8;
            }
        }
        DType::U16 => {
            let sum_data = unsafe { data_as_slice::<u16>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<u16>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f32 / count as f32) as u16;
            }
        }
        DType::U32 => {
            let sum_data = unsafe { data_as_slice::<u32>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<u32>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f64 / count) as u32;
            }
        }
        DType::U64 => {
            let sum_data = unsafe { data_as_slice::<u64>(&*sum_result) };
            let result_data = unsafe { data_as_slice_mut::<u64>(&mut *result) };
            for i in 0..sum_data.len() {
                result_data[i] = (sum_data[i] as f64 / count) as u64;
            }
        }
        DType::Bool => {
            return Err("Mean not supported for boolean type".to_string());
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Mean not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
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

    let mut result = array.zeros(output_shape)?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

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

    Ok(result)
}

/// Max of all elements
fn max_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = array.zeros(Shape::from([1]))?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut max_val = f32::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            let mut max_val = f64::NEG_INFINITY;
            for &val in tensor_data {
                max_val = max_val.max(val);
            }
            result_data[0] = max_val;
        }
        _ => return Err(format!("Max not implemented for {}", array.dtype())),
    }

    Ok(result)
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

    let mut result = array.zeros(output_shape)?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

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

    Ok(result)
}

/// Min of all elements
fn min_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = array.zeros(Shape::from([1]))?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut min_val = f32::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            let mut min_val = f64::INFINITY;
            for &val in tensor_data {
                min_val = min_val.min(val);
            }
            result_data[0] = min_val;
        }
        _ => return Err(format!("Min not implemented for {}", array.dtype())),
    }

    Ok(result)
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

    let mut result = array.ones(output_shape)?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

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

    Ok(result)
}

/// Product of all elements
fn prod_all<A: NdArray>(array: &A) -> Result<Box<dyn NdArray>, String> {
    let mut result = array.ones(Shape::from([1]))?;

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut prod = 1.0f32;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            let mut prod = 1.0f64;
            for &val in tensor_data {
                prod *= val;
            }
            result_data[0] = prod;
        }
        _ => return Err(format!("Prod not implemented for {}", array.dtype())),
    }

    Ok(result)
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

        // Verify sum result: 1+2+3+4 = 10
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 10.0);
    }

    #[test]
    fn test_sum_axis() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = sum(&array, Some(0)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Sum along axis 0
        assert_eq!(result.dtype(), dtype::F32);

        // Verify sum along axis 0: [[1,2],[3,4]] -> [1+3, 2+4] = [4, 6]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 4.0);
        assert_eq!(result_data[1], 6.0);
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

        // Verify mean result: (1+2+3+4)/4 = 2.5
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 2.5);
    }

    #[test]
    fn test_max_all() {
        let data = [1.0f32, 5.0, 3.0, 2.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = max(&array, None).unwrap();
        assert_eq!(result.shape(), &Shape::from([1]));
        assert_eq!(result.dtype(), dtype::F32);

        // Verify max result: max([1,5,3,2]) = 5
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 5.0);
    }

    #[test]
    fn test_min_axis() {
        let data = [3.0f32, 1.0, 4.0, 2.0];
        let array = Array::from_slice(&data, Shape::from([2, 2])).unwrap();

        let result = min(&array, Some(1)).unwrap();
        assert_eq!(result.shape(), &Shape::from([2])); // Min along axis 1
        assert_eq!(result.dtype(), dtype::F32);

        // Verify min along axis 1: [[3,1],[4,2]] -> [min(3,1), min(4,2)] = [1, 2]
        let result_data = unsafe { data_as_slice::<f32>(&*result) };
        assert_eq!(result_data[0], 1.0);
        assert_eq!(result_data[1], 2.0);
    }
}
