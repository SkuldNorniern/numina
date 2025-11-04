//! Sorting and searching operations

use crate::array::{NdArray, data_as_slice, data_as_slice_mut, ensure_host_accessible};
use crate::{DType, Shape};

/// Sort array along specified axis
pub fn sort<A: NdArray>(
    array: &A,
    axis: Option<usize>,
    descending: bool,
) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "sort")?;

    match axis {
        Some(axis) => sort_axis(array, axis, descending),
        None => sort_flatten(array, descending),
    }
}

/// Sort along a specific axis
fn sort_axis<A: NdArray>(
    array: &A,
    axis: usize,
    descending: bool,
) -> Result<Box<dyn NdArray>, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D array",
            axis,
            array.shape().ndim()
        ));
    }

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::BF16 => {
            // For floating point types, convert to f64 for sorting, then convert back
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
                if descending {
                    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
                } else {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
                for i in 0..values.len() {
                    result_data[i] = values[i] as f32;
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| tensor_data[i * cols + j] as f64)
                            .collect();
                        if descending {
                            column.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            column.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for i in 0..rows {
                            result_data[i * cols + j] = column[i] as f32;
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| tensor_data[i * cols + j] as f64)
                            .collect();
                        if descending {
                            row.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for j in 0..cols {
                            result_data[i * cols + j] = row[j] as f32;
                        }
                    }
                }
            }
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };
            let result_data = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<i8> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::I16 => {
            let tensor_data = unsafe { data_as_slice::<i16>(array) };
            let result_data = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<i16> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::I32 => {
            let tensor_data = unsafe { data_as_slice::<i32>(array) };
            let result_data = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<i32> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::I64 => {
            let tensor_data = unsafe { data_as_slice::<i64>(array) };
            let result_data = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<i64> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::U8 => {
            let tensor_data = unsafe { data_as_slice::<u8>(array) };
            let result_data = unsafe { data_as_slice_mut::<u8>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<u8> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::U16 => {
            let tensor_data = unsafe { data_as_slice::<u16>(array) };
            let result_data = unsafe { data_as_slice_mut::<u16>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<u16> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::U32 => {
            let tensor_data = unsafe { data_as_slice::<u32>(array) };
            let result_data = unsafe { data_as_slice_mut::<u32>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<u32> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::U64 => {
            let tensor_data = unsafe { data_as_slice::<u64>(array) };
            let result_data = unsafe { data_as_slice_mut::<u64>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<u64> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.cmp(a));
                } else {
                    values.sort_by(|a, b| a.cmp(b));
                }
                result_data.copy_from_slice(&values);
            }
        }
        DType::Bool => {
            return Err(format!("Sort not supported for boolean type"));
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Sort not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
}

/// Sort flattened array
fn sort_flatten<A: NdArray>(array: &A, descending: bool) -> Result<Box<dyn NdArray>, String> {
    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
            if descending {
                values.sort_by(|a, b| b.partial_cmp(a).unwrap());
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            for i in 0..values.len() {
                result_data[i] = values[i] as f32;
            }
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };
            let result_data = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            let mut values: Vec<i8> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::I16 => {
            let tensor_data = unsafe { data_as_slice::<i16>(array) };
            let result_data = unsafe { data_as_slice_mut::<i16>(&mut *result) };

            let mut values: Vec<i16> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::I32 => {
            let tensor_data = unsafe { data_as_slice::<i32>(array) };
            let result_data = unsafe { data_as_slice_mut::<i32>(&mut *result) };

            let mut values: Vec<i32> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::I64 => {
            let tensor_data = unsafe { data_as_slice::<i64>(array) };
            let result_data = unsafe { data_as_slice_mut::<i64>(&mut *result) };

            let mut values: Vec<i64> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::U8 => {
            let tensor_data = unsafe { data_as_slice::<u8>(array) };
            let result_data = unsafe { data_as_slice_mut::<u8>(&mut *result) };

            let mut values: Vec<u8> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::U16 => {
            let tensor_data = unsafe { data_as_slice::<u16>(array) };
            let result_data = unsafe { data_as_slice_mut::<u16>(&mut *result) };

            let mut values: Vec<u16> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::U32 => {
            let tensor_data = unsafe { data_as_slice::<u32>(array) };
            let result_data = unsafe { data_as_slice_mut::<u32>(&mut *result) };

            let mut values: Vec<u32> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::U64 => {
            let tensor_data = unsafe { data_as_slice::<u64>(array) };
            let result_data = unsafe { data_as_slice_mut::<u64>(&mut *result) };

            let mut values: Vec<u64> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort_by(|a, b| a.cmp(b));
            }
            result_data.copy_from_slice(&values);
        }
        DType::Bool => {
            return Err(format!("Sort not supported for boolean type"));
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Sort not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    Ok(result)
}

/// Get indices that would sort the array
pub fn argsort<A: NdArray>(
    array: &A,
    _axis: Option<usize>,
    descending: bool,
) -> Result<Box<dyn NdArray>, String> {
    ensure_host_accessible(array, "argsort")?;

    if array.shape().ndim() != 1 {
        return Err("Argsort currently only supports 1D arrays".to_string());
    }

    let mut indices: Vec<i32> = (0..array.len() as i32).collect();

    match array.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::BF16 => {
            // Convert to f64 for consistent comparison
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();

            if descending {
                indices
                    .sort_by(|&a, &b| values[b as usize].partial_cmp(&values[a as usize]).unwrap());
            } else {
                indices
                    .sort_by(|&a, &b| values[a as usize].partial_cmp(&values[b as usize]).unwrap());
            }
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::I16 => {
            let tensor_data = unsafe { data_as_slice::<i16>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::I32 => {
            let tensor_data = unsafe { data_as_slice::<i32>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::I64 => {
            let tensor_data = unsafe { data_as_slice::<i64>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::U8 => {
            let tensor_data = unsafe { data_as_slice::<u8>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::U16 => {
            let tensor_data = unsafe { data_as_slice::<u16>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::U32 => {
            let tensor_data = unsafe { data_as_slice::<u32>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::U64 => {
            let tensor_data = unsafe { data_as_slice::<u64>(array) };

            if descending {
                indices.sort_by(|&a, &b| tensor_data[b as usize].cmp(&tensor_data[a as usize]));
            } else {
                indices.sort_by(|&a, &b| tensor_data[a as usize].cmp(&tensor_data[b as usize]));
            }
        }
        DType::Bool => {
            return Err(format!("Argsort not supported for boolean type"));
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Argsort not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    let mut result = array.new_array(Shape::from([array.len()]), DType::I32)?;
    let result_data = unsafe { data_as_slice_mut::<i32>(&mut *result) };
    for (i, &idx) in indices.iter().enumerate() {
        result_data[i] = idx;
    }
    Ok(result)
}

/// Find indices where condition is true (basic boolean indexing support)
pub fn where_condition<A, F>(array: &A, condition: F) -> Result<Vec<usize>, String>
where
    A: NdArray,
    F: Fn(f32) -> bool,
{
    ensure_host_accessible(array, "where")?;
    let mut indices = Vec::new();

    match array.dtype() {
        DType::F16 | DType::F32 | DType::F64 | DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val) {
                    indices.push(i);
                }
            }
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::I16 => {
            let tensor_data = unsafe { data_as_slice::<i16>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::I32 => {
            let tensor_data = unsafe { data_as_slice::<i32>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::I64 => {
            let tensor_data = unsafe { data_as_slice::<i64>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::U8 => {
            let tensor_data = unsafe { data_as_slice::<u8>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::U16 => {
            let tensor_data = unsafe { data_as_slice::<u16>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::U32 => {
            let tensor_data = unsafe { data_as_slice::<u32>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::U64 => {
            let tensor_data = unsafe { data_as_slice::<u64>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::Bool => {
            return Err(format!("Where not supported for boolean type"));
        }
        DType::QI4 | DType::QU8 => {
            return Err(format!(
                "Where not implemented for quantized types {}",
                array.dtype()
            ));
        }
    }

    Ok(indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::Array;

    #[test]
    fn test_sort_1d() {
        let data = [3.0f32, 1.0, 4.0, 1.5, 9.0];
        let array = Array::from_slice(&data, Shape::from([5])).unwrap();
        let result = sort(&array, None, false).unwrap();

        assert_eq!(result.shape(), &Shape::from([5]));
        assert_eq!(result.dtype(), DType::F32);
    }

    #[test]
    fn test_sort_1d_descending() {
        let data = [3.0f32, 1.0, 4.0];
        let array = Array::from_slice(&data, Shape::from([3])).unwrap();
        let result = sort(&array, None, true).unwrap();

        assert_eq!(result.shape(), &Shape::from([3]));
        assert_eq!(result.dtype(), DType::F32);
    }

    #[test]
    fn test_argsort() {
        let data = [3.0f32, 1.0, 4.0, 1.5];
        let array = Array::from_slice(&data, Shape::from([4])).unwrap();
        let result = argsort(&array, None, false).unwrap();

        assert_eq!(result.shape(), &Shape::from([4]));
        assert_eq!(result.dtype(), DType::I32);
    }

    #[test]
    fn test_where_condition() {
        let data = [1.0f32, 5.0, 2.0, 8.0, 3.0];
        let array = Array::from_slice(&data, Shape::from([5])).unwrap();
        let indices = where_condition(&array, |x| x > 3.0).unwrap();

        assert_eq!(indices, vec![1, 3]); // indices of 5.0 and 8.0
    }
}
