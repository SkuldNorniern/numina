//! Sorting and searching operations

use crate::array::{NdArray, CpuBytesArray, data_as_slice, ensure_host_accessible};
use crate::{Shape, DType};

/// Sort array along specified axis
pub fn sort<A: NdArray>(array: &A, axis: Option<usize>, descending: bool) -> Result<CpuBytesArray, String> {
    ensure_host_accessible(array, "sort")?;

    match axis {
        Some(axis) => sort_axis(array, axis, descending),
        None => sort_flatten(array, descending),
    }
}

/// Sort along a specific axis
fn sort_axis<A: NdArray>(array: &A, axis: usize, descending: bool) -> Result<CpuBytesArray, String> {
    if axis >= array.shape().ndim() {
        return Err(format!(
            "Axis {} out of bounds for {}D array",
            axis,
            array.shape().ndim()
        ));
    }

    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            if array.shape().ndim() == 1 {
                // 1D tensor
                let mut values: Vec<f32> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
                } else {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
                result_data.copy_from_slice(&values);
            } else if array.shape().ndim() == 2 {
                // 2D tensor
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    // Sort along columns
                    for j in 0..cols {
                        let mut column: Vec<f32> = (0..rows).map(|i| tensor_data[i * cols + j]).collect();
                        if descending {
                            column.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            column.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for i in 0..rows {
                            result_data[i * cols + j] = column[i];
                        }
                    }
                } else if axis == 1 {
                    // Sort along rows
                    for i in 0..rows {
                        let mut row: Vec<f32> = (0..cols).map(|j| tensor_data[i * cols + j]).collect();
                        if descending {
                            row.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for j in 0..cols {
                            result_data[i * cols + j] = row[j];
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> = tensor_data.to_vec();
                if descending {
                    values.sort_by(|a, b| b.partial_cmp(a).unwrap());
                } else {
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                }
                result_data.copy_from_slice(&values);
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows).map(|i| tensor_data[i * cols + j]).collect();
                        if descending {
                            column.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            column.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for i in 0..rows {
                            result_data[i * cols + j] = column[i];
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols).map(|j| tensor_data[i * cols + j]).collect();
                        if descending {
                            row.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        } else {
                            row.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        }
                        for j in 0..cols {
                            result_data[i * cols + j] = row[j];
                        }
                    }
                }
            }
        }
        _ => return Err(format!("Sort not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Sort flattened array
fn sort_flatten<A: NdArray>(array: &A, descending: bool) -> Result<CpuBytesArray, String> {
    let mut result = CpuBytesArray::zeros(array.dtype(), array.shape().clone());

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { result.data_as_slice_mut::<f32>() };

            let mut values: Vec<f32> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.partial_cmp(a).unwrap());
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            result_data.copy_from_slice(&values);
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { result.data_as_slice_mut::<f64>() };

            let mut values: Vec<f64> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.partial_cmp(a).unwrap());
            } else {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            }
            result_data.copy_from_slice(&values);
        }
        _ => return Err(format!("Sort not implemented for {}", array.dtype())),
    }

    Ok(result)
}

/// Get indices that would sort the array
pub fn argsort<A: NdArray>(array: &A, _axis: Option<usize>, descending: bool) -> Result<CpuBytesArray, String> {
    ensure_host_accessible(array, "argsort")?;

    if array.shape().ndim() != 1 {
        return Err("Argsort currently only supports 1D arrays".to_string());
    }

    let mut indices: Vec<i32> = (0..array.len() as i32).collect();

    match array.dtype() {
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };

            if descending {
                indices.sort_by(|&a, &b| {
                    tensor_data[b as usize].partial_cmp(&tensor_data[a as usize]).unwrap()
                });
            } else {
                indices.sort_by(|&a, &b| {
                    tensor_data[a as usize].partial_cmp(&tensor_data[b as usize]).unwrap()
                });
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };

            if descending {
                indices.sort_by(|&a, &b| {
                    tensor_data[b as usize].partial_cmp(&tensor_data[a as usize]).unwrap()
                });
            } else {
                indices.sort_by(|&a, &b| {
                    tensor_data[a as usize].partial_cmp(&tensor_data[b as usize]).unwrap()
                });
            }
        }
        _ => return Err(format!("Argsort not implemented for {}", array.dtype())),
    }

    Ok(CpuBytesArray::new(
        unsafe {
            std::slice::from_raw_parts(
                indices.as_ptr() as *const u8,
                indices.len() * std::mem::size_of::<i32>(),
            )
            .to_vec()
        },
        Shape::from([array.len()]),
        DType::I32,
    ))
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
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val) {
                    indices.push(i);
                }
            }
        }
        _ => return Err(format!("Where not implemented for {}", array.dtype())),
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
