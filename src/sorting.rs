//! Sorting and searching operations.
//!
//! These operations currently require host-accessible, contiguous storage.
//!
//! For floating-point dtypes, sorting uses a total ordering (`f64::total_cmp`) so NaNs are handled
//! deterministically (as opposed to `partial_cmp` which can return `None`).

use crate::array::{NdArray, data_as_slice, data_as_slice_mut, ensure_host_accessible};
use crate::{DType, Shape};

fn sort_f64_total(values: &mut [f64], descending: bool) {
    if descending {
        values.sort_by(|a, b| b.total_cmp(a));
    } else {
        values.sort_by(|a, b| a.total_cmp(b));
    }
}

/// Sort an array along `axis`, or sort the flattened array when `axis` is `None`.
///
/// For float-like dtypes this uses a total ordering, which gives deterministic placement of NaNs.
///
/// # Errors
/// Returns `Err` if the backend is not host-accessible/contiguous, the axis is out of bounds, or
/// the dtype/shape combination is unsupported.
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

    if array.shape().ndim() > 2 {
        return Err("sort with axis currently supports only 1D or 2D arrays".to_string());
    }

    let mut result = array.zeros(array.shape().clone())?;

    match array.dtype() {
        DType::F16 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float16>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> =
                    tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
                sort_f64_total(&mut values, descending);
                for i in 0..values.len() {
                    result_data[i] = crate::Float16::from(values[i] as f32);
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = crate::Float16::from(column[i] as f32);
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = crate::Float16::from(row[j] as f32);
                        }
                    }
                }
            }
        }
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
                sort_f64_total(&mut values, descending);
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
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = column[i] as f32;
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| tensor_data[i * cols + j] as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = row[j] as f32;
                        }
                    }
                }
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> = tensor_data.to_vec();
                sort_f64_total(&mut values, descending);
                result_data.copy_from_slice(&values);
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> =
                            (0..rows).map(|i| tensor_data[i * cols + j]).collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = column[i];
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> =
                            (0..cols).map(|j| tensor_data[i * cols + j]).collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = row[j];
                        }
                    }
                }
            }
        }
        DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> = tensor_data.iter().map(|&x| x.to_f32() as f64).collect();
                sort_f64_total(&mut values, descending);
                for i in 0..values.len() {
                    result_data[i] = crate::BFloat16::from_f32(values[i] as f32);
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| tensor_data[i * cols + j].to_f32() as f64)
                            .collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = crate::BFloat16::from_f32(column[i] as f32);
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| tensor_data[i * cols + j].to_f32() as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = crate::BFloat16::from_f32(row[j] as f32);
                        }
                    }
                }
            }
        }
        DType::BF8 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> =
                    tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
                sort_f64_total(&mut values, descending);
                for i in 0..values.len() {
                    result_data[i] = crate::BFloat8::from(values[i] as f32);
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = crate::BFloat8::from(column[i] as f32);
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = crate::BFloat8::from(row[j] as f32);
                        }
                    }
                }
            }
        }
        DType::F8E4M3FN => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> =
                    tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
                sort_f64_total(&mut values, descending);
                for i in 0..values.len() {
                    result_data[i] = crate::Float8E4M3Fn::from(values[i] as f32);
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = crate::Float8E4M3Fn::from(column[i] as f32);
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = crate::Float8E4M3Fn::from(row[j] as f32);
                        }
                    }
                }
            }
        }
        DType::F8E5M2 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            if array.shape().ndim() == 1 {
                let mut values: Vec<f64> =
                    tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
                sort_f64_total(&mut values, descending);
                for i in 0..values.len() {
                    result_data[i] = crate::Float8E5M2::from(values[i] as f32);
                }
            } else if array.shape().ndim() == 2 {
                let (rows, cols) = (array.shape().dim(0), array.shape().dim(1));

                if axis == 0 {
                    for j in 0..cols {
                        let mut column: Vec<f64> = (0..rows)
                            .map(|i| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut column, descending);
                        for i in 0..rows {
                            result_data[i * cols + j] = crate::Float8E5M2::from(column[i] as f32);
                        }
                    }
                } else if axis == 1 {
                    for i in 0..rows {
                        let mut row: Vec<f64> = (0..cols)
                            .map(|j| f32::from(tensor_data[i * cols + j]) as f64)
                            .collect();
                        sort_f64_total(&mut row, descending);
                        for j in 0..cols {
                            result_data[i * cols + j] = crate::Float8E5M2::from(row[j] as f32);
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
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
                    values.sort();
                }
                result_data.copy_from_slice(&values);
            } else {
                return Err(
                    "sort with axis currently supports only 1D arrays for integer dtypes"
                        .to_string(),
                );
            }
        }
        DType::Bool => {
            return Err("Sort not supported for boolean type".to_string());
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Sort not implemented for {}", array.dtype()));
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
        DType::F16 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float16>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float16>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = crate::Float16::from(values[i] as f32);
            }
        }
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let result_data = unsafe { data_as_slice_mut::<f32>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = values[i] as f32;
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let result_data = unsafe { data_as_slice_mut::<f64>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.to_vec();
            sort_f64_total(&mut values, descending);
            result_data.copy_from_slice(&values);
        }
        DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| x.to_f32() as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = crate::BFloat16::from_f32(values[i] as f32);
            }
        }
        DType::BF8 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = crate::BFloat8::from(values[i] as f32);
            }
        }
        DType::F8E4M3FN => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = crate::Float8E4M3Fn::from(values[i] as f32);
            }
        }
        DType::F8E5M2 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let result_data = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut *result) };

            let mut values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();
            sort_f64_total(&mut values, descending);
            for i in 0..values.len() {
                result_data[i] = crate::Float8E5M2::from(values[i] as f32);
            }
        }
        DType::I8 => {
            let tensor_data = unsafe { data_as_slice::<i8>(array) };
            let result_data = unsafe { data_as_slice_mut::<i8>(&mut *result) };

            let mut values: Vec<i8> = tensor_data.to_vec();
            if descending {
                values.sort_by(|a, b| b.cmp(a));
            } else {
                values.sort();
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
                values.sort();
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
                values.sort();
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
                values.sort();
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
                values.sort();
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
                values.sort();
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
                values.sort();
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
                values.sort();
            }
            result_data.copy_from_slice(&values);
        }
        DType::Bool => {
            return Err("Sort not supported for boolean type".to_string());
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Sort not implemented for {}", array.dtype()));
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

/// Return indices that would sort the array (argsort).
///
/// Currently this only supports 1D arrays.
///
/// # Errors
/// Returns `Err` if the backend is not host-accessible/contiguous, the array is not 1D, or the
/// dtype is unsupported.
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
        DType::F16 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float16>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };
            let values: Vec<f64> = tensor_data.to_vec();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat16>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| x.to_f32() as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::BF8 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat8>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::F8E4M3FN => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
            }
        }
        DType::F8E5M2 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E5M2>(array) };
            let values: Vec<f64> = tensor_data.iter().map(|&x| f32::from(x) as f64).collect();

            if descending {
                indices.sort_by(|&a, &b| values[b as usize].total_cmp(&values[a as usize]));
            } else {
                indices.sort_by(|&a, &b| values[a as usize].total_cmp(&values[b as usize]));
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
            return Err("Argsort not supported for boolean type".to_string());
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Argsort not implemented for {}", array.dtype()));
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

/// Find indices where `condition` is true.
///
/// This is a minimal "where" helper (basic boolean indexing support). Currently this only
/// supports 1D arrays and evaluates the predicate against `f32` values.
///
/// # Errors
/// Returns `Err` if the backend is not host-accessible/contiguous, the array is not 1D, or the
/// dtype is unsupported.
pub fn where_condition<A, F>(array: &A, condition: F) -> Result<Vec<usize>, String>
where
    A: NdArray,
    F: Fn(f32) -> bool,
{
    ensure_host_accessible(array, "where")?;
    let mut indices = Vec::new();

    match array.dtype() {
        DType::F16 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float16>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(f32::from(val)) {
                    indices.push(i);
                }
            }
        }
        DType::F32 => {
            let tensor_data = unsafe { data_as_slice::<f32>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val) {
                    indices.push(i);
                }
            }
        }
        DType::F64 => {
            let tensor_data = unsafe { data_as_slice::<f64>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val as f32) {
                    indices.push(i);
                }
            }
        }
        DType::BF16 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat16>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(val.to_f32()) {
                    indices.push(i);
                }
            }
        }
        DType::BF8 => {
            let tensor_data = unsafe { data_as_slice::<crate::BFloat8>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(f32::from(val)) {
                    indices.push(i);
                }
            }
        }
        DType::F8E4M3FN => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E4M3Fn>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(f32::from(val)) {
                    indices.push(i);
                }
            }
        }
        DType::F8E5M2 => {
            let tensor_data = unsafe { data_as_slice::<crate::Float8E5M2>(array) };

            for (i, &val) in tensor_data.iter().enumerate() {
                if condition(f32::from(val)) {
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
            return Err("Where not supported for boolean type".to_string());
        }
        DType::Complex32 | DType::Complex64 | DType::Complex128 => {
            return Err(format!("Where not implemented for {}", array.dtype()));
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
