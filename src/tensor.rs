//! Tensor data structures and operations

use crate::DType;
use std::fmt;

/// Represents the shape of a tensor (dimensions)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a new shape from a slice
    pub fn from(dimensions: impl Into<Vec<usize>>) -> Self {
        let dims = dimensions.into();
        assert!(!dims.is_empty(), "Shape cannot be empty");
        assert!(
            dims.iter().all(|&d| d > 0),
            "All dimensions must be positive"
        );
        Shape(dims)
    }

    /// Get the number of dimensions (rank)
    pub fn ndim(&self) -> usize {
        self.0.len()
    }

    /// Get the total number of elements
    pub fn len(&self) -> usize {
        self.0.iter().product()
    }

    /// Check if shape is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get dimensions as slice
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Get a specific dimension
    pub fn dim(&self, index: usize) -> usize {
        self.0[index]
    }

    /// Create a new shape with modified dimension
    pub fn with_dim(mut self, index: usize, value: usize) -> Self {
        self.0[index] = value;
        self
    }

    /// Check if this shape is compatible with another for broadcasting
    pub fn can_broadcast_to(&self, other: &Shape) -> bool {
        if self.ndim() > other.ndim() {
            return false;
        }

        // Pad with leading dimensions of size 1
        let self_padded =
            std::iter::repeat_n(1, other.ndim() - self.ndim()).chain(self.0.iter().cloned());

        for (a, &b) in self_padded.zip(other.dims()) {
            if a != 1 && a != b {
                return false;
            }
        }

        true
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, &dim) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::from(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::from(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::from(dims.to_vec())
    }
}

/// Represents memory layout strides
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
    /// Create strides from shape (row-major/C order)
    pub fn from_shape(shape: &Shape) -> Self {
        let mut strides = vec![0; shape.ndim()];
        let mut stride = 1;

        // Row-major order (C-style)
        for i in (0..shape.ndim()).rev() {
            strides[i] = stride;
            stride *= shape.dim(i);
        }

        Strides(strides)
    }

    /// Get strides as slice
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }

    /// Calculate flat index from multi-dimensional indices
    pub fn flatten_index(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.0.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Check if layout is contiguous
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        let expected = Strides::from_shape(shape);
        self == &expected
    }
}

/// Main tensor structure
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Vec<u8>,    // Raw byte storage
    shape: Shape,     // Tensor dimensions
    strides: Strides, // Memory layout
    dtype: DType,     // Data type
    len: usize,       // Number of elements
}

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        let len = shape.len();
        let expected_size = len * dtype.dtype_size_bytes();

        assert_eq!(
            data.len(),
            expected_size,
            "Data size {} does not match expected size {} for shape {} and dtype {}",
            data.len(),
            expected_size,
            shape,
            dtype
        );

        let strides = Strides::from_shape(&shape);

        Tensor {
            data,
            shape,
            strides,
            dtype,
            len,
        }
    }

    /// Create tensor from slice (copies data)
    pub fn from_slice<T>(data: &[T], shape: Shape) -> Self
    where
        T: Copy + Into<DType>,
    {
        let dtype = data[0].into();
        let expected_len = shape.len();

        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} does not match shape {}",
            data.len(),
            shape
        );

        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        Self::new(bytes.to_vec(), shape, dtype)
    }

    /// Create tensor filled with zeros
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let len = shape.len();
        let size_bytes = len * dtype.dtype_size_bytes();
        let data = vec![0u8; size_bytes];

        Self::new(data, shape, dtype)
    }

    /// Create tensor filled with ones
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        let len = shape.len();
        let mut data = vec![0u8; len * dtype.dtype_size_bytes()];

        // Fill with appropriate representation of 1 for each dtype
        match dtype {
            DType::F32 => {
                let ones: Vec<f32> = vec![1.0; len];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ones.as_ptr() as *const u8,
                        data.as_mut_ptr(),
                        data.len(),
                    );
                }
            }
            DType::F64 => {
                let ones: Vec<f64> = vec![1.0; len];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        ones.as_ptr() as *const u8,
                        data.as_mut_ptr(),
                        data.len(),
                    );
                }
            }
            // For integer types, 1 is represented as 0x01 in little-endian
            _ => {
                // Simple implementation - fill with 1s for now
                // In practice, you'd need proper endianness handling
                for i in 0..len {
                    data[i * dtype.dtype_size_bytes()] = 1;
                }
            }
        }

        Self::new(data, shape, dtype)
    }

    /// Create identity matrix
    pub fn eye(dtype: DType, n: usize) -> Self {
        let shape = Shape::from([n, n]);
        let mut tensor = Self::zeros(dtype, shape);

        // Set diagonal elements to 1
        for i in 0..n {
            // This is a simplified implementation
            // Real implementation would need proper indexing
            let idx = i * n + i;
            match dtype {
                DType::F32 => {
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            tensor.data.as_mut_ptr() as *mut f32,
                            tensor.len,
                        )
                    };
                    data[idx] = 1.0;
                }
                DType::F64 => {
                    let data = unsafe {
                        std::slice::from_raw_parts_mut(
                            tensor.data.as_mut_ptr() as *mut f64,
                            tensor.len,
                        )
                    };
                    data[idx] = 1.0;
                }
                _ => unimplemented!("Identity matrix for {} not implemented", dtype),
            }
        }

        tensor
    }

    /// Get tensor shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get strides
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Get raw data as slice (for internal operations)
    /// # Safety
    /// T must match the actual data type stored in this tensor
    pub(crate) unsafe fn data_as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.len) }
    }

    /// Get raw data as mutable slice (for internal operations)
    /// # Safety
    /// T must match the actual data type stored in this tensor
    pub(crate) unsafe fn data_as_slice_mut<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    /// Reshape tensor (creates view if possible)
    pub fn reshape(self, new_shape: Shape) -> Result<Self, String> {
        if new_shape.len() != self.len {
            return Err(format!(
                "Cannot reshape {} elements into {}",
                self.len, new_shape
            ));
        }

        Ok(Tensor {
            data: self.data,
            shape: new_shape.clone(),
            strides: Strides::from_shape(&new_shape),
            dtype: self.dtype,
            len: self.len,
        })
    }

    /// Transpose tensor (2D only for now)
    pub fn transpose(self) -> Result<Self, String> {
        if self.ndim() != 2 {
            return Err("Transpose only supported for 2D tensors".to_string());
        }

        let new_shape = Shape::from([self.shape.dim(1), self.shape.dim(0)]);
        let mut new_data = vec![0u8; self.data.len()];

        // Simple transpose implementation
        match self.dtype {
            DType::F32 => {
                let old_data = unsafe {
                    std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.len)
                };
                let new_data_typed = unsafe {
                    std::slice::from_raw_parts_mut(new_data.as_mut_ptr() as *mut f32, self.len)
                };

                for i in 0..self.shape.dim(0) {
                    for j in 0..self.shape.dim(1) {
                        new_data_typed[j * self.shape.dim(0) + i] =
                            old_data[i * self.shape.dim(1) + j];
                    }
                }
            }
            _ => return Err(format!("Transpose not implemented for {}", self.dtype)),
        }

        Ok(Tensor {
            data: new_data,
            shape: new_shape.clone(),
            strides: Strides::from_shape(&new_shape),
            dtype: self.dtype,
            len: self.len,
        })
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {})",
            self.shape,
            self.dtype,
            self.strides
                .as_slice()
                .iter()
                .map(|&x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn shape_creation() {
        let shape = Shape::from([2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.len(), 24);
        assert_eq!(shape.dim(0), 2);
        assert_eq!(shape.dim(1), 3);
        assert_eq!(shape.dim(2), 4);
    }

    #[test]
    fn shape_display() {
        let shape = Shape::from([2, 3]);
        assert_eq!(format!("{}", shape), "[2, 3]");
    }

    #[test]
    fn strides_from_shape() {
        let shape = Shape::from([2, 3, 4]);
        let strides = Strides::from_shape(&shape);
        // Row-major: [12, 4, 1]
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn tensor_zeros() {
        let shape = Shape::from([2, 3]);
        let tensor = Tensor::zeros(dtype::F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), dtype::F32);
        assert_eq!(tensor.len(), 6);
    }

    #[test]
    fn tensor_ones() {
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::ones(dtype::F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), dtype::F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_from_slice() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);
        let tensor = Tensor::from_slice(&data, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), dtype::F32);
        assert_eq!(tensor.len(), 4);
    }

    #[test]
    fn tensor_reshape() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape1 = Shape::from([2, 3]);
        let tensor1 = Tensor::from_slice(&data, shape1);

        let shape2 = Shape::from([3, 2]);
        let tensor2 = tensor1.reshape(shape2.clone()).unwrap();
        assert_eq!(tensor2.shape(), &shape2);
        assert_eq!(tensor2.len(), 6);
    }

    #[test]
    fn tensor_display() {
        let shape = Shape::from([2, 3]);
        let tensor = Tensor::zeros(dtype::F32, shape);
        let display = format!("{}", tensor);
        assert!(display.contains("Tensor([2, 3], f32"));
    }
}
