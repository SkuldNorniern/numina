//! # Numina - Safe Tensor Library for Rust
//!
//! Numina provides a safe, efficient tensor library with ndarray-compatible API,
//! designed as the foundation for high-performance computing in Rust.

pub mod array;
pub mod dtype;
pub mod ops;
pub mod reductions;
pub mod sorting;
pub mod tensor;

pub use array::{Array, CpuBytesArray, NdArray, Shape, Strides};
pub use dtype::{
    DType, DTypeCandidate, DTypeLike,
    types::{BFloat16, QuantizedI4, QuantizedU8},
};
pub use ops::{
    add, mul, add_scalar, exp, log, sqrt, sin, cos, tan, asin, acos, atan, pow, abs, sign, matmul
};
pub use reductions::{sum, mean, max, min, prod};
pub use sorting::{sort, argsort, where_condition};
pub use tensor::Tensor;

// Re-export commonly used types
pub use DType::*;

/// Create a tensor from a slice
pub fn from_slice<T: Into<DType> + Copy>(data: &[T], shape: Shape) -> Tensor {
    Tensor::from_slice(data, shape)
}

/// Create a tensor filled with zeros
pub fn zeros(dtype: DType, shape: Shape) -> Tensor {
    Tensor::zeros(dtype, shape)
}

/// Create a tensor filled with ones
pub fn ones(dtype: DType, shape: Shape) -> Tensor {
    Tensor::ones(dtype, shape)
}

/// Create an identity matrix
pub fn eye(dtype: DType, n: usize) -> Tensor {
    Tensor::eye(dtype, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn basic_tensor_creation() {
        let shape = Shape::from([2, 3]);
        let tensor = zeros(dtype::F32, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.dtype(), dtype::F32);
    }

    #[test]
    fn tensor_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);
        let tensor = from_slice(&data, shape.clone());
        assert_eq!(tensor.shape(), &shape);
        assert_eq!(tensor.len(), 4);
    }
}
