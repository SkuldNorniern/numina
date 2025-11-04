//! # Numina - Safe Tensor Library for Rust
//!
//! Numina provides a safe, efficient tensor library with ndarray-compatible API,
//! designed as the foundation for high-performance computing in Rust.

pub mod array;
pub mod dtype;
pub mod ops;
pub mod reductions;
pub mod sorting;

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

// Re-export commonly used types
pub use DType::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype;

    #[test]
    fn basic_array_creation() {
        let shape = Shape::from([2, 3]);
        let cpu_bytes = CpuBytesArray::zeros(dtype::F32, shape.clone());
        assert_eq!(cpu_bytes.shape(), &shape);
        assert_eq!(cpu_bytes.dtype(), dtype::F32);
    }

    #[test]
    fn array_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);
        let array = Array::from_slice(&data, shape.clone()).unwrap();
        assert_eq!(array.shape(), &shape);
        assert_eq!(array.len(), 4);
    }
}
