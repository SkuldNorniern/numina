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
    types::{BFloat16, QuantizedI4, QuantizedU8},
    DType, DTypeCandidate, DTypeLike,
};
pub use ops::{
    abs, acos, add, add_scalar, asin, atan, cos, exp, log, matmul, mul, pow, sign, sin, sqrt, tan,
};
pub use reductions::{max, mean, min, prod, sum};
pub use sorting::{argsort, sort, where_condition};

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

    #[test]
    fn bfloat16_operations() {
        use crate::BFloat16;

        // Test BFloat16 array creation
        let bf16_val1 = BFloat16::from_f32(3.14159);
        let bf16_val2 = BFloat16::from_f32(2.71828);
        let bf16_data = [bf16_val1, bf16_val2];
        let shape = Shape::from([2]);
        let bf16_array = Array::from_slice(&bf16_data, shape.clone()).unwrap();

        assert_eq!(bf16_array.shape(), &shape);
        assert_eq!(bf16_array.dtype(), BF16);
        assert_eq!(bf16_array.len(), 2);

        // Test BFloat16 operations work
        let result = add(&bf16_array, &bf16_array).unwrap();
        assert_eq!(result.shape(), &shape);
        assert_eq!(result.dtype(), BF16);
    }

    #[test]
    fn backend_type_preservation() {
        // Test that operations preserve the backend type of the input
        let f32_data = [1.0f32, 2.0, 3.0, 4.0];
        let shape = Shape::from([2, 2]);

        // Test with Array<f32>
        let array_f32 = Array::from_slice(&f32_data, shape.clone()).unwrap();
        let result_typed = add(&array_f32, &array_f32).unwrap();

        // Test with CpuBytesArray
        let cpu_bytes = CpuBytesArray::new(
            unsafe { std::slice::from_raw_parts(f32_data.as_ptr() as *const u8, f32_data.len() * 4).to_vec() },
            shape.clone(),
            F32,
        );
        let result_bytes = add(&cpu_bytes, &cpu_bytes).unwrap();

        // Both should work and preserve their respective types
        assert_eq!(result_typed.shape(), &shape);
        assert_eq!(result_typed.dtype(), F32);
        assert_eq!(result_bytes.shape(), &shape);
        assert_eq!(result_bytes.dtype(), F32);
    }
}
