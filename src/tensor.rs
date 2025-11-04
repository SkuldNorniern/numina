//! Tensor data structures and operations
use std::fmt;
use std::mem;

use crate::array::{Shape, Strides, Array, CpuBytesArray, NdArray, data_as_slice, data_as_slice_mut};
use crate::dtype::DTypeLike;
use crate::DType;



/// Main tensor structure
#[derive(Debug)]
pub struct Tensor {
    storage: Box<dyn NdArray>,
}

impl Tensor {
    /// Create a new tensor from raw data
    pub fn new(data: Vec<u8>, shape: Shape, dtype: DType) -> Self {
        Tensor {
            storage: CpuBytesArray::new(data, shape, dtype).into_boxed(),
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

        let byte_len = data.len() * std::mem::size_of::<T>();
        let mut bytes = Vec::<u8>::with_capacity(byte_len);
        unsafe {
            bytes.set_len(byte_len);
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                byte_len,
            );
        }

        Self::new(bytes, shape, dtype)
    }

    /// Create tensor filled with zeros
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        Tensor {
            storage: CpuBytesArray::zeros(dtype, shape).into_boxed(),
        }
    }

    /// Create tensor filled with ones
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        Tensor {
            storage: CpuBytesArray::ones(dtype, shape).into_boxed(),
        }
    }

    /// Create identity matrix
    pub fn eye(dtype: DType, n: usize) -> Self {
        Tensor {
            storage: CpuBytesArray::eye(dtype, n).into_boxed(),
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> &Shape {
        self.storage.shape()
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if tensor is empty
    pub fn is_empty(&self) -> bool {
        self.storage.len() == 0
    }

    /// Get number of dimensions
    pub fn ndim(&self) -> usize {
        self.storage.shape().ndim()
    }

    /// Get strides
    pub fn strides(&self) -> &Strides {
        self.storage.strides()
    }

    /// Create a tensor from any NdArray implementation
    pub fn from_ndarray(array: Box<dyn NdArray>) -> Self {
        Tensor { storage: array }
    }

    /// Create a tensor from CpuBytesArray
    pub fn from_cpu_bytes_array(array: CpuBytesArray) -> Self {
        Tensor {
            storage: array.into_boxed(),
        }
    }

    /// Clone this tensor with its storage
    pub fn clone_tensor(&self) -> Self {
        Tensor {
            storage: self.storage.clone_array(),
        }
    }

    /// Get raw data as slice (for internal operations)
    /// # Safety
    /// T must match the actual data type stored in this tensor
    #[allow(dead_code)]
    pub(crate) unsafe fn data_as_slice<T>(&self) -> &[T] {
        unsafe { data_as_slice::<T>(self.storage.as_ref()) }
    }

    /// Get raw data as mutable slice (for internal operations)
    /// # Safety
    /// T must match the actual data type stored in this tensor
    pub(crate) unsafe fn data_as_slice_mut<T>(&mut self) -> &mut [T] {
        unsafe { data_as_slice_mut::<T>(self.storage.as_mut()) }
    }

    /// Check if this tensor supports reshaping operations
    pub fn can_reshape(&self) -> bool {
        // For now, only CpuBytesArray supports reshape/transpose
        // In the future, we could check for trait implementations
        false // TODO: implement proper type checking
    }

    /// Reshape tensor (creates new storage if supported)
    pub fn reshape(self, new_shape: Shape) -> Result<Self, String> {
        let reshaped_storage = self.storage.reshape(new_shape)?;
        Ok(Tensor {
            storage: reshaped_storage,
        })
    }

    /// Transpose tensor (2D only, creates new storage if supported)
    pub fn transpose(self) -> Result<Self, String> {
        let transposed_storage = self.storage.transpose()?;
        Ok(Tensor {
            storage: transposed_storage,
        })
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        use crate::ops;
        let result = ops::add(self, other)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        use crate::ops;
        let result = ops::mul(self, other)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Tensor, String> {
        use crate::ops;
        let result = ops::exp(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise logarithm
    pub fn log(&self) -> Result<Tensor, String> {
        use crate::ops;
        let result = ops::log(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor, String> {
        use crate::ops;
        let result = ops::sqrt(self)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Sum reduction
    pub fn sum(&self, axis: Option<usize>) -> Result<Tensor, String> {
        use crate::reductions;
        let result = reductions::sum(self, axis)?;
        Ok(Tensor::from_ndarray(result))
    }

    /// Mean reduction
    pub fn mean(&self, axis: Option<usize>) -> Result<Tensor, String> {
        use crate::reductions;
        let result = reductions::mean(self, axis)?;
        Ok(Tensor::from_ndarray(result))
    }
}

impl NdArray for Tensor {
    fn shape(&self) -> &Shape {
        self.storage.shape()
    }

    fn strides(&self) -> &Strides {
        self.storage.strides()
    }

    fn len(&self) -> usize {
        self.storage.len()
    }

    fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        unsafe { self.storage.as_bytes() }
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe { self.storage.as_mut_bytes() }
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        self.storage.clone_array()
    }
}

impl Tensor {
    /// Borrowed conversion into a typed N-dimensional array
    pub fn to_array<T>(&self) -> Result<Array<T>, String>
    where
        T: DTypeLike + Copy + std::fmt::Debug + 'static,
    {
        if self.dtype() != T::DTYPE {
            return Err(format!(
                "Cannot convert tensor of dtype {} to array of {}",
                self.dtype(),
                T::DTYPE
            ));
        }

        if mem::size_of::<T>() != self.dtype().dtype_size_bytes() {
            return Err(format!(
                "Size mismatch converting {} to {} ({} bytes vs {} bytes)",
                self.dtype(),
                std::any::type_name::<T>(),
                self.dtype().dtype_size_bytes(),
                mem::size_of::<T>()
            ));
        }

        let mut data = Vec::<T>::with_capacity(self.len());
        unsafe {
            data.set_len(self.len());
            std::ptr::copy_nonoverlapping(
                self.storage.as_bytes().as_ptr() as *const T,
                data.as_mut_ptr(),
                self.len(),
            );
        }

        Ok(Array::from_raw_parts(
            data,
            self.storage.shape().clone(),
            self.storage.strides().clone(),
        ))
    }

    /// Consume the tensor and convert it into a typed N-dimensional array
    pub fn into_array<T>(self) -> Result<Array<T>, String>
    where
        T: DTypeLike + Copy + std::fmt::Debug + 'static,
    {
        if self.dtype() != T::DTYPE {
            return Err(format!(
                "Cannot convert tensor of dtype {} to array of {}",
                self.dtype(),
                T::DTYPE
            ));
        }

        if mem::size_of::<T>() != self.dtype().dtype_size_bytes() {
            return Err(format!(
                "Size mismatch converting {} to {} ({} bytes vs {} bytes)",
                self.dtype(),
                std::any::type_name::<T>(),
                self.dtype().dtype_size_bytes(),
                mem::size_of::<T>()
            ));
        }

        let Tensor { storage } = self;
        let len = storage.len();
        let mut data = Vec::<T>::with_capacity(len);
        unsafe {
            data.set_len(len);
            std::ptr::copy_nonoverlapping(
                storage.as_bytes().as_ptr() as *const T,
                data.as_mut_ptr(),
                len,
            );
        }

        Ok(Array::from_raw_parts(
            data,
            storage.shape().clone(),
            storage.strides().clone(),
        ))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor({}, {}, {})",
            self.storage.shape(),
            self.storage.dtype(),
            self.storage
                .strides()
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
    use crate::array::{Shape, Strides};

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
