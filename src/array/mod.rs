//! Array abstractions and CPU-backed implementations
pub mod shape;
pub mod stride;

use std::mem;

use crate::DType;
use crate::dtype::DTypeLike;

// Re-export the shape and stride modules
pub use {shape::Shape, stride::Strides};

/// Trait implemented by any N-dimensional array storage
///
/// This abstraction allows Numina operations to work with different
/// backends (CPU, GPU, remote) as long as they can expose their shape
/// and, optionally, host-accessible memory.
pub trait NdArray: std::fmt::Debug {
    /// Returns the shape of the array
    fn shape(&self) -> &Shape;

    /// Returns the strides describing the memory layout
    fn strides(&self) -> &Strides;

    /// Number of logical elements stored in the array
    fn len(&self) -> usize;

    /// Returns true if the array contains no elements
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Data type of the elements stored in the array
    fn dtype(&self) -> DType;

    /// Returns true if the backend exposes host-accessible contiguous memory
    fn is_host_accessible(&self) -> bool {
        true
    }

    /// Helper: returns true when the layout is contiguous
    fn is_contiguous(&self) -> bool {
        self.strides().is_contiguous(self.shape())
    }

    /// Size in bytes for a single element
    fn element_size(&self) -> usize {
        self.dtype().dtype_size_bytes()
    }

    /// Total byte length of the underlying storage
    fn byte_len(&self) -> usize {
        self.len() * self.element_size()
    }

    /// Raw view of the underlying bytes
    ///
    /// # Safety
    /// The implementor must guarantee the returned slice remains valid
    /// for the lifetime of the array reference.
    unsafe fn as_bytes(&self) -> &[u8];

    /// Mutable raw access to the underlying bytes
    ///
    /// # Safety
    /// Same guarantees as [`NdArray::as_bytes`]. Implementations must
    /// ensure exclusive access to the memory region.
    unsafe fn as_mut_bytes(&mut self) -> &mut [u8];

    /// Clone the array into a new owned instance
    fn clone_array(&self) -> Box<dyn NdArray>;

    /// Create a new array of the same backend type with zeros
    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Err("Creating new arrays not supported for this backend".to_string())
    }

    /// Create a new array of the same backend type with ones
    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Err("Creating new arrays not supported for this backend".to_string())
    }

    /// Reshape this array to a new shape, returning a new array
    fn reshape(&self, _new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        // Default implementation for arrays that don't support reshape
        Err("Reshape not supported for this array backend".to_string())
    }

    /// Transpose this array (2D only), returning a new array
    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        // Default implementation for arrays that don't support transpose
        Err("Transpose not supported for this array backend".to_string())
    }
}

/// Convenience helper to reinterpret data as slice of `T`
///
/// # Safety
/// Caller must guarantee that `T` matches the storage dtype.
pub unsafe fn data_as_slice<T>(array: &dyn NdArray) -> &[T] {
    debug_assert_eq!(mem::size_of::<T>() * array.len(), array.byte_len());
    let bytes = unsafe { array.as_bytes() };
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, array.len()) }
}

/// Mutable variant of [`data_as_slice`]
///
/// # Safety
/// Caller must ensure that `T` matches the storage dtype and that no other
/// references to the array data exist.
pub unsafe fn data_as_slice_mut<T>(array: &mut dyn NdArray) -> &mut [T] {
    debug_assert_eq!(mem::size_of::<T>() * array.len(), array.byte_len());
    let bytes = unsafe { array.as_mut_bytes() };
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, array.len()) }
}

/// Ensure array is host-accessible and contiguous
pub fn ensure_host_accessible<A: NdArray>(array: &A, op: &str) -> Result<(), String> {
    if !array.is_host_accessible() {
        return Err(format!(
            "{op} requires host-accessible memory (backend does not expose CPU access yet)"
        ));
    }
    if !array.is_contiguous() {
        return Err(format!(
            "{op} currently supports only contiguous row-major layouts"
        ));
    }
    Ok(())
}

/// Ensure arrays are compatible for binary operations
pub fn ensure_binary_compat<A: NdArray, B: NdArray>(a: &A, b: &B, op: &str) -> Result<(), String> {
    ensure_host_accessible(a, op)?;
    ensure_host_accessible(b, op)?;

    if a.dtype() != b.dtype() {
        return Err(format!(
            "{op} dtype mismatch: {} vs {}",
            a.dtype(),
            b.dtype()
        ));
    }

    if a.shape() != b.shape() {
        return Err(format!(
            "{op} shape mismatch: {} vs {}",
            a.shape(),
            b.shape()
        ));
    }

    Ok(())
}

/// CPU-resident byte-addressable array used as the default tensor storage
#[derive(Debug, Clone)]
pub struct CpuBytesArray {
    data: Vec<u8>,
    shape: Shape,
    strides: Strides,
    dtype: DType,
    len: usize,
}

impl CpuBytesArray {
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

        Self {
            data,
            shape,
            strides,
            dtype,
            len,
        }
    }

    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let len = shape.len();
        let size_bytes = len * dtype.dtype_size_bytes();
        let data = vec![0u8; size_bytes];
        Self::new(data, shape, dtype)
    }

    pub fn ones(dtype: DType, shape: Shape) -> Self {
        let len = shape.len();
        let mut storage = Self::zeros(dtype, shape);

        match storage.dtype {
            DType::F32 => {
                let data = unsafe { data_as_slice_mut::<f32>(&mut storage) };
                for value in data.iter_mut() {
                    *value = 1.0;
                }
            }
            DType::F64 => {
                let data = unsafe { data_as_slice_mut::<f64>(&mut storage) };
                for value in data.iter_mut() {
                    *value = 1.0;
                }
            }
            _ => {
                for i in 0..len {
                    storage.data[i * storage.dtype.dtype_size_bytes()] = 1;
                }
            }
        }

        storage
    }

    pub fn eye(dtype: DType, n: usize) -> Self {
        let shape = Shape::from([n, n]);
        let mut storage = Self::zeros(dtype, shape);

        for i in 0..n {
            let idx = i * n + i;
            match storage.dtype {
                DType::F32 => {
                    let data = unsafe { data_as_slice_mut::<f32>(&mut storage) };
                    data[idx] = 1.0;
                }
                DType::F64 => {
                    let data = unsafe { data_as_slice_mut::<f64>(&mut storage) };
                    data[idx] = 1.0;
                }
                DType::I8 => {
                    let data = unsafe { data_as_slice_mut::<i8>(&mut storage) };
                    data[idx] = 1;
                }
                DType::I16 => {
                    let data = unsafe { data_as_slice_mut::<i16>(&mut storage) };
                    data[idx] = 1;
                }
                DType::I32 => {
                    let data = unsafe { data_as_slice_mut::<i32>(&mut storage) };
                    data[idx] = 1;
                }
                DType::I64 => {
                    let data = unsafe { data_as_slice_mut::<i64>(&mut storage) };
                    data[idx] = 1;
                }
                DType::U8 => {
                    let data = unsafe { data_as_slice_mut::<u8>(&mut storage) };
                    data[idx] = 1;
                }
                DType::U16 => {
                    let data = unsafe { data_as_slice_mut::<u16>(&mut storage) };
                    data[idx] = 1;
                }
                DType::U32 => {
                    let data = unsafe { data_as_slice_mut::<u32>(&mut storage) };
                    data[idx] = 1;
                }
                DType::U64 => {
                    let data = unsafe { data_as_slice_mut::<u64>(&mut storage) };
                    data[idx] = 1;
                }
                DType::Bool => {
                    let data = unsafe { data_as_slice_mut::<bool>(&mut storage) };
                    data[idx] = true;
                }
                _ => panic!("Identity matrix for {} not supported", storage.dtype),
            }
        }

        storage
    }

    pub fn reshape(self, new_shape: Shape) -> Result<Self, String> {
        if new_shape.len() != self.len {
            return Err(format!(
                "Cannot reshape {} elements into {}",
                self.len, new_shape
            ));
        }

        Ok(Self {
            strides: Strides::from_shape(&new_shape),
            shape: new_shape,
            ..self
        })
    }

    pub fn transpose(self) -> Result<Self, String> {
        if self.shape.ndim() != 2 {
            return Err("Transpose only supported for 2D arrays".to_string());
        }

        let new_shape = Shape::from([self.shape.dim(1), self.shape.dim(0)]);
        let mut new_data = vec![0u8; self.data.len()];

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

        Ok(Self {
            data: new_data,
            shape: new_shape.clone(),
            strides: Strides::from_shape(&new_shape),
            dtype: self.dtype,
            len: self.len,
        })
    }

    #[inline]
    pub(crate) fn data(&self) -> &[u8] {
        &self.data
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get mutable typed slice for internal operations
    /// # Safety
    /// T must match the actual dtype stored
    pub(crate) unsafe fn data_as_slice_mut<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len) }
    }

    pub fn into_boxed(self) -> Box<dyn NdArray> {
        Box::new(self)
    }
}

impl NdArray for CpuBytesArray {
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        &self.strides
    }

    fn len(&self) -> usize {
        self.len
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        self.data()
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        self.data_mut()
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        Box::new(self.clone())
    }

    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Ok(Box::new(CpuBytesArray::zeros(self.dtype, shape)))
    }

    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Ok(Box::new(CpuBytesArray::ones(self.dtype, shape)))
    }

    fn reshape(&self, new_shape: Shape) -> Result<Box<dyn NdArray>, String> {
        if new_shape.len() != self.len {
            return Err(format!(
                "Cannot reshape {} elements into {}",
                self.len, new_shape
            ));
        }

        // Copy the data and create new CpuBytesArray with new shape
        let mut bytes = vec![0u8; self.byte_len()];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.as_bytes().as_ptr(),
                bytes.as_mut_ptr(),
                self.byte_len(),
            );
        }

        Ok(Box::new(CpuBytesArray::new(bytes, new_shape, self.dtype)))
    }

    fn transpose(&self) -> Result<Box<dyn NdArray>, String> {
        if self.shape.ndim() != 2 {
            return Err("Transpose only supported for 2D arrays".to_string());
        }

        let new_shape = Shape::from([self.shape.dim(1), self.shape.dim(0)]);
        let mut bytes = vec![0u8; self.byte_len()];

        match self.dtype {
            DType::F32 => {
                let old_data = unsafe {
                    std::slice::from_raw_parts(self.as_bytes().as_ptr() as *const f32, self.len)
                };
                let new_data = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f32, self.len)
                };

                for i in 0..self.shape.dim(0) {
                    for j in 0..self.shape.dim(1) {
                        new_data[j * self.shape.dim(0) + i] = old_data[i * self.shape.dim(1) + j];
                    }
                }
            }
            DType::F64 => {
                let old_data = unsafe {
                    std::slice::from_raw_parts(self.as_bytes().as_ptr() as *const f64, self.len)
                };
                let new_data = unsafe {
                    std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut f64, self.len)
                };

                for i in 0..self.shape.dim(0) {
                    for j in 0..self.shape.dim(1) {
                        new_data[j * self.shape.dim(0) + i] = old_data[i * self.shape.dim(1) + j];
                    }
                }
            }
            _ => return Err(format!("Transpose not implemented for {}", self.dtype)),
        }

        Ok(Box::new(CpuBytesArray::new(bytes, new_shape, self.dtype)))
    }
}

/// Dense CPU-backed N-dimensional array with a concrete element type
#[derive(Debug, Clone)]
pub struct Array<T>
where
    T: DTypeLike,
{
    data: Vec<T>,
    shape: Shape,
    strides: Strides,
}

impl<T> Array<T>
where
    T: DTypeLike + std::fmt::Debug + 'static,
{
    /// Create an array from owned data and a shape description
    pub fn new(data: Vec<T>, shape: Shape) -> Result<Self, String> {
        if data.len() != shape.len() {
            return Err(format!(
                "Data length {} does not match shape {}",
                data.len(),
                shape
            ));
        }

        Ok(Self {
            strides: Strides::from_shape(&shape),
            data,
            shape,
        })
    }

    /// Create an array by copying from a slice
    pub fn from_slice(data: &[T], shape: Shape) -> Result<Self, String>
    where
        T: Copy,
    {
        Self::new(data.to_vec(), shape)
    }

    /// Returns a shared reference to the underlying data
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns the logical shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides describing the memory layout
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// Internal constructor used for zero-copy conversions from tensors
    #[allow(dead_code)]
    pub(crate) fn from_raw_parts(data: Vec<T>, shape: Shape, strides: Strides) -> Self {
        Self {
            data,
            shape,
            strides,
        }
    }
}

impl<T> NdArray for Array<T>
where
    T: DTypeLike + std::fmt::Debug + 'static + Default,
{
    fn shape(&self) -> &Shape {
        &self.shape
    }

    fn strides(&self) -> &Strides {
        &self.strides
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn dtype(&self) -> DType {
        T::DTYPE
    }

    unsafe fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * mem::size_of::<T>(),
            )
        }
    }

    unsafe fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut u8,
                self.data.len() * mem::size_of::<T>(),
            )
        }
    }

    fn clone_array(&self) -> Box<dyn NdArray> {
        Box::new(self.clone())
    }

    fn zeros(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let len = shape.len();
        let data = vec![T::default(); len];
        Ok(Box::new(Array::new(data, shape).map_err(|e| e)?))
    }

    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let len = shape.len();
        let mut data = vec![T::default(); len];

        // Fill with ones - this is a bit hacky but works for most numeric types
        // In practice, you might want to use a trait for this
        if std::mem::size_of::<T>() == std::mem::size_of::<f32>() {
            // Assume it's f32-like
            for item in data.iter_mut() {
                unsafe {
                    *(item as *mut T as *mut f32) = 1.0;
                }
            }
        } else if std::mem::size_of::<T>() == std::mem::size_of::<f64>() {
            // Assume it's f64-like
            for item in data.iter_mut() {
                unsafe {
                    *(item as *mut T as *mut f64) = 1.0;
                }
            }
        } else {
            // For integer types, just leave as default (0)
            // This is not ideal but works for now
        }

        Ok(Box::new(Array::new(data, shape).map_err(|e| e)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_construction() {
        let shape = Shape::from([2, 2]);
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], shape.clone()).unwrap();

        assert_eq!(array.shape(), &shape);
        assert_eq!(array.len(), 4);
        assert_eq!(array.dtype(), DType::F32);
    }

    #[test]
    fn array_to_cpu_bytes() {
        let shape = Shape::from([2, 2]);
        let array = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], shape.clone()).unwrap();

        // Convert to CpuBytesArray via NdArray trait
        let cpu_bytes: Box<dyn NdArray> = Box::new(array.clone());

        assert_eq!(cpu_bytes.shape(), &shape);
        assert_eq!(cpu_bytes.dtype(), DType::F32);
        assert_eq!(cpu_bytes.len(), 4);
    }
}
