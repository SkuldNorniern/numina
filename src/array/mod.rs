//! Array abstractions and CPU-backed implementations.
//!
//! The core abstraction is [`NdArray`], which describes an N-dimensional array's shape/strides and
//! exposes raw bytes for host-side operations.
//!
//! Most operations in this crate currently require:
//! - host-accessible memory (`NdArray::is_host_accessible() == true`)
//! - a contiguous, row-major layout (`NdArray::is_contiguous() == true`)
//!
//! ## Safety and alignment
//! [`data_as_slice`] and [`data_as_slice_mut`] reinterpret raw byte storage as typed slices. This
//! requires the underlying buffer to be correctly aligned for the target type; `CpuBytesArray`
//! guarantees this by allocating with the dtype's alignment.
mod aligned_bytes;
pub mod shape;
pub mod stride;

use std::mem;

use crate::DType;
use crate::dtype::DTypeLike;
use aligned_bytes::AlignedBytes;

// Re-export the shape and stride modules
pub use {shape::Shape, stride::Strides};

fn fill_vec_as<T, U: Copy>(data: &mut [T], value: U) {
    debug_assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
    for item in data {
        unsafe {
            *(item as *mut T as *mut U) = value;
        }
    }
}

/// Trait implemented by any N-dimensional array storage backend.
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

    /// Returns true if the backend exposes host-accessible memory.
    ///
    /// Many Numina operations currently run on the host and will return an error when this is
    /// `false`.
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

    /// Raw view of the underlying bytes.
    ///
    /// # Safety
    /// Implementations must guarantee:
    /// - the returned slice points to `self.byte_len()` bytes
    /// - the slice remains valid for the lifetime of `&self`
    /// - the bytes represent elements in the canonical in-memory layout for `self.dtype()`
    unsafe fn as_bytes(&self) -> &[u8];

    /// Mutable raw access to the underlying bytes.
    ///
    /// # Safety
    /// Same guarantees as [`NdArray::as_bytes`], plus:
    /// - the returned slice remains valid for the lifetime of `&mut self`
    /// - no other references alias the returned memory region
    unsafe fn as_mut_bytes(&mut self) -> &mut [u8];

    /// Clone the array into a new owned instance
    fn clone_array(&self) -> Box<dyn NdArray>;

    /// Create a new array of the same backend type with zeros
    fn zeros(&self, _shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Err("Creating new arrays not supported for this backend".to_string())
    }

    /// Create a new array of the same backend type with ones
    fn ones(&self, _shape: Shape) -> Result<Box<dyn NdArray>, String> {
        Err("Creating new arrays not supported for this backend".to_string())
    }

    /// Create a new array of the same backend type with a specific dtype
    fn new_array(&self, _shape: Shape, _dtype: DType) -> Result<Box<dyn NdArray>, String> {
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

/// Convenience helper to reinterpret array storage as a slice of `T`.
///
/// This function checks alignment at runtime and will panic if the underlying data pointer is not
/// aligned for `T`.
///
/// # Safety
/// - `T` must match the array's dtype and canonical byte layout.
/// - The backing buffer must be valid for reads of `array.len()` elements.
pub unsafe fn data_as_slice<T>(array: &dyn NdArray) -> &[T] {
    debug_assert_eq!(mem::size_of::<T>() * array.len(), array.byte_len());
    let bytes = unsafe { array.as_bytes() };
    let align = mem::align_of::<T>();
    assert_eq!(
        (bytes.as_ptr() as usize) % align,
        0,
        "misaligned data pointer for {}-byte alignment",
        align
    );
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, array.len()) }
}

/// Mutable variant of [`data_as_slice`].
///
/// This function checks alignment at runtime and will panic if the underlying data pointer is not
/// aligned for `T`.
///
/// # Safety
/// - `T` must match the array's dtype and canonical byte layout.
/// - The caller must ensure exclusive access to the underlying elements for the duration of the
///   returned borrow.
pub unsafe fn data_as_slice_mut<T>(array: &mut dyn NdArray) -> &mut [T] {
    debug_assert_eq!(mem::size_of::<T>() * array.len(), array.byte_len());
    let bytes = unsafe { array.as_mut_bytes() };
    let align = mem::align_of::<T>();
    assert_eq!(
        (bytes.as_ptr() as usize) % align,
        0,
        "misaligned data pointer for {}-byte alignment",
        align
    );
    unsafe { std::slice::from_raw_parts_mut(bytes.as_mut_ptr() as *mut T, array.len()) }
}

/// Validate that `array` is host-accessible and contiguous for an operation named `op`.
///
/// # Errors
/// Returns `Err` when:
/// - the backend does not expose host-accessible memory
/// - the layout is not contiguous row-major
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

/// Validate that `a` and `b` can participate in a binary operation named `op`.
///
/// This checks host accessibility, contiguity, dtype equality, and shape equality.
///
/// # Errors
/// Returns `Err` when the arrays are incompatible.
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

/// CPU-resident, byte-addressable N-dimensional array.
///
/// This is the "bytes" backend used by Numina operations. The backing allocation is aligned
/// according to the dtype's alignment so that it can be reinterpreted as typed slices.
///
/// Invariants:
/// - `data.len() == len * dtype.dtype_size_bytes()`
/// - `data` is aligned to `dtype.info().align`
/// - if `dtype == DType::Bool`, every byte is either `0` or `1`
#[derive(Debug, Clone)]
pub struct CpuBytesArray {
    data: AlignedBytes,
    shape: Shape,
    strides: Strides,
    dtype: DType,
    len: usize,
}

impl CpuBytesArray {
    /// Construct a CPU bytes array from raw bytes, validating length and (for `Bool`) contents.
    ///
    /// The input bytes are copied into an aligned allocation.
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
        let align = dtype.info().align;

        // Avoid ever forming `&[bool]` from invalid bytes.
        if dtype == DType::Bool {
            for (idx, &b) in data.iter().enumerate() {
                assert!(
                    b == 0 || b == 1,
                    "Invalid bool byte at index {idx}: {b} (expected 0 or 1)"
                );
            }
        }

        let data = AlignedBytes::from_slice(&data, align);

        Self {
            data,
            shape,
            strides,
            dtype,
            len,
        }
    }

    /// Construct a zero-initialized array of the given dtype and shape.
    ///
    /// For numeric dtypes this corresponds to `0` / `0.0`. For `Bool` it is `false`. For quantized
    /// dtypes this initializes the raw stored byte(s) to `0` (quantization metadata is external).
    pub fn zeros(dtype: DType, shape: Shape) -> Self {
        let len = shape.len();
        let size_bytes = len * dtype.dtype_size_bytes();
        let align = dtype.info().align;
        let data = AlignedBytes::new_zeroed(size_bytes, align);
        let strides = Strides::from_shape(&shape);
        Self {
            data,
            shape,
            strides,
            dtype,
            len,
        }
    }

    /// Construct an array filled with "one" for the given dtype and shape.
    ///
    /// - Float/integer types: `1` / `1.0`
    /// - Complex types: `1 + 0i`
    /// - `Bool`: `true`
    /// - Quantized types: raw value `1` (meaning depends on external quantization metadata)
    pub fn ones(dtype: DType, shape: Shape) -> Self {
        let mut storage = Self::zeros(dtype, shape);

        match storage.dtype {
            DType::F16 => {
                let data = unsafe { data_as_slice_mut::<crate::Float16>(&mut storage) };
                data.fill(crate::Float16::from(1.0f32));
            }
            DType::F32 => {
                let data = unsafe { data_as_slice_mut::<f32>(&mut storage) };
                data.fill(1.0);
            }
            DType::F64 => {
                let data = unsafe { data_as_slice_mut::<f64>(&mut storage) };
                data.fill(1.0);
            }
            DType::BF16 => {
                let data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut storage) };
                data.fill(crate::BFloat16::from_f32(1.0));
            }
            DType::BF8 => {
                let data = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut storage) };
                data.fill(crate::BFloat8::from(1.0f32));
            }
            DType::F8E4M3FN => {
                let data = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut storage) };
                data.fill(crate::Float8E4M3Fn::from(1.0f32));
            }
            DType::F8E5M2 => {
                let data = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut storage) };
                data.fill(crate::Float8E5M2::from(1.0f32));
            }
            DType::Complex32 => {
                let data = unsafe { data_as_slice_mut::<crate::Complex32>(&mut storage) };
                data.fill(crate::Complex32::new(1.0, 0.0));
            }
            DType::Complex64 => {
                let data = unsafe { data_as_slice_mut::<crate::Complex64>(&mut storage) };
                data.fill(crate::Complex64::new(1.0, 0.0));
            }
            DType::Complex128 => {
                let data = unsafe { data_as_slice_mut::<crate::Complex128>(&mut storage) };
                data.fill(crate::Complex128::new(1.0, 0.0));
            }
            DType::I8 => {
                let data = unsafe { data_as_slice_mut::<i8>(&mut storage) };
                data.fill(1);
            }
            DType::I16 => {
                let data = unsafe { data_as_slice_mut::<i16>(&mut storage) };
                data.fill(1);
            }
            DType::I32 => {
                let data = unsafe { data_as_slice_mut::<i32>(&mut storage) };
                data.fill(1);
            }
            DType::I64 => {
                let data = unsafe { data_as_slice_mut::<i64>(&mut storage) };
                data.fill(1);
            }
            DType::U8 => {
                let data = unsafe { data_as_slice_mut::<u8>(&mut storage) };
                data.fill(1);
            }
            DType::U16 => {
                let data = unsafe { data_as_slice_mut::<u16>(&mut storage) };
                data.fill(1);
            }
            DType::U32 => {
                let data = unsafe { data_as_slice_mut::<u32>(&mut storage) };
                data.fill(1);
            }
            DType::U64 => {
                let data = unsafe { data_as_slice_mut::<u64>(&mut storage) };
                data.fill(1);
            }
            DType::Bool => {
                let data = unsafe { data_as_slice_mut::<bool>(&mut storage) };
                data.fill(true);
            }
            DType::QI4 => {
                let data = unsafe { data_as_slice_mut::<crate::QuantizedI4>(&mut storage) };
                data.fill(crate::QuantizedI4::from_i8(1));
            }
            DType::QU8 => {
                let data = unsafe { data_as_slice_mut::<crate::QuantizedU8>(&mut storage) };
                data.fill(crate::QuantizedU8::from_raw(1));
            }
        }

        storage
    }

    /// Construct an `n x n` identity matrix for the given dtype.
    ///
    /// The diagonal is filled with "one" (see [`CpuBytesArray::ones`]) and off-diagonal entries are
    /// zero.
    pub fn eye(dtype: DType, n: usize) -> Self {
        let shape = Shape::from([n, n]);
        let mut storage = Self::zeros(dtype, shape);

        for i in 0..n {
            let idx = i * n + i;
            match storage.dtype {
                DType::F16 => {
                    let data = unsafe { data_as_slice_mut::<crate::Float16>(&mut storage) };
                    data[idx] = crate::Float16::from(1.0f32);
                }
                DType::F32 => {
                    let data = unsafe { data_as_slice_mut::<f32>(&mut storage) };
                    data[idx] = 1.0;
                }
                DType::F64 => {
                    let data = unsafe { data_as_slice_mut::<f64>(&mut storage) };
                    data[idx] = 1.0;
                }
                DType::BF16 => {
                    let data = unsafe { data_as_slice_mut::<crate::BFloat16>(&mut storage) };
                    data[idx] = crate::BFloat16::from_f32(1.0);
                }
                DType::BF8 => {
                    let data = unsafe { data_as_slice_mut::<crate::BFloat8>(&mut storage) };
                    data[idx] = crate::BFloat8::from(1.0f32);
                }
                DType::F8E4M3FN => {
                    let data = unsafe { data_as_slice_mut::<crate::Float8E4M3Fn>(&mut storage) };
                    data[idx] = crate::Float8E4M3Fn::from(1.0f32);
                }
                DType::F8E5M2 => {
                    let data = unsafe { data_as_slice_mut::<crate::Float8E5M2>(&mut storage) };
                    data[idx] = crate::Float8E5M2::from(1.0f32);
                }
                DType::Complex32 => {
                    let data = unsafe { data_as_slice_mut::<crate::Complex32>(&mut storage) };
                    data[idx] = crate::Complex32::new(1.0, 0.0);
                }
                DType::Complex64 => {
                    let data = unsafe { data_as_slice_mut::<crate::Complex64>(&mut storage) };
                    data[idx] = crate::Complex64::new(1.0, 0.0);
                }
                DType::Complex128 => {
                    let data = unsafe { data_as_slice_mut::<crate::Complex128>(&mut storage) };
                    data[idx] = crate::Complex128::new(1.0, 0.0);
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
                DType::QI4 => {
                    let data = unsafe { data_as_slice_mut::<crate::QuantizedI4>(&mut storage) };
                    data[idx] = crate::QuantizedI4::from_i8(1);
                }
                DType::QU8 => {
                    let data = unsafe { data_as_slice_mut::<crate::QuantizedU8>(&mut storage) };
                    data[idx] = crate::QuantizedU8::from_raw(1);
                }
            }
        }

        storage
    }

    /// Reshape the array without changing the underlying storage.
    ///
    /// # Errors
    /// Returns `Err` if the total element count would change.
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

    /// Transpose a 2D array, returning a new array with copied data.
    ///
    /// # Errors
    /// Returns `Err` if the array is not 2D or if transpose is not implemented for the dtype.
    pub fn transpose(self) -> Result<Self, String> {
        if self.shape.ndim() != 2 {
            return Err("Transpose only supported for 2D arrays".to_string());
        }

        let new_shape = Shape::from([self.shape.dim(1), self.shape.dim(0)]);
        let mut new_data = AlignedBytes::new_zeroed(self.data.len(), self.dtype.info().align);

        match self.dtype {
            DType::F32 => {
                let old_data = unsafe {
                    std::slice::from_raw_parts(
                        self.data.as_slice().as_ptr() as *const f32,
                        self.len,
                    )
                };
                let new_data_typed = unsafe {
                    std::slice::from_raw_parts_mut(
                        new_data.as_mut_slice().as_mut_ptr() as *mut f32,
                        self.len,
                    )
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
        self.data.as_slice()
    }

    #[inline]
    pub(crate) fn data_mut(&mut self) -> &mut [u8] {
        self.data.as_mut_slice()
    }

    /// Erase the concrete type and return a boxed `dyn NdArray`.
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

    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        Ok(Box::new(CpuBytesArray::zeros(dtype, shape)))
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
        let mut out = CpuBytesArray::zeros(self.dtype, new_shape.clone());

        match self.dtype {
            DType::F32 => {
                let old_data = unsafe { data_as_slice::<f32>(self) };
                let new_data = unsafe { data_as_slice_mut::<f32>(&mut out) };

                for i in 0..self.shape.dim(0) {
                    for j in 0..self.shape.dim(1) {
                        new_data[j * self.shape.dim(0) + i] = old_data[i * self.shape.dim(1) + j];
                    }
                }
            }
            DType::F64 => {
                let old_data = unsafe { data_as_slice::<f64>(self) };
                let new_data = unsafe { data_as_slice_mut::<f64>(&mut out) };

                for i in 0..self.shape.dim(0) {
                    for j in 0..self.shape.dim(1) {
                        new_data[j * self.shape.dim(0) + i] = old_data[i * self.shape.dim(1) + j];
                    }
                }
            }
            _ => return Err(format!("Transpose not implemented for {}", self.dtype)),
        }

        Ok(Box::new(out))
    }
}

/// Dense CPU-backed N-dimensional array with a concrete element type `T`.
///
/// This backend stores elements as `Vec<T>` and therefore has the strongest guarantees about
/// element validity.
///
/// `T` must implement [`DTypeLike`]. Note that [`DTypeLike`] is an **unsafe** trait: implementors
/// must ensure their representation matches the canonical dtype layout.
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
        Ok(Box::new(Array::new(data, shape)?))
    }

    fn ones(&self, shape: Shape) -> Result<Box<dyn NdArray>, String> {
        let len = shape.len();
        let mut data = vec![T::default(); len];

        match T::DTYPE {
            DType::F16 => fill_vec_as(&mut data, crate::Float16::from(1.0f32)),
            DType::F32 => fill_vec_as(&mut data, 1.0f32),
            DType::F64 => fill_vec_as(&mut data, 1.0f64),
            DType::BF16 => fill_vec_as(&mut data, crate::BFloat16::from_f32(1.0)),
            DType::BF8 => fill_vec_as(&mut data, crate::BFloat8::from(1.0f32)),
            DType::F8E4M3FN => fill_vec_as(&mut data, crate::Float8E4M3Fn::from(1.0f32)),
            DType::F8E5M2 => fill_vec_as(&mut data, crate::Float8E5M2::from(1.0f32)),
            DType::Complex32 => fill_vec_as(&mut data, crate::Complex32::new(1.0, 0.0)),
            DType::Complex64 => fill_vec_as(&mut data, crate::Complex64::new(1.0, 0.0)),
            DType::Complex128 => fill_vec_as(&mut data, crate::Complex128::new(1.0, 0.0)),
            DType::I8 => fill_vec_as(&mut data, 1i8),
            DType::I16 => fill_vec_as(&mut data, 1i16),
            DType::I32 => fill_vec_as(&mut data, 1i32),
            DType::I64 => fill_vec_as(&mut data, 1i64),
            DType::U8 => fill_vec_as(&mut data, 1u8),
            DType::U16 => fill_vec_as(&mut data, 1u16),
            DType::U32 => fill_vec_as(&mut data, 1u32),
            DType::U64 => fill_vec_as(&mut data, 1u64),
            DType::Bool => fill_vec_as(&mut data, true),
            DType::QI4 => fill_vec_as(&mut data, crate::QuantizedI4::from_i8(1)),
            DType::QU8 => fill_vec_as(&mut data, crate::QuantizedU8::from_raw(1)),
        }

        Ok(Box::new(Array::new(data, shape)?))
    }

    fn new_array(&self, shape: Shape, dtype: DType) -> Result<Box<dyn NdArray>, String> {
        if dtype == T::DTYPE {
            // Can create Array<T> directly
            let len = shape.len();
            let data = vec![T::default(); len];
            Ok(Box::new(Array::new(data, shape)?))
        } else {
            // Fall back to CpuBytesArray for different dtypes
            Ok(Box::new(CpuBytesArray::zeros(dtype, shape)))
        }
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

    #[test]
    fn cpu_bytes_ones_float16_is_one() {
        let shape = Shape::from([4]);
        let ones = CpuBytesArray::ones(DType::F16, shape);
        let values = unsafe { data_as_slice::<crate::Float16>(&ones) };
        for &v in values {
            assert_eq!(f32::from(v), 1.0);
        }
    }

    #[test]
    fn cpu_bytes_eye_complex64_sets_diagonal() {
        let eye = CpuBytesArray::eye(DType::Complex64, 3);
        let values = unsafe { data_as_slice::<crate::Complex64>(&eye) };
        for i in 0..3 {
            for j in 0..3 {
                let (re, im) = values[i * 3 + j].to_f32_tuple();
                if i == j {
                    assert_eq!(re, 1.0);
                    assert_eq!(im, 0.0);
                } else {
                    assert_eq!(re, 0.0);
                    assert_eq!(im, 0.0);
                }
            }
        }
    }

    #[test]
    fn cpu_bytes_ones_qi4_is_one() {
        let ones = CpuBytesArray::ones(DType::QI4, Shape::from([4]));
        let values = unsafe { data_as_slice::<crate::QuantizedI4>(&ones) };
        for &v in values {
            assert_eq!(v.to_i8(), 1);
        }
    }

    #[test]
    fn typed_array_ones_i32_is_one() {
        let array = Array::from_slice(&[0i32], Shape::from([1])).unwrap();
        let ones = array.ones(Shape::from([4])).unwrap();
        let values = unsafe { data_as_slice::<i32>(&*ones) };
        assert_eq!(values, &[1, 1, 1, 1]);
    }
}
