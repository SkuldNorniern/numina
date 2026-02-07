use std::alloc::{self, Layout};
use std::fmt;
use std::ptr::NonNull;

/// A heap allocation of `len` bytes with a guaranteed alignment.
///
/// This is an internal utility used to back `CpuBytesArray` while still allowing typed
/// reinterpretation (e.g. `&[f32]`) which requires proper alignment.
///
/// Notes:
/// - The memory is always initialized (zeroed or copied from a slice).
/// - Alignment is rounded up to the next power of two when needed.
pub struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

impl AlignedBytes {
    /// Allocate `len` zeroed bytes with at least `align` alignment.
    pub fn new_zeroed(len: usize, align: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                align: 1,
            };
        }

        let align = align.max(1);
        let align = if align.is_power_of_two() {
            align
        } else {
            align.next_power_of_two()
        };

        let layout = Layout::from_size_align(len, align).expect("invalid AlignedBytes layout");
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).expect("allocation failure");

        Self { ptr, len, align }
    }

    /// Allocate and copy `bytes` into a new aligned buffer.
    pub fn from_slice(bytes: &[u8], align: usize) -> Self {
        let mut out = Self::new_zeroed(bytes.len(), align);
        out.as_mut_slice().copy_from_slice(bytes);
        out
    }

    /// Length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Immutable view of the raw bytes.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutable view of the raw bytes.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn layout(&self) -> Option<Layout> {
        if self.len == 0 {
            None
        } else {
            Some(
                Layout::from_size_align(self.len, self.align).expect("invalid AlignedBytes layout"),
            )
        }
    }
}

impl Clone for AlignedBytes {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice(), self.align)
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        if let Some(layout) = self.layout() {
            unsafe {
                alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .field("align", &self.align)
            .finish()
    }
}
