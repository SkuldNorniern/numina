use std::fmt;

use crate::array::Shape;

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