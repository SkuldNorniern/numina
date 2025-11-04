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
