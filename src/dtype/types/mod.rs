//! Concrete dtype implementations.
//!
//! These types define Numina's canonical in-memory layout for non-primitive dtypes (and a few
//! wrapper types). They implement [`crate::dtype::DTypeCandidate`] and are used when encoding and
//! decoding values to/from bytes.

// Module declarations
pub mod bfloat16;
pub mod bfloat8;
pub mod complex;
pub mod float16;
pub mod float32;
pub mod float8;
pub mod quantized;

// Re-exports for convenience
pub use bfloat8::BFloat8;
pub use bfloat16::BFloat16;
pub use complex::{Complex32, Complex64, Complex128};
pub use float8::{Float8E4M3Fn, Float8E5M2};
pub use float16::Float16;
pub use float32::Float32;
pub use quantized::*;
