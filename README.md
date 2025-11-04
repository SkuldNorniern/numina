# Numina - Backend-Agnostic Array Library for Rust

A safe, efficient array library with ndarray-compatible operations, designed as the foundation for high-performance computing backends in Rust.

## Features

- **Safe & Ergonomic**: Memory-safe array operations with Rust's guarantees
- **Type Safe**: Compile-time shape and data type validation
- **Backend Agnostic**: `NdArray` trait enables multiple backends (CPU, GPU, remote)
- **Extensible Types**: Support for custom data types (BFloat16, quantized types)
- **Zero Dependencies**: Pure Rust implementation

## Quick Start

```rust
use numina::{Array, Shape, add, matmul, sum, F32};

// Create arrays
let a = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], Shape::from([2, 2]))?;
let b = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], Shape::from([2, 2]))?;

// Operations work on any NdArray backend
let c = add(&a, &b)?;           // Element-wise addition
let d = matmul(&a, &b)?;        // Matrix multiplication
let total = sum(&a, None)?;     // Sum all elements
let row_sums = sum(&a, Some(1))?; // Sum along axis
```

## Core Types

- **`Array<T>`**: Typed N-dimensional arrays for CPU operations
- **`CpuBytesArray`**: Byte-based N-dimensional arrays for CPU operations
- **`NdArray`**: Backend-agnostic trait for all array operations
- **`Shape`**: Multi-dimensional array dimensions
- **`DType`**: Data types (f32, f64, i8-i64, u8-u64, bool, custom types)

**Design Philosophy**: Numina provides the low-level backend infrastructure. High-level tensor APIs (like `Tensor` types) are provided by dependent crates like [`laminax-types`](../laminax-types/) which build upon Numina's `NdArray` trait.

## Custom Data Types

```rust
use numina::{BFloat16, QuantizedU8, QuantizedI4};

// Brain Float 16
let bf16 = BFloat16::from_f32(3.14159);
assert_eq!(bf16.size_bytes(), 2);

// 8-bit quantized
let q8 = QuantizedU8::quantize(2.5, 0.01);
assert!((q8.dequantize() - 2.5).abs() < 0.1);

// 4-bit quantized (2 values per byte)
let q4 = QuantizedI4::pack(3, -2, 1.0);
assert_eq!(q4.size_bytes(), 1); // 87.5% memory savings!
```

## Multiple Backends

```rust
use numina::{Array, CpuBytesArray, Shape, add, F32};

// Different backend implementations
let typed_array = Array::from_slice(&[1.0f32, 2.0], Shape::from([2]))?;
let bytes = [1.0f32, 2.0].iter().flat_map(|&x| x.to_le_bytes()).collect();
let byte_array = CpuBytesArray::new(bytes, Shape::from([2]), F32);

// Same operations work on all backends
let sum1 = add(&typed_array, &byte_array)?;
let sum2 = add(&byte_array, &typed_array)?;

// Cross-backend operations are fully supported
assert_eq!(sum1.shape(), sum2.shape());
```

## Architecture

```
src/
├── array/           # NdArray trait and CPU implementations
├── dtype/           # Data type system and custom types
├── ops.rs           # Mathematical operations
├── reductions.rs    # Reduction operations
├── sorting.rs       # Sorting and searching
└── lib.rs           # Library interface
```

## Status

**Implemented:**
- Array operations (add, mul, matmul, reductions)
- Multiple backends via NdArray trait (Array<T>, CpuBytesArray)
- Custom data types (BFloat16, QuantizedU8, QuantizedI4)
- Shape manipulation (reshape, transpose)
- Sorting and searching operations
- 31 tests passing

**Planned:**
- Broadcasting, advanced indexing, linear algebra
- File I/O, statistics
- Memory-mapped arrays
- More custom data types (FP8, FP4, NF4)

## Integration

Numina serves as one of the core libraries for [Laminax](https://github.com/SkuldNorniern/laminax), enabling high-performance GPU/CPU computing.
