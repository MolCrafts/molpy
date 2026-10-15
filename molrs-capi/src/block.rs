//! `extern "C"` functions for Block column access.
//!
//! A **Block** is a heterogeneous column table: each column is a
//! contiguous, typed ndarray keyed by an interned string.  This module
//! One read returns the column's bytes. The dtype is an out-parameter,
//! not part of the function name.
//!
//! | Pattern   | Function | Semantics |
//! |-----------|----------|-----------|
//! | **Pointer (read)**  | `molrs_block_get` | Zero-copy bytes; `out_dtype` is the stored variant |
//! | **Pointer (write)** | `molrs_block_get_mut` | Zero-copy mutable bytes; version bumped |
//! | **Copy**  | `molrs_block_copy` | Copies the column's bytes into a caller buffer |
//! | **Insert**| `molrs_block_set` | Copies caller data of the stated `MolrsDType` into a new column |
//!
//! `get`, `get_mut`, `copy` and `set` work for every fixed-width dtype. A
//! string column has no flat scalar buffer: reading one returns
//! `TypeMismatch`, which is not `KeyNotFound`, and `set` refuses
//! `MOLRS_D_TYPE_STRING` with `TypeMismatch`. Shape and dtype stay on
//! [`molrs_block_column_shape`] and [`molrs_block_column_dtype`].
//!
//! `F` is `f64`, `I` is `i32`, `Idx` is `u64`. `out_len` is an element
//! count. `molrs_block_copy`'s `buf_bytes` is a byte capacity.

use std::ffi::c_void;

use molrs::core::{BlockDtype, Column};
use molrs::op::{F, I, Idx};
use ndarray::ArrayD;
use num_complex::Complex;

use crate::error::{self, MolrsDType, MolrsStatus, ffi_err_to_status};
use crate::handle::{MolrsBlockHandle, c_to_block_handle};
use crate::handle_registry::lock_registry;
use crate::{ffi_try, null_check};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a C block handle to a Rust `BlockHandle`.
macro_rules! resolve_block {
    ($registry:expr, $c_handle:expr) => {
        match c_to_block_handle($c_handle, &$registry.interned_keys) {
            Some(bh) => bh,
            None => {
                error::set_last_error("invalid block handle or unknown key_id");
                return MolrsStatus::InvalidBlockHandle;
            }
        }
    };
}

/// Look up a column key string from an interned key_id.
macro_rules! resolve_col_key {
    ($registry:expr, $col_key_id:expr) => {
        match $registry.key_str($col_key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown col_key_id {}", $col_key_id));
                return MolrsStatus::KeyNotFound;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Get the number of rows in a block.
///
/// If the block has no columns yet, `*out` is set to 0.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_n_rows(MolrsBlockHandle block, size_t* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `out` -- On success, receives the row count.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale or invalid.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_n_rows(
    block: MolrsBlockHandle,
    out: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        match registry.frames.with_block(&bh, |b| b.n_rows()) {
            Ok(Some(n)) => {
                unsafe { *out = n };
                MolrsStatus::Ok
            }
            Ok(None) => {
                unsafe { *out = 0 };
                MolrsStatus::Ok
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Get the number of columns in a block.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_n_columns(MolrsBlockHandle block, size_t* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `out` -- On success, receives the column count.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale or invalid.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_n_columns(
    block: MolrsBlockHandle,
    out: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        match registry.frames.with_block(&bh, |b| b.len()) {
            Ok(n) => {
                unsafe { *out = n };
                MolrsStatus::Ok
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Query the data type of a column.
///
/// The returned [`MolrsDType`] is the stored variant. An `i64` column is
/// `Int64`, not `Int`. Pair it with [`molrs_block_get`](crate::molrs_block_get).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_column_dtype(MolrsBlockHandle block,
///                                    uint32_t col_key_id,
///                                    MolrsDType* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `col_key_id` -- Interned column name.
/// * `out` -- On success, receives the data type discriminant.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::KeyNotFound` if `col_key_id` was not interned or
///   the column does not exist.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `MolrsDType`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_column_dtype(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        let col_key = resolve_col_key!(registry, col_key_id);
        let result = registry.frames.with_block(&bh, |b| {
            b.get(&col_key).map(|col| MolrsDType::from(col.dtype()))
        });
        match result {
            Ok(Some(dt)) => {
                unsafe { *out = dt };
                MolrsStatus::Ok
            }
            Ok(None) => {
                error::set_last_error(format!("column '{}' not found", col_key));
                MolrsStatus::KeyNotFound
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Query the shape (dimensionality) of a column.
///
/// On entry, `*inout_ndim` must hold the capacity of the `out_shape`
/// buffer.  On return, `*inout_ndim` is set to the actual number of
/// dimensions, and the first `min(capacity, ndim)` elements of
/// `out_shape` are filled.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_column_shape(MolrsBlockHandle block,
///                                    uint32_t col_key_id,
///                                    size_t*  out_shape,
///                                    size_t*  inout_ndim);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `col_key_id` -- Interned column name.
/// * `out_shape` -- Buffer of at least `*inout_ndim` elements.
/// * `inout_ndim` -- In: buffer capacity.  Out: actual number of dimensions.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out_shape` or `inout_ndim` is null.
/// * `MolrsStatus::KeyNotFound` if the column does not exist.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_shape` must point to a buffer of at least `*inout_ndim`
///   writable `size_t` elements.
/// * `inout_ndim` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_column_shape(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_shape: *mut usize,
    inout_ndim: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_shape);
        null_check!(inout_ndim);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        let col_key = resolve_col_key!(registry, col_key_id);
        let result = registry
            .frames
            .with_block(&bh, |b| b.get(&col_key).map(|col| col.shape().to_vec()));
        match result {
            Ok(Some(shape)) => {
                let max_ndim = unsafe { *inout_ndim };
                let actual_ndim = shape.len();
                unsafe { *inout_ndim = actual_ndim };
                let copy_len = actual_ndim.min(max_ndim);
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_shape, copy_len) };
                out_slice.copy_from_slice(&shape[..copy_len]);
                MolrsStatus::Ok
            }
            Ok(None) => {
                error::set_last_error(format!("column '{}' not found", col_key));
                MolrsStatus::KeyNotFound
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

// ---------------------------------------------------------------------------
// Zero-copy read, mutable read, and copy
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ReadFail {
    Missing,
    String,
    NonContiguous,
    TooSmall,
}

fn fail_status(fail: ReadFail, key: &str) -> MolrsStatus {
    let (status, msg) = match fail {
        ReadFail::Missing => (
            MolrsStatus::KeyNotFound,
            format!("column '{key}' not found"),
        ),
        ReadFail::String => (
            MolrsStatus::TypeMismatch,
            format!("column '{key}' is string; a string column has no flat scalar buffer"),
        ),
        ReadFail::NonContiguous => (
            MolrsStatus::NonContiguous,
            format!("column '{key}' is not contiguous"),
        ),
        ReadFail::TooSmall => (MolrsStatus::InvalidArgument, "buffer too small".to_string()),
    };
    error::set_last_error(msg);
    status
}

/// Run `$f::<T>($args)` with `T` the element type of the fixed-width
/// [`MolrsDType`] `$dtype`; `string => $string` is the arm for
/// [`MolrsDType::String`], which has no flat scalar buffer.
macro_rules! dispatch_scalar {
    ($dtype:expr, $f:ident($($arg:expr),*), string => $string:expr) => {
        match $dtype {
            MolrsDType::Float => $f::<F>($($arg),*),
            MolrsDType::Int => $f::<I>($($arg),*),
            MolrsDType::Bool => $f::<bool>($($arg),*),
            MolrsDType::Uint => $f::<Idx>($($arg),*),
            MolrsDType::I8 => $f::<i8>($($arg),*),
            MolrsDType::I16 => $f::<i16>($($arg),*),
            MolrsDType::I64 => $f::<i64>($($arg),*),
            MolrsDType::U8 => $f::<u8>($($arg),*),
            MolrsDType::U16 => $f::<u16>($($arg),*),
            MolrsDType::U32 => $f::<u32>($($arg),*),
            MolrsDType::C64 => $f::<Complex<f32>>($($arg),*),
            MolrsDType::C128 => $f::<Complex<f64>>($($arg),*),
            MolrsDType::String => $string,
        }
    };
}

fn scalar_view(col: &Column) -> Result<(*const u8, usize, MolrsDType), ReadFail> {
    fn view<T: BlockDtype>(col: &Column) -> Result<(*const u8, usize), ReadFail> {
        // `col.dtype()` named `T`, so the column holds `T` elements.
        let arr = col.get::<T>().ok_or(ReadFail::String)?;
        let slice = arr.as_slice_memory_order().ok_or(ReadFail::NonContiguous)?;
        Ok((slice.as_ptr().cast(), slice.len()))
    }
    let dtype = MolrsDType::from(col.dtype());
    let (ptr, len) = dispatch_scalar!(dtype, view(col), string => Err(ReadFail::String))?;
    Ok((ptr, len, dtype))
}

fn scalar_view_mut(col: &mut Column) -> Result<(*mut u8, usize, MolrsDType), ReadFail> {
    fn view_mut<T: BlockDtype>(col: &mut Column) -> Result<(*mut u8, usize), ReadFail> {
        let arr = col.get_mut::<T>().ok_or(ReadFail::String)?;
        let slice = arr
            .as_slice_memory_order_mut()
            .ok_or(ReadFail::NonContiguous)?;
        Ok((slice.as_mut_ptr().cast(), slice.len()))
    }
    let dtype = MolrsDType::from(col.dtype());
    let (ptr, len) = dispatch_scalar!(dtype, view_mut(col), string => Err(ReadFail::String))?;
    Ok((ptr, len, dtype))
}

fn scalar_bytes(col: &Column) -> Result<Vec<u8>, ReadFail> {
    let (ptr, len, _) = scalar_view(col)?;
    let width = col.dtype().itemsize().ok_or(ReadFail::String)?;
    let nbytes = len.saturating_mul(width);
    // `ptr` addresses `len` elements of `width` bytes still owned by `col`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec();
    Ok(bytes)
}

/// Byte pointer and element count for one column.
///
/// `*out_len` is the number of elements. The byte length is `*out_len` times
/// the dtype's item size (`Bool` is 1, `C64` is 8, `C128` is 16).
/// `*out_dtype` is the stored variant.
///
/// A missing column is `KeyNotFound`. A string column is `TypeMismatch`:
/// a string has no flat scalar buffer. Those are different statuses.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_ptr`, `out_len`, and `out_dtype` must be non-null and writable.
/// * The returned pointer is valid until the block is mutated or the frame
///   is dropped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_get(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_dtype: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_ptr);
        null_check!(out_len);
        null_check!(out_dtype);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        let col_key = resolve_col_key!(registry, col_key_id);
        let result = registry.frames.with_block(&bh, |b| match b.get(&col_key) {
            None => Err(ReadFail::Missing),
            Some(col) => scalar_view(col),
        });
        match result {
            Ok(Ok((ptr, len, dtype))) => {
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                    *out_dtype = dtype;
                }
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Mutable byte pointer for one column. Bumps `block->block_version`.
///
/// Same dtype and status rules as [`molrs_block_get`]. The call bumps the
/// version itself, so a write through the pointer needs no follow-up call.
///
/// # Safety
///
/// * `block` must point to a live, writable `MolrsBlockHandle`.
/// * `out_ptr`, `out_len`, and `out_dtype` must be non-null and writable.
/// * The returned pointer is valid until another mutating call on the same
///   block, or until the frame is dropped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_get_mut(
    block: *mut MolrsBlockHandle,
    col_key_id: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    out_dtype: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(block);
        null_check!(out_ptr);
        null_check!(out_len);
        null_check!(out_dtype);
        let c_handle = unsafe { &*block };
        let mut registry = lock_registry();
        let col_key = resolve_col_key!(registry, col_key_id);
        let mut bh = resolve_block!(registry, c_handle);
        let result = registry
            .frames
            .with_block_mut(&mut bh, |b| match b.get_mut(&col_key) {
                None => Err(ReadFail::Missing),
                Some(col) => scalar_view_mut(col),
            });
        match result {
            Ok(Ok((ptr, len, dtype))) => {
                let c_block = unsafe { &mut *block };
                c_block.block_version = bh.version();
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                    *out_dtype = dtype;
                }
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Copy one column's bytes into `out_buf`.
///
/// `buf_bytes` is the buffer's capacity in bytes, not elements. A buffer
/// shorter than `n_elements * itemsize` returns `InvalidArgument`.
/// A missing column is `KeyNotFound`. A string column is `TypeMismatch`.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_buf` must point at `buf_bytes` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_copy(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_buf: *mut u8,
    buf_bytes: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_buf);
        let registry = lock_registry();
        let bh = resolve_block!(registry, &block);
        let col_key = resolve_col_key!(registry, col_key_id);
        let result = registry.frames.with_block(&bh, |b| {
            let Some(col) = b.get(&col_key) else {
                return Err(ReadFail::Missing);
            };
            let bytes = scalar_bytes(col)?;
            if buf_bytes < bytes.len() {
                return Err(ReadFail::TooSmall);
            }
            Ok(bytes)
        });
        match result {
            Ok(Ok(data)) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, data.len()) };
                out_slice.copy_from_slice(&data);
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

// ---------------------------------------------------------------------------
// Insert columns
// ---------------------------------------------------------------------------

/// A fixed-width element a C caller hands over as a flat buffer.
trait CScalar: BlockDtype + Copy {
    /// Copy `n` elements out of `data`.
    ///
    /// # Safety
    ///
    /// `data` points to `n` readable, aligned elements of the C type of
    /// `Self`.
    unsafe fn read(data: *const c_void, n: usize) -> Result<Vec<Self>, String> {
        Ok(unsafe { std::slice::from_raw_parts(data.cast::<Self>(), n) }.to_vec())
    }
}

impl CScalar for F {}
impl CScalar for I {}
impl CScalar for Idx {}
impl CScalar for i8 {}
impl CScalar for i16 {}
impl CScalar for i64 {}
impl CScalar for u8 {}
impl CScalar for u16 {}
impl CScalar for u32 {}
impl CScalar for Complex<f32> {}
impl CScalar for Complex<f64> {}

/// A C `bool` is one byte; any byte other than 0 or 1 is no `bool` and is
/// refused rather than reinterpreted.
impl CScalar for bool {
    unsafe fn read(data: *const c_void, n: usize) -> Result<Vec<Self>, String> {
        let bytes = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), n) };
        bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| match b {
                0 => Ok(false),
                1 => Ok(true),
                other => Err(format!(
                    "bool element {i} is byte {other}; a bool column holds 0 or 1"
                )),
            })
            .collect()
    }
}

/// The `shape`d array of `T` the caller's buffer holds.
///
/// # Safety
///
/// `data` points to `product(shape)` readable, aligned elements of `T`.
unsafe fn column_from_c<T: CScalar>(
    data: *const c_void,
    shape: &[usize],
) -> Result<Column, String> {
    let total_len: usize = shape.iter().product();
    let values = unsafe { T::read(data, total_len) }?;
    ArrayD::<T>::from_shape_vec(shape.to_vec(), values)
        .map(Column::from)
        .map_err(|e| format!("invalid shape/data: {e}"))
}

/// Insert (or replace) a column, copying the caller's data. `dtype` states
/// the element type `data` points to, and becomes the column's dtype.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_set(MolrsBlockHandle* block, uint32_t col_key_id,
///                             MolrsDType dtype, const void* data,
///                             const size_t* shape, size_t ndim);
/// ```
///
/// # Arguments
///
/// * `block` -- Pointer to the block handle; its `block_version` is updated.
/// * `col_key_id` -- Interned column name.
/// * `dtype` -- Element type of `data`: `MOLRS_D_TYPE_FLOAT` (`F`),
///   `MOLRS_D_TYPE_INT` (`int32_t`), `MOLRS_D_TYPE_UINT` (`uint64_t`),
///   `MOLRS_D_TYPE_BOOL` (one byte, 0 or 1), `MOLRS_D_TYPE_I8` … `U32`,
///   `MOLRS_D_TYPE_C64` / `C128` (a `float` / `double` pair per element).
///   `MOLRS_D_TYPE_STRING` has no flat buffer and is refused.
/// * `data` -- Pointer to the source data (row-major).
/// * `shape` -- Pointer to `ndim` dimension sizes.
/// * `ndim` -- Number of dimensions (must be >= 1).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any pointer is null.
/// * `MolrsStatus::TypeMismatch` if `dtype` is `MOLRS_D_TYPE_STRING`.
/// * `MolrsStatus::InvalidArgument` if `ndim == 0`, the element count is 0,
///   a `bool` element is neither 0 nor 1, or the column does not fit the
///   block (a row count other than the block's).
/// * `MolrsStatus::KeyNotFound` if `col_key_id` was not interned.
/// * `MolrsStatus::InvalidBlockHandle` if the block handle is stale.
///
/// # Safety
///
/// * `block` must point to a live, writable `MolrsBlockHandle`.
/// * `data` must point to at least `product(shape[0..ndim])` readable,
///   aligned elements of the C type `dtype` names.
/// * `shape` must point to `ndim` elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_set(
    block: *mut MolrsBlockHandle,
    col_key_id: u32,
    dtype: MolrsDType,
    data: *const c_void,
    shape: *const usize,
    ndim: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(block);
        null_check!(data);
        null_check!(shape);
        if ndim == 0 {
            error::set_last_error("ndim must be > 0");
            return MolrsStatus::InvalidArgument;
        }
        let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
        if shape_slice.iter().product::<usize>() == 0 {
            error::set_last_error("total element count is 0");
            return MolrsStatus::InvalidArgument;
        }
        // SAFETY: the caller guarantees `data` holds `product(shape)`
        // aligned elements of the C type `dtype` names.
        let column = unsafe {
            dispatch_scalar!(
                dtype,
                column_from_c(data, shape_slice),
                string => {
                    error::set_last_error(
                        "a string column has no flat scalar buffer; molrs_block_set takes a \
                         fixed-width MolrsDType",
                    );
                    return MolrsStatus::TypeMismatch;
                }
            )
        };
        let column = match column {
            Ok(c) => c,
            Err(msg) => {
                error::set_last_error(msg);
                return MolrsStatus::InvalidArgument;
            }
        };

        let c_handle = unsafe { &*block };
        let mut registry = lock_registry();
        let col_key = resolve_col_key!(registry, col_key_id);
        let mut bh = resolve_block!(registry, c_handle);

        let result = registry
            .frames
            .with_block_mut(&mut bh, |b| b.insert(col_key.clone(), column, None));
        match result {
            Ok(Ok(())) => {
                let c_block = unsafe { &mut *block };
                c_block.block_version = bh.version();
                MolrsStatus::Ok
            }
            Ok(Err(block_err)) => {
                error::set_last_error(format!("block insert error: {block_err}"));
                MolrsStatus::InvalidArgument
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

#[cfg(test)]
mod scalar_view_tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn i64_view_reports_int64_and_the_elements() {
        let col = Column::from(Array1::from_vec(vec![7_i64, 8]).into_dyn());
        let (ptr, len, dtype) = scalar_view(&col).unwrap();
        assert_eq!(len, 2);
        assert_eq!(dtype, MolrsDType::I64);
        let vals = unsafe { std::slice::from_raw_parts(ptr as *const i64, len) };
        assert_eq!(vals, &[7, 8]);
    }

    #[test]
    fn string_is_not_a_missing_column() {
        let col = Column::from(Array1::from_vec(vec!["a".to_string()]).into_dyn());
        assert!(matches!(scalar_view(&col), Err(ReadFail::String)));
        assert!(matches!(scalar_bytes(&col), Err(ReadFail::String)));
    }
}
