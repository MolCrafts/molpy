//! Error types and thread-local error buffer for the C API.
//!
//! Every `extern "C"` function returns a [`MolrsStatus`] code.  On
//! failure, a human-readable message is stored in a thread-local buffer
//! and can be retrieved via [`crate::molrs_last_error`].
//!
//! Column data types are exposed to C as [`MolrsDType`] discriminants
//! that map one-to-one to the internal [`molrs::core::DType`] enum.

use std::cell::RefCell;
use std::ffi::c_char;

use molrs::core::DType;
use molrs_ffi::FfiError;

/// Status codes returned by every `extern "C"` function.
///
/// A value of `Ok` (0) indicates success.  All other values are errors.
/// Call [`molrs_last_error`](crate::molrs_last_error) to obtain a
/// human-readable description of the most recent error.
///
/// # C mapping
///
/// In the generated header this is `enum MolrsStatus` with the same
/// integer discriminants listed below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// The supplied `MolrsFrameHandle` does not refer to a live frame.
    InvalidFrameHandle = 1,
    /// The supplied `MolrsBlockHandle` does not refer to a live block,
    /// or its version has been invalidated.
    InvalidBlockHandle = 2,
    /// The supplied `MolrsBoxHandle` does not refer to a live Box.
    InvalidBoxHandle = 3,
    /// The supplied `MolrsForceFieldHandle` does not refer to a live
    /// force field.
    InvalidForceFieldHandle = 4,
    /// A requested key (block name, column name, metadata key) was not
    /// found.
    KeyNotFound = 5,
    /// A column's internal storage is not contiguous in memory, so a
    /// zero-copy pointer cannot be returned.
    NonContiguous = 6,
    /// A function argument has an invalid value (e.g. zero-length shape,
    /// out-of-range index, buffer too small).
    InvalidArgument = 7,
    /// A required pointer argument was `NULL`.
    NullPointer = 8,
    /// The column's data type does not match the requested accessor
    /// (e.g. asking for `float` on an `int` column).
    TypeMismatch = 9,
    /// An unexpected internal error (e.g. a caught panic).
    InternalError = 10,
    /// A C string argument was not valid UTF-8.
    Utf8Error = 11,
    /// The provided 3x3 cell matrix is singular (determinant is zero)
    /// and cannot form a valid simulation box.
    SingularCell = 12,
    /// A parse error occurred (e.g. invalid SMILES or JSON string).
    ParseError = 13,
    /// The supplied `MolrsRegionHandle` does not refer to a live region.
    InvalidRegionHandle = 14,
}

/// Data type discriminants for Block columns.
///
/// Each column in a [`Block`](molrs::core::Block) stores a
/// homogeneously-typed ndarray.  This enum is the stored variant, not a
/// width bucket: an `i64` column is [`I64`](Self::I64), not [`Int`](Self::Int).
/// Discriminants 0–4 stay where they were; later variants are appended.
///
/// Each variant is named after the core dtype name (`DType::name()`), so the
/// C constant is `MOLRS_D_TYPE_` + that name upper-cased: `MOLRS_D_TYPE_FLOAT`,
/// `MOLRS_D_TYPE_UINT`, `MOLRS_D_TYPE_I64`, `MOLRS_D_TYPE_C128`, ….
///
/// A string column has no flat scalar buffer. [`molrs_block_get`](crate::molrs_block_get)
/// reports that as `TypeMismatch`, not as a missing key.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsDType {
    /// `float`: `f64` column.
    Float = 0,
    /// `int`: `i32` column.
    Int = 1,
    /// `bool`: one byte per element.
    Bool = 2,
    /// `uint`: `u64` column (indices and identifiers).
    Uint = 3,
    /// `string`: no flat scalar buffer.
    String = 4,
    /// `i8` column.
    I8 = 5,
    /// `i16` column.
    I16 = 6,
    /// `i64` column.
    I64 = 7,
    /// `u8` column.
    U8 = 8,
    /// `u16` column.
    U16 = 9,
    /// `u32` column.
    U32 = 10,
    /// `c64`: numpy `complex64`, a pair of `f32` per element.
    C64 = 11,
    /// `c128`: numpy `complex128`, a pair of `f64` per element.
    C128 = 12,
}

impl From<DType> for MolrsDType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float => Self::Float,
            DType::Int => Self::Int,
            DType::Bool => Self::Bool,
            DType::Uint => Self::Uint,
            DType::String => Self::String,
            DType::I8 => Self::I8,
            DType::I16 => Self::I16,
            DType::I64 => Self::I64,
            DType::U8 => Self::U8,
            DType::U16 => Self::U16,
            DType::U32 => Self::U32,
            DType::C64 => Self::C64,
            DType::C128 => Self::C128,
            // `DType` is `non_exhaustive`. Every variant that exists today is
            // named above; a future one must not be reported as `String`.
            other => unreachable!("no C dtype for {other:?}"),
        }
    }
}

// Thread-local error buffer: null-terminated bytes.
thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Store an error message in the thread-local buffer.
pub(crate) fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| {
        let s = msg.into();
        let mut buf = e.borrow_mut();
        buf.clear();
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
    });
}

/// Return a pointer to the last error message (null-terminated).
///
/// Valid until the next error is set on this thread.
pub(crate) fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|e| {
        let buf = e.borrow();
        if buf.is_empty() {
            c"".as_ptr()
        } else {
            buf.as_ptr() as *const c_char
        }
    })
}

/// Convert an `FfiError` to a `MolrsStatus`, storing the message.
pub(crate) fn ffi_err_to_status(err: &FfiError) -> MolrsStatus {
    set_last_error(err.to_string());
    match err {
        FfiError::InvalidFrameId => MolrsStatus::InvalidFrameHandle,
        FfiError::InvalidBlockHandle => MolrsStatus::InvalidBlockHandle,
        FfiError::KeyNotFound { .. } => MolrsStatus::KeyNotFound,
        FfiError::NonContiguous { .. } => MolrsStatus::NonContiguous,
        FfiError::DTypeMismatch { .. } => MolrsStatus::TypeMismatch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_keeps_its_discriminants() {
        assert_eq!(MolrsDType::from(DType::Float) as u8, 0);
        assert_eq!(MolrsDType::from(DType::Int) as u8, 1);
        assert_eq!(MolrsDType::from(DType::Bool) as u8, 2);
        assert_eq!(MolrsDType::from(DType::Uint) as u8, 3);
        assert_eq!(MolrsDType::from(DType::String) as u8, 4);
    }

    #[test]
    fn every_variant_is_named_after_the_core_dtype() {
        // The C constant is MOLRS_D_TYPE_<name>: the variant's name must be
        // the core `DType::name()` for every dtype the core has.
        for dtype in DType::ALL {
            let variant = format!("{:?}", MolrsDType::from(dtype)).to_lowercase();
            assert_eq!(variant, dtype.name());
        }
    }
}
