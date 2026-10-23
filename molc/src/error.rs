//! Error types and thread-local error buffer for the C API.
//!
//! Every `extern "C"` function returns a [`MolcStatus`] code.  On
//! failure, a human-readable message is stored in a thread-local buffer
//! and can be retrieved via [`crate::molc_last_error`].
//!
//! Column data types are exposed to C as [`MolcDType`] discriminants
//! that map one-to-one to the internal [`molrs::core::DType`] enum.

use std::cell::RefCell;
use std::ffi::c_char;

use molrs::core::DType;
use molrs_ffi::FfiError;

/// Status codes returned by every `extern "C"` function.
///
/// A value of `Ok` (0) indicates success.  All other values are errors.
/// Call [`molc_last_error`](crate::molc_last_error) to obtain a
/// human-readable description of the most recent error.
///
/// # C mapping
///
/// In the generated header this is `enum MolcStatus` with the same
/// integer discriminants listed below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolcStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// The supplied `MolcFrameHandle` does not refer to a live frame.
    InvalidFrameHandle = 1,
    /// The supplied `MolcBlockHandle` does not refer to a live block,
    /// or its version has been invalidated.
    InvalidBlockHandle = 2,
    /// The supplied `MolcBoxHandle` does not refer to a live Box.
    InvalidBoxHandle = 3,
    /// The supplied `MolcForceFieldHandle` does not refer to a live
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
    /// The supplied `MolcRegionHandle` does not refer to a live region.
    InvalidRegionHandle = 14,
}

/// Data type discriminants for Block columns.
///
/// Each column in a [`Block`](molrs::core::Block) stores a
/// homogeneously-typed ndarray.  This enum is the stored variant, not a
/// width bucket: an `i64` column is [`I64`](Self::I64), not [`I32`](Self::I32).
/// Discriminants 0–4 stay where they were; later variants are appended.
///
/// Each variant is named after the core dtype name (`DType::name()`), so the
/// C constant is `MOLC_D_TYPE_` + that name upper-cased: `MOLC_D_TYPE_F64`,
/// `MOLC_D_TYPE_U64`, `MOLC_D_TYPE_I64`, `MOLC_D_TYPE_C128`, ….
///
/// A string column has no flat scalar buffer. [`molc_block_get`](crate::molc_block_get)
/// reports that as `TypeMismatch`, not as a missing key.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolcDType {
    /// `f64` column.
    F64 = 0,
    /// `i32` column.
    I32 = 1,
    /// `bool`: one byte per element.
    Bool = 2,
    /// `u64` column (indices and identifiers).
    U64 = 3,
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

impl TryFrom<DType> for MolcDType {
    type Error = String;

    /// The C code of `dt`; a dtype this build of the C API has no code for
    /// (molrs's `DType` is `non_exhaustive`) is an error naming it, never a
    /// code standing in for it.
    fn try_from(dt: DType) -> Result<Self, String> {
        Ok(match dt {
            DType::Float => Self::F64,
            DType::Int => Self::I32,
            DType::Bool => Self::Bool,
            DType::Uint => Self::U64,
            DType::String => Self::String,
            DType::I8 => Self::I8,
            DType::I16 => Self::I16,
            DType::I64 => Self::I64,
            DType::U8 => Self::U8,
            DType::U16 => Self::U16,
            DType::U32 => Self::U32,
            DType::C64 => Self::C64,
            DType::C128 => Self::C128,
            other => return Err(format!("the C API has no dtype code for {other:?}")),
        })
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

/// Convert an `FfiError` to a `MolcStatus`, storing the message.
pub(crate) fn ffi_err_to_status(err: &FfiError) -> MolcStatus {
    set_last_error(err.to_string());
    match err {
        FfiError::InvalidFrameId => MolcStatus::InvalidFrameHandle,
        FfiError::InvalidBlockHandle => MolcStatus::InvalidBlockHandle,
        FfiError::KeyNotFound { .. } => MolcStatus::KeyNotFound,
        FfiError::NonContiguous { .. } => MolcStatus::NonContiguous,
        FfiError::DTypeMismatch { .. } => MolcStatus::TypeMismatch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_keeps_its_discriminants() {
        assert_eq!(MolcDType::try_from(DType::Float).unwrap() as u8, 0);
        assert_eq!(MolcDType::try_from(DType::Int).unwrap() as u8, 1);
        assert_eq!(MolcDType::try_from(DType::Bool).unwrap() as u8, 2);
        assert_eq!(MolcDType::try_from(DType::Uint).unwrap() as u8, 3);
        assert_eq!(MolcDType::try_from(DType::String).unwrap() as u8, 4);
    }

    #[test]
    fn every_variant_is_named_after_the_core_dtype() {
        // The C constant is MOLC_D_TYPE_<name>: the variant's name must be
        // the core `DType::name()` for every dtype the core has.
        for dtype in DType::ALL {
            let variant = format!("{:?}", MolcDType::try_from(dtype).unwrap()).to_lowercase();
            assert_eq!(variant, dtype.name());
        }
    }
}
