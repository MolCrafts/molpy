//! The exceptions the bindings define and every shared Rust-error → Python
//! mapping.
//!
//! Each exception is declared under the module that owns it:
//! `molpy.core.BlockDtypeError` (a column value the Store cannot hold),
//! `molpy.core.UnitsError` and `molpy.io.smiles.SmilesError`. A subsystem whose
//! refusals form a family of their own (`molpy.ff.ir`'s `IrError` tree)
//! declares it beside its bindings.
//!
//! `BlockDtypeError` is the single error a column write raises when the value
//! is not numpy-representable by the Rust Store (object dtype, None-bearing, or
//! ragged/mixed). It subclasses Python `TypeError` so downstream code can
//! `except molpy.core.BlockDtypeError` precisely while still being caught by
//! broad `except TypeError` handlers.

use molrs_ffi::FfiError;
use pyo3::create_exception;
use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

create_exception!(
    molpy.core,
    BlockDtypeError,
    PyTypeError,
    "Raised when a Block column value is not a numpy-representable dtype \
     (object, None-bearing, or ragged/mixed). The Rust Store holds only \
     float / int / bool / str columns."
);

create_exception!(
    molpy.core,
    UnitsError,
    PyValueError,
    "Raised when unit parsing, definition, arithmetic, or conversion fails."
);

create_exception!(
    molpy.io.smiles,
    SmilesError,
    PyValueError,
    "Raised when a SMILES / SMARTS / CGsmiles string is refused, by the \
     parser or by a later stage (expansion, emit). Carries the four facts \
     the Rust `SmilesError` owns: `kind` (the `SmilesErrorKind` variant \
     name), `span` (byte range into `input`, end clamped to its length), \
     `input` (the offending text, empty when the error was raised away from \
     the scanner) and `notation` (`'smiles' | 'smarts' | 'cgsmiles'`). \
     Subclasses Python `ValueError`, so broad `except ValueError` handlers \
     keep catching it."
);

/// Preserve the native units error message at the Python boundary.
pub fn units_error(error: ::molrs::core::UnitsError) -> PyErr {
    UnitsError::new_err(error.to_string())
}

/// Build a column-named `BlockDtypeError` for a rejected array.
///
/// Best-effort introspects the offending array to name the detected dtype and,
/// for object arrays, whether it is None-bearing (vs ragged/mixed) so the
/// message guides the caller to coerce or drop the column.
pub fn dtype_reject(key: &str, array: &Bound<'_, PyAny>) -> PyErr {
    let dtype = array
        .getattr("dtype")
        .and_then(|d| d.str())
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());

    let kind = array
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .unwrap_or_default();

    let mut reason = format!("column '{key}' has unsupported dtype '{dtype}'");
    if kind == "O" {
        if object_array_is_none_bearing(array) {
            reason.push_str("; the object array is None-bearing");
        } else {
            reason.push_str("; the object array is ragged/mixed");
        }
    }
    reason.push_str(
        ". Coerce to a numpy float/int/bool/str column or drop the column \
         (the Rust Store has no object-column overflow).",
    );
    BlockDtypeError::new_err(reason)
}

/// True if any element of a flattened object array is Python `None`.
fn object_array_is_none_bearing(array: &Bound<'_, PyAny>) -> bool {
    let Ok(flat) = array.call_method0("ravel") else {
        return false;
    };
    let Ok(iter) = flat.try_iter() else {
        return false;
    };
    iter.flatten().any(|item| item.is_none())
}

/// Convert an [`FfiError`] to the most appropriate Python exception.
///
/// | Variant               | Python exception   |
/// |-----------------------|--------------------|
/// | `InvalidFrameId`      | `RuntimeError`     |
/// | `InvalidBlockHandle`  | `RuntimeError`     |
/// | `KeyNotFound`         | `KeyError`         |
/// | `NonContiguous`       | `ValueError`       |
/// | `DTypeMismatch`       | `TypeError`        |
pub(crate) fn ffi_error_to_pyerr(err: FfiError) -> PyErr {
    match err {
        FfiError::InvalidFrameId => PyRuntimeError::new_err("invalid frame handle"),
        FfiError::InvalidBlockHandle => PyRuntimeError::new_err("invalid block handle"),
        FfiError::KeyNotFound { key } => PyKeyError::new_err(key),
        FfiError::NonContiguous { key } => {
            PyValueError::new_err(format!("column '{key}' is not contiguous in memory"))
        }
        FfiError::DTypeMismatch {
            key,
            expected,
            actual,
        } => PyTypeError::new_err(format!(
            "column '{key}' has dtype {actual} but expected {expected}"
        )),
    }
}

/// Convert a [`std::io::Error`] to a Python `IOError`.
pub fn io_error_to_pyerr(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// Convert a [`molrs::core::MolRsError`] to a Python `ValueError`.
pub fn molrs_error_to_pyerr(e: molrs::core::MolRsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a [`molrs::io::smiles::SmilesError`] to a Python
/// [`SmilesError`](crate::error::SmilesError).
///
/// The rendered Rust message stays `args[0]`, so `str(e)` is unchanged and the
/// class subclasses `ValueError`; on top of it the four facts the Rust error
/// owns cross as plain attributes — `kind`, `span`, `input`, `notation` — so a
/// caller can branch on the reason instead of matching message text.
///
/// The helper takes no [`Python`] token (it is used as a bare `map_err`
/// argument all over the io bindings), so it attaches to the interpreter
/// itself. Nothing here can panic across the seam: if the interpreter refuses
/// to carry the attributes — only reachable under allocation failure or
/// interpreter shutdown — that failure is returned as the raised error rather
/// than unwrapped.
pub fn smiles_error_to_pyerr(e: molrs::io::smiles::SmilesError) -> PyErr {
    let message = e.to_string();
    // A span is a byte range into `input`, and the scanner reports
    // end-of-input one byte past the text (`Span::new(len, len + 1)`), which
    // is not a valid slice bound of the string published alongside it — so the
    // end is clamped here. The start is clamped to that end so the pair stays
    // an ordered range for errors raised away from the scanner, which carry a
    // span into text they do not publish (`input` is then empty).
    let end = e.span.end.min(e.input.len());
    let start = e.span.start.min(end);

    Python::attach(|py| {
        let err = SmilesError::new_err(message);
        let value = err.value(py);
        let attached = (|| -> PyResult<()> {
            value.setattr("kind", smiles_error_kind_name(&e.kind))?;
            value.setattr("span", (start, end))?;
            value.setattr("input", e.input.as_str())?;
            value.setattr("notation", notation_name(e.notation))?;
            Ok(())
        })();
        match attached {
            Ok(()) => err,
            Err(failed) => failed,
        }
    })
}

/// The bare variant name of a [`SmilesErrorKind`], payload dropped.
///
/// Derived from the derived [`Debug`] rendering rather than from a hand-written
/// table: a table would have to be extended every time the Rust enum grows a
/// variant, and a forgotten row is a wrong `kind` at the seam rather than a
/// compile error. `Debug` writes the variant name first in all three shapes —
/// `UnexpectedEnd`, `UnexpectedChar('X')`, `RingBondConflict { rnum: 1 }` — so
/// cutting at the first `(` or space leaves exactly the name.
///
/// [`SmilesErrorKind`]: molrs::io::smiles::SmilesErrorKind
fn smiles_error_kind_name(kind: &molrs::io::smiles::SmilesErrorKind) -> String {
    let rendered = format!("{kind:?}");
    rendered
        .split(['(', ' '])
        .next()
        .unwrap_or(rendered.as_str())
        .to_owned()
}

/// The lowercase name of a [`Notation`], matching how every other enum crosses
/// this seam (bond kinds, pair ends).
///
/// Matched totally: a new notation in Rust must be spelled here, not silently
/// rendered as some fallback.
///
/// [`Notation`]: molrs::io::smiles::Notation
fn notation_name(notation: molrs::io::smiles::Notation) -> &'static str {
    match notation {
        molrs::io::smiles::Notation::Smiles => "smiles",
        molrs::io::smiles::Notation::Smarts => "smarts",
        molrs::io::smiles::Notation::CgSmiles => "cgsmiles",
    }
}

/// Convert any `Display` error (typically a `molrs::compute` / `molrs::signal`
/// analysis error) to a Python `ValueError`. Shared by the analysis bindings.
pub fn py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}
