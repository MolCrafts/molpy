//! `molpy.io.read_csv_block` / `write_csv_block` — `molrs::io`'s CSV doors
//! of the same names: a `Block` to and from CSV.

use crate::error::io_error_to_pyerr;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{PySink, PySource};
use crate::core::block::PyBlock;

// These read and write a store container rather than a molecular file
// format, but they are IO and they live here for the same reason `read_pdb`
// does: turning text into a container is a reader's job. `Block` carries no
// `from_csv` constructor — a container that parses its own formats grows one
// entry point per format and duplicates this module.

/// `delimiter` as the one character the Rust doors take.
fn delimiter_char(delimiter: &str) -> PyResult<char> {
    let mut chars = delimiter.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyValueError::new_err(format!(
            "a CSV delimiter is one character, got {delimiter:?}"
        ))),
    }
}

/// Read CSV into a :class:`~molpy.core.Block`.
///
/// Each column's dtype is inferred int → float → str. When ``header`` is
/// given the text is treated as headerless and those names are used;
/// otherwise the first non-empty line names the columns. Blank lines are
/// skipped; fields are split on ``delimiter`` and trimmed, with no quoting.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes | molpy.io.Text
///     The file, or its whole content (UTF-8).
/// delimiter : str
///     The one-character field separator.
/// header : list[str], optional
///     Column names of a headerless file.
///
/// Returns
/// -------
/// molpy.core.Block
///
/// Raises
/// ------
/// ValueError
///     The text is empty, a row is ragged, or ``delimiter`` is not one
///     character.
/// OSError
///     The file cannot be read.
#[pyfunction]
#[pyo3(signature = (source, delimiter = ",", header = None))]
pub fn read_csv_block(
    source: PySource<'_>,
    delimiter: &str,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let delimiter = delimiter_char(delimiter)?;
    let block = source
        .read(|s| molrs::io::read_csv_block(s, delimiter, header.as_deref()))
        .map_err(io_error_to_pyerr)?;
    PyBlock::from_core_block(block)
}

/// Write a :class:`~molpy.core.Block` as CSV — the inverse of
/// :func:`read_csv_block`.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The file, or a buffer that then holds exactly its bytes.
/// block : molpy.core.Block
///     The block to write.
/// delimiter : str
///     The one-character field separator.
/// header : bool
///     Whether the first line names the columns.
///
/// Raises
/// ------
/// ValueError
///     ``delimiter`` is not one character, or a column has a dtype CSV
///     cannot encode.
/// OSError
///     The file cannot be written.
#[pyfunction]
#[pyo3(signature = (sink, block, delimiter = ",", header = true))]
pub fn write_csv_block(
    sink: PySink<'_>,
    block: &PyBlock,
    delimiter: &str,
    header: bool,
) -> PyResult<()> {
    let delimiter = delimiter_char(delimiter)?;
    block.with_block(|b| {
        sink.put(
            |s| molrs::io::write_csv_block(s, b, delimiter, header),
            io_error_to_pyerr,
        )
    })?
}

/// Register this module's doors.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molpy.io", wrap_pyfunction!(read_csv_block, m)?)?;
    crate::add_function(m, "molpy.io", wrap_pyfunction!(write_csv_block, m)?)
}
