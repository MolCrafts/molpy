//! A frame in the wire encodings of `molrs::stream` — MessagePack and JSON —
//! as `molrs::io`'s `read_wire_frame` / `write_wire_frame`. The encoding is
//! the stream's ([`molrs::stream::Publisher`] puts it on the wire); reading
//! or writing a frame in it is `molpy.io`'s, as for every other format.

use crate::error::io_error_to_pyerr;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::core::frame::PyFrame;
use crate::stream::frame_encoding;

/// Rebuild a :class:`Frame` from a wire payload — what a
/// :class:`molpy.stream.Publisher` streams.
///
/// Parameters
/// ----------
/// data : bytes
///     A payload written by :func:`write_wire_frame` or a publisher.
/// encoding : {"msgpack", "json"}
///     The payload's encoding (a JSON payload is its UTF-8 text).
///
/// Returns
/// -------
/// Frame
///
/// Raises
/// ------
/// ValueError
///     If ``encoding`` is not one of the two, or the payload does not
///     decode.
#[pyfunction]
pub fn read_wire_frame(data: &[u8], encoding: &str) -> PyResult<PyFrame> {
    let frame =
        molrs::io::read_wire_frame(data, frame_encoding(encoding)?).map_err(io_error_to_pyerr)?;
    PyFrame::from_core_frame(frame)
}

/// Encode a :class:`Frame` as a wire payload — the inverse of
/// :func:`read_wire_frame`.
///
/// Parameters
/// ----------
/// frame : Frame
///     The frame to encode.
/// encoding : {"msgpack", "json"}
///     The encoding (JSON comes back as its UTF-8 bytes).
///
/// Returns
/// -------
/// bytes
///
/// Raises
/// ------
/// ValueError
///     If ``encoding`` is not one of the two, or the frame does not encode.
#[pyfunction]
pub fn write_wire_frame<'py>(
    py: Python<'py>,
    frame: &PyFrame,
    encoding: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let encoding = frame_encoding(encoding)?;
    let bytes = frame
        .with_frame(|f| molrs::io::write_wire_frame(f, encoding))?
        .map_err(io_error_to_pyerr)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Register the wire-encoding doors on the native module.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molpy.io", wrap_pyfunction!(read_wire_frame, m)?)?;
    crate::add_function(m, "molpy.io", wrap_pyfunction!(write_wire_frame, m)?)
}
