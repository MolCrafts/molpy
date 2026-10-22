//! Where a `molpy.io` door reads from and writes to: [`PySource`] and
//! [`PySink`], the Python spellings of `molrs::io::{Source, Sink}`, and
//! [`PyText`] (`molpy.io.Text`), the text of a whole file.
//!
//! A source is an ``os.PathLike`` (a file), ``bytes`` (a whole file in
//! memory) or a ``molpy.io.Text`` (a whole file's text). A plain ``str`` is
//! refused: it is ambiguous between a path and the text of a file, so the
//! caller says which. A sink is an ``os.PathLike`` (created or truncated) or
//! a ``bytearray`` (cleared, then holding exactly the file's bytes).

use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::rc::Rc;

use molrs::io::{Sink, Source};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PySlice, PyString};

/// The text of a whole file, for a ``molpy.io`` reader to read.
///
/// A reader takes a path (``pathlib.Path``), ``bytes``, or this: a plain
/// ``str`` is refused, because it could name a file or hold one.
///
/// Parameters
/// ----------
/// text : str
///     The file's whole content.
///
/// Examples
/// --------
/// >>> frame = molpy.io.read_xyz(molpy.io.Text("1\n\nO 0 0 0\n"))
/// >>> frame["atoms"].nrows
/// 1
#[pyclass(module = "molpy.io", name = "Text", frozen)]
pub struct PyText {
    text: String,
}

#[pymethods]
impl PyText {
    #[new]
    fn new(text: String) -> Self {
        Self { text }
    }

    /// The text itself.
    fn __str__(&self) -> String {
        self.text.clone()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Text({})", PyString::new(py, &self.text).repr()?))
    }
}

/// What a reader reads: a file on disk, or a whole file in memory
/// (``bytes``, or a [`PyText`]'s text) — `molrs::io::Source`, borrowed from
/// the Python objects without a copy.
pub(crate) enum PySource<'py> {
    /// A file (any ``os.PathLike``).
    Path(PathBuf),
    /// The bytes of a whole file.
    Bytes(Bound<'py, PyBytes>),
    /// The text of a whole file.
    Text(Bound<'py, PyText>),
}

impl PySource<'_> {
    /// The core source this names.
    pub(crate) fn source(&self) -> Source<'_> {
        match self {
            PySource::Path(path) => Source::Path(path),
            PySource::Bytes(bytes) => Source::Bytes(bytes.as_bytes()),
            PySource::Text(text) => Source::Bytes(text.get().text.as_bytes()),
        }
    }

    /// Run the core door `read` on this source; a failure on a file names
    /// the file (an OS error does not), keeping the error's kind.
    pub(crate) fn read<'s, T>(
        &'s self,
        read: impl FnOnce(Source<'s>) -> std::io::Result<T>,
    ) -> std::io::Result<T> {
        read(self.source()).map_err(|e| match self {
            PySource::Path(path) => named(path, e),
            _ => e,
        })
    }
}

/// `e` with `path` in front of its message, unless the message names it.
fn named(path: &std::path::Path, e: std::io::Error) -> std::io::Error {
    let shown = path.display().to_string();
    let message = e.to_string();
    if message.contains(&shown) {
        e
    } else {
        std::io::Error::new(e.kind(), format!("{shown}: {message}"))
    }
}

/// The name of `obj`'s type, for a refusal message.
fn type_name(obj: &Borrowed<'_, '_, PyAny>) -> String {
    obj.get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "?".to_owned())
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySource<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "a str is ambiguous between a path and the text of a file: pass \
                 pathlib.Path(s) for a file or molpy.io.Text(s) for text",
            ));
        }
        if let Ok(text) = obj.cast::<PyText>() {
            return Ok(PySource::Text(text.to_owned()));
        }
        if let Ok(bytes) = obj.cast::<PyBytes>() {
            return Ok(PySource::Bytes(bytes.to_owned()));
        }
        if obj.is_instance_of::<PyByteArray>() {
            return Err(PyTypeError::new_err(
                "a reader takes bytes, not a bytearray: pass bytes(buffer)",
            ));
        }
        if obj.hasattr("__fspath__")? {
            return Ok(PySource::Path(obj.extract::<PathBuf>()?));
        }
        Err(PyTypeError::new_err(format!(
            "a source is an os.PathLike, bytes or molpy.io.Text, not {}",
            type_name(&obj)
        )))
    }
}

/// Where a writer writes: a file (created or truncated), or a ``bytearray``
/// (cleared, then holding exactly the file's bytes) — `molrs::io::Sink`.
pub(crate) enum PySink<'py> {
    /// A file (any ``os.PathLike``).
    Path(PathBuf),
    /// A buffer in memory.
    Buffer(Bound<'py, PyByteArray>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for PySink<'py> {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<PyString>() {
            return Err(PyTypeError::new_err(
                "a writer takes a path as an os.PathLike, not a str: pass pathlib.Path(s) \
                 (or a bytearray to write into memory)",
            ));
        }
        if let Ok(buffer) = obj.cast::<PyByteArray>() {
            return Ok(PySink::Buffer(buffer.to_owned()));
        }
        if obj.is_instance_of::<PyBytes>() {
            return Err(PyTypeError::new_err(
                "bytes cannot be written into: pass a bytearray",
            ));
        }
        if obj.hasattr("__fspath__")? {
            return Ok(PySink::Path(obj.extract::<PathBuf>()?));
        }
        Err(PyTypeError::new_err(format!(
            "a sink is an os.PathLike or a bytearray, not {}",
            type_name(&obj)
        )))
    }
}

/// Replace `buffer`'s whole content with `bytes` (``buffer[:] = bytes``).
fn replace_all(buffer: &Bound<'_, PyByteArray>, bytes: &[u8]) -> PyResult<()> {
    let py = buffer.py();
    buffer.set_item(PySlice::full(py), PyBytes::new(py, bytes))
}

impl PySink<'_> {
    /// Run `write` on the core sink this names, mapping its error with `err`.
    /// A bytearray then holds what `write` left in its buffer — as
    /// `Sink::Buffer`, cleared and then written, even when `write` fails.
    pub(crate) fn put<E>(
        &self,
        write: impl FnOnce(Sink<'_>) -> Result<(), E>,
        err: impl FnOnce(E) -> PyErr,
    ) -> PyResult<()> {
        match self {
            PySink::Path(path) => write(Sink::Path(path)).map_err(err),
            PySink::Buffer(buffer) => {
                let mut bytes = Vec::new();
                let written = write(Sink::Buffer(&mut bytes));
                replace_all(buffer, &bytes)?;
                written.map_err(err)
            }
        }
    }

    /// An open target for a multi-frame writer class — the file, created or
    /// truncated, or a buffer — and, for a bytearray (cleared here), the
    /// [`Mirror`] that copies the buffer into it after each write.
    pub(crate) fn open(&self) -> PyResult<(WriteTarget, Option<Mirror>)> {
        match self {
            PySink::Path(path) => {
                let file = File::create(path)
                    .map_err(|e| crate::error::io_error_to_pyerr(named(path, e)))?;
                Ok((WriteTarget::File(BufWriter::new(file)), None))
            }
            PySink::Buffer(buffer) => {
                replace_all(buffer, &[])?;
                let state = Rc::new(RefCell::new(BufferState::default()));
                let target = WriteTarget::Buffer(SharedBuffer {
                    state: Rc::clone(&state),
                    position: 0,
                });
                let mirror = Mirror {
                    state,
                    buffer: buffer.clone().unbind(),
                };
                Ok((target, Some(mirror)))
            }
        }
    }
}

/// The bytes a [`SharedBuffer`] holds, with the byte ranges written since
/// they were last mirrored into the Python ``bytearray``.
#[derive(Default)]
struct BufferState {
    bytes: Vec<u8>,
    dirty: Vec<(usize, usize)>,
}

/// An in-memory, seekable write target whose bytes are mirrored into a
/// Python ``bytearray`` range by range, so writing a long trajectory costs
/// the bytes written, not the bytes held.
pub(crate) struct SharedBuffer {
    state: Rc<RefCell<BufferState>>,
    position: u64,
}

/// The open target of a multi-frame writer class: a buffered file, or a
/// buffer mirrored into a ``bytearray``.
pub(crate) enum WriteTarget {
    /// A file on disk.
    File(BufWriter<File>),
    /// A buffer in memory.
    Buffer(SharedBuffer),
}

/// The Python side of a [`SharedBuffer`]: what [`Mirror::sync`] copies the
/// written bytes into.
pub(crate) struct Mirror {
    state: Rc<RefCell<BufferState>>,
    buffer: Py<PyByteArray>,
}

impl Mirror {
    /// Copy every range written since the last call into the ``bytearray``,
    /// so it holds exactly the bytes written so far.
    pub(crate) fn sync(&self, py: Python<'_>) -> PyResult<()> {
        let mut state = self.state.borrow_mut();
        let buffer = self.buffer.bind(py);
        let dirty = std::mem::take(&mut state.dirty);
        for (start, end) in dirty {
            let start = start.min(buffer.len());
            let slice = PySlice::new(py, start as isize, end as isize, 1);
            buffer.set_item(slice, PyBytes::new(py, &state.bytes[start..end]))?;
        }
        Ok(())
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let mut state = self.state.borrow_mut();
        let mut cursor = Cursor::new(&mut state.bytes);
        cursor.set_position(self.position);
        let n = cursor.write(buf)?;
        let start = self.position as usize;
        self.position = cursor.position();
        state.dirty.push((start, start + n));
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for SharedBuffer {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let mut state = self.state.borrow_mut();
        let mut cursor = Cursor::new(&mut state.bytes);
        cursor.set_position(self.position);
        self.position = cursor.seek(pos)?;
        Ok(self.position)
    }
}

impl Write for WriteTarget {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            WriteTarget::File(w) => w.write(buf),
            WriteTarget::Buffer(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            WriteTarget::File(w) => w.flush(),
            WriteTarget::Buffer(w) => w.flush(),
        }
    }
}

impl Seek for WriteTarget {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            WriteTarget::File(w) => w.seek(pos),
            WriteTarget::Buffer(w) => w.seek(pos),
        }
    }
}
