//! The multi-frame classes of `molrs.io`, one reader and (where the core has
//! one) one writer per format, each `molrs::io`'s class of the same name:
//! `<Fmt>Reader(source)` is `<Fmt>Reader::open(src)`, a lazy, indexed
//! [`TrajectoryReader`] (``len``, ``reader[i]`` / ``reader[a:b]``,
//! iteration), and `<Fmt>Writer(sink)` is `<Fmt>Writer::create(sink)`, a
//! [`FrameWriter`] that writes one frame per ``write`` call.
//!
//! Readers: `molrs.io.pdb.PdbReader`, `molrs.io.xyz.XyzReader`,
//! `molrs.io.gro.GroReader`, `molrs.io.mol2.Mol2Reader`,
//! `molrs.io.cif.CifReader`, `molrs.io.sdf.SdfReader`,
//! `molrs.io.cube.CubeReader`, `molrs.io.lammps.LammpsDumpReader`,
//! `molrs.io.dcd.DcdReader`, `molrs.io.trr.TrrReader`,
//! `molrs.io.xtc.XtcReader`. Writers: the same formats but SDF and cube.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyList, PySlice};

use super::source::{Mirror, PySink, PySource, WriteTarget};
use crate::core::frame::PyFrame;
use crate::error::io_error_to_pyerr;
use molrs::io::reader::{ReadSeek, TrajectoryReader};
use molrs::io::writer::{FrameWriter, Writer};

/// A Python index (negative from the end) as an in-range position.
fn resolve(index: isize, n: usize) -> PyResult<usize> {
    let len = n as isize;
    let i = if index < 0 { index + len } else { index };
    if i < 0 || i >= len {
        return Err(PyIndexError::new_err(format!(
            "frame index {index} out of range for {n} frames"
        )));
    }
    Ok(i as usize)
}

/// The positions a Python slice over `n` frames selects.
fn slice_positions(slice: &Bound<'_, PySlice>, n: usize) -> PyResult<Vec<usize>> {
    let ix = slice.indices(n as isize)?;
    let mut positions = Vec::new();
    let mut i = ix.start;
    while (ix.step > 0 && i < ix.stop) || (ix.step < 0 && i > ix.stop) {
        positions.push(i as usize);
        i += ix.step;
    }
    Ok(positions)
}

/// Frame `index` of `reader`, which must hold it.
fn frame_at<R: TrajectoryReader>(reader: &mut R, index: usize) -> PyResult<PyFrame> {
    let frame = reader
        .read_frame(index)
        .map_err(io_error_to_pyerr)?
        .ok_or_else(|| PyIndexError::new_err(format!("no frame {index}")))?;
    PyFrame::from_core_frame(frame)
}

macro_rules! frame_reader_class {
    ($py:ident, $name:literal, $module:literal, $core:ty, $format:literal) => {
        #[doc = concat!(
            "Lazy, indexed reader of every frame of a ", $format, " file —\n",
            "``molrs::io``'s ``", $name, "::open``.\n\n",
            "Frames are parsed on demand: random access (``reader[i]``) indexes\n",
            "the file once, then seeks; iteration walks it in order.\n\n",
            "Parameters\n",
            "----------\n",
            "source : os.PathLike | bytes | molrs.io.Text\n",
            "    The file (``.gz`` is decompressed), or its whole content.\n\n",
            "Examples\n",
            "--------\n",
            ">>> reader = ", $module, ".", $name, "(pathlib.Path(\"run\"))  # doctest: +SKIP\n",
            ">>> last = reader[-1]  # doctest: +SKIP\n"
        )]
        #[pyclass(module = $module, name = $name, unsendable)]
        pub struct $py {
            inner: Option<$core>,
            /// The iteration cursor: the next frame `__next__` reads.
            cursor: usize,
        }

        impl $py {
            fn reader(&mut self) -> PyResult<&mut $core> {
                self.inner.as_mut().ok_or_else(|| {
                    PyValueError::new_err(concat!("I/O operation on a closed ", $name))
                })
            }

            fn n(&mut self) -> PyResult<usize> {
                self.reader()?.len().map_err(io_error_to_pyerr)
            }
        }

        #[pymethods]
        impl $py {
            #[new]
            fn py_new(source: PySource<'_>) -> PyResult<Self> {
                let inner = source.read(<$core>::open).map_err(io_error_to_pyerr)?;
                Ok(Self {
                    inner: Some(inner),
                    cursor: 0,
                })
            }

            /// Index the whole file now rather than on the first random
            /// access.
            fn build_index(&mut self) -> PyResult<()> {
                self.reader()?.build_index().map_err(io_error_to_pyerr)
            }

            /// Release the file. Further reads raise ``ValueError``.
            fn close(&mut self) {
                self.inner = None;
            }

            /// Number of frames (indexes the file once).
            fn __len__(&mut self) -> PyResult<usize> {
                self.n()
            }

            /// Frame ``key`` (negative counts from the end), or the list of
            /// frames a slice selects.
            ///
            /// Raises
            /// ------
            /// IndexError
            ///     If an index is out of range.
            fn __getitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let py = key.py();
                let n = self.n()?;
                if let Ok(slice) = key.cast::<PySlice>() {
                    let frames = slice_positions(slice, n)?
                        .into_iter()
                        .map(|i| frame_at(self.reader()?, i))
                        .collect::<PyResult<Vec<_>>>()?;
                    Ok(PyList::new(py, frames)?.into_any().unbind())
                } else {
                    let i = resolve(key.extract()?, n)?;
                    Ok(Py::new(py, frame_at(self.reader()?, i)?)?.into_any())
                }
            }

            /// Walk the frames in order, from the first.
            fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
                slf.cursor = 0;
                slf
            }

            fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
                let index = self.cursor;
                match self.reader()?.read_frame(index).map_err(io_error_to_pyerr)? {
                    Some(frame) => {
                        self.cursor = index + 1;
                        Ok(Some(PyFrame::from_core_frame(frame)?))
                    }
                    None => Ok(None),
                }
            }

            fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
            fn __exit__(
                &mut self,
                _exc_type: Option<Py<PyAny>>,
                _exc_value: Option<Py<PyAny>>,
                _traceback: Option<Py<PyAny>>,
            ) -> bool {
                self.inner = None;
                false
            }

            fn __repr__(&mut self) -> String {
                match self.inner.as_mut() {
                    Some(inner) => match inner.len() {
                        Ok(n) => format!(concat!($name, "(n_frames={})"), n),
                        Err(_) => concat!($name, "(<unreadable>)").to_owned(),
                    },
                    None => concat!($name, "(<closed>)").to_owned(),
                }
            }
        }
    };
}

frame_reader_class!(
    PyPdbReader,
    "PdbReader",
    "molrs.io.pdb",
    molrs::io::pdb::PdbReader<Box<dyn ReadSeek>>,
    "PDB (one frame per ``MODEL``)"
);
frame_reader_class!(
    PyXyzReader,
    "XyzReader",
    "molrs.io.xyz",
    molrs::io::xyz::XyzReader<Box<dyn ReadSeek>>,
    "(extended) XYZ"
);
frame_reader_class!(
    PyGroReader,
    "GroReader",
    "molrs.io.gro",
    molrs::io::gro::GroReader<Box<dyn ReadSeek>>,
    "GROMACS GRO"
);
frame_reader_class!(
    PyMol2Reader,
    "Mol2Reader",
    "molrs.io.mol2",
    molrs::io::mol2::Mol2Reader<Box<dyn ReadSeek>>,
    "Tripos MOL2 (one frame per molecule)"
);
frame_reader_class!(
    PyCifReader,
    "CifReader",
    "molrs.io.cif",
    molrs::io::cif::CifReader<Box<dyn ReadSeek>>,
    "CIF (one frame per ``data_`` block)"
);
frame_reader_class!(
    PySdfReader,
    "SdfReader",
    "molrs.io.sdf",
    molrs::io::sdf::SdfReader<Box<dyn ReadSeek>>,
    "MDL SDF (one frame per record)"
);
frame_reader_class!(
    PyCubeReader,
    "CubeReader",
    "molrs.io.cube",
    molrs::io::CubeReader<Box<dyn ReadSeek>>,
    "Gaussian cube (cubes back to back)"
);
frame_reader_class!(
    PyLammpsDumpReader,
    "LammpsDumpReader",
    "molrs.io.lammps",
    molrs::io::lammps::LammpsDumpReader<Box<dyn ReadSeek>>,
    "LAMMPS dump"
);
frame_reader_class!(
    PyDcdReader,
    "DcdReader",
    "molrs.io.dcd",
    molrs::io::dcd::DcdReader<Box<dyn ReadSeek>>,
    "DCD"
);
frame_reader_class!(
    PyTrrReader,
    "TrrReader",
    "molrs.io.trr",
    molrs::io::trr::TrrReader<Box<dyn ReadSeek>>,
    "GROMACS TRR"
);
frame_reader_class!(
    PyXtcReader,
    "XtcReader",
    "molrs.io.xtc",
    molrs::io::xtc::XtcReader<Box<dyn ReadSeek>>,
    "GROMACS XTC"
);

/// The writer state every writer class holds: the core writer over the open
/// target, and, for a ``bytearray`` sink, its mirror.
struct Open<W> {
    writer: W,
    mirror: Option<Mirror>,
}

impl<W: FrameWriter> Open<W> {
    fn write(&mut self, py: Python<'_>, frame: &PyFrame) -> PyResult<()> {
        frame
            .with_frame(|f| self.writer.write(f))?
            .map_err(io_error_to_pyerr)?;
        self.sync(py)
    }

    fn sync(&self, py: Python<'_>) -> PyResult<()> {
        match &self.mirror {
            Some(mirror) => mirror.sync(py),
            None => Ok(()),
        }
    }
}

macro_rules! frame_writer_class {
    ($py:ident, $name:literal, $module:literal, $core:ident, $format:literal) => {
        #[doc = concat!(
            "Writer of a ", $format, " file, one frame per :meth:`write` —\n",
            "``molrs::io``'s ``", $name, "::create``.\n\n",
            "Parameters\n",
            "----------\n",
            "sink : os.PathLike | bytearray\n",
            "    The file (created or truncated), or a buffer: cleared, then\n",
            "    holding exactly the bytes written so far after each write.\n\n",
            "Examples\n",
            "--------\n",
            ">>> with ", $module, ".", $name, "(pathlib.Path(\"out\")) as w:  # doctest: +SKIP\n",
            "...     for frame in frames:\n",
            "...         w.write(frame)\n"
        )]
        #[pyclass(module = $module, name = $name, unsendable)]
        pub struct $py {
            inner: Option<Open<$core<WriteTarget>>>,
        }

        impl $py {
            fn open(&mut self) -> PyResult<&mut Open<$core<WriteTarget>>> {
                self.inner.as_mut().ok_or_else(|| {
                    PyValueError::new_err(concat!("I/O operation on a closed ", $name))
                })
            }
        }

        #[pymethods]
        impl $py {
            #[new]
            fn py_new(sink: PySink<'_>) -> PyResult<Self> {
                let (target, mirror) = sink.open()?;
                Ok(Self {
                    inner: Some(Open {
                        writer: $core::new(target),
                        mirror,
                    }),
                })
            }

            /// Append ``frame`` to the file.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If the writer is closed.
            /// OSError
            ///     If the frame cannot be written.
            fn write(&mut self, py: Python<'_>, frame: &PyFrame) -> PyResult<()> {
                self.open()?.write(py, frame)
            }

            /// Release the file (every write is already flushed).
            /// Idempotent; writes after it raise ``ValueError``.
            fn close(&mut self, py: Python<'_>) -> PyResult<()> {
                if let Some(open) = self.inner.take() {
                    open.sync(py)?;
                }
                Ok(())
            }

            fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
                slf
            }

            #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
            fn __exit__(
                &mut self,
                py: Python<'_>,
                _exc_type: Option<Py<PyAny>>,
                _exc_value: Option<Py<PyAny>>,
                _traceback: Option<Py<PyAny>>,
            ) -> PyResult<bool> {
                self.close(py)?;
                Ok(false)
            }

            fn __repr__(&self) -> String {
                match self.inner {
                    Some(_) => concat!($name, "(<open>)").to_owned(),
                    None => concat!($name, "(<closed>)").to_owned(),
                }
            }
        }
    };
}

use molrs::io::cif::CifWriter;
use molrs::io::dcd::DcdWriter;
use molrs::io::gro::GroWriter;
use molrs::io::lammps::LammpsDumpWriter;
use molrs::io::mol2::Mol2Writer;
use molrs::io::pdb::PdbWriter;
use molrs::io::trr::TrrWriter;
use molrs::io::xtc::XtcWriter;
use molrs::io::xyz::XyzWriter;

frame_writer_class!(PyPdbWriter, "PdbWriter", "molrs.io.pdb", PdbWriter, "PDB");
frame_writer_class!(
    PyXyzWriter,
    "XyzWriter",
    "molrs.io.xyz",
    XyzWriter,
    "(extended) XYZ"
);
frame_writer_class!(
    PyGroWriter,
    "GroWriter",
    "molrs.io.gro",
    GroWriter,
    "GROMACS GRO"
);
frame_writer_class!(
    PyMol2Writer,
    "Mol2Writer",
    "molrs.io.mol2",
    Mol2Writer,
    "Tripos MOL2"
);
frame_writer_class!(PyCifWriter, "CifWriter", "molrs.io.cif", CifWriter, "CIF");
frame_writer_class!(
    PyLammpsDumpWriter,
    "LammpsDumpWriter",
    "molrs.io.lammps",
    LammpsDumpWriter,
    "LAMMPS dump"
);
frame_writer_class!(PyDcdWriter, "DcdWriter", "molrs.io.dcd", DcdWriter, "DCD");
frame_writer_class!(
    PyTrrWriter,
    "TrrWriter",
    "molrs.io.trr",
    TrrWriter,
    "GROMACS TRR"
);
frame_writer_class!(
    PyXtcWriter,
    "XtcWriter",
    "molrs.io.xtc",
    XtcWriter,
    "GROMACS XTC"
);

#[pymethods]
impl PyLammpsDumpWriter {
    /// Set the ``dump custom`` column line of the snapshots written after
    /// this call, e.g. ``["id", "element", "x", "y", "z"]``, in the order
    /// given; a name a frame's ``atoms`` block cannot supply raises on
    /// :meth:`write`. Without it every column the block holds is written.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the writer is closed.
    fn set_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        let names: Vec<&str> = columns.iter().map(String::as_str).collect();
        self.open()?.writer.set_columns(&names);
        Ok(())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPdbReader>()?;
    m.add_class::<PyXyzReader>()?;
    m.add_class::<PyGroReader>()?;
    m.add_class::<PyMol2Reader>()?;
    m.add_class::<PyCifReader>()?;
    m.add_class::<PySdfReader>()?;
    m.add_class::<PyCubeReader>()?;
    m.add_class::<PyLammpsDumpReader>()?;
    m.add_class::<PyDcdReader>()?;
    m.add_class::<PyTrrReader>()?;
    m.add_class::<PyXtcReader>()?;
    m.add_class::<PyPdbWriter>()?;
    m.add_class::<PyXyzWriter>()?;
    m.add_class::<PyGroWriter>()?;
    m.add_class::<PyMol2Writer>()?;
    m.add_class::<PyCifWriter>()?;
    m.add_class::<PyLammpsDumpWriter>()?;
    m.add_class::<PyDcdWriter>()?;
    m.add_class::<PyTrrWriter>()?;
    m.add_class::<PyXtcWriter>()?;
    Ok(())
}
