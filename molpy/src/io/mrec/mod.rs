//! Scientific records (`*.mrec`): [`write_mrec`], flat on `molpy.io` like
//! every other format's writer ([`register_doors`]), and the `_core.mrec`
//! submodule, `molpy.io.mrec` ([`register`]): [`PyMrecFile`] (a record's
//! per-section reads), the store reader and writer ([`PyMrecReader`],
//! [`PySequenceSchema`], [`PyMrecWriter`]), [`pack_mrec_zip`], the
//! `ForceFieldSection` and the `validation` submodule
//! (`molpy.io.mrec.validation`) — each named as `molrs::io::mrec`'s.

pub mod section;

use std::path::PathBuf;

use crate::core::frame::{
    PyFrame, PyMetaValue, infer_meta_value, json_map_to_plain_dict, meta_document_arg,
    meta_value_from_dtype,
};
use crate::core::simbox::PyBox;
use crate::core::trajectory::PyTrajectory;
use crate::error::molrs_error_to_pyerr;
use crate::io::{PySink, PySource, data_error};
use molrs::io::mrec::{
    Compression, MolRec, MrecFile, MrecReader, MrecWriter, SequenceSchema, dtype_from_schema_tag,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use section::PyForceFieldSection;

/// Write a record — its ``meta`` document and whichever state sections are
/// given — as a ``*.mrec`` store (``molrs::io::write_mrec``).
///
/// A path is written as a directory store; a ``bytearray`` receives the
/// packed record (the bytes of a ``*.mrec.zip``), which
/// :class:`molpy.io.mrec.MrecFile` reads back.
///
/// Parameters
/// ----------
/// sink : os.PathLike | bytearray
///     The store's path, or a buffer that then holds the packed record.
/// meta : Mapping, optional
///     The record's identity document, written to ``meta/`` as given: a
///     ``dict``, a :class:`~molpy.core.MetaDocument`, or any mapping
///     (``frame.meta`` included). Nested tuples and documents are JSON
///     arrays and objects.
/// frame : Frame, optional
///     The snapshot (``frame/``).
/// system : Frame, optional
///     The system definition — topology and types (``system/``).
/// trajectory : Trajectory, optional
///     The ordered frame sequence (``trajectory/``).
/// forcefield : ForceFieldSection | ForceField, optional
///     The force field the record's types link into (``forcefield/``): a
///     :class:`~molpy.io.mrec.ForceFieldSection` as given, or a
///     :class:`~molpy.ff.forcefield.ForceField` through
///     :meth:`~molpy.io.mrec.ForceFieldSection.from_forcefield`.
///
/// Raises
/// ------
/// ValueError
///     If the record has no state section, a section fails to encode, or
///     the force field is invalid or has no section form.
/// TypeError
///     If ``meta`` is not a mapping or holds a value JSON cannot, or
///     ``forcefield`` is neither a ``ForceField`` nor a
///     ``ForceFieldSection``.
/// OSError
///     If the store cannot be written.
#[pyfunction]
#[pyo3(signature = (sink, *, meta=None, frame=None, system=None, trajectory=None, forcefield=None))]
pub fn write_mrec(
    sink: PySink<'_>,
    meta: Option<&Bound<'_, PyAny>>,
    frame: Option<&Bound<'_, PyFrame>>,
    system: Option<&Bound<'_, PyFrame>>,
    trajectory: Option<PyRef<'_, PyTrajectory>>,
    forcefield: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut record = MolRec::new();
    if let Some(meta) = meta {
        record.meta = meta_document_arg(meta)?;
    }
    record.frame = frame.map(|f| f.borrow().clone_core_frame()).transpose()?;
    record.system = system.map(|s| s.borrow().clone_core_frame()).transpose()?;
    record.trajectory = trajectory.map(|t| t.inner.clone());
    record.forcefield = forcefield.map(PyForceFieldSection::from_arg).transpose()?;
    sink.put(|s| molrs::io::write_mrec(s, &record), data_error)
}

/// A ``*.mrec`` record opened for its sections (``molrs::io::mrec::MrecFile``):
/// each read decodes only the section it names, so another section — a
/// trajectory, observables, one this build does not know — cannot fail it.
/// A trajectory section is :class:`MrecReader`'s.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes
///     A ``*.mrec`` directory store, a packed ``*.mrec.zip`` (read in
///     place), or the bytes of a packed one.
///
/// Raises
/// ------
/// ValueError
///     If the source is neither a directory store nor a packed record.
///
/// Examples
/// --------
/// >>> record = molpy.io.mrec.MrecFile(pathlib.Path("water.mrec"))  # doctest: +SKIP
/// >>> frame = record.frame()  # doctest: +SKIP
#[pyclass(module = "molpy.io.mrec", name = "MrecFile", frozen, unsendable)]
pub struct PyMrecFile {
    inner: MrecFile,
}

#[pymethods]
impl PyMrecFile {
    #[new]
    fn new(source: PySource<'_>) -> PyResult<Self> {
        let inner = source.read(MrecFile::open).map_err(data_error)?;
        Ok(Self { inner })
    }

    /// The record's top-level section names (``meta``, ``frame``,
    /// ``system``, …), sorted, without decoding any of them.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn sections(&self) -> PyResult<Vec<String>> {
        self.inner.sections().map_err(data_error)
    }

    /// The record's ``meta`` document, as the producer wrote it; empty when
    /// the store has none.
    ///
    /// Returns
    /// -------
    /// dict
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let map = self.inner.meta().map_err(data_error)?;
        json_map_to_plain_dict(py, &map)
    }

    /// The ``frame`` section — the snapshot — or ``None`` when the record
    /// has none.
    ///
    /// Returns
    /// -------
    /// Frame | None
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the section fails to decode.
    fn frame(&self) -> PyResult<Option<PyFrame>> {
        self.inner
            .frame()
            .map_err(data_error)?
            .map(PyFrame::from_core_frame)
            .transpose()
    }

    /// The ``system`` section — the topology — or ``None`` when the record
    /// has none.
    ///
    /// Returns
    /// -------
    /// Frame | None
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the section fails to decode.
    fn system(&self) -> PyResult<Option<PyFrame>> {
        self.inner
            .system()
            .map_err(data_error)?
            .map(PyFrame::from_core_frame)
            .transpose()
    }

    /// The ``forcefield`` section, validated, or ``None`` when the record
    /// has none. It comes back whole — every document key, every table,
    /// units as stored; :meth:`ForceFieldSection.to_forcefield` turns it
    /// into a force field molrs can compile.
    ///
    /// Returns
    /// -------
    /// ForceFieldSection | None
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the section is malformed.
    fn forcefield(&self) -> PyResult<Option<PyForceFieldSection>> {
        Ok(self
            .inner
            .forcefield()
            .map_err(data_error)?
            .map(|inner| PyForceFieldSection { inner }))
    }
}

/// Lazy one-frame cursor over the ``trajectory`` section of a ``*.mrec``
/// record (``molrs::io::mrec::MrecReader``).
///
/// Construction opens the index only, and each read decodes exactly the
/// asked-for frame, keeping the last decoded chunk of every column so
/// playback through consecutive frames is a slice rather than a decode.
/// Supports ``len(reader)``, ``reader[i]`` (negative indices included),
/// iteration, and the ``with`` statement.
///
/// Parameters
/// ----------
/// source : os.PathLike | bytes
///     A ``*.mrec`` directory store, a packed ``*.mrec.zip``, or the bytes of
///     a packed one.
///
/// Raises
/// ------
/// ValueError
///     If the record has no ``trajectory`` section (read the others with
///     :class:`MrecFile`), or its index does not decode.
#[pyclass(module = "molpy.io.mrec", name = "MrecReader", unsendable)]
pub struct PyMrecReader {
    inner: MrecReader,
    /// The iteration cursor: the next frame `__next__` decodes.
    cursor: u64,
}

impl PyMrecReader {
    /// Decode committed frame `index` (already resolved).
    fn frame_at(&self, index: u64) -> PyResult<PyFrame> {
        let frame = self
            .inner
            .frame(index)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }
}

#[pymethods]
impl PyMrecReader {
    #[new]
    fn py_new(source: PySource<'_>) -> PyResult<Self> {
        let inner = source.read(MrecReader::open).map_err(data_error)?;
        Ok(Self { inner, cursor: 0 })
    }

    /// Decode one frame carrying only the named ``(block, column)`` pairs.
    ///
    /// A viewer that needs coordinates decodes ``("atoms", "x")``,
    /// ``("atoms", "y")``, ``("atoms", "z")`` and nothing else. Blocks none of
    /// whose columns are named are left out; the cell and per-step metadata
    /// always come along.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Zero-based frame index; negative counts from the end.
    /// columns : list[tuple[str, str]]
    ///     The ``(block, column)`` pairs to decode.
    ///
    /// Returns
    /// -------
    /// Frame
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``index`` is past the commit marker.
    fn frame_columns(&self, index: isize, columns: Vec<(String, String)>) -> PyResult<PyFrame> {
        let resolved = self.resolve_index(index)?;
        let pairs: Vec<(&str, &str)> = columns
            .iter()
            .map(|(block, column)| (block.as_str(), column.as_str()))
            .collect();
        let frame = self
            .inner
            .frame_columns(resolved, &pairs)
            .map_err(molrs_error_to_pyerr)?
            .ok_or_else(|| PyIndexError::new_err("trajectory index out of range"))?;
        PyFrame::from_core_frame(frame)
    }

    /// The update of block ``name`` that frame ``index`` resolves to, or
    /// ``None`` when the block is absent there.
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows, so a consumer can skip re-uploading a block whose update did not
    /// change without comparing a single value.
    fn block_update_at(&self, name: &str, index: isize) -> PyResult<Option<u64>> {
        let resolved = self.resolve_index(index)?;
        self.inner
            .block_update_at(name, resolved)
            .map_err(molrs_error_to_pyerr)
    }

    /// The cell at frame ``index``, or ``None`` when no cell has been written
    /// at or before it.
    fn box_at(&self, index: isize) -> PyResult<Option<PyBox>> {
        let resolved = self.resolve_index(index)?;
        Ok(self
            .inner
            .box_at(resolved)
            .map_err(molrs_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Number of committed frames.
    fn __len__(&self) -> usize {
        self.inner.steps().len()
    }

    /// Decode the committed frame at ``index`` (negative counts from the
    /// end).
    ///
    /// Raises
    /// ------
    /// IndexError
    ///     If ``index`` is past the commit marker.
    /// ValueError
    ///     If a section fails to decode.
    fn __getitem__(&self, index: isize) -> PyResult<PyFrame> {
        let resolved = self.resolve_index(index)?;
        self.frame_at(resolved)
    }

    /// Walk the committed frames in order, from the first.
    fn __iter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.cursor = 0;
        slf
    }

    fn __next__(&mut self) -> PyResult<Option<PyFrame>> {
        if self.cursor >= self.inner.steps().len() as u64 {
            return Ok(None);
        }
        let frame = self.frame_at(self.cursor)?;
        self.cursor += 1;
        Ok(Some(frame))
    }

    /// Every committed frame, with the steps and the times when the run
    /// wrote any, as an in-memory :class:`~molpy.core.Trajectory`. A run too
    /// large for memory is what ``reader[i]`` is for.
    ///
    /// Returns
    /// -------
    /// Trajectory
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On the first frame that fails to decode.
    fn to_trajectory(&self) -> PyResult<PyTrajectory> {
        let inner = self.inner.to_trajectory().map_err(molrs_error_to_pyerr)?;
        Ok(PyTrajectory { inner })
    }

    /// Step numbers of the committed frames.
    #[getter]
    fn step(&self) -> Vec<i64> {
        self.inner.steps().to_vec()
    }

    /// Time of each committed frame, when the run wrote any. The record
    /// carries no unit for it; the producer's convention applies.
    #[getter]
    fn time(&self) -> Option<Vec<f64>> {
        self.inner.times().map(<[f64]>::to_vec)
    }

    /// Whether the store carries a block section of this name (e.g. ``"bonds"``)
    /// with at least one committed update.
    fn has_block(&self, name: &str) -> bool {
        self.inner.has_block(name)
    }

    /// Names of every block section present in the store.
    fn block_names(&self) -> Vec<String> {
        self.inner.block_names().map(str::to_string).collect()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> bool {
        false
    }
}

impl PyMrecReader {
    /// Resolve a possibly-negative Python index against the committed length.
    fn resolve_index(&self, index: isize) -> PyResult<u64> {
        let len = self.inner.steps().len() as isize;
        let normalized = if index < 0 { index + len } else { index };
        if normalized < 0 || normalized >= len {
            return Err(PyIndexError::new_err("trajectory index out of range"));
        }
        Ok(normalized as u64)
    }
}

/// A frame-sequence schema pinned before a run's frames are written.
///
/// The schema fixes every block, column and dtype the run may carry;
/// :class:`MrecWriter` refuses a frame that steps outside it.
/// Derive it from representative frames (:meth:`from_frames`), then refine
/// it column by column with ``declare_*``.
#[pyclass(module = "molpy.io.mrec", name = "SequenceSchema")]
pub struct PySequenceSchema {
    inner: SequenceSchema,
}

#[pymethods]
impl PySequenceSchema {
    /// Derive a schema from the union of several frames' blocks and columns
    /// (``molrs::io::mrec::SequenceSchema::from_frames``). ``from_frames([])``
    /// is the empty schema, to be filled with ``declare_*``.
    ///
    /// Parameters
    /// ----------
    /// frames : Sequence[Frame]
    ///     Representative frames of the run.
    ///
    /// Returns
    /// -------
    /// SequenceSchema
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A reserved block or column name, or two frames that declare one
    ///     block's shape, or one column's dtype or precision, differently.
    #[staticmethod]
    fn from_frames(frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<Self> {
        let cores = frames
            .iter()
            .map(|frame| frame.clone_core_frame())
            .collect::<PyResult<Vec<_>>>()?;
        let inner = SequenceSchema::from_frames(&cores).map_err(molrs_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Declare a block, with the row count a typical frame of it carries.
    ///
    /// ``rows`` sizes the block's frame-aligned inner chunk; ``None`` leaves
    /// the byte floor to decide.
    #[pyo3(signature = (name, rows=None))]
    fn declare_block<'py>(
        mut slf: PyRefMut<'py, Self>,
        name: &str,
        rows: Option<u64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_block(name, rows)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare a column of ``block`` by dtype tag (``"f64"``, ``"u64"``,
    /// ``"string"``, …) and trailing shape (``[3]`` for an xyz column).
    #[pyo3(signature = (block, column, dtype, trailing=None))]
    fn declare_column<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        dtype: &str,
        trailing: Option<Vec<u64>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let dtype = dtype_from_schema_tag(dtype).map_err(molrs_error_to_pyerr)?;
        slf.inner
            .declare_column(block, column, dtype, &trailing.unwrap_or_default())
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare the precision of ``column`` of ``block``: an absolute tolerance
    /// in the column's units (``1e-3`` keeps Å coordinates to a thousandth).
    ///
    /// Every frame's values are rounded to the largest power of two not above
    /// it (ties to even) before the change check and before they land, so a
    /// change below half that step is no change, and the column is stored
    /// shuffled and compressed. Pinned with the schema. A schema derived with
    /// :meth:`from_frames` takes each column's :meth:`Block.precision`.
    fn declare_precision<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        precision: f64,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_precision(block, column, precision)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare that ``column`` of ``block`` (``u64``) holds row indices into
    /// *target*: ``"<block>"`` of the same resolved frame, or
    /// ``"/<section>/<block>"``. Pinned with the schema; the writer refuses a
    /// frame whose resolved blocks break it. :meth:`from_frames` takes each
    /// column's :meth:`Block.target`.
    fn declare_target<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        column: &str,
        target: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_target(block, column, target)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// The declared target of ``column`` of ``block``, or ``None``.
    fn target(&self, block: &str, column: &str) -> Option<String> {
        self.inner.target(block, column).map(str::to_string)
    }

    /// Declare ``block`` aligned with ``target``: its rows are ``target``'s
    /// rows, one for one, at every frame after carry-forward. A frame whose
    /// ``target`` changes row count must restate ``block``; one that keeps
    /// it may let ``block`` carry forward. The two blocks keep disjoint
    /// columns, alignments do not chain, and the aligned block declares no
    /// structural shape. A reader hands back the two blocks.
    fn declare_aligned<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        target: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_aligned(block, target)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// The block ``block`` is aligned with, or ``None``.
    fn aligned_with(&self, block: &str) -> Option<String> {
        self.inner.aligned_with(block).map(str::to_string)
    }

    /// The declared precision of ``column`` of ``block``, or ``None``.
    fn precision(&self, block: &str, column: &str) -> Option<f64> {
        self.inner.precision(block, column)
    }

    /// Declare the structural shape of ``block`` (a volumetric grid); every
    /// update then carries exactly ``prod(shape)`` rows.
    fn declare_structural_shape<'py>(
        mut slf: PyRefMut<'py, Self>,
        block: &str,
        shape: Vec<usize>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .declare_structural_shape(block, &shape)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// Declare a per-step metadata key by dtype tag (``"f64"``, ``"i64"``,
    /// ``"f64x3"``, ``"string"``, ``"json"``, …). A frame that omits the key
    /// is an error unless ``fill`` is given: the value written for the steps
    /// that omit it, read at ``dtype`` (a :class:`~molpy.core.MetaValue` is
    /// taken as it is).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An unknown dtype, a fill that does not read at it, or a key
    ///     declared before.
    #[pyo3(signature = (key, dtype, fill=None))]
    fn declare_meta<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        dtype: &str,
        fill: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let fill = fill
            .map(|fill| {
                if fill.extract::<PyRef<'_, PyMetaValue>>().is_ok() {
                    infer_meta_value(fill)
                } else {
                    meta_value_from_dtype(dtype, fill)
                        .map_err(|e| PyValueError::new_err(format!("meta key {key:?} fill: {e}")))
                }
            })
            .transpose()?;
        slf.inner
            .declare_meta(key, dtype, fill)
            .map_err(molrs_error_to_pyerr)?;
        Ok(slf)
    }

    /// The declared block names.
    fn block_names(&self) -> Vec<String> {
        self.inner.block_names().map(str::to_string).collect()
    }

    /// The declared column names of ``block``, or ``None`` when undeclared.
    fn column_names(&self, block: &str) -> Option<Vec<String>> {
        self.inner
            .column_names(block)
            .map(|names| names.map(str::to_string).collect())
    }

    /// The declared per-step metadata keys with their dtype tags.
    fn meta_keys(&self) -> Vec<(String, String)> {
        self.inner
            .meta_keys()
            .map(|(key, dtype)| (key.to_string(), dtype.to_string()))
            .collect()
    }
}

/// Parse a compression spec: ``None`` / ``"none"``, ``"gzip"`` (level 1),
/// ``"gzip:5"``, ``"zstd"`` (level 3), ``"zstd:7"``.
fn parse_compression(spec: Option<&str>) -> PyResult<Compression> {
    let Some(spec) = spec else {
        return Ok(Compression::None);
    };
    let (name, level) = match spec.split_once(':') {
        Some((name, level)) => (name, Some(level)),
        None => (spec, None),
    };
    let level = |default: i64| -> PyResult<i64> {
        match level {
            Some(text) => text.parse::<i64>().map_err(|_| {
                PyValueError::new_err(format!("compression level {text:?} is not an integer"))
            }),
            None => Ok(default),
        }
    };
    match name {
        "none" => Ok(Compression::None),
        "gzip" => Ok(Compression::Gzip(level(1)? as u32)),
        "zstd" => Ok(Compression::Zstd(level(3)? as i32)),
        other => Err(PyValueError::new_err(format!(
            "unknown compression {other:?}; use None, \"gzip[:level]\" or \"zstd[:level]\""
        ))),
    }
}

/// Append-first writer for a ``*.mrec`` trajectory store.
///
/// Wraps the Rust ``MrecWriter`` over the positional-write store.
/// Frames are buffered and landed whole inner chunks at a time on a cadence
/// derived from the frame size (or ``flush_every``); ``flush()`` and
/// ``close()`` commit whatever is buffered, durably unless ``durable=False``.
///
/// Parameters
/// ----------
/// path
///     Destination filesystem path for the store.
/// schema
///     The :class:`SequenceSchema` every appended frame is checked
///     against.
/// flush_every
///     Land every this many frames instead of the derived cadence.
/// compression
///     How floating-point columns are compressed: ``None``,
///     ``"gzip[:level]"`` or ``"zstd[:level]"``. Everything else always
///     carries gzip level 1. A column with a declared precision is
///     byte-shuffled and compressed whatever this says: ``None`` means
///     zstd level 3, and a named compressor replaces it.
/// durable
///     Whether ``flush()`` / ``close()`` fsync the touched files.
/// meta
///     The record's identity document, written to ``meta/`` as
///     given: a ``dict``, a
///     :class:`~molpy.core.MetaDocument`, or any mapping (``frame.meta``
///     included). Nested tuples and documents are JSON arrays and objects.
#[pyclass(module = "molpy.io.mrec", name = "MrecWriter", unsendable)]
pub struct PyMrecWriter {
    inner: Option<MrecWriter>,
}

#[pymethods]
impl PyMrecWriter {
    #[new]
    #[pyo3(signature = (path, schema, *, flush_every=None, compression=None, durable=true, meta=None))]
    fn py_new(
        path: PathBuf,
        schema: &Bound<'_, PySequenceSchema>,
        flush_every: Option<u64>,
        compression: Option<&str>,
        durable: bool,
        meta: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema = schema.borrow().inner.clone();
        let mut writer = MrecWriter::create(&path, schema)
            .map_err(molrs_error_to_pyerr)?
            .with_compression(parse_compression(compression)?)
            .map_err(molrs_error_to_pyerr)?
            .with_durable(durable);
        if let Some(frames) = flush_every {
            writer = writer
                .with_flush_every(frames)
                .map_err(molrs_error_to_pyerr)?;
        }
        if let Some(meta) = meta {
            writer = writer
                .with_meta(&meta_document_arg(meta)?)
                .map_err(molrs_error_to_pyerr)?;
        }
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Reattach to an existing store and continue appending after its last
    /// committed frame. Anything a crash left past the commit marker is
    /// rolled back first.
    #[staticmethod]
    #[pyo3(signature = (path, *, flush_every=None, durable=true))]
    fn open(path: PathBuf, flush_every: Option<u64>, durable: bool) -> PyResult<Self> {
        let mut writer = MrecWriter::open(&path)
            .map_err(molrs_error_to_pyerr)?
            .with_durable(durable);
        if let Some(frames) = flush_every {
            writer = writer
                .with_flush_every(frames)
                .map_err(molrs_error_to_pyerr)?;
        }
        Ok(Self {
            inner: Some(writer),
        })
    }

    /// Buffer a frame. With no ``step`` the writer numbers frames 0, 1, 2, …;
    /// pass ``step`` (and optionally ``time``, in the producer's own unit — the
    /// record stores none) for real MD numbering, or
    /// ``time`` alone to keep the automatic numbering and still record times.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the writer is closed, the frame steps outside the
    ///     pinned schema, or ``step`` does not strictly increase.
    #[pyo3(signature = (frame, step=None, time=None))]
    fn append(
        &mut self,
        frame: &Bound<'_, PyFrame>,
        step: Option<i64>,
        time: Option<f64>,
    ) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        frame
            .borrow()
            .with_frame(|core| match (step, time) {
                (Some(step), time) => writer.append_at(core, step, time),
                (None, Some(time)) => writer.append_timed(core, time),
                (None, None) => writer.append(core),
            })?
            .map_err(molrs_error_to_pyerr)
    }

    /// Commit every buffered frame (durably, unless ``durable=False``).
    fn flush(&mut self) -> PyResult<()> {
        let writer = self
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("writer is closed"))?;
        writer.flush().map_err(molrs_error_to_pyerr)
    }

    /// Commit and release the store. Idempotent; appends after close raise.
    fn close(&mut self) -> PyResult<()> {
        if let Some(writer) = self.inner.take() {
            writer.close().map_err(molrs_error_to_pyerr)?;
        }
        Ok(())
    }

    /// The landing cadence in force, in frames.
    #[getter]
    fn flush_every(&self) -> PyResult<u64> {
        self.inner
            .as_ref()
            .map(MrecWriter::flush_every)
            .ok_or_else(|| PyValueError::new_err("writer is closed"))
    }

    /// Frames committed so far.
    #[getter]
    fn committed(&self) -> PyResult<u64> {
        self.inner
            .as_ref()
            .map(MrecWriter::committed)
            .ok_or_else(|| PyValueError::new_err("writer is closed"))
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
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }
}

/// Pack a closed ``*.mrec`` directory into a sibling ``*.mrec.zip`` (every
/// entry stored, byte-identical to the files it replaces) and remove the
/// directory.
///
/// Returns
/// -------
/// str
///     The path of the archive.
#[pyfunction]
pub fn pack_mrec_zip(path: PathBuf) -> PyResult<String> {
    let archive = molrs::io::mrec::pack_mrec_zip(&path).map_err(molrs_error_to_pyerr)?;
    Ok(archive.to_string_lossy().into_owned())
}

/// Judge a snapshot or system-definition frame against the Frame vocabulary.
///
/// Parameters
/// ----------
/// frame
///     In-memory :class:`~molpy.core.Frame`.
///
/// Raises
/// ------
/// ValueError
///     If the frame fails the canonical vocabulary.
#[pyfunction]
pub fn validate_frame(frame: &Bound<'_, PyFrame>) -> PyResult<()> {
    frame
        .borrow()
        .with_frame(molrs::io::mrec::validation::validate_frame)?
        .map_err(molrs_error_to_pyerr)
}

/// Register [`write_mrec`] flat on the native module, as `molpy.io`'s.
pub(crate) fn register_doors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molpy.io", wrap_pyfunction!(write_mrec, m)?)
}

/// Register `molpy.io.mrec` on its `_core.mrec` submodule, with
/// `molpy.io.mrec.validation` as that submodule's `validation`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_mrec_zip, m)?)?;
    m.add_class::<PyMrecFile>()?;
    m.add_class::<PyMrecReader>()?;
    m.add_class::<PySequenceSchema>()?;
    m.add_class::<PyMrecWriter>()?;
    m.add_class::<section::PyForceFieldSection>()?;
    crate::add_submodule(
        m,
        "validation",
        "molpy.io.mrec.validation",
        register_validation,
    )
}

/// Register `molpy.io.mrec.validation`: the record contract's runtime checks.
fn register_validation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_frame, m)?)?;
    Ok(())
}
