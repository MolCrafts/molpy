//! Scientific records (`*.mrec`): [`write_mrec`], flat on `molpy.io` like
//! every other format's writer ([`register_doors`]), and the `_core.mrec`
//! submodule, `molpy.io.mrec` ([`register`]): [`PyMrecFile`] (a record's
//! per-section reads), the store reader and writer ([`PyMrecReader`],
//! [`PySequenceSchema`], [`PyMrecWriter`]), [`pack_mrec_zip`] and the
//! `ForceFieldSection` — each named as `molrs::io::mrec`'s.

pub mod section;

use crate::error::io_error_to_pyerr;

use crate::core::block::{column_from_py, column_to_numpy};
use crate::core::frame::{
    PyFrame, PyMetaValue, infer_meta_value, json_map_to_plain_dict, meta_document_arg,
    meta_value_from_dtype,
};
use crate::core::simbox::PyBox;
use crate::core::trajectory::PyTrajectory;
use crate::error::molrs_error_to_pyerr;
use crate::io::source::PyPath;
use crate::io::{PySink, PySource};
use molrs::core::{DType, ObservableKind, ObservableRecord};
use molrs::io::mrec::{Compression, MolRec, MrecFile, MrecReader, MrecWriter, SequenceSchema};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
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
/// status : Mapping, optional
///     The lifecycle document (``status/``), written as given.
/// method : Mapping, optional
///     The method document (``method/``), written as given.
/// metrics : Mapping, optional
///     The metrics section. ``catalog`` is the group document; ``series``,
///     ``steps`` and ``wall_time`` map a logical series name (``train/loss``)
///     to a 1-d array (``f64``, ``i64``, string); ``wal`` is the
///     ``metrics.jsonl`` text. A name is stored under its safe name.
/// observables : Mapping, optional
///     ``name → {"meta": document, "data": array}``. ``meta`` is the
///     observable's document (``kind``, ``description``, ``time_dependent``,
///     and whatever else the producer set). ``data`` is the array at its
///     own dtype and shape, a 0-d array included.
///
/// Raises
/// ------
/// ValueError
///     If the record has no state section, a section fails to encode, a
///     metrics axis disagrees with its series, an observable name is
///     reserved or its array has no axis while ``time_dependent``, or the
///     force field is invalid or has no section form.
/// TypeError
///     If a document is not a mapping or holds a value JSON cannot, or
///     ``forcefield`` is neither a ``ForceField`` nor a
///     ``ForceFieldSection``.
/// OSError
///     If the store cannot be written.
#[pyfunction]
#[pyo3(signature = (
    sink,
    *,
    meta=None,
    frame=None,
    system=None,
    trajectory=None,
    forcefield=None,
    status=None,
    method=None,
    metrics=None,
    observables=None,
))]
#[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
pub fn write_mrec(
    sink: PySink<'_>,
    meta: Option<&Bound<'_, PyAny>>,
    frame: Option<&Bound<'_, PyFrame>>,
    system: Option<&Bound<'_, PyFrame>>,
    trajectory: Option<PyRef<'_, PyTrajectory>>,
    forcefield: Option<&Bound<'_, PyAny>>,
    status: Option<&Bound<'_, PyAny>>,
    method: Option<&Bound<'_, PyAny>>,
    metrics: Option<&Bound<'_, PyAny>>,
    observables: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let mut record = MolRec::new();
    if let Some(meta) = meta {
        record.meta = meta_document_arg(meta)?;
    }
    record.frame = frame.map(|f| f.borrow().clone_core_frame()).transpose()?;
    record.system = system.map(|s| s.borrow().clone_core_frame()).transpose()?;
    record.trajectory = trajectory.map(|t| t.to_core(t.py())).transpose()?;
    record.forcefield = forcefield.map(PyForceFieldSection::from_arg).transpose()?;
    if let Some(status) = status {
        record.status = meta_document_arg(status)?;
    }
    if let Some(method) = method {
        record.method = meta_document_arg(method)?;
    }
    if let Some(metrics) = metrics {
        apply_metrics(&mut record, metrics)?;
    }
    if let Some(observables) = observables {
        record.observables = observables_from_py(observables)?;
    }
    sink.put(|s| molrs::io::write_mrec(s, &record), io_error_to_pyerr)
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
        let inner = source.read(MrecFile::open).map_err(io_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// The record's top-level section names (``meta``, ``frame``,
    /// ``system``, …), sorted, without decoding any of them.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn sections(&self) -> PyResult<Vec<String>> {
        self.inner.sections().map_err(io_error_to_pyerr)
    }

    /// The record's ``meta`` document, as the producer wrote it; empty when
    /// the store has none.
    ///
    /// Returns
    /// -------
    /// dict
    fn meta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let map = self.inner.meta().map_err(io_error_to_pyerr)?;
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
            .map_err(io_error_to_pyerr)?
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
            .map_err(io_error_to_pyerr)?
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
            .map_err(io_error_to_pyerr)?
            .map(|inner| PyForceFieldSection { inner }))
    }

    /// The ``status`` document, or ``None`` when the record has none.
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        optional_document(py, self.inner.status().map_err(io_error_to_pyerr)?)
    }

    /// The ``method`` document, or ``None`` when the record has none.
    fn method<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        optional_document(py, self.inner.method().map_err(io_error_to_pyerr)?)
    }

    /// The ``metrics`` section, or ``None`` when the record has none.
    ///
    /// Returns
    /// -------
    /// dict | None
    ///     ``catalog``, ``series``, ``steps`` and ``wall_time`` (logical
    ///     series name → 1-d array), and ``wal`` when the JSONL file is there.
    fn metrics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(metrics) = self.inner.metrics().map_err(io_error_to_pyerr)? else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("catalog", json_map_to_plain_dict(py, &metrics.catalog)?)?;
        out.set_item(
            "series",
            number_series_dict(py, &metrics.series, "float64")?,
        )?;
        out.set_item("steps", number_series_dict(py, &metrics.steps, "int64")?)?;
        out.set_item("wall_time", string_series_dict(py, &metrics.wall_time)?)?;
        if let Some(wal) = metrics.wal {
            out.set_item("wal", wal)?;
        }
        Ok(Some(out))
    }

    /// The ``observables`` section, or ``None`` when the record has none.
    ///
    /// Returns
    /// -------
    /// dict | None
    ///     ``name → {"meta": document, "data": array}``, the document as
    ///     stored and the array at its stored dtype and shape.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A data array without its metadata, metadata without its array,
    ///     or a document without ``kind``.
    fn observables<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(entries) = self.inner.observables().map_err(io_error_to_pyerr)? else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        for entry in entries {
            let item = PyDict::new(py);
            item.set_item("meta", json_map_to_plain_dict(py, &entry.meta)?)?;
            item.set_item("data", column_to_numpy(py, &entry.values)?)?;
            out.set_item(entry.name, item)?;
        }
        Ok(Some(out))
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
        let inner = source.read(MrecReader::open).map_err(io_error_to_pyerr)?;
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
    fn to_trajectory(&self, py: Python<'_>) -> PyResult<PyTrajectory> {
        let inner = self.inner.to_trajectory().map_err(molrs_error_to_pyerr)?;
        PyTrajectory::from_core(py, inner)
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
        let dtype = DType::from_name(dtype)
            .ok_or_else(|| PyValueError::new_err(format!("unknown column dtype {dtype:?}")))?;
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

/// The compression of a spec: ``None`` is no compression, a string is
/// [`Compression`]'s own spelling (``"gzip:5"``, ``"zstd"``, …).
fn parse_compression(spec: Option<&str>) -> PyResult<Compression> {
    spec.map_or(Ok(Compression::None), |text| {
        text.parse::<Compression>().map_err(PyValueError::new_err)
    })
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
        path: PyPath,
        schema: &Bound<'_, PySequenceSchema>,
        flush_every: Option<u64>,
        compression: Option<&str>,
        durable: bool,
        meta: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let schema = schema.borrow().inner.clone();
        let mut writer = MrecWriter::create(&path.0, schema)
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
    fn open(path: PyPath, flush_every: Option<u64>, durable: bool) -> PyResult<Self> {
        let mut writer = MrecWriter::open(&path.0)
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
pub fn pack_mrec_zip(path: PyPath) -> PyResult<String> {
    let archive = molrs::io::mrec::pack_mrec_zip(&path.0).map_err(molrs_error_to_pyerr)?;
    Ok(archive.to_string_lossy().into_owned())
}

fn optional_document<'py>(
    py: Python<'py>,
    document: Option<serde_json::Map<String, serde_json::Value>>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    document
        .map(|map| json_map_to_plain_dict(py, &map))
        .transpose()
}

fn as_dict<'py>(value: &Bound<'py, PyAny>, what: &str) -> PyResult<Bound<'py, PyDict>> {
    if let Ok(dict) = value.cast::<PyDict>() {
        return Ok(dict.clone());
    }
    if let Ok(mapping) = value.cast::<pyo3::types::PyMapping>() {
        let dict = PyDict::new(value.py());
        dict.update(mapping)?;
        return Ok(dict);
    }
    Err(PyTypeError::new_err(format!(
        "{what} must be a mapping, got {}",
        value.get_type().name()?
    )))
}

fn number_series_dict<'py, T>(
    py: Python<'py>,
    values: &std::collections::BTreeMap<String, Vec<T>>,
    dtype: &str,
) -> PyResult<Bound<'py, PyDict>>
where
    T: Clone + IntoPyObject<'py>,
{
    let out = PyDict::new(py);
    let numpy = py.import("numpy")?;
    let dtype = numpy.getattr(dtype)?;
    for (name, column) in values {
        let items = PyList::new(py, column.iter().cloned())?;
        out.set_item(name, numpy.call_method1("array", (items, &dtype))?)?;
    }
    Ok(out)
}

fn string_series_dict<'py>(
    py: Python<'py>,
    values: &std::collections::BTreeMap<String, Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("str_")?)?;
    for (name, column) in values {
        out.set_item(
            name,
            numpy.call_method("array", (column.clone(),), Some(&kwargs))?,
        )?;
    }
    Ok(out)
}

fn apply_metrics(record: &mut MolRec, value: &Bound<'_, PyAny>) -> PyResult<()> {
    for (key, item) in as_dict(value, "metrics")?.iter() {
        match key.extract::<String>()?.as_str() {
            "catalog" => record.metrics = meta_document_arg(&item)?,
            "series" => record.metrics_series = numeric_series(&item, "series")?,
            "steps" => record.metrics_steps = numeric_series(&item, "steps")?,
            "wall_time" => record.metrics_wall_time = text_series(&item)?,
            "wal" if item.is_none() => record.metrics_wal = None,
            "wal" => record.metrics_wal = Some(item.extract()?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "metrics has no {other} field"
                )));
            }
        }
    }
    Ok(())
}

fn numeric_series<T>(
    value: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<std::collections::BTreeMap<String, Vec<T>>>
where
    T: numpy::Element + Copy + molrs::core::BlockDtype,
{
    let mut out = std::collections::BTreeMap::new();
    for (key, array) in as_dict(value, &format!("metrics {what}"))?.iter() {
        let name: String = key.extract()?;
        let column = column_from_py(&array)?;
        if column.shape().len() != 1 {
            return Err(PyValueError::new_err(format!(
                "metrics {what} '{name}' is one-dimensional"
            )));
        }
        let values = column.get::<T>().ok_or_else(|| {
            PyValueError::new_err(format!(
                "metrics {what} '{name}' has dtype {}, not the series dtype",
                column.dtype().name()
            ))
        })?;
        out.insert(name, values.iter().copied().collect());
    }
    Ok(out)
}

fn text_series(
    value: &Bound<'_, PyAny>,
) -> PyResult<std::collections::BTreeMap<String, Vec<String>>> {
    let mut out = std::collections::BTreeMap::new();
    for (key, array) in as_dict(value, "metrics wall_time")?.iter() {
        let name: String = key.extract()?;
        let column = column_from_py(&array)?;
        if column.shape().len() != 1 {
            return Err(PyValueError::new_err(format!(
                "metrics wall_time '{name}' is one-dimensional"
            )));
        }
        let values = column.get::<String>().ok_or_else(|| {
            PyValueError::new_err(format!(
                "metrics wall_time '{name}' is a string array, found {}",
                column.dtype().name()
            ))
        })?;
        out.insert(name, values.iter().cloned().collect());
    }
    Ok(out)
}

fn observables_from_py(value: &Bound<'_, PyAny>) -> PyResult<molrs::io::mrec::Observables> {
    let mut out = molrs::io::mrec::Observables::new();
    for (key, item) in as_dict(value, "observables")?.iter() {
        let name: String = key.extract()?;
        let item = as_dict(&item, &format!("observable '{name}'"))?;
        let meta = item.get_item("meta")?.ok_or_else(|| {
            PyValueError::new_err(format!("observable '{name}' has no meta document"))
        })?;
        let data = item.get_item("data")?.ok_or_else(|| {
            PyValueError::new_err(format!("observable '{name}' has no data array"))
        })?;
        let mut document = meta_document_arg(&meta)?;
        let kind = required_string(&mut document, "kind", &name)?;
        let description = required_string(&mut document, "description", &name)?;
        let time_dependent = match document.remove("time_dependent") {
            Some(serde_json::Value::Bool(flag)) => flag,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "observable '{name}' needs a boolean time_dependent"
                )));
            }
        };
        let axes = match document.remove("axes") {
            None => Vec::new(),
            Some(serde_json::Value::Array(items)) => items
                .into_iter()
                .map(|item| match item {
                    serde_json::Value::String(text) => Ok(text),
                    _ => Err(PyValueError::new_err(format!(
                        "observable '{name}' axes are strings"
                    ))),
                })
                .collect::<PyResult<Vec<_>>>()?,
            Some(_) => {
                return Err(PyValueError::new_err(format!(
                    "observable '{name}' axes are a list of strings"
                )));
            }
        };
        let mut obs = ObservableRecord::new(
            name.clone(),
            ObservableKind::from(kind),
            column_from_py(&data)?,
        );
        obs.description = description;
        obs.time_dependent = time_dependent;
        obs.unit = optional_string(&mut document, "unit", &name)?;
        obs.axes = axes;
        obs.sampling = optional_string(&mut document, "sampling", &name)?;
        obs.domain = optional_string(&mut document, "domain", &name)?;
        obs.target = optional_string(&mut document, "target", &name)?;
        obs.extra = document;
        out.insert(obs).map_err(molrs_error_to_pyerr)?;
    }
    Ok(out)
}

fn required_string(
    document: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    name: &str,
) -> PyResult<String> {
    match document.remove(key) {
        Some(serde_json::Value::String(text)) if !text.is_empty() || key == "description" => {
            Ok(text)
        }
        Some(serde_json::Value::String(_)) => Err(PyValueError::new_err(format!(
            "observable '{name}' needs a non-empty {key}"
        ))),
        _ => Err(PyValueError::new_err(format!(
            "observable '{name}' needs a string {key}"
        ))),
    }
}

fn optional_string(
    document: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    name: &str,
) -> PyResult<Option<String>> {
    match document.remove(key) {
        None => Ok(None),
        Some(serde_json::Value::String(text)) => Ok(Some(text)),
        Some(_) => Err(PyValueError::new_err(format!(
            "observable '{name}' {key} is a string"
        ))),
    }
}

/// Register [`write_mrec`] flat on the native module, as `molpy.io`'s.
pub(crate) fn register_doors(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(m, "molpy.io", wrap_pyfunction!(write_mrec, m)?)
}

/// Register `molpy.io.mrec` on its `_core.mrec` submodule.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_mrec_zip, m)?)?;
    m.add_class::<PyMrecFile>()?;
    m.add_class::<PyMrecReader>()?;
    m.add_class::<PySequenceSchema>()?;
    m.add_class::<PyMrecWriter>()?;
    m.add_class::<section::PyForceFieldSection>()?;
    Ok(())
}
