//! Python class `molpy.core.Block`, a heterogeneous column store backed by the
//! shared FFI store.
//!
//! A [`PyBlock`] holds typed columns keyed by name. Numeric, bool and complex
//! columns are contiguous ndarrays and index to a zero-copy numpy view. A
//! string column indexes the same way, to a numpy ``str`` array of the
//! column's shape; that array is a copy, because a Rust ``String`` has no
//! zero-copy numpy view. This is the only `Block` class: construction from a
//! mapping, schema-dtype adoption, row / multi-column indexing, rename, deep
//! copy, sort and pickling are all implemented here, once.
//!
//! # Supported Column Types
//!
//! | Rust type        | numpy dtype (default) | Typical usage                |
//! |------------------|-----------------------|------------------------------|
//! | `F`  (f64)       | `float64` (narrow float input widened here) | positions, masses, charges |
//! | `I`  (i32/i64)   | `int32`   / `int64`   | atom type IDs                |
//! | `Idx` (u64), `u8`/`u16`/`u32` | `uint64` / `uint8` / `uint16` / `uint32` | bond indices |
//! | `bool`           | `bool`                | selection masks               |
//! | `String`         | numpy ``str`` on index (a copy); shape preserved | element symbols |

use std::sync::Arc;

use molrs::core::{Block as CoreBlock, BlockDtype, BlockError, Column, ColumnArray, DType};
use molrs::op::{F, I, Idx};
use molrs_ffi::BlockRef;
use ndarray::{Array1, ArrayD, IxDyn};
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PySlice, PyTuple};

use crate::core::schema::extract_column_key;
use crate::error::ffi_error_to_pyerr;

/// Heterogeneous column store exposed to Python as `molpy.core.Block`.
///
/// Each column is a named, typed array. All columns share the same number of
/// rows (axis-0 length). The underlying storage lives in an FFI `Store` and is
/// accessed through a version-tracked [`BlockHandle`]; a block read from a
/// frame (``frame["atoms"]``) is a handle on the frame's stored block, so every
/// method below reads and writes the frame's data.
///
/// A column is dense, so a per-row component that only some rows carry is
/// stored alongside a validity mask: `Block.validity(key)` returns it, or
/// `None` when every cell of that column is a real value.
///
/// Every column-key argument accepts a ``str`` or a :class:`molpy.core.keys.Key`.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molpy.core import Block
///
/// b = Block({"x": [1.0, 2.0, 3.0], "element": ["C", "H", "H"]})
/// assert b.n_rows == 3
/// assert "x" in b
/// arr = b["x"]                 # zero-copy numpy view
/// names = b["element"]         # numpy str array; a copy, shape preserved
/// xyz = b["x", "x", "x"]             # (3, 3) stacked columns
/// sub = b[b["x"] > 1.5]       # a new Block of the selected rows
/// assert b.validity("x") is None   # no holes in that column
/// ```
#[pyclass(
    module = "molpy.core",
    name = "Block",
    from_py_object,
    unsendable,
    subclass
)]
#[derive(Clone)]
pub struct PyBlock {
    pub(crate) inner: BlockRef,
}

#[pymethods]
impl PyBlock {
    /// Create a block, optionally from a mapping of column name -> array.
    ///
    /// Each value goes through :meth:`__setitem__`, so a canonical key adopts
    /// the dtype the Frame schema declares for it. Copying an existing block
    /// is :meth:`copy`, not ``Block(block)``.
    ///
    /// Parameters
    /// ----------
    /// data : Mapping[str | Key, ArrayLike], optional
    ///     Column name -> array. Every column must have the same length.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``data`` is a ``Block`` or not a mapping.
    /// ValueError
    ///     If a value is not array-like, the lengths differ, or a canonical
    ///     column's values do not survive the schema dtype.
    ///
    /// Examples
    /// --------
    /// >>> Block().n_rows
    /// 0
    /// >>> Block({"id": [1, 2]}).dtype("id")
    /// 'uint'
    #[new]
    #[pyo3(signature = (data = None))]
    fn new(data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut block = Self::from_core_block(CoreBlock::new())?;
        if let Some(data) = data {
            block.absorb_mapping(data)?;
        }
        Ok(block)
    }

    /// Insert a numpy array (or list of strings) as a named column, at the
    /// width given.
    ///
    /// The typed write: unlike ``block[key] = array`` it does not convert to
    /// the schema dtype (an ``i64`` column stays ``i64``; a narrow float is
    /// promoted to ``float64``, the one float width); a dtype the schema
    /// refuses raises. If a column with the same key already exists it is
    /// replaced. The array length must match the row count of existing
    /// columns, or the block must be empty.
    ///
    /// ``validity`` optionally attaches a per-row mask — the write side of
    /// :meth:`validity`. ``validity[i]`` is ``False`` where row ``i`` of
    /// ``values`` is a hole: the array still carries something there
    /// (typically a zero) and the mask is the only place that says it is not
    /// a stated value. An all-``True`` mask states nothing and is dropped, so
    /// :meth:`validity` keeps answering ``None``.
    ///
    /// Parameters
    /// ----------
    /// key : str | Key
    ///     Column name (e.g. ``"x"``, ``"element"``).
    /// values : numpy.ndarray | list[str]
    ///     Column data.
    /// validity : numpy.ndarray | Sequence[bool] | None
    ///     1-D boolean mask in row order, one entry per row of ``values``.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If the array dtype is not supported, or ``validity`` is not a 1-D
    ///     bool array or a sequence of bools.
    /// ValueError
    ///     If the row count does not match existing columns, or ``validity``
    ///     does not have exactly one entry per row of ``values``.
    ///
    /// Examples
    /// --------
    /// >>> b = Block()
    /// >>> b.insert("x", np.zeros(3, dtype=np.float32))
    /// >>> b.insert("frag_id", np.array([7, 0, 0]), [True, False, False])
    /// >>> b.validity("frag_id")
    /// array([ True, False, False])
    #[pyo3(signature = (key, values, validity = None))]
    fn insert(
        &mut self,
        key: &Bound<'_, PyAny>,
        values: &Bound<'_, PyAny>,
        validity: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let key = extract_column_key(key)?;
        let mask = validity.map(|v| bool_mask(&key, v)).transpose()?;
        self.insert_any(&key, values, mask)
    }

    /// The validity mask of a column, or ``None`` when the column has no holes.
    ///
    /// A block column is dense, so a component that only some rows carry —
    /// ``frag_id`` on a partially labelled molecule, ``h_count`` declared on
    /// one bracket atom — needs a second array saying which cells are real.
    /// ``None`` is the common case and means "every cell is a stated value";
    /// it is not an all-``True`` array, and a caller must not read a missing
    /// mask as "all holes". A key that names no column is a different question
    /// and raises ``KeyError``, as indexing and :meth:`dtype` do — were it
    /// to answer ``None``, a misspelled key would read as a dense column.
    ///
    /// Parameters
    /// ----------
    /// key : str | Key
    ///     Column name.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray | None
    ///     A 1-D ``bool`` array in row order — ``True`` where the cell holds a
    ///     real value, ``False`` where it is a hole — or ``None`` when the
    ///     column carries no mask.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    fn validity<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyArray1<bool>>>> {
        let key = extract_column_key(key)?;
        // The mask is borrowed from inside the store, so it is copied out
        // before the borrow ends; it is one byte per row and, unlike a column,
        // has no `Arc` to hand numpy for a zero-copy view.
        let mask = self.with_block(|b| {
            if !b.contains_key(&key) {
                return Err(missing_column(b, &key));
            }
            Ok(b.validity(&key).map(<[bool]>::to_vec))
        })??;
        Ok(mask.map(|m| Array1::from(m).into_pyarray(py)))
    }

    /// Attach a validity mask to an existing column, whatever its dtype.
    ///
    /// ``mask[i]`` is ``False`` where row ``i`` holds no value. The mask
    /// replaces any the column had; an all-``True`` mask clears it, so
    /// :meth:`validity` answers ``None`` afterwards.
    ///
    /// Parameters
    /// ----------
    /// key : str | Key
    ///     Column name.
    /// mask : numpy.ndarray | Sequence[bool]
    ///     1-D boolean mask, one entry per row.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    /// TypeError
    ///     If ``mask`` is not a 1-D bool array or a sequence of bools.
    /// ValueError
    ///     If ``mask`` does not have exactly one entry per row.
    fn set_validity(&mut self, key: &Bound<'_, PyAny>, mask: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = extract_column_key(key)?;
        let mask = bool_mask(&key, mask)?;
        self.inner
            .with_mut(|b| b.set_validity(&key, mask))
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| match e {
                BlockError::MissingColumn { key } => PyKeyError::new_err(key),
                other => PyValueError::new_err(other.to_string()),
            })
    }

    /// Declare (or, with ``None``, withdraw) the precision of an ``f64``
    /// column: an absolute tolerance in the column's own units.
    ///
    /// A record writer rounds the column to the largest power of two not
    /// above ``precision`` (ties to even) before storing it, so the stored
    /// values are within ``precision / 2`` of these and compress several
    /// times better; the values in memory are not touched. The declaration
    /// is stored with the column (``frame`` / ``system``) or in the
    /// trajectory's schema, and reads back. A rename keeps it; replacing the
    /// column with another dtype, or removing it, drops it.
    ///
    /// Parameters
    /// ----------
    /// key : str | Key
    ///     Column name.
    /// precision : float | None
    ///     Finite, within ``[2**-1000, 2**1000]``; ``None`` withdraws it.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    /// ValueError
    ///     If the column is not ``float64`` or ``precision`` is out of bounds.
    ///
    /// Examples
    /// --------
    /// >>> b = molpy.core.Block({"x": np.array([0.12345, 1.5])})
    /// >>> b.set_precision("x", 1e-3)
    /// >>> b.precision("x")
    /// 0.001
    #[pyo3(signature = (key, precision))]
    fn set_precision(&mut self, key: &Bound<'_, PyAny>, precision: Option<f64>) -> PyResult<()> {
        let key = extract_column_key(key)?;
        self.inner
            .with_mut(|b| match precision {
                Some(p) => b.set_precision(&key, p),
                None if b.contains_key(&key) => {
                    b.clear_precision(&key);
                    Ok(())
                }
                None => Err(BlockError::MissingColumn { key: key.clone() }),
            })
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| match e {
                BlockError::MissingColumn { key } => PyKeyError::new_err(key),
                other => PyValueError::new_err(other.to_string()),
            })
    }

    /// Declare (or, with ``None``, withdraw) that a ``uint64`` column holds
    /// 0-based row indices into *target*: ``"<block>"`` of the same frame, or
    /// ``"/<section>/<block>"`` of a frame-shaped section of the record
    /// (``"/frame/atoms"``).
    ///
    /// The relation endpoints ``atomi`` … ``atoml`` reference ``atoms``
    /// without one; a declaration overrides that, and names what any other
    /// referencing column (``members.atom``) points into. Subsetting and
    /// replicating renumber same-frame references; a ``*.mrec`` writer and
    /// reader refuse a reference that does not resolve.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    /// ValueError
    ///     If the column is not ``uint64`` or *target* is malformed or names a
    ///     trajectory block (``"/trajectory/…"``).
    #[pyo3(signature = (key, target))]
    fn set_target(&mut self, key: &Bound<'_, PyAny>, target: Option<&str>) -> PyResult<()> {
        let key = extract_column_key(key)?;
        self.inner
            .with_mut(|b| match target {
                Some(target) => b.set_target(&key, target),
                None if b.contains_key(&key) => {
                    b.clear_target(&key);
                    Ok(())
                }
                None => Err(BlockError::MissingColumn { key: key.clone() }),
            })
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| match e {
                BlockError::MissingColumn { key } => PyKeyError::new_err(key),
                other => PyValueError::new_err(other.to_string()),
            })
    }

    /// The declared target of a column, or ``None``.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    fn target(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        let key = extract_column_key(key)?;
        self.with_block(|b| {
            if !b.contains_key(&key) {
                return Err(missing_column(b, &key));
            }
            Ok(b.target(&key).map(str::to_string))
        })?
    }

    /// Every declared target, as ``{column: target}``.
    fn targets(&self) -> PyResult<std::collections::BTreeMap<String, String>> {
        self.with_block(|b| {
            b.targets()
                .map(|(k, t)| (k.to_string(), t.to_string()))
                .collect()
        })
    }

    /// The declared precision of a column, or ``None`` when it declares none.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist in this block.
    fn precision(&self, key: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
        let key = extract_column_key(key)?;
        self.with_block(|b| {
            if !b.contains_key(&key) {
                return Err(missing_column(b, &key));
            }
            Ok(b.precision(&key))
        })?
    }

    /// Row-wise union of ``parts``: their rows in order, under the union of
    /// their columns (first-seen order).
    ///
    /// A column a part lacks is filled with the dtype's default for that
    /// part's rows and marked null in the column's validity mask. A part's
    /// own masks travel with its rows. The parts are not modified.
    ///
    /// Parameters
    /// ----------
    /// parts : Sequence[Block]
    ///
    /// Returns
    /// -------
    /// Block
    ///     A new block (new buffers).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If two parts carry one column under different dtypes or per-row
    ///     shapes.
    ///
    /// Examples
    /// --------
    /// >>> s = Block.stack([Block({"x": [0.0]}), Block({"x": [1.0], "q": [2.0]})])
    /// >>> s.validity("q")
    /// array([False,  True])
    #[staticmethod]
    fn stack(parts: Vec<PyRef<'_, PyBlock>>) -> PyResult<PyBlock> {
        let blocks = parts
            .iter()
            .map(|p| p.clone_core_block())
            .collect::<PyResult<Vec<_>>>()?;
        let stacked =
            CoreBlock::stack(&blocks).map_err(|e| PyValueError::new_err(e.to_string()))?;
        PyBlock::from_core_block(stacked)
    }

    /// Positions as an ``(n_rows, 3)`` float64 array, gathered from the
    /// ``x`` / ``y`` / ``z`` columns (a copy).
    ///
    /// Assigning an ``(N, 3)`` array-like writes it back into ``x`` / ``y`` /
    /// ``z`` (each replaced as a float64 column; a missing one is added).
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     On read, if ``x``, ``y`` or ``z`` is missing.
    /// ValueError
    ///     On write, if the array is not ``(N, 3)`` or ``N`` differs from the
    ///     block's row count.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<F>>> {
        let xyz = self.with_block(CoreBlock::coords)?.map_err(coords_error)?;
        Ok(xyz.into_pyarray(py))
    }

    #[setter]
    fn set_coords(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let xyz = coords_array(value)?;
        self.inner
            .with_mut(|b| b.set_coords(xyz.view()))
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Number of rows (axis-0 length); ``0`` for a block with no rows.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_rows(&self) -> PyResult<usize> {
        self.with_block(|b| b.n_rows().unwrap_or(0))
    }

    /// Axis-0 length of an empty block, or reshape every column.
    fn resize(&mut self, n_rows: usize) -> PyResult<()> {
        self.inner
            .with_mut(|b| b.resize(n_rows))
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Declare this block as N-dimensional. Product of ``shape`` must equal
    /// the row count when the block has columns.
    fn set_shape(&mut self, shape: Vec<usize>) -> PyResult<()> {
        self.inner
            .with_mut(|b| b.set_shape(&shape))
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Reported shape: ``[n_rows]`` for a table, the declared N-D shape for a
    /// grid, ``[]`` for an empty block.
    #[getter]
    fn shape(&self) -> PyResult<Vec<usize>> {
        self.with_block(|b| b.shape())
    }

    /// Explicit N-D structural shape, or ``None`` for a plain row table.
    #[getter]
    fn structural_shape(&self) -> PyResult<Option<Vec<usize>>> {
        self.with_block(|b| b.structural_shape().map(|s| s.to_vec()))
    }

    /// Number of columns in this block.
    fn __len__(&self) -> PyResult<usize> {
        self.with_block(|b| b.len())
    }

    /// List all column names.
    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_block(|b| b.keys().map(|s| s.to_string()).collect())
    }

    /// Iterate over the column names, so ``dict(block)`` is its columns.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyList::new(py, self.keys()?)?.try_iter()
    }

    /// Whether a column exists; a key that is not a ``str`` or ``Key`` is
    /// absent.
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        match extract_column_key(key) {
            Ok(key) => self.with_block(|b| b.contains_key(&key)),
            Err(_) => Ok(false),
        }
    }

    /// Read a column, several stacked columns, or a selection of rows.
    ///
    /// * ``block["x"]`` (``str`` or ``Key``) — the column as a numpy array of
    ///   its stored shape. Numeric, bool and complex columns are a zero-copy
    ///   view. A string column is a numpy ``str`` array and a copy: a Rust
    ///   string has no zero-copy numpy view, and writing the array does not
    ///   change the column.
    /// * ``block["x", "y", "z"]`` / ``block[["x", "y", "z"]]`` — equal-shaped,
    ///   equal-dtype columns side by side, one ``(n_rows, k)`` array.
    /// * ``block[mask]`` (1-D bool array, one entry per row) or
    ///   ``block[indices]`` (1-D int array; ``-n_rows <= i < 0`` wraps) — a new
    ///   ``Block`` of those rows, in order, validity masks included.
    /// * ``block[a:b:c]`` — a new ``Block`` of the sliced rows.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     A missing column, or an empty name list.
    /// ValueError
    ///     Stacked columns differ in shape or dtype; a row index past the end.
    /// IndexError
    ///     A mask of the wrong length, an index below ``-n_rows``, or a
    ///     selector that is not 1-D.
    /// TypeError
    ///     Any other key, including a float row array.
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(name) = extract_column_key(key) {
            return self.column_view(py, &name);
        }
        if key.cast::<PyTuple>().is_ok() || key.cast::<PyList>().is_ok() {
            return self.stacked_columns(py, key);
        }
        if let Ok(slice) = key.cast::<PySlice>() {
            let n = isize::try_from(self.n_rows()?)
                .map_err(|_| PyValueError::new_err("row count overflows isize"))?;
            let span = slice.indices(n)?;
            let rows = (0..span.slicelength)
                .map(|k| (span.start + k as isize * span.step) as usize)
                .collect();
            return Ok(Py::new(py, self.gather_rows(rows)?)?.into_any());
        }
        if key.is_instance(&py.import("numpy")?.getattr("ndarray")?)? {
            let rows = Self::row_selection(key, self.n_rows()?)?;
            return Ok(Py::new(py, self.gather_rows(rows)?)?.into_any());
        }
        Err(PyTypeError::new_err(format!(
            "a Block is indexed by a column name, a tuple or list of names, a \
             slice, or a 1-D bool/int row array; got {}",
            key.get_type().name()?
        )))
    }

    /// Store a column, or spread an array over several columns.
    ///
    /// ``block["x"] = values`` stores one column (``values`` goes through
    /// ``numpy.asarray``), replacing a column already under that name. A key
    /// the Frame schema declares adopts the declared dtype when the values
    /// survive the conversion (``np.arange(n)`` under ``id`` is stored
    /// ``uint``) and is refused when they do not (``-1`` under ``id``).
    ///
    /// ``block["x", "y", "z"] = arr`` spreads an ``(N, k)`` array over the
    /// ``k`` named columns, column ``i`` receiving ``arr[:, i]``; it is the
    /// inverse of the stacked read. Every check runs before the first column
    /// is written, so a refusal leaves the block unchanged.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     An empty name list.
    /// ValueError
    ///     A scalar value; values the schema dtype cannot hold; a row count
    ///     that does not match; a repeated name or an array that is not
    ///     ``(N, k)``.
    /// BlockDtypeError
    ///     An object / None-bearing / ragged column.
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(name) = extract_column_key(key) {
            return self.set_column(&name, value);
        }
        if key.cast::<PyTuple>().is_ok() || key.cast::<PyList>().is_ok() {
            return self.spread_columns(key, value);
        }
        Err(PyTypeError::new_err(format!(
            "a Block column is set by a name or a tuple/list of names; got {}",
            key.get_type().name()?
        )))
    }

    /// Remove a column by name — ``del block["x"]``.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If no column of that name exists.
    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = extract_column_key(key)?;
        let removed = self
            .inner
            .with_mut(|b| b.remove(&key).is_some())
            .map_err(ffi_error_to_pyerr)?;
        if removed {
            Ok(())
        } else {
            Err(PyKeyError::new_err(key))
        }
    }

    /// Rename a column in place, keeping its data and validity mask.
    ///
    /// Renaming is a write into ``new_key``: when the Frame schema declares
    /// ``new_key`` with a different dtype, the column adopts it if the values
    /// survive the conversion (a file's ``int64`` serial renamed onto ``id``
    /// becomes ``uint``), and the rename is refused otherwise.
    ///
    /// Parameters
    /// ----------
    /// old_key : str | Key
    ///     Existing column name.
    /// new_key : str | Key
    ///     New column name.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``old_key`` does not exist or ``new_key`` already does.
    /// ValueError
    ///     If the values do not survive the schema dtype of ``new_key``.
    fn rename(
        &mut self,
        py: Python<'_>,
        old_key: &Bound<'_, PyAny>,
        new_key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let old_key = extract_column_key(old_key)?;
        let new_key = extract_column_key(new_key)?;
        let (index, taken, mask) = self.with_block(|b| {
            (
                b.keys().position(|k| k == old_key),
                b.contains_key(&new_key),
                b.validity(&old_key).map(<[bool]>::to_vec),
            )
        })?;
        let Some(index) = index else {
            return Err(PyKeyError::new_err(old_key));
        };
        if taken {
            return Err(PyKeyError::new_err(format!(
                "cannot rename '{old_key}' to '{new_key}': column already exists"
            )));
        }
        let column = self.column_view(py, &old_key)?.into_bound(py);
        let adopted = adopt_schema_dtype(&new_key, &column)?;
        if !adopted.is(&column) {
            // Write the converted column first: a refusal leaves the block
            // as it was. It lands at the end, so it is then moved into the
            // old column's slot — a rename keeps the column in place.
            self.insert_any(&new_key, &adopted, mask)?;
            self.inner
                .with_mut(|b| {
                    b.remove(&old_key);
                    b.move_column(&new_key, index)
                })
                .map_err(ffi_error_to_pyerr)?
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            return Ok(());
        }
        self.inner
            .with_mut(|b| b.rename_column(&old_key, &new_key))
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| match e {
                molrs::core::BlockError::Validation { .. } => PyKeyError::new_err(old_key.clone()),
                other => PyValueError::new_err(other.to_string()),
            })
    }

    /// Return a new Block sorted by the ``key`` column (original unchanged).
    ///
    /// Parameters
    /// ----------
    /// key : str | Key
    ///     Column to sort by.
    /// reverse : bool, optional
    ///     Descending order (the ascending order reversed). Default ``False``.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` is not a column.
    #[pyo3(signature = (key, reverse = false))]
    fn sort(&self, key: &Bound<'_, PyAny>, reverse: bool) -> PyResult<PyBlock> {
        let key = extract_column_key(key)?;
        let sorted = self.with_block(|b| {
            if !b.contains_key(&key) {
                return Err(missing_column(b, &key));
            }
            b.sort_by(&key, reverse)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })??;
        PyBlock::from_core_block(sorted)
    }

    /// A deep copy: new buffers for every column, masks and shape included.
    ///
    /// Writing into the copy's columns (through numpy) never reaches this
    /// block, nor the numpy array a column was built from.
    fn copy(&self) -> PyResult<PyBlock> {
        PyBlock::from_core_block(self.with_block(CoreBlock::deep_copy)?)
    }

    /// Return the dtype string for the given column.
    ///
    /// Returns
    /// -------
    /// str
    ///     ``"float"``, ``"int"``, ``"i64"``, ``"uint"``,
    ///     ``"bool"``, ``"u8"``, ``"string"``, …
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    fn dtype(&self, key: &Bound<'_, PyAny>) -> PyResult<String> {
        let key = extract_column_key(key)?;
        self.with_block(|b| {
            let col = b
                .get(&key)
                .ok_or_else(|| PyKeyError::new_err(key.clone()))?;
            Ok::<String, PyErr>(format!("{}", col.dtype()))
        })?
    }

    /// Pickle by logical state: columns, validity masks, declared precisions,
    /// row count and structural shape.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let columns = PyDict::new(py);
        let masks = PyDict::new(py);
        for key in this.keys()? {
            let value = this.column_view(py, &key)?;
            columns.set_item(&key, value)?;
            if let Some(mask) = this.with_block(|b| b.validity(&key).map(<[bool]>::to_vec))? {
                masks.set_item(&key, mask)?;
            }
        }
        let precisions = PyDict::new(py);
        for (key, p) in this.with_block(|b| {
            b.precisions()
                .map(|(k, p)| (k.to_string(), p))
                .collect::<Vec<_>>()
        })? {
            precisions.set_item(key, p)?;
        }
        let state = PyDict::new(py);
        state.set_item("columns", columns)?;
        state.set_item("validity", masks)?;
        state.set_item("precision", precisions)?;
        state.set_item("targets", this.targets()?)?;
        state.set_item("n_rows", this.with_block(|b| b.n_rows())?)?;
        state.set_item(
            "shape",
            this.with_block(|b| b.structural_shape().map(<[usize]>::to_vec))?,
        )?;
        crate::pickle::reduce_with_state(slf.as_any(), PyTuple::empty(py), state.into_any())
    }

    /// Restore the state [`__reduce__`](Self::__reduce__) produced.
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let field = |name: &str| {
            state
                .get_item(name)?
                .ok_or_else(|| PyKeyError::new_err(format!("Block state lacks '{name}'")))
        };
        let columns = field("columns")?;
        let columns = columns.cast::<PyDict>()?;
        let masks = field("validity")?;
        let masks = masks.cast::<PyDict>()?;
        for (key, array) in columns.iter() {
            let key: String = key.extract()?;
            let mask = masks
                .get_item(&key)?
                .map(|mask| mask.extract::<Vec<bool>>())
                .transpose()?;
            self.insert_any(&key, &array, mask)?;
        }
        if columns.is_empty()
            && let Some(n_rows) = field("n_rows")?.extract::<Option<usize>>()?
        {
            self.resize(n_rows)?;
        }
        if let Some(shape) = field("shape")?.extract::<Option<Vec<usize>>>()? {
            self.set_shape(shape)?;
        }
        if let Some(targets) = state.get_item("targets")? {
            for (key, target) in targets.cast::<PyDict>()?.iter() {
                let key: String = key.extract()?;
                let target: String = target.extract()?;
                self.inner
                    .with_mut(|b| b.set_target(&key, &target))
                    .map_err(ffi_error_to_pyerr)?
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        // Absent from a state pickled before precisions were carried.
        let precisions = match state.get_item("precision")? {
            Some(precisions) => precisions,
            None => PyDict::new(state.py()).into_any(),
        };
        for (key, p) in precisions.cast::<PyDict>()?.iter() {
            let key: String = key.extract()?;
            let p: f64 = p.extract()?;
            self.inner
                .with_mut(|b| b.set_precision(&key, p))
                .map_err(ffi_error_to_pyerr)?
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_block(|b| {
            let keys: Vec<&str> = b.keys().collect();
            format!("Block(n_rows={}, keys={:?})", b.n_rows().unwrap_or(0), keys)
        })
    }
}

impl PyBlock {
    /// Create a `PyBlock` from a Rust `CoreBlock`, allocating a new
    /// single-frame FFI store.
    pub(crate) fn from_core_block(block: CoreBlock) -> PyResult<Self> {
        let arena = molrs_ffi::FrameArenaCell::default();
        let frame = arena.borrow_mut().frame_new();
        arena
            .borrow_mut()
            .set_block(frame, "__block__", block.deep_copy())
            .map_err(ffi_error_to_pyerr)?;
        let handle = arena
            .borrow()
            .get_block(frame, "__block__")
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self {
            inner: BlockRef::new(arena, handle),
        })
    }

    /// A standalone block built from a mapping of column name -> array — the
    /// constructor body, shared with ``frame[name] = {...}``.
    pub(crate) fn from_mapping(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut block = Self::from_core_block(CoreBlock::new())?;
        block.absorb_mapping(data)?;
        Ok(block)
    }

    /// Independent snapshot: writable NumPy views cannot mutate its buffers.
    pub(crate) fn clone_core_block(&self) -> PyResult<CoreBlock> {
        self.with_block(CoreBlock::deep_copy)
    }

    /// Run a read-only closure on the underlying `CoreBlock`.
    pub(crate) fn with_block<R>(&self, f: impl FnOnce(&CoreBlock) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }

    /// Normalise a row selector over `n_rows` rows to row indices.
    ///
    /// The selector goes through ``numpy.asarray``. A bool mask of length
    /// `n_rows` selects its ``True`` rows in order; integer indices in
    /// ``-n_rows..-1`` wrap to ``n_rows + i`` and non-negative ones pass through
    /// (the upper bound is the gather's to check); an empty selector selects
    /// nothing whatever its dtype. Shared by ``Block[...]`` and
    /// ``Frame.subset``.
    pub(crate) fn row_selection(
        selector: &Bound<'_, PyAny>,
        n_rows: usize,
    ) -> PyResult<Vec<usize>> {
        let py = selector.py();
        let arr = py.import("numpy")?.call_method1("asarray", (selector,))?;
        let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
        let [len] = shape[..] else {
            return Err(PyIndexError::new_err(format!(
                "row selector must be 1-D, got shape {shape:?}"
            )));
        };
        let dtype = arr.getattr("dtype")?;
        let kind: String = dtype.getattr("kind")?.extract()?;
        if kind == "b" {
            if len != n_rows {
                return Err(PyIndexError::new_err(format!(
                    "boolean index did not match block: block has {n_rows} rows \
                     but mask has length {len}"
                )));
            }
            let mask: Vec<bool> = arr.call_method0("tolist")?.extract()?;
            return Ok(mask
                .iter()
                .enumerate()
                .filter_map(|(row, &keep)| keep.then_some(row))
                .collect());
        }
        if len == 0 {
            return Ok(Vec::new());
        }
        if kind != "i" && kind != "u" {
            return Err(PyTypeError::new_err(format!(
                "row indices must be bool or integer, got dtype {}",
                dtype.str()?
            )));
        }
        let n = i64::try_from(n_rows).map_err(|_| PyValueError::new_err("row count overflows"))?;
        let indices: Vec<i64> = arr.call_method0("tolist")?.extract()?;
        indices
            .into_iter()
            .map(|i| match i {
                i if i < -n => Err(PyIndexError::new_err(format!(
                    "row index {i} is out of range for {n_rows} rows"
                ))),
                i if i < 0 => Ok((i + n) as usize),
                i => Ok(i as usize),
            })
            .collect()
    }

    /// A new block of rows `indices`, masks included.
    fn gather_rows(&self, indices: Vec<usize>) -> PyResult<PyBlock> {
        let rows = self.with_block(|b| {
            b.select_rows(&indices)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })??;
        PyBlock::from_core_block(rows)
    }

    /// Store every entry of a mapping through [`set_column`](Self::set_column).
    fn absorb_mapping(&mut self, data: &Bound<'_, PyAny>) -> PyResult<()> {
        if data.cast::<PyBlock>().is_ok() {
            return Err(PyTypeError::new_err(
                "Block() takes a mapping of column name -> array; copy a block \
                 with block.copy()",
            ));
        }
        if !data.hasattr("keys")? {
            return Err(PyTypeError::new_err(format!(
                "Block() takes a mapping of column name -> array, got {}",
                data.get_type().name()?
            )));
        }
        for key in data.call_method0("keys")?.try_iter()? {
            let key = key?;
            let value = data.get_item(&key)?;
            self.set_column(&extract_column_key(&key)?, &value)?;
        }
        Ok(())
    }

    /// Numpy array of one column, with the column's stored shape.
    ///
    /// Numeric, bool and complex columns are a zero-copy view: cloning an
    /// `Arc<ArrayD<T>>` inside the closure is an O(1) refcount bump, and the
    /// owner struct carries that Arc out of the store borrow so the view stays
    /// valid after the closure returns. A string column is a numpy ``str``
    /// array built after the borrow ends. Numpy strings are fixed-width, so
    /// that array is a copy; its shape is the column's, not a flattened vector.
    fn column_view(&self, py: Python<'_>, key: &str) -> PyResult<Py<PyAny>> {
        enum Read {
            View(Py<PyAny>),
            String(Vec<String>, Vec<usize>),
        }
        let read = self
            .inner
            .with(|b| -> PyResult<Read> {
                let col = b.get(key).ok_or_else(|| missing_column(b, key))?;
                Ok(match col {
                    Column::Float(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::Int(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::I8(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::I16(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::I64(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::Bool(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::Uint(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::U8(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::U16(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::U32(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::C64(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::C128(a) => Read::View(typed_array_view(py, Arc::clone(a))?),
                    Column::String(a) => {
                        Read::String(a.iter().cloned().collect(), a.shape().to_vec())
                    }
                })
            })
            .map_err(ffi_error_to_pyerr)??;
        match read {
            Read::View(array) => Ok(array),
            Read::String(flat, shape) => numpy_string_array(py, &flat, shape),
        }
    }

    /// ``block["x", "y", "z"]``: equal-shaped, equal-dtype columns stacked
    /// side by side into one ``(n_rows, k)`` array.
    fn stacked_columns(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let names = column_names(key)?;
        let arrays = names
            .iter()
            .map(|name| Ok(self.column_view(py, name)?.into_bound(py)))
            .collect::<PyResult<Vec<_>>>()?;
        let (first_shape, first_dtype) = (arrays[0].getattr("shape")?, arrays[0].getattr("dtype")?);
        for (name, array) in names.iter().zip(&arrays).skip(1) {
            let shape = array.getattr("shape")?;
            if !shape.eq(&first_shape)? {
                return Err(PyValueError::new_err(format!(
                    "stacked columns must share one shape: '{}' has shape {}, \
                     '{name}' has shape {}",
                    names[0],
                    first_shape.str()?,
                    shape.str()?
                )));
            }
            let dtype = array.getattr("dtype")?;
            if !dtype.eq(&first_dtype)? {
                return Err(PyValueError::new_err(format!(
                    "stacked columns must share one dtype: '{}' is {}, '{name}' \
                     is {}",
                    names[0],
                    first_dtype.str()?,
                    dtype.str()?
                )));
            }
        }
        Ok(py
            .import("numpy")?
            .call_method1("column_stack", (PyList::new(py, arrays)?,))?
            .unbind())
    }

    /// ``block[name] = values`` — the one schema-adopting column write.
    fn set_column(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let array = value
            .py()
            .import("numpy")?
            .call_method1("asarray", (value,))?;
        if array.getattr("ndim")?.extract::<usize>()? == 0 {
            return Err(PyValueError::new_err(format!(
                "Block column '{name}' must be an array-like of at least 1-D; got \
                 a scalar ({}). Wrap it in a sequence or broadcast it to the \
                 column length.",
                value.repr()?
            )));
        }
        let array = adopt_schema_dtype(name, &array)?;
        self.insert_any(name, &array, None)
    }

    /// ``block["x", "y", "z"] = arr`` — every check first, then one
    /// [`insert_any`](Self::insert_any) per column.
    fn spread_columns(&mut self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let names = column_names(key)?;
        let mut seen = std::collections::HashSet::new();
        if !names.iter().all(|name| seen.insert(name)) {
            return Err(PyValueError::new_err(format!(
                "Block column names {names:?} repeat a name"
            )));
        }
        let numpy = py.import("numpy")?;
        let array = numpy.call_method1("asarray", (value,))?;
        let shape: Vec<usize> = array.getattr("shape")?.extract()?;
        let k = names.len();
        if shape.len() != 2 || shape[1] != k {
            return Err(PyValueError::new_err(format!(
                "Writing {k} columns {names:?} needs an (N, {k}) array; got shape {shape:?}"
            )));
        }
        let (len, n_rows) = self.with_block(|b| (b.len(), b.n_rows().unwrap_or(0)))?;
        if len > 0 && shape[0] != n_rows {
            return Err(PyValueError::new_err(format!(
                "Writing columns {names:?} needs {n_rows} rows, the block's row \
                 count; got {}",
                shape[0]
            )));
        }
        let all_rows = PySlice::full(py);
        let columns = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let column = array.get_item((&all_rows, i))?;
                let column = numpy.call_method1("ascontiguousarray", (column,))?;
                adopt_schema_dtype(name, &column)
            })
            .collect::<PyResult<Vec<_>>>()?;
        for (name, column) in names.iter().zip(&columns) {
            self.insert_any(name, column, None)?;
        }
        Ok(())
    }

    /// Shared body of [`insert`](Self::insert) and the subscript write:
    /// dispatch a Python value to its typed column, optionally carrying a
    /// validity mask.
    ///
    /// A masked insert takes the same zero-copy forge path as a plain one: the
    /// mask is attached to the installed column by `Block::set_validity`,
    /// which accepts a column of any dtype.
    fn insert_any(
        &mut self,
        key: &str,
        array: &Bound<'_, PyAny>,
        validity: Option<Vec<bool>>,
    ) -> PyResult<()> {
        // numpy-only Store contract: reject object-kind arrays (object dtype,
        // None-bearing, ragged/mixed — numpy renders all of these as kind 'O')
        // up front, before the typed-cast / Vec<String> extraction below, so
        // even an empty object array fails fast instead of slipping through as
        // an empty string column. Python lists (the list[str] path) carry no
        // `.dtype` and are untouched here.
        if let Ok(dtype) = array.getattr("dtype")
            && let Ok(kind) = dtype.getattr("kind").and_then(|k| k.extract::<String>())
            && kind == "O"
        {
            return Err(crate::error::dtype_reject(key, array));
        }

        // Matched-dtype, C-contiguous numpy arrays get forged into a
        // foreign-backed Column (zero memcpy). When the layout forbids
        // forging, the same numpy array is copied into a Rust-owned column.
        // Integer widths are preserved: i64 stays i64. A float column is
        // always stored as `F` (f64), so a narrow float **array handed in
        // here** is widened to f64 — that is input normalization at the
        // boundary, the same thing the trajectory readers do for a
        // float32-on-disk `time`. It is not the same as reading a store whose
        // arrays are *physically* float16/float32: those are refused.
        if let Ok(dtype) = array.getattr("dtype")
            && dtype.getattr("kind")?.extract::<String>()? == "f"
            && dtype.getattr("itemsize")?.extract::<usize>()? < std::mem::size_of::<F>()
        {
            let promoted = array.call_method1("astype", ("float64",))?;
            return self.insert_any(key, &promoted, validity);
        }
        macro_rules! try_exact {
            ($t:ty, $holder:expr) => {
                if let Ok(pyarr) = array.cast::<PyArrayDyn<$t>>() {
                    if let Some(col) = try_forge_foreign_column(pyarr, $holder) {
                        return self.insert_column(key, col, validity);
                    }
                    return self.insert_column(key, column_from_py(array)?, validity);
                }
            };
        }
        try_exact!(F, Column::from);
        try_exact!(I, Column::from);
        try_exact!(i64, Column::from);
        try_exact!(i16, Column::from);
        try_exact!(i8, Column::from);
        try_exact!(Idx, Column::from);
        try_exact!(u32, Column::from);
        try_exact!(u16, Column::from);
        try_exact!(u8, Column::from);
        try_exact!(bool, Column::from);
        try_exact!(Complex<f64>, Column::from);
        try_exact!(Complex<f32>, Column::from);

        if let Ok(strings) = array.extract::<Vec<String>>() {
            return self.insert_array::<String>(key, Array1::from(strings).into_dyn(), validity);
        }
        Err(crate::error::dtype_reject(key, array))
    }

    /// Insert a typed ndarray column, validating row count and — when a
    /// validity mask is given — that the mask covers exactly those rows.
    fn insert_array<T: BlockDtype>(
        &mut self,
        key: &str,
        array: ndarray::ArrayD<T>,
        validity: Option<Vec<bool>>,
    ) -> PyResult<()> {
        self.inner
            .with_mut(|b| {
                b.insert(key, array, validity)
                    .map_err(|e| PyValueError::new_err(e.to_string()))
            })
            .map_err(ffi_error_to_pyerr)??;
        Ok(())
    }

    /// Install a pre-built `Column` into the store, validating row count.
    ///
    /// Parallel to [`insert_array`](Self::insert_array), which takes an
    /// `ArrayD<T>` and wraps it into a Rust-owned Column. Use this when the
    /// caller already holds a Column (typically a foreign-backed one from
    /// [`try_forge_foreign_column`]).
    ///
    /// A `validity` mask is checked against the column's rows before anything
    /// is installed, so a refused mask leaves the block unchanged.
    fn insert_column(
        &mut self,
        key: &str,
        col: Column,
        validity: Option<Vec<bool>>,
    ) -> PyResult<()> {
        self.inner
            .with_mut(|b| {
                let rows = col.n_rows().unwrap_or(0);
                if let Some(mask) = &validity
                    && mask.len() != rows
                {
                    return Err(BlockError::ValidityLength {
                        key: key.to_owned(),
                        expected: rows,
                        got: mask.len(),
                    });
                }
                b.insert(key, col, validity)
            })
            .map_err(ffi_error_to_pyerr)?
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

/// A validity mask from a 1-D bool numpy array or a sequence of bools.
///
/// numpy hands out `np.bool_`, which is not a Python `bool`, so the array
/// cast is tried before the generic sequence extraction.
fn bool_mask(key: &str, mask: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    if let Ok(arr) = mask.cast::<PyArray1<bool>>() {
        return Ok(arr.readonly().as_array().to_vec());
    }
    mask.extract::<Vec<bool>>().map_err(|_| {
        PyTypeError::new_err(format!(
            "validity for column '{key}' must be a 1-D bool array or a \
             sequence of bools"
        ))
    })
}

/// An `(N, 3)` float64 array from any array-like, for a coordinate write.
pub(crate) fn coords_array(value: &Bound<'_, PyAny>) -> PyResult<ndarray::Array2<F>> {
    let numpy = value.py().import("numpy")?;
    let array = numpy.call_method1("asarray", (value, numpy.getattr("float64")?))?;
    let array = array.cast::<PyArray2<F>>().map_err(|_| {
        PyValueError::new_err(format!(
            "coordinates must be an (N, 3) array, got shape {}",
            array
                .getattr("shape")
                .and_then(|s| s.str())
                .map(|s| s.to_string())
                .unwrap_or_default()
        ))
    })?;
    Ok(array.readonly().as_array().to_owned())
}

/// A coordinate read's error: a missing axis column is a `KeyError`.
pub(crate) fn coords_error(e: BlockError) -> PyErr {
    match e {
        BlockError::MissingColumn { key } => PyKeyError::new_err(key),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Numpy ``str`` array of a string column's row-major values, at `shape`.
///
/// Built outside the store borrow. ``reshape`` keeps a rank above 1; it does
/// not flatten the column down to a vector.
pub(crate) fn numpy_string_array(
    py: Python<'_>,
    flat: &[String],
    shape: Vec<usize>,
) -> PyResult<Py<PyAny>> {
    // The dtype is explicit: numpy reads an empty list as float64.
    let numpy = py.import("numpy")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", numpy.getattr("str_")?)?;
    let arr = numpy.call_method("asarray", (PyList::new(py, flat)?,), Some(&kwargs))?;
    Ok(arr.call_method1("reshape", (shape,))?.unbind())
}

/// The `KeyError` for a column `key` the block lacks, naming what it has.
fn missing_column(block: &CoreBlock, key: &str) -> PyErr {
    let names: Vec<&str> = block.keys().collect();
    PyKeyError::new_err(format!(
        "no column {key:?}; available: {}",
        names.join(", ")
    ))
}

/// The names of a tuple / list key; empty is a `KeyError`.
fn column_names(key: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let names = key
        .try_iter()?
        .map(|name| extract_column_key(&name?))
        .collect::<PyResult<Vec<_>>>()?;
    if names.is_empty() {
        return Err(PyKeyError::new_err("Empty list not allowed for indexing"));
    }
    Ok(names)
}

/// Store a canonical column at the dtype the Frame schema declares.
///
/// Width is not semantics: ``np.arange(n)`` yields int64 because that is
/// numpy's default, not because the caller meant a signed 64-bit id. When the
/// values are representable in the declared dtype (`F`, `I` or `Idx`), adopt
/// it. When they are not — a negative under an unsigned key, a fractional
/// value under an integer one — that *is* semantics, and it raises. A key the
/// schema does not declare, and a non-numeric array, pass through untouched.
fn adopt_schema_dtype<'py>(key: &str, array: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let Some(spec) = molrs::core::schema::column(key) else {
        return Ok(array.clone());
    };
    let py = array.py();
    let want = match spec.dtype {
        DType::Float => numpy::dtype::<F>(py),
        DType::Int => numpy::dtype::<I>(py),
        DType::Uint => numpy::dtype::<Idx>(py),
        DType::I64 => numpy::dtype::<i64>(py),
        _ => return Ok(array.clone()),
    };
    let have = array.getattr("dtype")?;
    let kind: String = have.getattr("kind")?.extract()?;
    if !matches!(kind.as_str(), "u" | "i" | "f" | "b") || have.eq(&want)? {
        return Ok(array.clone());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("casting", "unsafe")?;
    let numpy = py.import("numpy")?;
    // The conversion is a probe, judged by the round trip below: numpy's
    // "invalid value" / overflow warnings on it are not the caller's business.
    let quiet = PyDict::new(py);
    quiet.set_item("invalid", "ignore")?;
    quiet.set_item("over", "ignore")?;
    let errstate = numpy.call_method("errstate", (), Some(&quiet))?;
    errstate.call_method0("__enter__")?;
    let probe = (|| -> PyResult<_> {
        let converted = array.call_method("astype", (&want,), Some(&kwargs))?;
        // Both directions: a value that wraps (-1 under u64) compares unequal
        // to its conversion, and one the target rounds (2**53 + 1 under f64)
        // does not come back.
        let back = converted.call_method1("astype", (&have,))?;
        Ok((converted, back))
    })();
    errstate.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
    let (converted, back) = probe?;
    let survives = numpy
        .call_method1("array_equal", (&converted, array))?
        .is_truthy()?
        && numpy
            .call_method1("array_equal", (back, array))?
            .is_truthy()?;
    if !survives {
        return Err(PyValueError::new_err(format!(
            "column {key:?} is declared {:?} by the Frame schema, and the given {} \
             values do not survive the conversion",
            spec.dtype.to_string(),
            have.str()?
        )));
    }
    Ok(converted)
}

/// Owned Column conversion shared by block fallbacks and observable values.
/// Storage widths survive; only narrow real floats normalize to core f64.
pub(crate) fn column_from_py(value: &Bound<'_, PyAny>) -> PyResult<Column> {
    let array = value
        .py()
        .import("numpy")?
        .call_method1("asarray", (value,))?;
    let kind: String = array.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "f"
        && array
            .getattr("dtype")?
            .getattr("itemsize")?
            .extract::<usize>()?
            < 8
    {
        return column_from_py(&array.call_method1("astype", ("float64",))?);
    }
    macro_rules! exact {
        ($t:ty) => {
            if let Ok(a) = array.cast::<PyArrayDyn<$t>>() {
                return Ok(Column::from(a.readonly().as_array().to_owned()));
            }
        };
    }
    exact!(f64);
    exact!(i8);
    exact!(i16);
    exact!(i32);
    exact!(i64);
    exact!(u8);
    exact!(u16);
    exact!(u32);
    exact!(u64);
    exact!(bool);
    exact!(Complex<f32>);
    exact!(Complex<f64>);
    if kind == "U" {
        let shape: Vec<usize> = array.getattr("shape")?.extract()?;
        let strings: Vec<String> = array
            .call_method0("flatten")?
            .call_method0("tolist")?
            .extract()?;
        return Ok(Column::from(
            ArrayD::from_shape_vec(IxDyn(&shape), strings)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        ));
    }
    Err(PyTypeError::new_err(
        "values must have a supported numeric, bool, complex or string dtype",
    ))
}

/// Forge a foreign-backed [`Column`] from a numpy array without copying its
/// buffer.
///
/// Returns `Some(col)` when the numpy array is safe to alias:
///
///   * rank >= 1 (Block rejects rank-0 anyway),
///   * C-contiguous (row-major — `ArrayD::from_shape_vec` assumes this),
///   * non-empty (degenerate zero-length arrays are cheaper to copy).
///
/// Returns `None` otherwise. Callers decide how to recover — typically by
/// copying the array through [`PyBlock::insert_array`].
///
/// The forged Column holds a `ColumnArray::from_foreign` over numpy's
/// buffer, pinned alive by a `Py<PyArrayDyn<T>>` keeper. Mutations are
/// visible in both directions; Rust-side mutation via `as_*_mut` triggers
/// copy-on-write inside the column array.
fn try_forge_foreign_column<T>(
    pyarr: &Bound<'_, PyArrayDyn<T>>,
    wrap: fn(ColumnArray<T>) -> Column,
) -> Option<Column>
where
    T: numpy::Element + BlockDtype + Clone,
{
    if pyarr.ndim() == 0 || !pyarr.is_c_contiguous() {
        return None;
    }

    let shape: Vec<usize> = pyarr.shape().to_vec();
    let total: usize = shape.iter().product();
    if total == 0 {
        return None;
    }

    let ptr = pyarr.data();
    // SAFETY:
    //   * `ptr` points to `total` elements of `T` owned by numpy; the buffer
    //     stays alive through the `keeper` refcount stored on the column array.
    //   * `cap == len == total`, so ndarray never calls realloc/reserve on
    //     the forged Vec.
    //   * `ColumnArray::from_foreign` wraps the inner ArrayD in ManuallyDrop,
    //     suppressing the forged Vec's Drop. Only the keeper's Drop runs,
    //     which decrefs numpy and lets numpy free the buffer through its own
    //     allocator.
    let forged = unsafe {
        let vec = Vec::from_raw_parts(ptr, total, total);
        match ArrayD::from_shape_vec(IxDyn(&shape), vec) {
            Ok(arr) => arr,
            Err(_) => {
                // Can't happen (len == product(shape) by construction), but
                // if it did, leak the Vec rather than double-free: the numpy
                // keeper would still release the memory on drop, and a Vec
                // drop here would run Rust's allocator over foreign memory.
                std::mem::forget(Vec::from_raw_parts(ptr, total, total));
                return None;
            }
        }
    };
    let keeper: Py<PyArrayDyn<T>> = pyarr.clone().unbind();
    // SAFETY: keeper pins numpy's buffer for the column array's lifetime.
    let holder = unsafe { ColumnArray::from_foreign(forged, keeper) };
    Some(wrap(holder))
}

// ---------------------------------------------------------------------------
// Zero-copy numpy views backed by Arc-shared Rust storage
// ---------------------------------------------------------------------------

/// Keep a typed column buffer alive for a numpy view of any storage width.
#[pyclass(module = "molpy._core", unsendable, subclass)]
struct ArrayOwner {
    _keep: Box<dyn std::any::Any + Send + Sync>,
}

/// Zero-copy numpy view of a typed column. The owner holds an `Arc` clone of
/// the column array so the buffer outlives the Python array.
fn typed_array_view<T>(
    py: Python<'_>,
    array: Arc<ColumnArray<T>>,
) -> PyResult<Py<pyo3::types::PyAny>>
where
    T: numpy::Element + 'static,
{
    let owner = Py::new(
        py,
        ArrayOwner {
            _keep: Box::new(Arc::clone(&array)),
        },
    )?;
    let owner = owner.into_bound(py);
    let view = unsafe { PyArrayDyn::<T>::borrow_from_array(array.array(), owner.into_any()) };
    Ok(view.into_any().unbind())
}
