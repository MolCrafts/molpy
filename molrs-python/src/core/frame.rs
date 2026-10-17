//! Python wrapper for `Frame`, a hierarchical data container backed by the
//! shared FFI store.
//!
//! A [`PyFrame`] maps string keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to
//! [`PyBlock`] column stores. It may optionally carry a [`PyBox`] (simulation
//! box) and an exact-dtype metadata map.
//!
//! # Conventional Block Layout
//!
//! | Block key   | Expected columns                                      | Notes                                    |
//! |-------------|-------------------------------------------------------|------------------------------------------|
//! | `"atoms"`   | `symbol` (str), `x`/`y`/`z` (float), `mass` (float)  | Atom positions and properties             |
//! | `"bonds"`   | `atomi`/`atomj` (uint), `order` (float)               | Bond topology (indices into atoms)        |
//! | `"angles"`  | `atomi`/`atomj`/`atomk` (uint), `type` (int)          | Angle topology                            |
//!
//! The frame itself does **not** enforce cross-block row consistency; that is
//! the caller's responsibility (use [`PyFrame::validate`] to check).

use crate::core::block::{PyBlock, coords_array, coords_error};
use crate::core::simbox::PyBox;
use crate::error::ffi_error_to_pyerr;
use crate::error::molrs_error_to_pyerr;
use molrs::core::Frame as CoreFrame;
use molrs::core::{MetaMap, MetaValue};
use molrs_ffi::FrameRef;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyCapsule, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::Value as JsonValue;

/// Exact-dtype frame metadata value.
#[pyclass(
    module = "molrs.core",
    name = "MetaValue",
    frozen,
    from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyMetaValue {
    pub(crate) inner: MetaValue,
}

#[pymethods]
impl PyMetaValue {
    /// Construct a metadata value from a stable dtype tag and payload.
    #[new]
    fn new(dtype: &str, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            inner: meta_value_from_dtype(dtype, value)?,
        })
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(slf.as_any(), (this.dtype(), this.value(slf.py())?))
    }

    #[getter]
    fn value(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Plain, not frozen: this payload is what `__reduce__` pickles, and a
        // `MetaDocument` has no constructor. `.value` is not a `frame.meta` door.
        meta_value_to_py(py, &self.inner, JsonForm::Plain)
    }

    fn __repr__(&self) -> String {
        format!(
            "MetaValue(dtype='{}', value={:?})",
            self.inner.dtype(),
            self.inner
        )
    }
}

/// Collect a mapping (or an iterable of pairs) into a fresh [`MetaMap`].
///
/// A plain value is re-inferred. [`MetaValue`](PyMetaValue) fixes the dtype of
/// that one value — it does not pin the key. A tag survives a round trip
/// through a `*.mrec` frame or system (`_meta_types`), a declared sequence
/// schema and the serde frame document. Another frame's metadata copies across
/// whole, tags and all.
fn mapping_to_meta_map(source: &Bound<'_, PyAny>) -> PyResult<MetaMap> {
    // Another frame's metadata copies across whole, tags and all: going through
    // plain values would silently widen every non-default dtype.
    if let Ok(view) = source.extract::<PyRef<'_, PyFrameMeta>>() {
        return view
            .inner
            .with(|f| f.meta.clone())
            .map_err(ffi_error_to_pyerr);
    }
    let mut map = MetaMap::new();
    if source.hasattr("keys")? {
        for key in source.call_method0("keys")?.try_iter()? {
            let key = key?;
            let name: String = key.extract()?;
            map.insert(name, infer_meta_value(&source.get_item(&key)?)?);
        }
        return Ok(map);
    }
    for pair in source.try_iter()? {
        let (name, value): (String, Bound<'_, PyAny>) = pair?.extract()?;
        map.insert(name, infer_meta_value(&value)?);
    }
    Ok(map)
}

/// Live, write-through view of a frame's metadata.
///
/// Every door hands back a frozen value. Scalars unwrap, a fixed-length vector
/// is a `tuple` (not a `list`), a JSON array — including one nested inside a
/// document — is a `tuple`, and a JSON object is a [`MetaDocument`](PyMetaDocument).
/// Writing a value lands in the frame. [`copy`](Self::copy) is the only mutable
/// container this surface hands out; its values are still frozen, and
/// [`MetaDocument::copy`](PyMetaDocument::copy) is that unfreeze one level down.
///
/// [`dtype`](Self::dtype) reports the tag of the value stored right now; any
/// plain write re-infers it. [`MetaValue`](PyMetaValue) fixes the dtype of that
/// write only — it does not pin the key. A tag survives a round trip through a
/// `*.mrec` frame or system (`_meta_types`), a declared sequence schema and the
/// serde frame document.
///
/// `frame.meta["run"]["step"] = 3` raises `TypeError`. Read, unfreeze, write
/// back: `doc = frame.meta["run"].copy(); doc["step"] = 3; frame.meta["run"] = doc`.
/// `json.dumps` accepts a tuple and rejects a document; the idiom is
/// `json.dumps(frame.meta["run"].copy())`.
///
/// Iteration order of this mapping is insertion order. Order inside a nested
/// [`MetaDocument`](PyMetaDocument) is unspecified.
///
/// [`keys`](Self::keys), [`values`](Self::values), and [`items`](Self::items)
/// are live `collections.abc` views in insertion order. A non-`str` lookup is
/// absent; a non-`str` write raises `TypeError`. Deleting a not-yet-visited
/// key while iterating `values()` or `items()` raises `KeyError`.
#[pyclass(module = "molrs.core", name = "FrameMeta", unsendable, subclass)]
pub struct PyFrameMeta {
    inner: FrameRef,
}

/// A lookup key this map cannot hold is absent, not a type error.
fn lookup_key(key: &Bound<'_, PyAny>) -> Option<String> {
    if key.cast::<PyString>().is_err() {
        return None;
    }
    key.extract::<String>().ok()
}

fn missing_key(key: &Bound<'_, PyAny>) -> PyErr {
    PyKeyError::new_err(key.clone().unbind())
}

impl PyFrameMeta {
    /// `collections.abc` is imported on every call so a later rebinding is visible.
    fn abc_view<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        class_name: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import("collections.abc")?
            .getattr(class_name)?
            .call1((slf,))
    }

    fn tag_of(&self, key: &str) -> PyResult<Option<&'static str>> {
        self.inner
            .with(|f| f.meta.get(key).map(MetaValue::dtype))
            .map_err(ffi_error_to_pyerr)
    }

    fn insert_meta(&mut self, key: &str, value: MetaValue) -> PyResult<()> {
        self.inner
            .with_meta_mut(|meta| {
                meta.insert(key, value);
            })
            .map_err(ffi_error_to_pyerr)
    }

    fn take(&mut self, key: &str) -> PyResult<Option<MetaValue>> {
        self.inner
            .with_meta_mut(|meta| meta.remove(key))
            .map_err(ffi_error_to_pyerr)
    }

    fn snapshot(&self) -> PyResult<Vec<(String, MetaValue)>> {
        self.inner
            .with(|f| {
                f.meta
                    .iter()
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect()
            })
            .map_err(ffi_error_to_pyerr)
    }

    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.snapshot()? {
            dict.set_item(key, meta_value_to_py(py, &value, JsonForm::Frozen)?)?;
        }
        Ok(dict)
    }

    /// Infer with no borrow held, then one `extend`. `update(self)` re-enters
    /// this mapping; a `&mut self` or `with_mut` held across that read panics.
    fn absorb(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut pairs = Vec::new();
        if other.hasattr("keys")? {
            for key in other.call_method0("keys")?.try_iter()? {
                let key = key?;
                let name: String = key.extract()?;
                let value = other.get_item(&key)?;
                pairs.push((name, infer_meta_value(&value)?));
            }
        } else {
            for pair in other.try_iter()? {
                let (name, value): (String, Bound<'_, PyAny>) = pair?.extract()?;
                pairs.push((name, infer_meta_value(&value)?));
            }
        }
        slf.borrow()
            .inner
            .with_meta_mut(|meta| {
                meta.extend(pairs);
            })
            .map_err(ffi_error_to_pyerr)
    }
}

#[pymethods]
impl PyFrameMeta {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Some(name) = lookup_key(key) else {
            return Err(missing_key(key));
        };
        match self
            .inner
            .with(|f| f.meta.get(&name).cloned())
            .map_err(ffi_error_to_pyerr)?
        {
            Some(value) => meta_value_to_py(py, &value, JsonForm::Frozen),
            None => Err(PyKeyError::new_err(name)),
        }
    }

    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let typed = infer_meta_value(value)?;
        self.insert_meta(key, typed)
    }

    fn __delitem__(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let Some(name) = lookup_key(key) else {
            return Err(missing_key(key));
        };
        match self.take(&name)? {
            Some(_) => Ok(()),
            None => Err(PyKeyError::new_err(name)),
        }
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Some(name) = lookup_key(key) else {
            return Ok(false);
        };
        self.inner
            .with(|f| f.meta.contains_key(&name))
            .map_err(ffi_error_to_pyerr)
    }

    fn __len__(&self) -> PyResult<usize> {
        self.inner
            .with(|f| f.meta.len())
            .map_err(ffi_error_to_pyerr)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        // `KeysView` iterates the mapping, so this must not call `keys()`.
        let keys = self
            .inner
            .with(|f| f.meta.keys().cloned().collect::<Vec<String>>())
            .map_err(ffi_error_to_pyerr)?;
        PyList::new(py, keys)?.try_iter()
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.as_dict(py)?.as_any().eq(other)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("{}", self.as_dict(py)?))
    }

    /// The dtype tag stored for `key`, or ``None`` when the key is absent.
    fn dtype(&self, key: &str) -> PyResult<Option<&'static str>> {
        self.tag_of(key)
    }

    fn keys<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "KeysView")
    }

    fn values<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "ValuesView")
    }

    fn items<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "ItemsView")
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let Some(name) = lookup_key(key) else {
            return Ok(default.unwrap_or_else(|| py.None()));
        };
        match self
            .inner
            .with(|f| f.meta.get(&name).cloned())
            .map_err(ffi_error_to_pyerr)?
        {
            Some(value) => meta_value_to_py(py, &value, JsonForm::Frozen),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    #[pyo3(signature = (key, *default))]
    fn pop(
        &mut self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: &Bound<'_, pyo3::types::PyTuple>,
    ) -> PyResult<Py<PyAny>> {
        let Some(name) = lookup_key(key) else {
            return if default.len() == 1 {
                Ok(default.get_item(0)?.unbind())
            } else {
                Err(missing_key(key))
            };
        };
        match self.take(&name)? {
            Some(value) => meta_value_to_py(py, &value, JsonForm::Frozen),
            None if default.len() == 1 => Ok(default.get_item(0)?.unbind()),
            None => Err(PyKeyError::new_err(name)),
        }
    }

    fn popitem(&mut self, py: Python<'_>) -> PyResult<(String, Py<PyAny>)> {
        let popped = self
            .inner
            .with_meta_mut(|meta| -> Option<(String, MetaValue)> {
                let (key, value) = meta
                    .iter()
                    .next_back()
                    .map(|(key, value)| (key.clone(), value.clone()))?;
                let _ = meta.remove(&key);
                Some((key, value))
            })
            .map_err(ffi_error_to_pyerr)?;
        match popped {
            Some((key, value)) => Ok((key, meta_value_to_py(py, &value, JsonForm::Frozen)?)),
            None => Err(PyKeyError::new_err("popitem(): metadata is empty")),
        }
    }

    fn clear(&mut self) -> PyResult<()> {
        self.inner
            .with_meta_mut(|meta| meta.clear())
            .map_err(ffi_error_to_pyerr)
    }

    #[pyo3(signature = (key, default=None))]
    fn setdefault(
        &mut self,
        py: Python<'_>,
        key: &str,
        default: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if let Some(value) = self
            .inner
            .with(|f| f.meta.get(key).cloned())
            .map_err(ffi_error_to_pyerr)?
        {
            return meta_value_to_py(py, &value, JsonForm::Frozen);
        }
        // `dict.setdefault(k)` inserts None; so does this, since None is a
        // JSON null rather than a rejection.
        let value = match default {
            Some(value) => value,
            None => py.None().into_bound(py),
        };
        let typed = infer_meta_value(&value)?;
        self.insert_meta(key, typed.clone())?;
        meta_value_to_py(py, &typed, JsonForm::Frozen)
    }

    #[pyo3(signature = (other=None, **kwargs))]
    fn update(
        slf: &Bound<'_, Self>,
        other: Option<&Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        if let Some(other) = other {
            Self::absorb(slf, other)?;
        }
        if let Some(kwargs) = kwargs {
            Self::absorb(slf, kwargs.as_any())?;
        }
        Ok(())
    }

    /// A plain `dict` snapshot; mutating it does not touch the frame.
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.as_dict(py)
    }

    /// The same snapshot with every dtype kept, as `dict[str, MetaValue]`.
    ///
    /// `dict(meta)` throws the tags away, which is the right default for
    /// reading; this is what a copy or a pickle has to carry instead.
    fn typed<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, value) in self.snapshot()? {
            dict.set_item(key, Py::new(py, PyMetaValue { inner: value })?)?;
        }
        Ok(dict)
    }

    fn __or__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let merged = self.as_dict(py)?;
        merged.update(other.cast::<pyo3::types::PyMapping>()?)?;
        Ok(merged)
    }

    fn __ror__<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let merged = other.cast::<PyDict>()?.copy()?;
        merged.update(self.as_dict(py)?.as_mapping())?;
        Ok(merged)
    }

    fn __ior__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<()> {
        Self::absorb(slf, other)
    }
}

/// Frozen JSON object returned by every ``frame.meta`` door.
///
/// A fixed-length vector comes back as a ``tuple`` and a JSON object as a
/// ``MetaDocument``. Nested arrays are tuples; nested objects are documents.
/// Item assignment raises ``TypeError`` — the snapshot does not write through.
/// ``copy()`` is the unfreeze: a deep plain ``dict`` whose nested documents are
/// dicts and whose nested arrays are lists. ``json.dumps`` rejects a document
/// and accepts that copy: ``json.dumps(frame.meta["run"].copy())``.
///
/// Iteration order is unspecified. ``frame.meta`` itself enumerates in
/// insertion order; the two levels differ.
#[pyclass(module = "molrs.core", name = "MetaDocument", frozen, subclass)]
pub struct PyMetaDocument {
    inner: serde_json::Map<String, JsonValue>,
}

impl PyMetaDocument {
    fn decode<'py>(&self, py: Python<'py>, form: JsonForm) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (key, item) in &self.inner {
            dict.set_item(key, json_to_py(py, item, form)?)?;
        }
        Ok(dict)
    }

    /// `collections.abc` is imported on every call so a later rebinding is visible.
    fn abc_view<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
        class_name: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        py.import("collections.abc")?
            .getattr(class_name)?
            .call1((slf,))
    }
}

#[pymethods]
impl PyMetaDocument {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Some(name) = lookup_key(key) else {
            return Err(missing_key(key));
        };
        match self.inner.get(&name) {
            Some(value) => json_to_py(py, value, JsonForm::Frozen),
            None => Err(PyKeyError::new_err(name)),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyIterator>> {
        let keys: Vec<String> = self.inner.keys().cloned().collect();
        PyList::new(py, keys)?.try_iter()
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool {
        lookup_key(key).is_some_and(|name| self.inner.contains_key(&name))
    }

    fn keys<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "KeysView")
    }

    fn values<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "ValuesView")
    }

    fn items<'py>(slf: &Bound<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Self::abc_view(slf, py, "ItemsView")
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let Some(name) = lookup_key(key) else {
            return Ok(default.unwrap_or_else(|| py.None()));
        };
        match self.inner.get(&name) {
            Some(value) => json_to_py(py, value, JsonForm::Frozen),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// Frozen-side: one level of [`JsonForm::Frozen`], then `dict.__eq__`.
    ///
    /// A nested document compares through the same method. An array member
    /// equals a tuple and not a list. When `other` is a document, `dict.__eq__`
    /// returns `NotImplemented` and the reflected call lands here with a plain
    /// dict on the right.
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.decode(py, JsonForm::Frozen)?.as_any().eq(other)
    }

    /// Written out: a slot that answers only `==` makes `!=` fall back to identity.
    fn __ne__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.decode(py, JsonForm::Frozen)?.as_any().ne(other)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "MetaDocument({})",
            self.decode(py, JsonForm::Frozen)?
        ))
    }

    /// Deep plain decode. Nested documents become dicts, nested arrays lists.
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.decode(py, JsonForm::Plain)
    }

    /// Pickle by content: the plain [`copy`](Self::copy), rebuilt through
    /// [`_from_plain`](Self::_from_plain) — a document has no constructor.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let plain = slf.get().decode(py, JsonForm::Plain)?;
        Ok((
            slf.get_type().getattr("_from_plain")?,
            PyTuple::new(py, [plain])?,
        ))
    }

    /// Freeze a plain dict back into a document; the pickle reconstructor.
    #[staticmethod]
    fn _from_plain(plain: &Bound<'_, PyDict>) -> PyResult<Self> {
        match py_to_json(plain.as_any(), 0)? {
            JsonValue::Object(inner) => Ok(Self { inner }),
            _ => Err(PyTypeError::new_err("MetaDocument needs a dict")),
        }
    }
}

/// Hierarchical data container exposed to Python as `molrs.core.Frame`.
///
/// A `Frame` is a dictionary of named [`Block`](crate::core::block::PyBlock)s with
/// optional simulation box and metadata. It is the primary exchange format for
/// molecular data across the molrs ecosystem, and the only `Frame` class:
/// every reader, graph serialiser and builder returns it.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs.core import Box
/// from molrs.core import Frame
///
/// frame = Frame(
///     {"atoms": {"element": ["O", "H", "H"], "x": [0.0, 0.76, -0.76]}},
///     meta={"title": "water"},
///     box=Box.cube(10.0),
/// )
/// frame["atoms"]["y"] = np.zeros(3)   # writes into the stored block
/// print(frame)          # Frame(blocks=['atoms'], box=yes)
/// ```
#[pyclass(
    module = "molrs.core",
    name = "Frame",
    from_py_object,
    unsendable,
    subclass
)]
#[derive(Clone)]
pub struct PyFrame {
    pub(crate) inner: FrameRef,
}

#[pymethods]
impl PyFrame {
    /// Create a frame from blocks, metadata and a box (all optional).
    ///
    /// Copying an existing frame is :meth:`copy`, not ``Frame(frame)``.
    ///
    /// Parameters
    /// ----------
    /// blocks : Mapping[str, Block | Mapping[str, ArrayLike]], optional
    ///     Block name -> block; a mapping value is built as ``Block(value)``.
    /// meta : Mapping[str, Any], optional
    ///     Metadata, as for the :attr:`meta` setter.
    /// box : Box, optional
    ///     The simulation box.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``blocks`` is a ``Frame`` or not a mapping, or a block name is
    ///     not a ``str``.
    ///
    /// Examples
    /// --------
    /// >>> Frame({"atoms": {"x": [0.0, 1.0]}}, meta={"step": 0}).keys()
    /// ['atoms']
    #[new]
    #[pyo3(signature = (blocks = None, *, meta = None, r#box = None))]
    fn new(
        blocks: Option<&Bound<'_, PyAny>>,
        meta: Option<&Bound<'_, PyAny>>,
        r#box: Option<PyRef<'_, PyBox>>,
    ) -> PyResult<Self> {
        let mut frame = Self {
            inner: FrameRef::default(),
        };
        if let Some(blocks) = blocks {
            if blocks.cast::<PyFrame>().is_ok() {
                return Err(PyTypeError::new_err(
                    "Frame() takes a mapping of block name -> block; copy a frame \
                     with frame.copy()",
                ));
            }
            if !blocks.hasattr("keys")? {
                return Err(PyTypeError::new_err(format!(
                    "Frame() takes a mapping of block name -> block, got {}",
                    blocks.get_type().name()?
                )));
            }
            for name in blocks.call_method0("keys")?.try_iter()? {
                let name = name?;
                let block = blocks.get_item(&name)?;
                let name: String = name
                    .extract()
                    .map_err(|_| PyTypeError::new_err("Frame block names must be str"))?;
                frame.__setitem__(&name, &block)?;
            }
        }
        if let Some(meta) = meta {
            frame.set_meta(meta)?;
        }
        if let Some(simbox) = r#box {
            frame.set_box(Some(&simbox))?;
        }
        Ok(frame)
    }

    /// Retrieve a block by name: a handle on the stored block.
    ///
    /// Every :class:`Block` member reads and writes this frame's data through
    /// it (``frame["atoms"]["x"] = …`` lands in the frame). A handle goes
    /// stale once the block is replaced or restructured through another
    /// handle; read it again from the frame.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name (e.g. ``"atoms"``).
    ///
    /// Returns
    /// -------
    /// Block
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> atoms = frame["atoms"]
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Py<PyAny>> {
        if let Ok(inner) = self.inner.block(key) {
            return Ok(Py::new(py, PyBlock { inner })?.into_any());
        }
        Err(PyKeyError::new_err(key.to_string()))
    }

    /// Assign a block under the given name.
    ///
    /// If a block with the same key already exists it is replaced. A mapping
    /// of column name -> array is built as ``Block(value)`` first.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    /// value : Block | Mapping[str, ArrayLike]
    ///     The block to store.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``value`` is neither a ``Block`` nor a mapping.
    ///
    /// Examples
    /// --------
    /// >>> frame["atoms"] = {"x": [0.0, 1.0]}
    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let core_block = if let Ok(block) = value.extract::<PyRef<'_, PyBlock>>() {
            block.clone_core_block()?
        } else if value.hasattr("keys")? && value.cast::<PyFrame>().is_err() {
            PyBlock::from_mapping(value)?.clone_core_block()?
        } else {
            return Err(PyTypeError::new_err(format!(
                "a frame block must be a Block or a mapping of column name -> \
                 array, got {}",
                value.get_type().name()?
            )));
        };
        self.inner
            .arena
            .borrow_mut()
            .set_block(self.inner.id, key, core_block)
            .map_err(ffi_error_to_pyerr)
    }

    /// Delete a block by name.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name to remove.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If ``key`` does not exist.
    ///
    /// Examples
    /// --------
    /// >>> del frame["bonds"]
    fn __delitem__(&mut self, key: &str) -> PyResult<()> {
        self.inner
            .arena
            .borrow_mut()
            .del_block(self.inner.id, key)
            .map_err(ffi_error_to_pyerr)
    }

    /// Test whether a block name is present.
    ///
    /// Parameters
    /// ----------
    /// key : str
    ///     Block name.
    ///
    /// Returns
    /// -------
    /// bool
    ///
    /// Examples
    /// --------
    /// >>> "atoms" in frame
    /// True
    fn __contains__(&self, key: &str) -> PyResult<bool> {
        self.with_frame(|f| f.contains_key(key))
    }

    /// Number of blocks stored in this frame.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> PyResult<usize> {
        self.with_frame(|f| f.len())
    }

    /// List all block names.
    ///
    /// Returns
    /// -------
    /// list[str]
    fn keys(&self) -> PyResult<Vec<String>> {
        self.with_frame(|f| f.keys().map(|s| s.to_string()).collect())
    }

    /// The simulation :class:`Box` attached to this frame, or ``None``.
    ///
    ///
    /// Returns
    /// -------
    /// Box | None
    ///     Periodic simulation box, if set.
    ///
    /// Examples
    /// --------
    /// >>> if frame.box is not None:
    /// ...     print(frame.box.volume())
    #[getter]
    fn get_box(&self) -> PyResult<Option<PyBox>> {
        Ok(self
            .inner
            .box_clone()
            .map_err(ffi_error_to_pyerr)?
            .map(|inner| PyBox { inner }))
    }

    /// Set (or clear) the simulation :class:`Box`.
    ///
    /// Parameters
    /// ----------
    /// box : Box | None
    ///     Pass ``None`` to remove the simulation box.
    ///
    /// Examples
    /// --------
    /// >>> frame.box = Box.cube(20.0)
    /// >>> frame.box = None  # remove
    #[setter]
    fn set_box(&mut self, box_: Option<&PyBox>) -> PyResult<()> {
        self.inner
            .set_box(box_.map(|sb| sb.inner.clone()))
            .map_err(ffi_error_to_pyerr)
    }

    /// Live, write-through view of this frame's metadata.
    ///
    /// Every door hands back a frozen value: scalars unwrap, a fixed-length
    /// vector is a ``tuple``, a JSON array is a ``tuple``, and a JSON object
    /// is a :class:`MetaDocument`. ``frame.meta["run"]["step"] = 3`` raises
    /// ``TypeError``. ``json.dumps`` rejects a document; use
    /// ``json.dumps(frame.meta["run"].copy())``. Order inside a nested
    /// document is unspecified.
    ///
    /// Returns
    /// -------
    /// FrameMeta
    ///     Mutations of the mapping persist. ``dtype(k)`` reports the tag
    ///     stored right now; a plain write re-infers it, and a
    ///     :class:`MetaValue` fixes the dtype of that one write. A tag
    ///     survives a round trip through a ``*.mrec`` frame or system, a
    ///     declared sequence schema and the serde frame document.
    #[getter]
    fn meta(&self) -> PyFrameMeta {
        PyFrameMeta {
            inner: self.inner.clone(),
        }
    }

    /// Replace the metadata dictionary.
    ///
    /// Values may be :class:`MetaValue`, a :class:`MetaDocument`, a ``tuple``,
    /// or any JSON-serializable object (``str``, ``int``, ``float``, ``bool``,
    /// ``list``, ``dict``).
    #[setter]
    fn set_meta(&mut self, meta: &Bound<'_, PyAny>) -> PyResult<()> {
        // Build the replacement first: `frame.meta = frame.meta` and
        // `Frame.copy` both read the map they are about to overwrite.
        let map = mapping_to_meta_map(meta)?;
        self.inner
            .with_meta_mut(|meta| {
                *meta = map;
            })
            .map_err(ffi_error_to_pyerr)
    }

    /// Judge this frame against the canonical Frame schema.
    ///
    /// Delegates to ``molrs``'s ``Validator::canonical`` — dtype, shape,
    /// required columns, and endpoint ranges. Callers must not re-implement
    /// those checks in Python.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame does not conform (message is the full report).
    fn validate(&self) -> PyResult<()> {
        self.with_frame(|f| f.validate().map_err(molrs_error_to_pyerr))?
    }

    /// Return a deep copy of this frame.
    ///
    /// All blocks (new column buffers), the simulation box, and typed metadata
    /// are copied into a new, independent frame backed by its own store.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     An independent copy.
    fn copy(&self) -> PyResult<Self> {
        Self::from_core_frame(self.with_frame(CoreFrame::deep_copy)?)
    }

    /// A new frame holding rows ``rows`` of ``block``, with every relation
    /// block that indexes it cut down and renumbered.
    ///
    /// Old row ``rows[k]`` becomes new row ``k``; every column travels. A
    /// relation block whose endpoints index ``block`` (``bonds``, ``angles``,
    /// …, or any block carrying ``atomi``..``atoml``) keeps only the rows whose
    /// endpoints all lie in the selection, in their original order, with each
    /// endpoint rewritten. Every other block, the box and ``meta`` are copied
    /// unchanged. Values keep their units. This frame is never modified.
    ///
    /// Parameters
    /// ----------
    /// rows : ArrayLike
    ///     A 1-D bool mask of length ``block.nrows`` (e.g.
    ///     ``frame["atoms"]["mol_id"] == 1``), which keeps its ``True`` rows in
    ///     order, or 1-D integer row indices, which may be negative down to
    ///     ``-nrows``; the ``k``-th selected old row becomes new row ``k``.
    /// block : str, optional
    ///     The block to select from (default ``"atoms"``).
    ///
    /// Returns
    /// -------
    /// Frame
    ///     A new, independent frame (no buffer shared with this one).
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     If there is no block ``block``.
    /// IndexError
    ///     If ``rows`` is not 1-D, a mask has the wrong length, or an index is
    ///     below ``-nrows``.
    /// TypeError
    ///     If ``rows`` is neither bool nor integer.
    /// ValueError
    ///     If a row is past the end or repeated, a relation block indexing
    ///     ``block`` lacks a ``UInt`` endpoint column, or the frame carries a
    ///     ``members`` block.
    #[pyo3(signature = (rows, block = "atoms"))]
    fn subset(&self, rows: &Bound<'_, PyAny>, block: &str) -> PyResult<Self> {
        let nrows = self
            .with_frame(|f| f.get(block).map(|b| b.n_rows().unwrap_or(0)))?
            .ok_or_else(|| PyKeyError::new_err(block.to_string()))?;
        let rows = PyBlock::row_selection(rows, nrows)?;
        // `Frame::subset` already builds every block on new buffers.
        let sub = self
            .with_frame(|f| f.subset(block, &rows))?
            .map_err(molrs_error_to_pyerr)?;
        Self::from_core_frame(sub)
    }

    /// A new frame holding ``count`` copies of this one, concatenated block by
    /// block — the inverse of :meth:`subset` for ``count`` identical
    /// molecules.
    ///
    /// Copy ``c`` of row ``r`` lands at ``c * nrows + r``. Every endpoint of a
    /// relation block (``bonds``, ``angles``, …, or any block carrying
    /// ``atomi``..``atoml``) in copy ``c`` is offset by ``c`` times the row
    /// count of the block it indexes. Every other column — identifiers
    /// (``id``, ``mol_id``) included — is copied verbatim; regenerate them if
    /// the copies need distinct labels. Validity masks travel; ``meta`` and
    /// the box are copied unchanged. This frame is never modified.
    ///
    /// Parameters
    /// ----------
    /// count : int
    ///     Number of copies; ``0`` gives zero-row blocks.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     A new, independent frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a relation block indexes a block the frame lacks, lacks a
    ///     ``UInt`` endpoint column, or the frame carries a ``members`` block.
    fn replicate(&self, count: usize) -> PyResult<Self> {
        let copies = self
            .with_frame(|f| f.replicate(count))?
            .map_err(molrs_error_to_pyerr)?;
        Self::from_core_frame(copies)
    }

    /// The frames joined end to end, block by block — :meth:`replicate` for
    /// parts that differ.
    ///
    /// Every block name any part has becomes one block of the parts' rows in
    /// order; a column one part lacks is filled for its rows and marked null.
    /// Every endpoint of a relation block (``bonds``, ``angles``, …, or any
    /// block carrying ``atomi``..``atoml``) in part ``p`` is offset by the
    /// rows the block it indexes has in the parts before it. Every other
    /// column — ``id`` / ``mol_id`` included — is copied verbatim. ``meta`` and
    /// the box are the first part's. No part is modified.
    ///
    /// Parameters
    /// ----------
    /// frames : Sequence[Frame]
    ///     The parts, in order; none gives an empty frame.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     A new, independent frame.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a relation block indexes a block its part lacks or lacks a
    ///     ``UInt`` endpoint column, or two parts carry one column under
    ///     different dtypes.
    #[staticmethod]
    fn concat(frames: Vec<PyRef<'_, PyFrame>>) -> PyResult<Self> {
        let parts = frames
            .iter()
            .map(|f| f.clone_core_frame())
            .collect::<PyResult<Vec<CoreFrame>>>()?;
        let joined = CoreFrame::concat(&parts).map_err(molrs_error_to_pyerr)?;
        Self::from_core_frame(joined)
    }

    /// The ``atoms`` block's positions as an ``(N, 3)`` float64 array (a
    /// copy), gathered from its ``x`` / ``y`` / ``z`` columns.
    ///
    /// Assigning an ``(N, 3)`` array-like writes it back into ``atoms``
    /// (created when the frame has none).
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     On read, if there is no ``atoms`` block or it lacks ``x``, ``y`` or
    ///     ``z``.
    /// ValueError
    ///     On write, if the array is not ``(N, 3)`` or ``N`` differs from the
    ///     ``atoms`` row count.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<molrs::op::F>>> {
        use molrs::core::schema::block_names::ATOMS;
        use numpy::IntoPyArray;
        let xyz = self.with_frame(|f| {
            f.get(ATOMS)
                .ok_or_else(|| PyKeyError::new_err(ATOMS))
                .and_then(|atoms| atoms.coords().map_err(coords_error))
        })??;
        Ok(xyz.into_pyarray(py))
    }

    #[setter]
    fn set_coords(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let xyz = coords_array(value)?;
        self.inner
            .with_mut(|f| f.set_coords(xyz.view()))
            .map_err(ffi_error_to_pyerr)?
            .map_err(molrs_error_to_pyerr)
    }

    /// Pickle by logical state: the blocks, the typed metadata and the box.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let blocks = PyDict::new(py);
        for key in this.keys()? {
            blocks.set_item(&key, this.__getitem__(py, &key)?)?;
        }
        let state = PyDict::new(py);
        state.set_item("blocks", blocks)?;
        // The typed snapshot: `dict(meta)` would drop every dtype tag.
        state.set_item("meta", this.meta().typed(py)?)?;
        state.set_item("box", this.get_box()?)?;
        crate::pickle::reduce_with_state(slf.as_any(), PyTuple::empty(py), state.into_any())
    }

    /// Restore the state [`__reduce__`](Self::__reduce__) produced.
    fn __setstate__(&mut self, state: &Bound<'_, PyDict>) -> PyResult<()> {
        let field = |name: &str| {
            state
                .get_item(name)?
                .ok_or_else(|| PyKeyError::new_err(format!("Frame state lacks '{name}'")))
        };
        let blocks = field("blocks")?;
        for (key, block) in blocks.cast::<PyDict>()?.iter() {
            self.__setitem__(&key.extract::<String>()?, &block)?;
        }
        self.set_meta(&field("meta")?)?;
        let simbox = field("box")?;
        let simbox = simbox.extract::<Option<PyRef<'_, PyBox>>>()?;
        self.set_box(simbox.as_deref())
    }

    fn __repr__(&self) -> PyResult<String> {
        self.with_frame(|f| {
            let keys: Vec<&str> = f.keys().collect();
            format!(
                "Frame(blocks={:?}, box={})",
                keys,
                if f.simbox.is_some() { "yes" } else { "no" }
            )
        })
    }

    /// Export this frame's FFI handle as a ``PyCapsule``.
    ///
    /// The capsule wraps a *clone* of this frame's ``FrameRef`` handle. The
    /// clone shares the same underlying ``Store`` (``Rc<RefCell<Store>>``),
    /// so a consumer (e.g. Atomiverse C++ via the molrs-cxxapi bridge) that
    /// resolves the capsule reads and writes the *same* frame data: no deep
    /// copy is made. The capsule's destructor reclaims the boxed
    /// ``FrameRef`` on capsule destruction, dropping its two ``Rc``
    /// references.
    ///
    /// Pointer indirection: PyO3's ``PyCapsule::new`` heap-boxes its
    /// payload, and the payload here is a ``#[repr(transparent)]``
    /// ``FrameRefPtr`` (itself ``*mut FrameRef``). The capsule's ``void*``
    /// is therefore ``*mut FrameRefPtr`` ≡ ``*mut *mut FrameRef``: one
    /// dereference yields the ``*mut FrameRef`` clone. Atomiverse's
    /// ``frame_clone_from_addr`` does exactly that double-resolve.
    ///
    /// The capsule name is ``molrs_ffi::abi::frameref_capsule_name()`` —
    /// ``"molrs.FrameRef/<major.minor>"``. The name carries the ABI line so a
    /// consumer built on a different molrs minor fails the name check cleanly
    /// instead of dereferencing a possibly drifted layout.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.FrameRef/<major.minor>"`` whose
    ///     pointer is ``*mut *mut`` :class:`molrs_ffi.FrameRef` (a cloned
    ///     handle).
    fn _ffi_frameref_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a clone of the handle and hand the raw pointer to the capsule.
        // `FrameRef` holds an `Rc` and is therefore not `Send`; a bare
        // `*mut FrameRef` is not `Send` either, so wrap it in `FrameRefPtr`
        // which asserts `Send`. This is sound because the capsule is only
        // ever touched under the GIL (molrs FFI is single-threaded — see the
        // threading note in `molrs_ffi::shared`).
        let raw = FrameRefPtr(Box::into_raw(Box::new(self.inner.clone())));
        let name = molrs_ffi::abi::frameref_capsule_name().to_owned();
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: FrameRefPtr, _ctx| {
            // SAFETY: `ptr.0` is the pointer produced by `Box::into_raw`
            // above and is reclaimed exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }

    /// Build a `Frame` from a ``"molrs.FrameRef"`` capsule — the **return path**
    /// for a downstream Rust consumer (e.g. molpack handing back a packed frame).
    ///
    /// Symmetric with [`_ffi_frameref_capsule`](Self::_ffi_frameref_capsule):
    /// the consumer wraps its result frame in a `FrameRef`, exports a capsule of
    /// the same shape and name, and calls this. The new `Frame` **shares** the
    /// producer's store (one `Rc` bump), so no deep copy is made.
    #[staticmethod]
    fn _from_ffi_frameref_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        // `pointer_checked` validates the capsule name and rejects a null
        // payload in one step, returning the `*mut *mut FrameRef`. The name
        // carries the ABI line, so a producer on another molrs minor line
        // (or a capsule without the ABI line in its name) fails here cleanly instead of being dereferenced.
        let expected = molrs_ffi::abi::frameref_capsule_name();
        let ptr = capsule.pointer_checked(Some(expected)).map_err(|err| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "{err} — this build of molcrafts-molrs speaks FFI ABI line \
                 {line} (capsule name {expected:?}); the producing extension \
                 embeds a different molrs minor line. Align both packages on \
                 one minor line.",
                line = molrs_ffi::abi::abi_line(),
            ))
        })?;
        let pp = ptr.as_ptr() as *const *const FrameRef;
        // SAFETY: a `"molrs.FrameRef"` capsule's `void*` is `*mut *mut FrameRef`
        // (the exporter boxes a `*mut FrameRef`). Deref once to reach the cloned
        // handle and `.clone()` it (two `Rc` bumps) onto the same store. The
        // capsule is only ever touched under the GIL.
        let fref = unsafe { (**pp).clone() };
        Ok(Self { inner: fref })
    }
}

/// `Send` wrapper around a `*mut FrameRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `FrameRef` is `!Send` (it holds an `Rc`), and raw pointers are `!Send` by
/// default. The capsule is only ever created, read, and destroyed while the
/// Python GIL is held, so no cross-thread access of the `Rc` ever occurs —
/// the `unsafe impl Send` is upheld by that single-threaded discipline.
///
/// `#[repr(transparent)]` guarantees this newtype has exactly the layout of
/// the wrapped `*mut FrameRef`. PyO3's `PyCapsule::new` heap-boxes the
/// payload, so the capsule's `void*` is `*mut FrameRefPtr`; the transparent
/// repr makes that pointer reinterpretable as `*mut *mut FrameRef`, which is
/// how Atomiverse's molrs-cxxapi bridge resolves it
/// (`frame_clone_from_addr`).
#[repr(transparent)]
struct FrameRefPtr(*mut FrameRef);

// SAFETY: see the type-level doc — single-threaded, GIL-guarded use only.
unsafe impl Send for FrameRefPtr {}

impl PyFrame {
    /// Create a `PyFrame` from a Rust `CoreFrame`, allocating a new FFI store.
    pub(crate) fn from_core_frame(frame: CoreFrame) -> PyResult<Self> {
        let arena = molrs_ffi::FrameArenaCell::default();
        let id = arena.borrow_mut().frame_new();
        arena
            .borrow_mut()
            .set_frame(id, frame.deep_copy())
            .map_err(ffi_error_to_pyerr)?;
        Ok(Self {
            inner: FrameRef::new(arena, id),
        })
    }

    /// Clone the underlying `CoreFrame` out of the store (deep copy).
    pub(crate) fn clone_core_frame(&self) -> PyResult<CoreFrame> {
        self.with_frame(CoreFrame::deep_copy)
    }

    /// Run a read-only closure on the underlying `CoreFrame`.
    pub(crate) fn with_frame<R>(&self, f: impl FnOnce(&CoreFrame) -> R) -> PyResult<R> {
        self.inner.with(f).map_err(ffi_error_to_pyerr)
    }

    /// Run a closure that edits the underlying `CoreFrame` in place.
    pub(crate) fn with_frame_mut<R>(&self, f: impl FnOnce(&mut CoreFrame) -> R) -> PyResult<R> {
        self.inner.with_mut(f).map_err(ffi_error_to_pyerr)
    }
}

/// `Frozen` is every `frame.meta` door. `Plain` is [`PyMetaValue::value`]
/// (and therefore pickle) and [`PyMetaDocument::copy`]: JSON comes back as
/// dicts and lists, which a reducer can pack.
#[derive(Clone, Copy)]
enum JsonForm {
    Frozen,
    Plain,
}

/// Scalars unwrap. Fixed-length vectors are tuples either way. `form` applies
/// to JSON only.
fn meta_value_to_py(py: Python<'_>, value: &MetaValue, form: JsonForm) -> PyResult<Py<PyAny>> {
    macro_rules! scalar {
        ($value:expr) => {
            $value.into_pyobject(py)?.into_any().unbind()
        };
    }
    macro_rules! tuple {
        ($value:expr) => {
            PyTuple::new(py, $value)?.into_any().unbind()
        };
    }
    Ok(match value {
        MetaValue::Bool(v) => v.into_pyobject(py)?.to_owned().into_any().unbind(),
        MetaValue::I32(v) => scalar!(*v),
        MetaValue::I64(v) => scalar!(*v),
        MetaValue::U32(v) => scalar!(*v),
        MetaValue::U64(v) => scalar!(*v),
        MetaValue::F64(v) => scalar!(*v),
        MetaValue::String(v) => scalar!(v),
        MetaValue::Bool3(v) => tuple!(v),
        MetaValue::I32x3(v) => tuple!(v),
        MetaValue::I64x3(v) => tuple!(v),
        MetaValue::U32x3(v) => tuple!(v),
        MetaValue::U64x3(v) => tuple!(v),
        MetaValue::F64x3(v) => tuple!(v),
        MetaValue::F64x6(v) => tuple!(v),
        MetaValue::F64x9(v) => tuple!(v),
        MetaValue::Json(v) => json_to_py(py, v, form)?,
    })
}

/// Build a [`MetaValue`] carrying an explicit dtype tag, coercing `value` into it.
///
/// Coercion never truncates: a value the tag cannot hold raises.
pub(crate) fn meta_value_from_dtype(dtype: &str, value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    fn array<T, const N: usize>(value: &Bound<'_, PyAny>, dtype: &str) -> PyResult<[T; N]>
    where
        for<'a, 'py> T: FromPyObject<'a, 'py>,
    {
        let values: Vec<T> = value.extract()?;
        values.try_into().map_err(|values: Vec<T>| {
            PyTypeError::new_err(format!("{dtype} requires {N} values, got {}", values.len()))
        })
    }

    Ok(match dtype {
        "bool" => MetaValue::Bool(value.extract()?),
        "i32" => MetaValue::I32(value.extract()?),
        "i64" => MetaValue::I64(value.extract()?),
        "u32" => MetaValue::U32(value.extract()?),
        "u64" => MetaValue::U64(value.extract()?),
        "f64" => MetaValue::F64(value.extract()?),
        "string" => MetaValue::String(value.extract()?),
        "bool3" => MetaValue::Bool3(array(value, dtype)?),
        "i32x3" => MetaValue::I32x3(array(value, dtype)?),
        "i64x3" => MetaValue::I64x3(array(value, dtype)?),
        "u32x3" => MetaValue::U32x3(array(value, dtype)?),
        "u64x3" => MetaValue::U64x3(array(value, dtype)?),
        "f64x3" => MetaValue::F64x3(array(value, dtype)?),
        "f64x6" => MetaValue::F64x6(array(value, dtype)?),
        "f64x9" => MetaValue::F64x9(array(value, dtype)?),
        "json" => MetaValue::Json(py_to_json(value, 0)?),
        _ => {
            return Err(PyTypeError::new_err(format!(
                "unknown metadata dtype '{dtype}'"
            )));
        }
    })
}

/// Pick a dtype for a written value.
///
/// Scalars take their exact Python counterpart; a numeric list or tuple of 3,
/// 6 or 9 becomes the matching fixed-length vector — the shapes `mrec`
/// declares — and everything else (objects, ragged or non-numeric sequences,
/// `None`) is a JSON document. A [`MetaValue`](PyMetaValue) passes through
/// with its tag intact. The tuple arm is the list arm widened: a 6-tuple read
/// off an `f64x6` key must not fall through into JSON.
pub(crate) fn infer_meta_value(value: &Bound<'_, PyAny>) -> PyResult<MetaValue> {
    if let Ok(typed) = value.extract::<PyRef<'_, PyMetaValue>>() {
        return Ok(typed.inner.clone());
    }
    // bool before int: Python bools are ints.
    if let Ok(v) = value.cast::<PyBool>() {
        return Ok(MetaValue::Bool(v.is_true()));
    }
    if let Ok(v) = value.cast::<PyInt>() {
        if let Ok(n) = v.extract::<i64>() {
            return Ok(MetaValue::I64(n));
        }
        if let Ok(n) = v.extract::<u64>() {
            return Ok(MetaValue::U64(n));
        }
        return Err(PyTypeError::new_err(
            "integer metadata does not fit i64/u64",
        ));
    }
    if let Ok(v) = value.cast::<PyFloat>() {
        return Ok(MetaValue::F64(v.extract()?));
    }
    if let Ok(v) = value.cast::<PyString>() {
        return Ok(MetaValue::String(v.extract()?));
    }
    if let Some(typed) = infer_fixed_vector(value)? {
        return Ok(typed);
    }
    Ok(MetaValue::Json(py_to_json(value, 0)?))
}

/// A list or tuple of 3, 6 or 9 numbers, or `None` when this is not one.
fn infer_fixed_vector(value: &Bound<'_, PyAny>) -> PyResult<Option<MetaValue>> {
    let len = if let Ok(list) = value.cast::<PyList>() {
        list.len()
    } else if let Ok(tuple) = value.cast::<PyTuple>() {
        tuple.len()
    } else {
        return Ok(None);
    };
    let dtype = match len {
        3 => "f64x3",
        6 => "f64x6",
        9 => "f64x9",
        _ => return Ok(None),
    };
    Ok(meta_value_from_dtype(dtype, value).ok())
}

/// A record-level `meta` argument as a JSON object.
///
/// Takes a `dict`, a [`MetaDocument`](PyMetaDocument), or any other
/// `collections.abc.Mapping` (`frame.meta` among them). Nested values may be
/// anything a `frame.meta` door hands out — tuples and documents included —
/// so what the bindings give a caller round-trips back in. Used by `*.mrec`,
/// which needs `fs`.
#[cfg(feature = "fs")]
pub(crate) fn meta_document_arg(
    meta: &Bound<'_, PyAny>,
) -> PyResult<serde_json::Map<String, JsonValue>> {
    let plain = if meta.cast::<PyDict>().is_ok() || meta.is_instance_of::<PyMetaDocument>() {
        meta.clone()
    } else if let Ok(mapping) = meta.cast::<pyo3::types::PyMapping>() {
        let dict = PyDict::new(meta.py());
        dict.update(mapping)?;
        dict.into_any()
    } else {
        return Err(PyTypeError::new_err(format!(
            "meta must be a mapping, got {}",
            meta.get_type().name()?
        )));
    };
    match py_to_json(&plain, 0)? {
        JsonValue::Object(map) => Ok(map),
        _ => unreachable!("a dict or a document converts to a JSON object"),
    }
}

/// A JSON object as a plain `dict`: nested objects are dicts, arrays lists.
/// Used by `*.mrec`, which needs `fs`.
#[cfg(feature = "fs")]
pub(crate) fn json_map_to_plain_dict<'py>(
    py: Python<'py>,
    map: &serde_json::Map<String, JsonValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, item) in map {
        dict.set_item(key, json_to_py(py, item, JsonForm::Plain)?)?;
    }
    Ok(dict)
}

fn json_to_py(py: Python<'_>, value: &JsonValue, form: JsonForm) -> PyResult<Py<PyAny>> {
    Ok(match value {
        JsonValue::Null => py.None(),
        JsonValue::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64()
                    .unwrap_or(f64::NAN)
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        JsonValue::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        JsonValue::Array(items) => {
            let mut decoded = Vec::with_capacity(items.len());
            for item in items {
                decoded.push(json_to_py(py, item, form)?);
            }
            match form {
                JsonForm::Frozen => PyTuple::new(py, decoded)?.into_any().unbind(),
                JsonForm::Plain => PyList::new(py, decoded)?.into_any().unbind(),
            }
        }
        JsonValue::Object(map) => match form {
            JsonForm::Frozen => Py::new(py, PyMetaDocument { inner: map.clone() })?.into_any(),
            JsonForm::Plain => {
                let dict = PyDict::new(py);
                for (key, item) in map {
                    dict.set_item(key, json_to_py(py, item, JsonForm::Plain)?)?;
                }
                dict.into_any().unbind()
            }
        },
    })
}

/// Deepest container nesting a written value may have. Past it the value is
/// refused rather than recursed into: a self-referencing list or dict would
/// otherwise overflow the stack and kill the interpreter.
const MAX_JSON_DEPTH: usize = 128;

/// A Python value as JSON: `None`, bools, ints, floats, strings, dicts,
/// documents, lists and tuples.
pub(crate) fn py_to_json(value: &Bound<'_, PyAny>, depth: usize) -> PyResult<JsonValue> {
    if depth > MAX_JSON_DEPTH {
        return Err(PyValueError::new_err(format!(
            "meta value nests deeper than {MAX_JSON_DEPTH} levels (cyclic?)"
        )));
    }
    if value.is_none() {
        return Ok(JsonValue::Null);
    }
    if let Ok(b) = value.cast::<PyBool>() {
        return Ok(JsonValue::Bool(b.is_true()));
    }
    if let Ok(i) = value.cast::<PyInt>() {
        if let Ok(n) = i.extract::<i64>() {
            return Ok(JsonValue::from(n));
        }
        return Ok(JsonValue::from(i.extract::<u64>()?));
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        // A JSON document is finite: writing a NaN as `null` would be a
        // silent loss. A non-finite float belongs in a typed meta value
        // (`f64`), not inside a document.
        let v = f.extract::<f64>()?;
        return serde_json::Number::from_f64(v)
            .map(JsonValue::Number)
            .ok_or_else(|| {
                PyValueError::new_err(format!(
                    "a JSON document holds only finite numbers, found {v}; store it as an \
                     f64 meta value instead"
                ))
            });
    }
    if let Ok(s) = value.cast::<PyString>() {
        return Ok(JsonValue::String(s.extract::<String>()?));
    }
    if let Ok(doc) = value.extract::<PyRef<'_, PyMetaDocument>>() {
        return Ok(JsonValue::Object(doc.inner.clone()));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v, depth + 1)?);
        }
        return Ok(JsonValue::Object(map));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return py_sequence_to_json(list.iter(), depth);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return py_sequence_to_json(tuple.iter(), depth);
    }
    Err(PyTypeError::new_err(format!(
        "metadata value is not JSON-serializable: {value}"
    )))
}

fn py_sequence_to_json<'py>(
    items: impl IntoIterator<Item = Bound<'py, PyAny>>,
    depth: usize,
) -> PyResult<JsonValue> {
    let mut decoded = Vec::new();
    for item in items {
        decoded.push(py_to_json(&item, depth + 1)?);
    }
    Ok(JsonValue::Array(decoded))
}
