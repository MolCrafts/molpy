//! `molpy.core.schema` — the Frame vocabulary, inspectable from Python.
//!
//! Every value here is projected from the compiled-in Rust tables at import
//! time. Nothing is transcribed, so `molpy.core.schema` and
//! `molrs::core::schema` cannot describe different contracts.
//!
//! `molpy.core.keys` is the field-name convention projected from the same tables:
//! each constant is a :class:`Key` (not a bare ``str``).

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use molrs::core::DType;
use molrs::core::schema;
use molrs::op::{F, I, Idx};
use num_complex::Complex;

// ── Key ──────────────────────────────────────────────────────────────────────

/// Canonical name projected from the Rust key tables.
///
/// A column key, or a frame-meta key. Ordered groups are tuples of these.
/// Block names are :mod:`molpy.core.schema`, not keys. Use ``.key`` (or
/// ``str(key)``) wherever an API still takes a plain string.
#[pyclass(module = "molpy.core.keys", name = "Key", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyKey {
    name: &'static str,
}

impl PyKey {
    pub(crate) const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

#[pymethods]
impl PyKey {
    /// The canonical column name string (e.g. ``"x"``).
    #[getter]
    fn key(&self) -> &'static str {
        self.name
    }

    fn __str__(&self) -> &'static str {
        self.name
    }

    fn __repr__(&self) -> String {
        format!("Key({:?})", self.name)
    }

    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        pyo3::types::PyString::new(py, self.name).hash()
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: CompareOp) -> PyResult<bool> {
        let other_s = if let Ok(k) = other.extract::<PyRef<'_, PyKey>>() {
            k.name.to_owned()
        } else if let Ok(s) = other.extract::<String>() {
            s
        } else {
            return Ok(matches!(op, CompareOp::Ne));
        };
        Ok(match op {
            CompareOp::Eq => self.name == other_s,
            CompareOp::Ne => self.name != other_s,
            CompareOp::Lt => self.name < other_s.as_str(),
            CompareOp::Le => self.name <= other_s.as_str(),
            CompareOp::Gt => self.name > other_s.as_str(),
            CompareOp::Ge => self.name >= other_s.as_str(),
        })
    }
}

/// Accept either a :class:`Key` or a ``str`` as a column name.
pub(crate) fn extract_column_key(ob: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(k) = ob.extract::<PyRef<'_, PyKey>>() {
        return Ok(k.name.to_owned());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Ok(s);
    }
    Err(PyTypeError::new_err(
        "column key must be molpy.core.keys.Key or str",
    ))
}

/// One canonical column of the Frame vocabulary.
#[pyclass(
    module = "molpy.core.schema",
    name = "ColumnSpec",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Clone)]
pub struct PyColumnSpec {
    /// Canonical key as it appears in a Block.
    pub key: String,
    /// Constant name, also exported as `molpy.core.keys.<const_name>`.
    pub const_name: String,
    /// The storage dtype's name: `"f64"`, `"i32"`, `"i64"`, `"u64"`,
    /// `"bool"`, `"string"`, … (Rust `DType::name`).
    pub dtype: String,
    /// `"scalar"` or `"vec(n)"`.
    pub shape: String,
    /// Physical dimension as lower-case preset names joined by ``" * "``
    /// (``"length"``, ``"charge * length"``), ``"dimensionless"``, or empty
    /// for a column that is not a physical quantity.
    pub dimension: String,
    /// The dimension's unit in the ``real`` preset (the convention molrs's
    /// readers normalise to); empty when dimensionless or not a quantity.
    pub unit: String,
    /// One-line meaning.
    pub doc: String,
}

#[pymethods]
impl PyColumnSpec {
    #[new]
    fn new(
        key: String,
        const_name: String,
        dtype: String,
        shape: String,
        dimension: String,
        unit: String,
        doc: String,
    ) -> Self {
        Self {
            key,
            const_name,
            dtype,
            shape,
            dimension,
            unit,
            doc,
        }
    }

    fn __repr__(&self) -> String {
        format!("ColumnSpec(key='{}', dtype='{}')", self.key, self.dtype)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.key.clone(),
                this.const_name.clone(),
                this.dtype.clone(),
                this.shape.clone(),
                this.dimension.clone(),
                this.unit.clone(),
                this.doc.clone(),
            ),
        )
    }

    /// The numpy dtype name this column is stored at — the dtype
    /// ``block[key] = values`` adopts (``"float64"``, ``"int32"``,
    /// ``"int64"``, ``"uint64"``, ``"bool"``, …), and ``"str"`` for a string
    /// column.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``dtype`` names no storage dtype.
    #[getter]
    fn numpy_dtype(&self, py: Python<'_>) -> PyResult<String> {
        let dtype = DType::from_name(&self.dtype)
            .ok_or_else(|| PyValueError::new_err(format!("unknown dtype {:?}", self.dtype)))?;
        // Derived from the Rust element type each variant stores, so it
        // cannot drift from what a column of this dtype hands numpy.
        let descr = match dtype {
            DType::Float => numpy::dtype::<F>(py),
            DType::I8 => numpy::dtype::<i8>(py),
            DType::I16 => numpy::dtype::<i16>(py),
            DType::Int => numpy::dtype::<I>(py),
            DType::I64 => numpy::dtype::<i64>(py),
            DType::Bool => numpy::dtype::<bool>(py),
            DType::Uint => numpy::dtype::<Idx>(py),
            DType::U8 => numpy::dtype::<u8>(py),
            DType::U16 => numpy::dtype::<u16>(py),
            DType::U32 => numpy::dtype::<u32>(py),
            DType::C64 => numpy::dtype::<Complex<f32>>(py),
            DType::C128 => numpy::dtype::<Complex<f64>>(py),
            DType::String => return Ok("str".to_owned()),
            other => {
                return Err(PyValueError::new_err(format!(
                    "dtype {other} has no numpy equivalent"
                )));
            }
        };
        descr.getattr("name")?.extract()
    }
}

/// One canonical block of the Frame vocabulary.
#[pyclass(
    module = "molpy.core.schema",
    name = "BlockSpec",
    frozen,
    get_all,
    from_py_object
)]
#[derive(Clone)]
pub struct PyBlockSpec {
    /// Canonical block name.
    pub name: String,
    /// `"node"`, `"relation(k)"`, or `"grid"`.
    pub row_kind: String,
    /// Block the endpoints index into, for relation blocks.
    pub endpoint_target: Option<String>,
    /// Endpoint column keys, in position order.
    pub endpoint_columns: Vec<String>,
    /// Endpoint columns whose target is declared per block (``targets``).
    pub declared_endpoints: Vec<String>,
    /// Columns that must be present.
    pub required: Vec<String>,
    /// Conventional but optional columns.
    pub optional: Vec<String>,
    /// Whether columns outside the vocabulary are admissible here.
    pub open: bool,
    /// One-line meaning.
    pub doc: String,
}

#[pymethods]
impl PyBlockSpec {
    #[new]
    #[allow(clippy::too_many_arguments, reason = "Public Python keyword arguments")]
    fn new(
        name: String,
        row_kind: String,
        endpoint_target: Option<String>,
        endpoint_columns: Vec<String>,
        required: Vec<String>,
        optional: Vec<String>,
        open: bool,
        doc: String,
        declared_endpoints: Vec<String>,
    ) -> Self {
        Self {
            name,
            row_kind,
            endpoint_target,
            endpoint_columns,
            declared_endpoints,
            required,
            optional,
            open,
            doc,
        }
    }

    fn __repr__(&self) -> String {
        format!("BlockSpec(name='{}', rows='{}')", self.name, self.row_kind)
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.name.clone(),
                this.row_kind.clone(),
                this.endpoint_target.clone(),
                this.endpoint_columns.clone(),
                this.required.clone(),
                this.optional.clone(),
                this.open,
                this.doc.clone(),
                this.declared_endpoints.clone(),
            ),
        )
    }
}

fn column_specs() -> Vec<PyColumnSpec> {
    schema::document()
        .columns
        .into_iter()
        .map(|c| PyColumnSpec {
            key: c.key,
            const_name: c.const_name,
            dtype: c.dtype,
            shape: c.shape,
            dimension: c.dimension,
            unit: c.unit,
            doc: c.doc,
        })
        .collect()
}

fn block_specs() -> Vec<PyBlockSpec> {
    schema::document()
        .blocks
        .into_iter()
        .map(|b| PyBlockSpec {
            name: b.name,
            row_kind: b.row_kind,
            endpoint_target: b.endpoint_target,
            endpoint_columns: b.endpoint_columns,
            declared_endpoints: b.declared_endpoints,
            required: b.required,
            optional: b.optional,
            open: b.open,
            doc: b.doc,
        })
        .collect()
}

/// Spec for a column key, or `None` if the key is unconstrained.
///
/// *key* may be a ``str`` or a :class:`molpy.core.keys.Key`.
#[pyfunction]
#[pyo3(name = "column")]
fn py_column(key: &Bound<'_, PyAny>) -> PyResult<Option<PyColumnSpec>> {
    let key = extract_column_key(key)?;
    Ok(column_specs().into_iter().find(|c| c.key == key))
}

/// Spec for a block name, or `None` if the block is not in the vocabulary.
#[pyfunction]
#[pyo3(name = "block")]
fn py_block(name: &str) -> Option<PyBlockSpec> {
    block_specs().into_iter().find(|b| b.name == name)
}

/// The whole vocabulary as canonical JSON — stable across runs, so two
/// releases can be diffed.
#[pyfunction]
fn to_json() -> String {
    schema::document().to_json()
}

/// The whole vocabulary as Markdown tables.
#[pyfunction]
fn to_markdown() -> String {
    schema::document().to_markdown()
}

/// The row references of a block: ``[(column, target block), …]``, empty
/// when the block references nothing.
///
/// A canonical relation block answers from the vocabulary; any other block
/// references ``"atoms"`` through the endpoint columns (``atomi`` …
/// ``atoml``) *columns* holds, in position order. *targets* — the block's
/// declared ``targets`` (``Block.targets()``) — overrides those defaults and
/// adds every other referencing column (``members.atom`` references nothing
/// until it is declared). A target is ``"<block>"`` of the same frame or
/// ``"/<section>/<block>"``.
#[pyfunction]
#[pyo3(signature = (name, columns, targets = None))]
fn relation_endpoints(
    name: &str,
    columns: Vec<String>,
    targets: Option<std::collections::BTreeMap<String, String>>,
) -> Vec<(String, String)> {
    let targets = targets.unwrap_or_default();
    let declared: Vec<(&str, &str)> = targets
        .iter()
        .map(|(c, t)| (c.as_str(), t.as_str()))
        .collect();
    schema::relation_endpoints(name, |k| columns.iter().any(|c| c == k), &declared)
        .into_iter()
        .map(|r| (r.column, r.target))
        .collect()
}

/// Register `molpy.core.schema`.
pub fn register_schema(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyColumnSpec>()?;
    m.add_class::<PyBlockSpec>()?;
    m.add("columns", column_specs())?;
    m.add("blocks", block_specs())?;
    for spec in schema::BLOCK_NAMES {
        m.add(spec.const_name, spec.value)?;
    }
    for group in schema::BLOCK_GROUPS {
        m.add(
            group.const_name,
            pyo3::types::PyTuple::new(m.py(), group.keys.iter().copied())?,
        )?;
    }
    m.add_function(wrap_pyfunction!(py_column, m)?)?;
    m.add_function(wrap_pyfunction!(py_block, m)?)?;
    m.add_function(wrap_pyfunction!(to_json, m)?)?;
    m.add_function(wrap_pyfunction!(to_markdown, m)?)?;
    m.add_function(wrap_pyfunction!(relation_endpoints, m)?)?;
    Ok(())
}

/// Register `molpy.core.keys`, projected from the same tables.
///
/// A loop over the Rust tables, not a hand-written list: a new column, group,
/// or frame-meta key appears as `molpy.core.keys.<CONST>` with no edit here.
///
/// Each scalar is a :class:`Key`. Ordered groups (`COORDS`, …) are lists of
/// :class:`Key`. Block names are :mod:`molpy.core.schema`.
pub fn register_keys(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKey>()?;

    for spec in schema::SCHEMA_COLUMNS {
        m.add(spec.const_name, PyKey::new(spec.key))?;
    }
    for group in schema::KEY_GROUPS {
        let keys: Vec<PyKey> = group.keys.iter().copied().map(PyKey::new).collect();
        m.add(group.const_name, keys)?;
    }
    use molrs::core::keys::{GRAPH_KEYS, LAMMPS_META_KEYS, META_KEYS};
    for spec in META_KEYS.iter().chain(GRAPH_KEYS).chain(LAMMPS_META_KEYS) {
        m.add(spec.const_name, PyKey::new(spec.value))?;
    }
    Ok(())
}
