//! `molpy.io.mrec.ForceFieldSection`: molrec's `forcefield` record section as
//! data, and the `ForceField` ↔ section doors.
//!
//! The section is what a `*.mrec` stores — the force-field document and one
//! `Block` per style table — kept whole, units unconverted, unknown keys and
//! tables included. `ForceFieldSection.from_forcefield` /
//! `ForceFieldSection.to_forcefield` map it onto a compilable force field
//! (`molrs::io::mrec::ForceFieldSection`).

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping};

use molrs::io::mrec::ForceFieldSection;
use molrs::io::mrec::style_block_name;

use crate::core::block::PyBlock;
use crate::core::frame::{json_map_to_plain_dict, meta_document_arg};
use crate::error::{molrs_error_to_pyerr, py_value_err};
use crate::ff::forcefield::PyForceField;

/// The ``forcefield`` section of a ``*.mrec`` record: the force-field
/// document and one :class:`~molpy.core.Block` per style table (molrec
/// ``docs/spec/forcefield.md``).
///
/// It holds what a store holds, whole: the document keeps every key, the
/// tables every block, and no number is converted to another unit.
/// :meth:`from_forcefield` builds one from a force field;
/// :meth:`to_forcefield` turns one into a force field molrs can compile. Nothing is checked on construction; :meth:`validate`
/// checks the chapter's rules, and every writer runs it.
///
/// Parameters
/// ----------
/// document
///     The document (``name``, ``units``, ``styles``, …) — a
///     ``dict`` or any mapping of JSON values.
/// tables
///     Block name → :class:`~molpy.core.Block` (the style tables, at
///     :meth:`block_name` of their style, and any other block).
#[pyclass(
    module = "molpy.io.mrec",
    name = "ForceFieldSection",
    unsendable,
    subclass
)]
pub struct PyForceFieldSection {
    pub(crate) inner: ForceFieldSection,
}

impl PyForceFieldSection {
    /// The section a ``forcefield=`` argument names: a
    /// :class:`ForceFieldSection` as given, or a ``ForceField``'s
    /// :meth:`ForceFieldSection.from_forcefield`.
    pub(crate) fn from_arg(value: &Bound<'_, PyAny>) -> PyResult<ForceFieldSection> {
        if let Ok(section) = value.cast::<PyForceFieldSection>() {
            return Ok(section.borrow().inner.clone());
        }
        if let Ok(ff) = value.cast::<PyForceField>() {
            return ForceFieldSection::from_forcefield(&ff.borrow().inner).map_err(py_value_err);
        }
        Err(PyTypeError::new_err(format!(
            "forcefield must be a ForceField or a ForceFieldSection, got {}",
            value.get_type().name()?
        )))
    }
}

#[pymethods]
impl PyForceFieldSection {
    #[new]
    #[pyo3(signature = (document, tables = None))]
    fn new(document: &Bound<'_, PyAny>, tables: Option<&Bound<'_, PyMapping>>) -> PyResult<Self> {
        let mut inner = ForceFieldSection {
            document: meta_document_arg(document)?,
            ..ForceFieldSection::default()
        };
        if let Some(tables) = tables {
            for item in tables.items()?.iter() {
                let (name, block): (String, PyRef<'_, PyBlock>) = item.extract()?;
                inner.tables.insert(name, block.clone_core_block()?);
            }
        }
        Ok(Self { inner })
    }

    /// The force-field document, as a fresh ``dict``.
    #[getter]
    fn document<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        json_map_to_plain_dict(py, &self.inner.document)
    }

    /// Block name → :class:`~molpy.core.Block`, in stored order. Each block is a
    /// copy: editing it does not change the section.
    #[getter]
    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (name, block) in &self.inner.tables {
            out.set_item(name, PyBlock::from_core_block(block.deep_copy())?)?;
        }
        Ok(out)
    }

    /// The document's ``name``, or ``None``.
    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(str::to_owned)
    }

    /// The table of the ``(category, style)`` style (a copy), or ``None``.
    fn table(&self, category: &str, style: &str) -> PyResult<Option<PyBlock>> {
        self.inner
            .table(category, style)
            .map(|block| PyBlock::from_core_block(block.deep_copy()))
            .transpose()
    }

    /// The block name of a style's table: ``<category>.<style>``, every byte
    /// of the style outside ``A-Z a-z 0-9 - _`` written ``%XX``
    /// (``block_name("pair", "lj/cut")`` is ``"pair.lj%2Fcut"``).
    #[staticmethod]
    fn block_name(category: &str, style: &str) -> String {
        style_block_name(category, style)
    }

    /// Check the section against the ``forcefield`` chapter.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     Naming the first rule broken (units, a duplicate
    ///     style, a missing table, a duplicate or null type name, the
    ///     wrong endpoint columns, a parameter dtype, a ``pair`` table
    ///     pricing one unordered pair twice with different parameters, …).
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(molrs_error_to_pyerr)
    }

    /// A force field as a record's ``forcefield`` section.
    ///
    /// The units are the declared (or default ``real``) preset, stated
    /// beside its quantities; each style's types become its table's rows.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     When the force field has no section form (units that
    ///     are no preset, a param that is a number in one type and a
    ///     string in another, …).
    #[staticmethod]
    fn from_forcefield(forcefield: PyRef<'_, PyForceField>) -> PyResult<Self> {
        let inner = ForceFieldSection::from_forcefield(&forcefield.inner).map_err(py_value_err)?;
        Ok(Self { inner })
    }

    /// The force field this section describes. Its units become the force
    /// field's declared units; nothing is converted.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     When the section is invalid, or molrs cannot hold it:
    ///     a category outside atom/bond/angle/dihedral/improper/pair/cmap,
    ///     units that are no preset, a smirks-keyed style.
    fn to_forcefield(&self) -> PyResult<PyForceField> {
        let inner = self.inner.to_forcefield().map_err(py_value_err)?;
        Ok(PyForceField { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceFieldSection(name={:?}, tables={})",
            self.inner.name().unwrap_or(""),
            self.inner.tables.len()
        )
    }
}
