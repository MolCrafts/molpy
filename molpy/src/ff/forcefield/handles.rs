//! Style and type handles over a [`PyForceField`].
//!
//! A handle is the owning force field plus the identifiers of one style or
//! one type, and nothing else: every read and write goes through the one
//! native [`molrs::ff::forcefield::ForceField`]. `ForceField.def_style` returns the
//! category's style handle ([`PyAtomStyle`] … [`PyCmapStyle`], and
//! [`PyRelationStyle`] for every other category); its ``def_type`` door
//! defines a type and returns the type's handle ([`PyAtomType`] …
//! [`PyCmapType`], [`PyRelationType`]).

use std::sync::Arc;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use molrs::ff::forcefield::{ForceField, Style};

use super::{PyForceField, param_from_py, params_from_dict, params_to_dict};

/// A category and its handle classes: the seven with classes of their own,
/// and every other category — a custom one of the force-field IR registry,
/// molrec's `constraint` / `drude` / `virtual_site`, or one a record brought
/// in that nothing declares — as a relation.
#[derive(Clone, PartialEq, Eq)]
pub(crate) enum Category {
    Atom,
    Bond,
    Angle,
    Dihedral,
    Improper,
    Pair,
    Cmap,
    Relation(Arc<str>),
}

/// Which categories a ``get_styles`` / ``get_types`` selector names.
pub(crate) enum Selection {
    /// Every category (``Style`` / ``ForceFieldType``).
    Every,
    /// Every category beyond the seven (``RelationStyle`` / ``RelationType``).
    Relations,
    Only(Category),
}

impl Selection {
    pub(crate) fn contains(&self, category: &Category) -> bool {
        match self {
            Self::Every => true,
            Self::Relations => matches!(category, Category::Relation(_)),
            Self::Only(only) => only == category,
        }
    }
}

impl Category {
    const BUILTIN: [Self; 7] = [
        Self::Atom,
        Self::Bond,
        Self::Angle,
        Self::Dihedral,
        Self::Improper,
        Self::Pair,
        Self::Cmap,
    ];

    /// The category `name` names in `ff`: one of the seven, one the
    /// process-wide force-field IR registry declares, or one `ff` holds a
    /// style of.
    ///
    /// # Errors
    ///
    /// `ValueError` for any other name.
    pub(crate) fn of(name: &str, ff: &ForceField) -> PyResult<Self> {
        if let Some(builtin) = Self::BUILTIN.into_iter().find(|c| c.name() == name) {
            return Ok(builtin);
        }
        let declared =
            molrs::ff::style_registry::with_global_registry(|r| r.category(name).is_some())
                || ff.styles().iter().any(|s| s.category() == name);
        if declared {
            Ok(Self::Relation(Arc::from(name)))
        } else {
            Err(PyValueError::new_err(format!(
                "unknown force-field category '{name}'"
            )))
        }
    }

    /// The category of a defined style.
    pub(crate) fn of_style(style: &Style) -> Self {
        let name = style.category();
        Self::BUILTIN
            .into_iter()
            .find(|c| c.name() == name)
            .unwrap_or_else(|| Self::Relation(Arc::from(name)))
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Atom => "atom",
            Self::Bond => "bond",
            Self::Angle => "angle",
            Self::Dihedral => "dihedral",
            Self::Improper => "improper",
            Self::Pair => "pair",
            Self::Cmap => "cmap",
            Self::Relation(name) => name,
        }
    }

    fn style_class<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        match self {
            Self::Atom => py.get_type::<PyAtomStyle>(),
            Self::Bond => py.get_type::<PyBondStyle>(),
            Self::Angle => py.get_type::<PyAngleStyle>(),
            Self::Dihedral => py.get_type::<PyDihedralStyle>(),
            Self::Improper => py.get_type::<PyImproperStyle>(),
            Self::Pair => py.get_type::<PyPairStyle>(),
            Self::Cmap => py.get_type::<PyCmapStyle>(),
            Self::Relation(_) => py.get_type::<PyRelationStyle>(),
        }
    }

    fn type_class<'py>(&self, py: Python<'py>) -> Bound<'py, PyType> {
        match self {
            Self::Atom => py.get_type::<PyAtomType>(),
            Self::Bond => py.get_type::<PyBondType>(),
            Self::Angle => py.get_type::<PyAngleType>(),
            Self::Dihedral => py.get_type::<PyDihedralType>(),
            Self::Improper => py.get_type::<PyImproperType>(),
            Self::Pair => py.get_type::<PyPairType>(),
            Self::Cmap => py.get_type::<PyCmapType>(),
            Self::Relation(_) => py.get_type::<PyRelationType>(),
        }
    }

    /// The categories `selector` names in `ff`: a category name its own; a
    /// style or type class its category (``RelationStyle`` /
    /// ``RelationType`` every category beyond the seven); the base
    /// ``Style`` / ``ForceFieldType`` every one.
    ///
    /// # Errors
    ///
    /// `ValueError` for an unknown category name, `TypeError` when
    /// `selector` is neither a name nor a style or type class.
    pub(crate) fn selected(selector: &Bound<'_, PyAny>, ff: &ForceField) -> PyResult<Selection> {
        let py = selector.py();
        if let Ok(name) = selector.extract::<&str>() {
            return Ok(Selection::Only(Self::of(name, ff)?));
        }
        if selector.is(py.get_type::<PyStyle>()) || selector.is(py.get_type::<PyFfType>()) {
            return Ok(Selection::Every);
        }
        if selector.is(py.get_type::<PyRelationStyle>())
            || selector.is(py.get_type::<PyRelationType>())
        {
            return Ok(Selection::Relations);
        }
        Self::BUILTIN
            .into_iter()
            .find(|category| {
                selector.is(category.style_class(py)) || selector.is(category.type_class(py))
            })
            .map(Selection::Only)
            .ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "expected a category name or a Style / ForceFieldType class, got {}",
                    selector.repr().map(|r| r.to_string()).unwrap_or_default()
                ))
            })
    }

    pub(crate) fn style_handle(
        &self,
        py: Python<'_>,
        ff: &Py<PyForceField>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let base = PyStyle {
            ff: ff.clone_ref(py),
            category: self.clone(),
            name: name.to_owned(),
        };
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Atom => Py::new(py, init.add_subclass(PyAtomStyle {}))?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBondStyle {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngleStyle {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedralStyle {}))?.into_any(),
            Self::Improper => Py::new(py, init.add_subclass(PyImproperStyle {}))?.into_any(),
            Self::Pair => Py::new(py, init.add_subclass(PyPairStyle {}))?.into_any(),
            Self::Cmap => Py::new(py, init.add_subclass(PyCmapStyle {}))?.into_any(),
            Self::Relation(_) => Py::new(py, init.add_subclass(PyRelationStyle {}))?.into_any(),
        })
    }

    fn type_handle(
        &self,
        py: Python<'_>,
        ff: &Py<PyForceField>,
        style: Option<&str>,
        name: &str,
    ) -> PyResult<Py<PyAny>> {
        let base = PyFfType {
            ff: ff.clone_ref(py),
            category: self.clone(),
            style: style.map(str::to_owned),
            name: name.to_owned(),
        };
        let init = PyClassInitializer::from(base);
        Ok(match self {
            Self::Atom => Py::new(py, init.add_subclass(PyAtomType {}))?.into_any(),
            Self::Bond => Py::new(py, init.add_subclass(PyBondType {}))?.into_any(),
            Self::Angle => Py::new(py, init.add_subclass(PyAngleType {}))?.into_any(),
            Self::Dihedral => Py::new(py, init.add_subclass(PyDihedralType {}))?.into_any(),
            Self::Improper => Py::new(py, init.add_subclass(PyImproperType {}))?.into_any(),
            Self::Pair => Py::new(py, init.add_subclass(PyPairType {}))?.into_any(),
            Self::Cmap => Py::new(py, init.add_subclass(PyCmapType {}))?.into_any(),
            Self::Relation(_) => Py::new(py, init.add_subclass(PyRelationType {}))?.into_any(),
        })
    }
}

fn missing_style(category: &Category, name: &str) -> PyErr {
    PyValueError::new_err(format!("no {} style named '{name}'", category.name()))
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

/// Handle of one style of a :class:`ForceField`.
///
/// Made by ``ForceField.def_style`` / ``get_style`` / ``get_styles``; its
/// category's subclass (``BondStyle``, …) carries the typed ``def_type``.
/// Two handles are equal when they name the same category and style of the
/// same force field.
#[pyclass(module = "molpy.ff.forcefield", name = "Style", frozen, subclass)]
pub struct PyStyle {
    ff: Py<PyForceField>,
    category: Category,
    name: String,
}

impl PyStyle {
    fn with_style<R>(&self, py: Python<'_>, f: impl FnOnce(&Style) -> R) -> PyResult<R> {
        let ff = self.ff.bind(py).try_borrow()?;
        let style = ff
            .inner
            .get_style(self.category.name(), &self.name)
            .ok_or_else(|| missing_style(&self.category, &self.name))?;
        Ok(f(style))
    }

    fn type_handles(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let names: Vec<String> = self.with_style(py, |style| {
            style
                .type_rows()
                .into_iter()
                .map(|(name, _, _)| name.to_owned())
                .collect()
        })?;
        names
            .iter()
            .map(|name| {
                self.category
                    .type_handle(py, &self.ff, Some(&self.name), name)
            })
            .collect()
    }

    /// The one door every typed ``def_type`` goes through: define `name` on
    /// `endpoints` (``AtomType`` handles) with `params`.
    fn define(
        &self,
        py: Python<'_>,
        name: &str,
        endpoints: &[&Bound<'_, PyAny>],
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let mut ends = Vec::with_capacity(endpoints.len());
        for end in endpoints {
            let atom = end.cast::<PyAtomType>().map_err(|_| {
                PyTypeError::new_err(format!(
                    "{} type '{name}': an endpoint is an AtomType handle, got {} {}",
                    self.category.name(),
                    end.get_type()
                        .name()
                        .map(|n| n.to_string())
                        .unwrap_or_default(),
                    end.repr().map(|r| r.to_string()).unwrap_or_default()
                ))
            })?;
            ends.push(atom.as_super().get().name.clone());
        }
        let params = params_from_dict(params)?;
        let stored = {
            let mut ff = self.ff.bind(py).try_borrow_mut()?;
            let style = ff
                .inner
                .get_style_mut(self.category.name(), &self.name)
                .ok_or_else(|| missing_style(&self.category, &self.name))?;
            let ends: Vec<&str> = ends.iter().map(String::as_str).collect();
            style
                .def_type(name, &ends, params)
                .map_err(crate::ff::ir::def_err)?;
            // A pair restating a stored pair under another name is that row.
            match style.type_params(name) {
                Some(_) => name.to_owned(),
                None => style
                    .get_pairtype(ends[0], ends.get(1).copied())
                    .map(|row| row.name.clone())
                    .expect("def_type stored the row or found the pair it restates"),
            }
        };
        self.category
            .type_handle(py, &self.ff, Some(&self.name), &stored)
    }
}

#[pymethods]
impl PyStyle {
    /// The style name (``"harmonic"``, ``"lj/cut"``, …).
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The category (``"atom"``, ``"bond"``, …, ``"pair"``, ``"cmap"``, or
    /// any other a relation style is of).
    #[getter]
    fn category(&self) -> &str {
        self.category.name()
    }

    /// The types of this style that are instances of `type_cls` (every type,
    /// in definition order, when it is ``None``).
    #[pyo3(signature = (type_cls = None))]
    fn get_types(
        &self,
        py: Python<'_>,
        type_cls: Option<&Bound<'_, PyType>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let types = self.type_handles(py)?;
        let Some(cls) = type_cls else {
            return Ok(types);
        };
        let mut kept = Vec::with_capacity(types.len());
        for handle in types {
            if handle.bind(py).is_instance(cls)? {
                kept.push(handle);
            }
        }
        Ok(kept)
    }

    /// The type named `name`, or ``None``.
    fn get_type_by_name(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let defined = self.with_style(py, |style| style.type_endpoints(name).is_some())?;
        if !defined {
            return Ok(None);
        }
        self.category
            .type_handle(py, &self.ff, Some(&self.name), name)
            .map(Some)
    }

    /// The style-level params (e.g. a pair style's ``cutoff``, ``mixing``),
    /// as a new dict.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.with_style(py, |style| params_to_dict(py, style.params()))?
    }

    /// One style-level param, or ``None``.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.params(py)?.get_item(key)
    }

    /// Set one style-level param (``Style::set_param``): a number (how a
    /// caller declares a cutoff), a string (``s["mixing"] = "geometric"``) or
    /// an array of numbers (stored as float64).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``value`` is neither a number, a str nor an array of numbers.
    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = param_from_py(key, value)?;
        let mut ff = self.ff.bind(py).try_borrow_mut()?;
        let style = ff
            .inner
            .get_style_mut(self.category.name(), &self.name)
            .ok_or_else(|| missing_style(&self.category, &self.name))?;
        style.set_param(key, value);
        Ok(())
    }

    /// Remove the type named ``name`` from this style (``Style::del_type``)
    /// and return how many were removed: ``1``, or ``0`` when the style
    /// defines no such type. Handles onto the removed type go stale.
    fn del_type(&self, py: Python<'_>, name: &str) -> PyResult<usize> {
        let mut ff = self.ff.bind(py).try_borrow_mut()?;
        let style = ff
            .inner
            .get_style_mut(self.category.name(), &self.name)
            .ok_or_else(|| missing_style(&self.category, &self.name))?;
        Ok(style.del_type(name))
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.ff.as_ptr() as usize, self.category.name(), &self.name).hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyStyle>().is_ok_and(|other| {
            let other = other.get();
            other.ff.as_ptr() == self.ff.as_ptr()
                && other.category == self.category
                && other.name == self.name
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!("<{}: {}>", slf.get_type().name()?, slf.get().name))
    }
}

/// The atom style: ``def_type(name, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "AtomStyle", extends = PyStyle, frozen, subclass)]
pub struct PyAtomStyle {}

#[pymethods]
impl PyAtomStyle {
    /// Define the atom type ``name`` with ``params`` (numbers and strings,
    /// e.g. ``mass=12.011, element="C"``) and return its handle — the
    /// endpoint the bonded and pair styles take.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super().get().define(slf.py(), name, &[], params)
    }
}

/// The bond style: ``def_type(name, itom, jtom, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "BondStyle", extends = PyStyle, frozen, subclass)]
pub struct PyBondStyle {}

#[pymethods]
impl PyBondStyle {
    /// Define the bond type ``name`` between the atom types ``itom`` and
    /// ``jtom`` with ``params`` and return its handle. The name is stored as
    /// given and never read for endpoints.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom], params)
    }
}

/// The angle style: ``def_type(name, itom, jtom, ktom, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "AngleStyle", extends = PyStyle, frozen, subclass)]
pub struct PyAngleStyle {}

#[pymethods]
impl PyAngleStyle {
    /// Define the angle type ``name`` on ``itom``–``jtom``–``ktom``
    /// (``jtom`` the vertex) with ``params`` and return its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom], params)
    }
}

/// The dihedral style: ``def_type(name, itom, jtom, ktom, ltom, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "DihedralStyle", extends = PyStyle, frozen, subclass)]
pub struct PyDihedralStyle {}

#[pymethods]
impl PyDihedralStyle {
    /// Define the dihedral type ``name`` on ``itom``–``jtom``–``ktom``–``ltom``
    /// with ``params`` and return its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, ltom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        ltom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom, ltom], params)
    }
}

/// The improper style: ``def_type(name, itom, jtom, ktom, ltom, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "ImproperStyle", extends = PyStyle, frozen, subclass)]
pub struct PyImproperStyle {}

#[pymethods]
impl PyImproperStyle {
    /// Define the improper type ``name`` on ``itom``, ``jtom``, ``ktom``,
    /// ``ltom`` in the style's slot order with ``params`` and return its
    /// handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, ltom, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        ltom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom, ltom], params)
    }
}

/// The pair style: ``def_type(name, itom, jtom=None, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "PairStyle", extends = PyStyle, frozen, subclass)]
pub struct PyPairStyle {}

#[pymethods]
impl PyPairStyle {
    /// Define the pair type ``name`` between ``itom`` and ``jtom`` with
    /// ``params`` and return its handle; ``jtom=None`` is the self pair of
    /// ``itom``.
    ///
    /// A pair is its two atom types in either order, so a style holds one
    /// row per pair: restating a stored pair under any name with equal
    /// parameters (annotations such as ``desc`` aside) stores nothing and
    /// returns the stored row's handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition, or a restatement of a stored pair
    ///     with different parameters.
    #[pyo3(signature = (name, itom, jtom = None, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: Option<&Bound<'_, PyAny>>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let style = slf.as_super().get();
        match jtom {
            None => style.define(slf.py(), name, &[itom], params),
            Some(jtom) => style.define(slf.py(), name, &[itom, jtom], params),
        }
    }
}

/// The cmap style: ``def_type(name, itom, jtom, ktom, ltom, mtom, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "CmapStyle", extends = PyStyle, frozen, subclass)]
pub struct PyCmapStyle {}

#[pymethods]
impl PyCmapStyle {
    /// Define the cmap type ``name`` on the five atom types of two
    /// consecutive dihedrals, ``itom``–``jtom``–``ktom``–``ltom`` and
    /// ``jtom``–``ktom``–``ltom``–``mtom``, with ``params`` (its correction
    /// table is the array param ``grid``, an ``N × N`` float array) and
    /// return its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On a conflicting re-definition.
    #[pyo3(signature = (name, itom, jtom, ktom, ltom, mtom, **params))]
    #[allow(clippy::too_many_arguments, reason = "five endpoints, as in Python")]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        itom: &Bound<'_, PyAny>,
        jtom: &Bound<'_, PyAny>,
        ktom: &Bound<'_, PyAny>,
        ltom: &Bound<'_, PyAny>,
        mtom: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        slf.as_super()
            .get()
            .define(slf.py(), name, &[itom, jtom, ktom, ltom, mtom], params)
    }
}

/// The style of any category beyond the seven — a custom category of the
/// force-field IR, molrec's ``constraint`` / ``drude`` /
/// ``virtual_site``, or one a record brought in that nothing declares:
/// ``def_type(name, *endpoints, **params)``.
#[pyclass(module = "molpy.ff.forcefield", name = "RelationStyle", extends = PyStyle, frozen, subclass)]
pub struct PyRelationStyle {}

#[pymethods]
impl PyRelationStyle {
    /// How many endpoints a type of this category names.
    #[getter]
    fn arity(slf: &Bound<'_, Self>) -> PyResult<usize> {
        slf.as_super().get().with_style(slf.py(), Style::arity)
    }

    /// Define the type ``name`` on ``endpoints`` (``AtomType`` handles, as
    /// many as the category's arity, in order) with ``params`` and return
    /// its handle.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If an endpoint is not an ``AtomType``.
    /// ValueError
    ///     On the wrong number of endpoints, or a conflicting re-definition.
    #[pyo3(signature = (name, *endpoints, **params))]
    fn def_type(
        slf: &Bound<'_, Self>,
        name: &str,
        endpoints: &Bound<'_, PyTuple>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let ends: Vec<Bound<'_, PyAny>> = endpoints.iter().collect();
        let refs: Vec<&Bound<'_, PyAny>> = ends.iter().collect();
        slf.as_super().get().define(slf.py(), name, &refs, params)
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Handle of one type of a :class:`ForceField`.
///
/// Its params read as a mapping (``t["k"]``, ``"k" in t``, ``t.keys()``) and
/// write one at a time (``t["k"] = 300.0``, ``t["element"] = "C"``,
/// ``t["grid"] = np.zeros((24, 24))``).
/// ``endpoints`` are the ``AtomType`` handles it is defined on. Two handles
/// are equal when they name the same category, style and type of the same
/// force field.
#[pyclass(
    module = "molpy.ff.forcefield",
    name = "ForceFieldType",
    frozen,
    subclass
)]
pub struct PyFfType {
    ff: Py<PyForceField>,
    category: Category,
    /// `None` for an endpoint atom type no atom style defines.
    style: Option<String>,
    name: String,
}

impl PyFfType {
    fn params_of<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let Some(style) = &self.style else {
            return Ok(PyDict::new(py));
        };
        let ff = self.ff.bind(py).try_borrow()?;
        let style = ff
            .inner
            .get_style(self.category.name(), style)
            .ok_or_else(|| missing_style(&self.category, style))?;
        match style.type_params(&self.name) {
            Some(params) => params_to_dict(py, params),
            None => Ok(PyDict::new(py)),
        }
    }

    fn endpoint(slf: &Bound<'_, Self>, index: usize) -> PyResult<Py<PyAny>> {
        Ok(Self::endpoints(slf)?
            .bind(slf.py())
            .get_item(index)?
            .unbind())
    }
}

#[pymethods]
impl PyFfType {
    /// The type name — the label a Frame's ``type`` column carries.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// The category (``"atom"``, ``"bond"``, …, ``"pair"``, ``"cmap"``, or
    /// any other a relation style is of).
    #[getter]
    fn category(&self) -> &str {
        self.category.name()
    }

    /// The type's params (numbers, strings and float64 arrays), as a new
    /// dict.
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.params_of(py)
    }

    /// One param, or ``None``.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.params_of(py)?.get_item(key)
    }

    /// One param, or `default`.
    #[pyo3(signature = (key, default = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.params_of(py)?.get_item(key)?.or(default))
    }

    fn __contains__(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        self.params_of(py)?.contains(key)
    }

    /// Set one param: a number, a string (``t["element"] = "C"``) or an
    /// array (a numpy array or nested sequence of numbers, stored as
    /// float64: ``t["grid"] = np.zeros((24, 24))``).
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``value`` is neither a number, a str nor an array of numbers.
    /// ValueError
    ///     If the type is an endpoint no atom style defines.
    fn __setitem__(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let Some(style) = &self.style else {
            return Err(PyValueError::new_err(format!(
                "atom type '{}' is defined by no atom style; it has no params to set",
                self.name
            )));
        };
        let value = param_from_py(key, value)?;
        let mut ff = self.ff.bind(py).try_borrow_mut()?;
        let style = ff
            .inner
            .get_style_mut(self.category.name(), style)
            .ok_or_else(|| missing_style(&self.category, style))?;
        let set = style.set_type_param(&self.name, key, value);
        if !set {
            return Err(PyValueError::new_err(format!(
                "no {} type named '{}'",
                self.category.name(),
                self.name
            )));
        }
        Ok(())
    }

    /// The param names.
    fn keys(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.params_of(py)?.keys().extract()
    }

    /// ``(name, value)`` param pairs.
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Vec<(String, Bound<'py, PyAny>)>> {
        self.params_of(py)?.items().extract()
    }

    /// The ``AtomType`` handles this type is defined on (none for an atom
    /// type; two for a pair type, a self pair naming one atom type twice).
    #[getter]
    fn endpoints(slf: &Bound<'_, Self>) -> PyResult<Py<PyTuple>> {
        let py = slf.py();
        let this = slf.get();
        let Some(style) = &this.style else {
            return Ok(PyTuple::empty(py).unbind());
        };
        let ff = this.ff.bind(py).try_borrow()?;
        let names = ff
            .inner
            .get_style(this.category.name(), style)
            .and_then(|style| style.type_endpoints(&this.name))
            .unwrap_or_default();
        // Each endpoint resolves to the atom style that defines it.
        let atom_styles: Vec<&Style> = ff.inner.get_styles(Category::Atom.name());
        let resolved: Vec<(Option<String>, String)> = names
            .into_iter()
            .map(|name| {
                let style = atom_styles
                    .iter()
                    .find(|style| style.type_endpoints(&name).is_some())
                    .map(|style| style.name().to_owned());
                (style, name)
            })
            .collect();
        drop(ff);
        let handles = resolved
            .iter()
            .map(|(style, name)| Category::Atom.type_handle(py, &this.ff, style.as_deref(), name))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, handles)?.unbind())
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (
            self.ff.as_ptr() as usize,
            self.category.name(),
            &self.style,
            &self.name,
        )
            .hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyFfType>().is_ok_and(|other| {
            let other = other.get();
            other.ff.as_ptr() == self.ff.as_ptr()
                && other.category == self.category
                && other.style == self.style
                && other.name == self.name
        })
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(format!("<{}: {}>", slf.get_type().name()?, slf.get().name))
    }
}

/// An atom type; the endpoint every other type is defined on.
#[pyclass(module = "molpy.ff.forcefield", name = "AtomType", extends = PyFfType, frozen, subclass)]
pub struct PyAtomType {}

/// The `itom` / `jtom` endpoint accessors of a type class.
macro_rules! first_two_endpoints {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The first endpoint atom type.
            #[getter]
            fn itom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 0)
            }

            /// The second endpoint atom type.
            #[getter]
            fn jtom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 1)
            }
        }
    };
}

/// The `ktom` endpoint accessor of a type class with three or more endpoints.
macro_rules! third_endpoint {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The third endpoint atom type.
            #[getter]
            fn ktom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 2)
            }
        }
    };
}

/// The `ltom` endpoint accessor of a type class with four or more endpoints.
macro_rules! fourth_endpoint {
    ($ty:ty) => {
        #[pymethods]
        impl $ty {
            /// The fourth endpoint atom type.
            #[getter]
            fn ltom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                PyFfType::endpoint(slf.as_super(), 3)
            }
        }
    };
}

/// A bond type.
#[pyclass(module = "molpy.ff.forcefield", name = "BondType", extends = PyFfType, frozen, subclass)]
pub struct PyBondType {}
first_two_endpoints!(PyBondType);

/// An angle type (``jtom`` the vertex).
#[pyclass(module = "molpy.ff.forcefield", name = "AngleType", extends = PyFfType, frozen, subclass)]
pub struct PyAngleType {}
first_two_endpoints!(PyAngleType);
third_endpoint!(PyAngleType);

/// A dihedral type.
#[pyclass(module = "molpy.ff.forcefield", name = "DihedralType", extends = PyFfType, frozen, subclass)]
pub struct PyDihedralType {}
first_two_endpoints!(PyDihedralType);
third_endpoint!(PyDihedralType);
fourth_endpoint!(PyDihedralType);

/// An improper type.
#[pyclass(module = "molpy.ff.forcefield", name = "ImproperType", extends = PyFfType, frozen, subclass)]
pub struct PyImproperType {}
first_two_endpoints!(PyImproperType);
third_endpoint!(PyImproperType);
fourth_endpoint!(PyImproperType);

/// A pair type.
#[pyclass(module = "molpy.ff.forcefield", name = "PairType", extends = PyFfType, frozen, subclass)]
pub struct PyPairType {}
first_two_endpoints!(PyPairType);

/// A cmap type: two consecutive dihedrals on five atom types.
#[pyclass(module = "molpy.ff.forcefield", name = "CmapType", extends = PyFfType, frozen, subclass)]
pub struct PyCmapType {}
first_two_endpoints!(PyCmapType);
third_endpoint!(PyCmapType);
fourth_endpoint!(PyCmapType);

#[pymethods]
impl PyCmapType {
    /// The fifth endpoint atom type.
    #[getter]
    fn mtom(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        PyFfType::endpoint(slf.as_super(), 4)
    }
}

/// A type of a category beyond the seven; its ``endpoints`` are as many
/// atom types as the category's arity.
#[pyclass(module = "molpy.ff.forcefield", name = "RelationType", extends = PyFfType, frozen, subclass)]
pub struct PyRelationType {}

/// Register the style and type handle classes (base before subclasses).
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStyle>()?;
    m.add_class::<PyAtomStyle>()?;
    m.add_class::<PyBondStyle>()?;
    m.add_class::<PyAngleStyle>()?;
    m.add_class::<PyDihedralStyle>()?;
    m.add_class::<PyImproperStyle>()?;
    m.add_class::<PyPairStyle>()?;
    m.add_class::<PyCmapStyle>()?;
    m.add_class::<PyRelationStyle>()?;
    m.add_class::<PyFfType>()?;
    m.add_class::<PyAtomType>()?;
    m.add_class::<PyBondType>()?;
    m.add_class::<PyAngleType>()?;
    m.add_class::<PyDihedralType>()?;
    m.add_class::<PyImproperType>()?;
    m.add_class::<PyPairType>()?;
    m.add_class::<PyCmapType>()?;
    m.add_class::<PyRelationType>()?;
    Ok(())
}
