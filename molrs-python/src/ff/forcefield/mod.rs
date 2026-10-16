//! Python bindings for `molrs::ff::forcefield` (`molrs.ff.forcefield`): the
//! [`ForceField`] container, its style / type handles ([`handles`]), its
//! IR-form conversions ([`forms`]) and per-row parameter columns
//! ([`param_columns`]). Force-field files are `molrs.io`'s
//! (`molrs::io`).
//!
//! The Python `dict` ↔ [`Params`](molrs::ff::ir::Params) conversions
//! every force-field binding uses ([`params_from_dict`], [`params_to_dict`],
//! [`array_param`]) live here, beside the type they convert.

mod forms;
pub mod handles;
mod param_columns;

use crate::array::ArrayArgDyn;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyString, PyTuple};

use molrs::ff::forcefield::ForceField;
use molrs_ffi::ForceFieldRef;

use crate::core::frame::PyFrame;
use crate::error::py_value_err;
use crate::ff::ir;

use numpy::ToPyArray;

/// Force-field definition metadata exposed to Python as `molrs.ff.forcefield.ForceField`.
///
/// Subclassable, like every core data class (molnex's `ForceField` extends it).
/// Styles and types
/// are read and written through their handles (:mod:`handles`).
#[pyclass(module = "molrs.ff.forcefield", name = "ForceField", subclass)]
pub struct PyForceField {
    pub(crate) inner: ForceField,
}

/// `value` as an array param: a numpy array of at least one dimension, or a
/// list or tuple of numbers (any nesting), converted to float64 by
/// ``numpy.asarray``; `None` for any other value (a 0-d array is a number). A
/// sequence numpy cannot make a numeric array of (strings, ragged nesting)
/// raises ``TypeError``.
pub(crate) fn array_param(value: &Bound<'_, PyAny>) -> PyResult<Option<ndarray::ArrayD<f64>>> {
    let py = value.py();
    let np = py.import("numpy")?;
    let is_array = if value.is_instance(&np.getattr("ndarray")?)? {
        value.getattr("ndim")?.extract::<usize>()? > 0
    } else {
        value.cast::<PyList>().is_ok() || value.cast::<PyTuple>().is_ok()
    };
    if !is_array {
        return Ok(None);
    }
    let converted = np
        .call_method1("asarray", (value, np.getattr("float64")?))
        .map_err(|e| {
            PyTypeError::new_err(format!(
                "an array param must be numbers of one rectangular shape: {e}"
            ))
        })?;
    let array: ArrayArgDyn<'_, f64> = converted.extract()?;
    Ok(Some(array.as_array().to_owned()))
}

/// Convert an optional Python ``dict[str, float | str | array]`` of parameters
/// into [`Params`](molrs::ff::ir::Params). A ``str`` value goes to the
/// string side, a number to the numeric side, an array (a numpy array or a
/// nested list / tuple of numbers, stored as float64) to the array side;
/// anything else raises ``TypeError``. A missing dict yields no params.
pub(crate) fn params_from_dict(
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<molrs::ff::ir::Params> {
    let mut out = molrs::ff::ir::Params::new();
    let Some(d) = params else {
        return Ok(out);
    };
    for (k, v) in d.iter() {
        let key = k.extract::<String>()?;
        if let Ok(text) = v.cast::<PyString>() {
            out.set_str(&key, text.to_str()?);
        } else if let Some(array) = array_param(&v)? {
            out.set_array(&key, array);
        } else if let Ok(number) = v.extract::<f64>() {
            out.set(&key, number);
        } else {
            return Err(PyTypeError::new_err(format!(
                "param '{key}' must be a number, a str or an array of numbers, got {}",
                v.get_type().name()?
            )));
        }
    }
    Ok(out)
}

/// `Send` wrapper around a `*mut ForceFieldRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `ForceFieldRef` is `!Send` (it holds an `Rc`) and raw pointers are `!Send`,
/// but the capsule is only ever created, read, and destroyed while the Python
/// GIL is held, so no cross-thread `Rc` access occurs. `#[repr(transparent)]`
/// makes the capsule's `void*` reinterpretable as `*mut *mut ForceFieldRef`,
/// matching the frame convention a consumer resolves (mirrors
/// [`crate::core::frame`]'s `FrameRefPtr`).
#[repr(transparent)]
struct ForceFieldRefPtr(*mut ForceFieldRef);

// SAFETY: GIL-guarded, single-threaded use only — see the type-level doc.
unsafe impl Send for ForceFieldRefPtr {}

impl PyForceField {
    /// A new Python ``ForceField`` holding `inner`.
    pub(crate) fn from_core(py: Python<'_>, inner: ForceField) -> PyResult<Py<PyForceField>> {
        Py::new(py, PyForceField { inner })
    }

    /// Every style handle whose category `selection` names, in definition
    /// order.
    fn style_handles(
        slf: &Bound<'_, Self>,
        selection: &handles::Selection,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let styles: Vec<(handles::Category, String)> = slf
            .try_borrow()?
            .inner
            .styles()
            .iter()
            .filter_map(|style| {
                let category = handles::Category::of_style(style);
                selection
                    .contains(&category)
                    .then(|| (category, style.name().to_owned()))
            })
            .collect();
        let ff = slf.clone().unbind();
        styles
            .iter()
            .map(|(category, name)| category.style_handle(py, &ff, name))
            .collect()
    }

    /// The pickled definition: `(name, declared units, declared special
    /// bonds, [(category, arity, style, params, [(type, endpoints,
    /// params)])])`.
    fn definition<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let styles = PyList::empty(py);
        for style in self.inner.styles() {
            let types = PyList::empty(py);
            for (name, endpoints, params) in style.type_rows() {
                types.append((name, endpoints, params_to_dict(py, params)?))?;
            }
            styles.append((
                style.category(),
                style.arity(),
                style.name(),
                params_to_dict(py, style.params())?,
                types,
            ))?;
        }
        let special_bonds = self
            .inner
            .declared_special_bonds()
            .map(|sb| (sb.lj.to_vec(), sb.coul.to_vec()));
        (
            self.inner.name.clone(),
            self.inner.declared_units().map(str::to_owned),
            special_bonds,
            styles,
        )
            .into_pyobject(py)
    }

    /// The force field a [`definition`](Self::definition) describes.
    #[allow(
        clippy::type_complexity,
        reason = "the pickled definition, see `definition`"
    )]
    fn from_definition(definition: &Bound<'_, PyAny>) -> PyResult<ForceField> {
        let (name, units, special_bonds, styles): (
            String,
            Option<String>,
            Option<([f64; 3], [f64; 3])>,
            Vec<(
                String,
                usize,
                String,
                Bound<'_, PyDict>,
                Vec<(String, Vec<String>, Bound<'_, PyDict>)>,
            )>,
        ) = definition.extract()?;
        let mut inner = ForceField::new(&name);
        if let Some(units) = units {
            inner.set_units(&units);
        }
        if let Some((lj, coul)) = special_bonds {
            inner.set_special_bonds(molrs::ff::ir::SpecialBonds { lj, coul });
        }
        for (category, arity, style_name, params, types) in styles {
            // The arity travels with the style: a category no registry of
            // the unpickling process declares is still a relation of it.
            let style = inner
                .def_style_with_arity(
                    &category,
                    arity,
                    &style_name,
                    params_from_dict(Some(&params))?,
                )
                .map_err(ir::def_err)?;
            for (type_name, endpoints, params) in types {
                let endpoints: Vec<&str> = endpoints.iter().map(String::as_str).collect();
                style
                    .def_type(&type_name, &endpoints, params_from_dict(Some(&params))?)
                    .map_err(ir::def_err)?;
            }
        }
        Ok(inner)
    }
}

/// `params` as a dict: numbers, strings, and arrays as new float64 numpy
/// arrays.
pub(crate) fn params_to_dict<'py>(
    py: Python<'py>,
    params: &molrs::ff::ir::Params,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in params.iter() {
        out.set_item(key, value)?;
    }
    for (key, value) in params.iter_strings() {
        out.set_item(key, value)?;
    }
    for (key, value) in params.iter_arrays() {
        out.set_item(key, value.to_pyarray(py))?;
    }
    Ok(out)
}

#[pymethods]
impl PyForceField {
    /// Construct an empty force field. Populate it with :meth:`def_style` and
    /// the style handles' ``def_type``, or load one with a reader
    /// (:func:`molrs.io.read_openmm_forcefield`, …). ``units`` declares the unit system
    /// when given; left out, the force field declares none and :attr:`units`
    /// reads ``"real"``.
    #[new]
    #[pyo3(signature = (name = "forcefield", units = None))]
    fn new(name: &str, units: Option<&str>) -> Self {
        let mut inner = ForceField::new(name);
        if let Some(units) = units {
            inner.set_units(units);
        }
        Self { inner }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The unit system the parameters are expressed in (a LAMMPS ``units``
    /// name); ``"real"`` when none is declared.
    #[getter]
    fn units(&self) -> String {
        self.inner.units().to_owned()
    }

    /// Merge ``other`` into this force field, in place, and return ``self``.
    ///
    /// The union of both definitions: ``other``'s new styles and types are
    /// defined through the same primitives. A style or type both define is
    /// one definition when the two price alike — the parameters the style's
    /// spec declares, defaults filled in, numbers equal within a relative
    /// 1e-6 (annotations, estimate provenance and a reader's bookkeeping such
    /// as a prmtop atom type's ``id`` take no part) — and this force field's
    /// is kept; otherwise it raises ``ValueError``. Declared ``units`` and
    /// ``special_bonds`` are adopted when this force field declares none; two
    /// declared values that differ raise ``ValueError``. On error nothing
    /// changes.
    fn merge<'py>(
        slf: &Bound<'py, Self>,
        other: &Bound<'py, PyForceField>,
    ) -> PyResult<Bound<'py, Self>> {
        // Merging a force field into itself is the identical overlap: a no-op
        // (and borrowing it both ways at once would fail).
        if !slf.is(other) {
            let other = other.borrow();
            slf.borrow_mut()
                .inner
                .merge(&other.inner)
                .map_err(py_value_err)?;
        }
        Ok(slf.clone())
    }

    /// The special-bond triples ``(lj, coul)``, each ``[1-2, 1-3, 1-4]`` — what
    /// :meth:`set_special_bonds` declared, or the default ``[0, 0, 1]``.
    #[getter]
    fn special_bonds(&self) -> ([f64; 3], [f64; 3]) {
        let sb = self.inner.special_bonds();
        (sb.lj, sb.coul)
    }

    /// Declare both LJ and Coulomb special-bond triples (1-2, 1-3, 1-4).
    ///
    /// Length-3 sequences required; a wrong length raises ``ValueError``.
    /// Entries ``[0]``/``[1]`` are stored but not applied (1-2/1-3 exclusion
    /// is by omitting pairs from the neighbour list).
    fn set_special_bonds(&mut self, lj: [f64; 3], coul: [f64; 3]) {
        self.inner
            .set_special_bonds(molrs::ff::ir::SpecialBonds { lj, coul });
    }

    /// Write this force field's 1-4 pricing of ``frame``'s 1-4 pairs as
    /// per-pair override cells (``epsilon``, ``sigma``, ``lj_scale``,
    /// ``coul_scale``) on its ``pairs`` block, and return how many rows were
    /// filled.
    ///
    /// Every ``pairs`` row flagged ``is_14`` (each 1-4 pair once; the block is
    /// built when the frame has none) that no ``dihedral charmm`` ``w > 0``
    /// covers gets its null cells: ``epsilon``/``sigma`` are the pair's 1-4
    /// Lennard-Jones parameters — under ``lj/charmm`` with ``one_four =
    /// "epsilon14"`` the cross (NBFIX) row or the two types'
    /// ``epsilon14``/``sigma14`` mixed, else the regular pair parameters — and
    /// the scales are the ``special_bonds`` 1-4 weights. Cells already set are
    /// kept. A field whose ``lj/charmm`` declares ``one_four = "epsilon14"``
    /// (an OpenMM ``<LennardJonesForce>`` with ``sigma14``/``epsilon14``, a
    /// GROMACS ``[ pairtypes ]`` table) needs this on its frames before it
    /// compiles; LAMMPS's writers refuse the override columns.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     A typed frame (``atoms.type``), modified in place.
    ///
    /// Returns
    /// -------
    /// int
    ///     The number of ``pairs`` rows given override cells.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An untyped frame, a type without a pair row, two Lennard-Jones
    ///     styles or an invalid ``one_four``.
    fn materialize_one_four(&self, frame: &PyFrame) -> PyResult<usize> {
        frame
            .with_frame_mut(|core| self.inner.materialize_one_four(core))?
            .map_err(py_value_err)
    }

    /// Export this force field's FFI handle as a ``PyCapsule``.
    ///
    /// The force-field analogue of :meth:`Frame._ffi_frameref_capsule`. The
    /// capsule wraps a :class:`molrs_ffi.ForceFieldRef` that **shares** this
    /// force field's parameters (one ``Rc`` clone — no deep copy), so a
    /// downstream Rust consumer (e.g. the molpack relaxer) can resolve it and
    /// compile potentials with **no marshalling**. The capsule's ``void*`` is
    /// ``*mut *mut`` :class:`molrs_ffi.ForceFieldRef`, matching the frame
    /// convention; its name is ``molrs_ffi::abi::forcefield_capsule_name()``
    /// — ``"molrs.ForceFieldRef/<major.minor>"``, carrying the ABI line so a
    /// cross-minor consumer fails the name check cleanly. The capsule's
    /// destructor reclaims the boxed handle, dropping its ``Rc``.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.ForceFieldRef/<major.minor>"``.
    fn _ffi_forcefield_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a shared handle (Rc clone of this force field) and hand the raw
        // pointer to the capsule. See `ForceFieldRefPtr` for the Send / layout
        // contract.
        let raw = ForceFieldRefPtr(Box::into_raw(Box::new(ForceFieldRef::new(
            self.inner.clone(),
        ))));
        let name = molrs_ffi::abi::forcefield_capsule_name().to_owned();
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: ForceFieldRefPtr, _ctx| {
            // SAFETY: `ptr.0` came from `Box::into_raw` above and is reclaimed
            // exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }

    // -- styles: defined here, read and written through their handles ----------

    /// Define the ``category`` style ``name`` with style-level ``params``
    /// (numbers and strings, e.g. ``{"cutoff": 10.0, "mixing": "geometric"}``)
    /// and return its handle (``AtomStyle`` … ``PairStyle``, ``CmapStyle``),
    /// whose typed ``def_type`` defines types. Any other category the
    /// force-field IR registry declares (``drude``, a registered custom
    /// category, …), or one this force field holds already, returns a
    /// ``RelationStyle``, whose ``def_type(name, *endpoints, **params)``
    /// takes as many endpoints as the category's arity. Re-defining it with
    /// equal ``params`` keeps the existing style.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On different ``params`` for an existing style, or an unknown
    ///     category.
    #[pyo3(signature = (category, name, params = None))]
    fn def_style(
        slf: &Bound<'_, Self>,
        category: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let params = params_from_dict(params)?;
        let category = {
            let mut ff = slf.try_borrow_mut()?;
            let style = ff
                .inner
                .def_style(category, name, params)
                .map_err(ir::def_err)?;
            handles::Category::of_style(style)
        };
        category.style_handle(slf.py(), &slf.clone().unbind(), name)
    }

    /// Every style, in definition order.
    #[getter(styles)]
    fn every_style(slf: &Bound<'_, Self>) -> PyResult<Vec<Py<PyAny>>> {
        Self::style_handles(slf, &handles::Selection::Every)
    }

    /// The ``category`` style ``name``, or ``None``.
    fn get_style(slf: &Bound<'_, Self>, category: &str, name: &str) -> PyResult<Option<Py<PyAny>>> {
        let Some(category) = slf
            .try_borrow()?
            .inner
            .get_style(category, name)
            .map(handles::Category::of_style)
        else {
            return Ok(None);
        };
        category
            .style_handle(slf.py(), &slf.clone().unbind(), name)
            .map(Some)
    }

    /// The styles of a category — a name (``"bond"``) or a style class
    /// (``BondStyle``; ``RelationStyle`` selects every category beyond the
    /// seven, ``Style`` every category).
    fn get_styles(slf: &Bound<'_, Self>, category: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let selection = handles::Category::selected(category, &slf.try_borrow()?.inner)?;
        Self::style_handles(slf, &selection)
    }

    /// The types of a category — a name (``"bond"``) or a type class
    /// (``BondType``; ``RelationType`` selects every category beyond the
    /// seven, ``ForceFieldType`` every category) — style by style.
    fn get_types(slf: &Bound<'_, Self>, category: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let py = slf.py();
        let mut types = Vec::new();
        let selection = handles::Category::selected(category, &slf.try_borrow()?.inner)?;
        for style in Self::style_handles(slf, &selection)? {
            types.extend(
                style
                    .bind(py)
                    .call_method0(intern!(py, "get_types"))?
                    .extract::<Vec<Py<PyAny>>>()?,
            );
        }
        Ok(types)
    }

    // -- pickling ----------------------------------------------------------------

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let definition = slf.try_borrow()?.definition(py)?;
        crate::pickle::reduce_with_state(
            slf.as_any(),
            PyTuple::empty(py),
            PyTuple::new(py, [definition])?.into_any(),
        )
    }

    fn __setstate__(&mut self, state: (Bound<'_, PyAny>,)) -> PyResult<()> {
        self.inner = Self::from_definition(&state.0)?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceField(name='{}', styles={})",
            self.inner.name,
            self.inner.styles().len()
        )
    }
}

/// Register `molrs.ff.forcefield`: the container and its handles.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForceField>()?;
    handles::register(m)?;
    Ok(())
}
