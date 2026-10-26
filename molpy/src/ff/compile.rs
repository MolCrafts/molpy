//! Python bindings for `molrs::ff::compile` (on `molpy.ff` itself): a force
//! field bound to its kernels.
//!
//! * [`PyPotentialCompiler`] — a `ForceField` compiled into kernels:
//!   `Potentials` over a fixed topology, or `WeightedTerms` (each kernel with
//!   its special-bonds weights) for a neighbour-driven integrator.
//! * `ExplicitTerms(category, style, atoms, *, charges=None, **params).compile()` — the kernel
//!   of **any** registered style over explicit instances (atom indices and
//!   one parameter row per term, as stored: the force-field IR's units, angle
//!   values in degrees). It is [`molrs::ff::compile::ExplicitTerms`], so the
//!   kernel is priced by the code a compiled force field is.

use ndarray::{ArrayD, Axis};
use numpy::{PyArrayDyn, PyArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use molrs::ff::compile::{ExplicitTerms, PotentialCompiler};
use molrs::ff::forcefield::ForceField;
use molrs::ff::ir::{ParamKind, Params, StyleSpec};
use molrs::op::F;

use crate::core::frame::PyFrame;
use crate::ff::forcefield::PyForceField;
use crate::ff::ir;
use crate::ff::potential::{PyPotentials, PyWeightedTerms};
use crate::ff::style_registry::column;

/// The terms' atoms: an ``(n, arity)`` array of indices.
fn term_atoms(atoms: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<usize>>> {
    let rows: Vec<Vec<usize>> = atoms.extract().map_err(|_| {
        PyTypeError::new_err("atoms: an (n, arity) array of atom indices, one row per term")
    })?;
    Ok(rows)
}

/// The value of per-term parameter `key` for each of `n` terms.
enum Column {
    Num(Vec<F>),
    Array(Vec<ArrayD<F>>),
    Text(Vec<String>),
}

fn per_term(
    py: Python<'_>,
    what: &str,
    kind: Option<&ParamKind>,
    value: &Bound<'_, PyAny>,
    n: usize,
) -> PyResult<Column> {
    match kind {
        Some(ParamKind::Text { .. }) => {
            let col = match value.cast::<PyString>() {
                Ok(s) => vec![s.to_str()?.to_owned(); n],
                Err(_) => value.extract::<Vec<String>>()?,
            };
            if col.len() != n {
                return Err(PyValueError::new_err(format!(
                    "{what}: {} strings for {n} terms",
                    col.len()
                )));
            }
            Ok(Column::Text(col))
        }
        Some(ParamKind::Array { rank }) => {
            let a = py
                .import("numpy")?
                .call_method1("asarray", (value, "float64"))?
                .cast_into::<PyArrayDyn<F>>()?
                .readonly()
                .as_array()
                .to_owned();
            let rank = *rank as usize;
            match a.ndim() {
                d if d == rank => Ok(Column::Array(vec![a; n])),
                d if d == rank + 1 && a.shape()[0] == n => Ok(Column::Array(
                    a.axis_iter(Axis(0)).map(|row| row.to_owned()).collect(),
                )),
                _ => Err(PyValueError::new_err(format!(
                    "{what}: an array of rank {rank}, or {n} of them stacked; got shape {:?}",
                    a.shape()
                ))),
            }
        }
        Some(ParamKind::Scalar) | None => Ok(Column::Num(column(what, value, n)?)),
    }
}

/// The terms of one style over explicit instances —
/// ``molrs::ff::compile::ExplicitTerms``; :meth:`compile` builds their kernel.
///
/// Works for every style the force-field IR prices: a built-in, a style
/// registered from Python (by expression or kernel, :mod:`molpy.ff.ir`), a
/// style of a custom category, or an unregistered style given its
/// ``expression=``. The kernel is built exactly as
/// :meth:`PotentialCompiler.compile` builds it, one type per term.
///
/// Parameters
/// ----------
/// category, style : str
///     The style (``"bond", "harmonic"``; ``"pair", "lj/cut"``; …).
/// atoms : array_like of int, shape (n, arity)
///     Each term's atoms. A pair term is an atom pair, priced with its own
///     row (the pair's cross row).
/// charges : array_like of float, optional
///     Per-atom charges (``atoms.charge``), read by ``coul/cut`` and by
///     pair expressions through ``q1``, ``q2``.
/// **params
///     Style parameters (``cutoff``, ``mixing``, ``coulomb``, …, an
///     unregistered style's ``expression``) as a number or a string; every
///     other one per term **as stored** (angle values in degrees): a number
///     (broadcast) or one value per term, an array parameter one array (or
///     ``n`` stacked), a text parameter a string or one per term. Indexed
///     families are spelled ``k1``, ``k2``, …. A style whose numbers are per
///     instance (``coul/cut``) takes none.
///
/// Raises
/// ------
/// TypeError
///     A parameter the registered style does not declare.
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from molpy.ff import ExplicitTerms
/// >>> pots = ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.5).compile()
/// >>> e, f = pots.evaluate(np.array([0.0, 0, 0, 1.6, 0, 0]))
/// >>> round(e, 12)
/// 3.0
#[pyclass(module = "molpy.ff", name = "ExplicitTerms", frozen)]
pub struct PyExplicitTerms {
    inner: ExplicitTerms,
}

#[pymethods]
impl PyExplicitTerms {
    #[new]
    #[pyo3(signature = (category, style, atoms, *, charges=None, **params))]
    fn new(
        py: Python<'_>,
        category: &str,
        style: &str,
        atoms: &Bound<'_, PyAny>,
        charges: Option<Vec<F>>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: explicit_terms(py, category, style, atoms, charges, params)?,
        })
    }

    /// Build the kernel of these terms, exactly as
    /// :meth:`PotentialCompiler.compile` builds it, one type per term.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///     One member; ``Potentials.push`` moves it into another collection.
    ///
    /// Raises
    /// ------
    /// IrError
    ///     The IR's refusal, by its subclass: ``UnknownCategory``, ``Arity``
    ///     (a row of the wrong length), ``NoKernel``, ``MissingParam``, ….
    fn compile(&self) -> PyResult<PyPotentials> {
        crate::ff::style_registry::clear_kernel_err();
        let pots = self.inner.compile().map_err(ir::compile_err)?;
        crate::ff::style_registry::take_kernel_err()?;
        Ok(PyPotentials::of(pots))
    }
}

/// The [`ExplicitTerms`] the Python constructor's arguments describe.
fn explicit_terms(
    py: Python<'_>,
    category: &str,
    style: &str,
    atoms: &Bound<'_, PyAny>,
    charges: Option<Vec<F>>,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<ExplicitTerms> {
    let who = format!("{category} `{style}`");
    let atoms = term_atoms(atoms)?;
    let n = atoms.len();
    let (pair, spec): (bool, Option<StyleSpec>) =
        molrs::ff::style_registry::GlobalRegistry.with(|r| {
            (
                r.category(category).is_some_and(|c| c.is_pair_driven()),
                r.style(category, style).map(|(s, _)| s.clone()),
            )
        });
    let mut style_params = Params::new();
    let mut columns: Vec<(String, Column)> = Vec::new();
    for (k, v) in params.into_iter().flat_map(|d| d.iter()) {
        let key: String = k.extract()?;
        let what = format!("{who}: `{key}`");
        let decl =
            match &spec {
                Some(spec) => Some(spec.declaration(&key, pair).ok_or_else(|| {
                    PyTypeError::new_err(format!("{who} has no parameter `{key}`"))
                })?),
                None => None,
            };
        if pair && matches!(decl, Some((None, _))) {
            return Err(PyTypeError::new_err(format!(
                "{who}: `{key}` is per atom; pass charges= (a term's row is its cross row)"
            )));
        }
        let style_level = match decl {
            Some((_, style_level)) => style_level,
            // An unregistered style declares nothing: its expression is the
            // one style parameter, every other value is per term.
            None => key == "expression",
        };
        if style_level {
            match v.cast::<PyString>() {
                Ok(s) => style_params.set(&key, s.to_str()?),
                Err(_) => style_params.set(&key, v.extract::<F>()?),
            }
            continue;
        }
        let kind = decl.and_then(|(p, _)| p).map(|p| &p.kind);
        columns.push((key, per_term(py, &what, kind, &v, n)?));
    }
    let mut terms = ExplicitTerms::new(category, style).style_params(style_params);
    if columns.is_empty() {
        terms = terms.atoms(atoms);
    } else {
        for (t, atoms) in atoms.iter().enumerate() {
            let mut row = Params::new();
            for (key, col) in &columns {
                match col {
                    Column::Num(c) => row.set(key, c[t]),
                    Column::Array(c) => row.set(key, c[t].clone()),
                    Column::Text(c) => row.set(key, &c[t]),
                }
            }
            terms = terms.term(atoms, row);
        }
    }
    if let Some(q) = charges {
        terms = terms.charges(q);
    }
    Ok(terms)
}

/// Compiles a :class:`ForceField` into evaluable kernels.
///
/// Exposed to Python as ``molpy.ff.PotentialCompiler``. It owns a **copy** of
/// the force field, taken at construction: later edits to that
/// :class:`ForceField` do not reach a compiler that already exists — make a
/// new one.
///
/// Three doors, each doing one thing:
///
/// * :meth:`compile` — bind a typed :class:`Frame` now;
/// * :meth:`compile_typed` — the kernels of a neighbour-driven evaluation,
///   for MD.
///
/// Examples
/// --------
/// >>> compiler = molpy.ff.PotentialCompiler(typifier.forcefield())
/// >>> potentials = compiler.compile(frame)
/// >>> energy = potentials.calc_energy(frame)
#[pyclass(module = "molpy.ff", name = "PotentialCompiler", subclass)]
pub struct PyPotentialCompiler {
    ff: ForceField,
}

#[pymethods]
impl PyPotentialCompiler {
    /// Copy ``forcefield`` into a new compiler.
    ///
    /// Parameters
    /// ----------
    /// forcefield : ForceField
    ///     The force field to compile. Copied; later edits do not reach this
    ///     compiler.
    #[new]
    fn new(forcefield: PyRef<'_, PyForceField>) -> Self {
        Self {
            ff: forcefield.inner.clone(),
        }
    }

    /// Build evaluable :class:`Potentials` against a typed ``frame``.
    ///
    /// The frame must carry the topology + ``type`` columns each style
    /// resolves (``atoms``/``bonds``/``angles``/``dihedrals``/``impropers``)
    /// and its special pairs (``pairs``), as a typifier produces them. The
    /// pair styles price every pair of atoms — right for a molecule in free
    /// space — inside their ``cutoff`` (``r < cutoff``, with the switch of a
    /// style that has one), as :meth:`compile_typed` and LAMMPS do. A pair
    /// the ``pairs`` block names takes its weight (its ``lj_scale`` /
    /// ``coul_scale`` cell, else the force field's ``special_bonds`` for its
    /// ``n_bonds``); every other pair is at full strength. ``pair
    /// coul/long/pme`` prices the same weights in its own minimum-image loop
    /// over the frame's periodic ``box``.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``frame`` is not a :class:`Frame` (``None`` included).
    /// ValueError
    ///     If a style has no registered kernel, a type label is unknown, the
    ///     frame's bonds name a pair no ``pairs`` row does (generate them with
    ///     the angles and dihedrals), a pair style priced from every pair
    ///     meets a periodic frame (use :meth:`compile_typed`), or a style that
    ///     reads the box (``coul/long/pme``) meets a frame without one.
    fn compile(&self, frame: &PyFrame) -> PyResult<PyPotentials> {
        crate::ff::style_registry::clear_kernel_err();
        let potentials = frame
            .with_frame(|core| PotentialCompiler::new(&self.ff).compile(core))?
            .map_err(ir::compile_err)?;
        crate::ff::style_registry::take_kernel_err()?;
        Ok(PyPotentials::of(potentials))
    }

    /// Build the kernels for a **neighbour-driven** evaluation, with the
    /// special-bonds weights each one takes.
    ///
    /// The counterpart of :meth:`compile`, and what periodic MD needs. That
    /// one prices every pair of atoms at raw distances, right for a
    /// free-boundary molecule and wrong for a periodic system; over the same
    /// pairs the two price the same energy. This one resolves the pair styles
    /// against the **atoms**, so they price whatever pairs a neighbour table
    /// holds, and requires each style's declared cutoff. ``pair
    /// coul/long/pme`` prices its real-space erfc over the neighbour table;
    /// its reciprocal sum reads the periodic box and corrects the special
    /// pairs by their weights.
    ///
    /// The weights are the frame's special pairs (``pairs``) under the force
    /// field's ``special_bonds``, as :meth:`compile` weighs them. Without
    /// them a neighbour table would count a bonded pair twice: once by the
    /// bond term and once at full non-bonded strength, at bond length.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data.
    ///
    /// Returns
    /// -------
    /// WeightedTerms
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a style cannot be built, or a pair style has no
    ///     neighbour-driven form.
    fn compile_typed(&self, frame: &PyFrame) -> PyResult<PyWeightedTerms> {
        crate::ff::style_registry::clear_kernel_err();
        let members = frame
            .with_frame(|core| PotentialCompiler::new(&self.ff).compile_typed(core))?
            .map_err(ir::compile_err)?;
        crate::ff::style_registry::take_kernel_err()?;
        Ok(PyWeightedTerms {
            members: Some(members),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PotentialCompiler(forcefield='{}', styles={})",
            self.ff.name,
            self.ff.styles().len()
        )
    }
}

/// Register the compiler classes `molpy.ff` re-exports.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPotentialCompiler>()?;
    m.add_class::<PyExplicitTerms>()?;
    Ok(())
}
