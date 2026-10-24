//! Python bindings for `molrs::ff::compile` (`molpy.ff.compile`): a force
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
use crate::ff::potential::{PotBacking, PyPotentials, PyWeightedTerms};
use crate::ff::style_registry::{column, declared};

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
/// >>> from molpy.ff.compile import ExplicitTerms
/// >>> pots = ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.5).compile()
/// >>> e, f = pots.calc_energy_forces(np.array([0.0, 0, 0, 1.6, 0, 0]))
/// >>> round(e, 12)
/// 3.0
#[pyclass(module = "molpy.ff.compile", name = "ExplicitTerms", frozen)]
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
        Ok(PyPotentials {
            inner: PotBacking::Compiled(pots),
            err_slots: vec![crate::ff::style_registry::kernel_err_slot()],
        })
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
        molrs::ff::style_registry::with_global_registry(|r| {
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
                Some(spec) => Some(declared(spec, pair, &key).ok_or_else(|| {
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
/// Exposed to Python as ``molpy.ff.compile.PotentialCompiler``. It owns a **copy** of
/// the force field, taken at construction: later edits to that
/// :class:`ForceField` do not reach a compiler that already exists — make a
/// new one.
///
/// Three doors, each doing one thing:
///
/// * :meth:`compile` — bind a typed :class:`Frame` now;
/// * :meth:`defer` — a :class:`Potentials` that binds the topology of the
///   :class:`Frame` it is evaluated on;
/// * :meth:`compile_typed` — the kernels of a neighbour-driven evaluation,
///   for MD.
///
/// Examples
/// --------
/// >>> compiler = molpy.ff.compile.PotentialCompiler(typifier.forcefield())
/// >>> potentials = compiler.compile(frame)
/// >>> energy = potentials.calc_energy(frame)
#[pyclass(module = "molpy.ff.compile", name = "PotentialCompiler", subclass)]
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
    /// resolves (``atoms``/``bonds``/``angles``/``dihedrals``/``impropers``/
    /// ``pairs``), as produced by a typifier or an external emitter. Every
    /// pair style is resolved against the frame's ``pairs`` block — a fixed
    /// list, right for a molecule in free space — and prices a row only
    /// inside its ``cutoff`` (``r < cutoff``, with its switch where it has
    /// one), as :meth:`compile_typed` and LAMMPS do; a style stating no
    /// ``cutoff`` prices every row. ``pair coul/long/pme`` reads the
    /// frame's periodic ``box``. When the frame has a ``pairs`` block, the
    /// real-space erfc term is priced over that list (an exclusion row
    /// omitted, a 1-4 row at the Coulomb 1-4 scale); with no ``pairs``
    /// block that term stays the minimum-image loop inside the kernel.
    /// The reciprocal sum still reads the box.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data. Required; for potentials that bind at
    ///     evaluation time use :meth:`defer`.
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
    ///     force field's 1-2 / 1-3 weights are not 0 or 1, or a style that
    ///     reads the box (``coul/long/pme``) meets a frame without a periodic
    ///     one.
    fn compile(&self, frame: &PyFrame) -> PyResult<PyPotentials> {
        crate::ff::style_registry::clear_kernel_err();
        let potentials = frame
            .with_frame(|core| PotentialCompiler::new(&self.ff).compile(core))?
            .map_err(ir::compile_err)?;
        crate::ff::style_registry::take_kernel_err()?;
        Ok(PyPotentials {
            inner: PotBacking::Compiled(potentials),
            err_slots: vec![crate::ff::style_registry::kernel_err_slot()],
        })
    }

    /// A :class:`Potentials` that compiles when it is evaluated.
    ///
    /// It holds this compiler's force field and binds the topology and
    /// coordinates of the :class:`Frame` passed to ``calc_energy(frame)`` /
    /// ``calc_forces(frame)`` (the molpy evaluation model). It has no members
    /// until then, so ``len`` is ``0``, and it cannot be evaluated on a bare
    /// coordinate array or moved into an integrator.
    ///
    /// Returns
    /// -------
    /// Potentials
    fn defer(&self) -> PyPotentials {
        PyPotentials {
            inner: PotBacking::Deferred(self.ff.clone()),
            err_slots: vec![crate::ff::style_registry::kernel_err_slot()],
        }
    }

    /// Build the kernels for a **neighbour-driven** evaluation, with the
    /// special-bonds weights each one takes.
    ///
    /// The counterpart of :meth:`compile`, and what periodic MD needs. That
    /// one resolves every pair style against the frame's ``pairs`` block — a
    /// fixed list, right for a free-boundary molecule and wrong for a
    /// periodic system; over the same pairs the two price the same energy. This one resolves them against the
    /// **atoms**, reads no ``pairs`` block, and requires the style's declared
    /// cutoff. ``pair coul/long/pme`` prices its real-space erfc over the
    /// neighbour table; the reciprocal sum still reads the periodic box and
    /// takes no pair weight.
    ///
    /// The weights come from the force field's ``special_bonds`` walked over
    /// the frame's bond graph. Without them a neighbour table would count a
    /// bonded pair twice: once by the bond term and once at full non-bonded
    /// strength, at bond length.
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
        let (topo, members) = frame.with_frame(|core| -> PyResult<_> {
            let topo = molrs::core::Topology::from_frame(core)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            crate::ff::style_registry::clear_kernel_err();
            let members = PotentialCompiler::new(&self.ff)
                .compile_typed(core)
                .map_err(ir::compile_err)?;
            crate::ff::style_registry::take_kernel_err()?;
            Ok((topo, members))
        })??;
        let bound = members
            .into_iter()
            .map(|(pot, weights)| {
                let special = weights
                    .map(|w| w.special_weights(&topo))
                    .unwrap_or_default();
                (pot, special)
            })
            .collect();
        Ok(PyWeightedTerms {
            members: Some(bound),
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

/// Register `molpy.ff.compile`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPotentialCompiler>()?;
    m.add_class::<PyExplicitTerms>()?;
    Ok(())
}
