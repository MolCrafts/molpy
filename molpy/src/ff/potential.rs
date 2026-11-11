//! Python bindings for `molrs::ff::potential` (`molpy.ff.potential`): the
//! evaluable force terms.
//!
//! * [`PyPotentials`] — a collection of kernels evaluated together
//!   (`evaluate`, the mirror of `Potential::evaluate` into a fresh
//!   `Accumulator`); `push` moves one more member in. A force field compiles
//!   into one (or into [`PyWeightedTerms`], each kernel with its
//!   special-bonds weights) through `molpy.ff.PotentialCompiler`.
//! * [`PyPairLjCut`] — the one-type `lj/cut` kernel a neighbour loop feeds (the
//!   MD integrators' nonbond kernel): `pair_energy_force` for one pair,
//!   `evaluate_pairs` (`PairDriven::evaluate_pairs`) over a pair table.
//!
//! ```text
//! pots = Potentials()
//! pots.push(ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile())
//! pots.push(ExplicitTerms("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5).compile())
//! energy, forces = pots.evaluate(pos)
//! ```
//!
//! # The Python seam
//!
//! A Python object satisfying the `molpy.ff.potential.Potential` protocol
//! (``units``, ``cutoff``, ``bind``, ``compute``) is a potential like any
//! other: [`take_potential`] turns any exposed potential — `PairLjCut`,
//! `Potentials`, or a Python object wrapped as a
//! [`PythonMember`](super::python_potential::PythonMember) — into one
//! [`ForceTerm`]; [`take_members`] does the same for a whole
//! [`PyWeightedTerms`]. The `Potential` trait has no error channel, so a
//! Python exception raised mid-evaluation is parked in an [`ErrSlot`] and
//! re-raised by the caller that drove the evaluation ([`take_err`]). MD mounts
//! a Python potential itself (``MD.set_evaluator``), as a bound potential that
//! stops the run at a failing step.

use super::ir;
use crate::array::{ArrayArg1, ArrayArg2};
use crate::core::frame::PyFrame;
use crate::core::neighborlist::PyNeighbors;
use molrs::core::Frame;
use molrs::ff::potential::pair::{PairLjCut, PairPotential};
use molrs::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential, Potentials};
use molrs::op::F;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

use super::python_potential::{PythonMember, SharedMember};

/// Register `molpy.ff.potential`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPotentials>()?;
    m.add_class::<PyWeightedTerms>()?;
    m.add_class::<PyPairLjCut>()?;
    Ok(())
}

/// LAMMPS ``pair_style lj/cut``: the one-type cut Lennard-Jones / Mie kernel
/// a neighbour loop feeds pairs to (MD's nonbond kernel). A pair list with a
/// row per pair is ``ExplicitTerms("pair", "lj/cut", pairs, epsilon=…, sigma=…).compile()``.
#[pyclass(name = "PairLjCut", module = "molpy.ff.potential", subclass)]
pub struct PyPairLjCut {
    pub(crate) inner: PairLjCut,
}

#[pymethods]
impl PyPairLjCut {
    #[new]
    #[pyo3(signature = (epsilon, sigma, cutoff, *, n=12, m=6, shifted=true, smeared=false))]
    fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PairLjCut::new(epsilon, sigma, cutoff, n, m, shifted, smeared)
                .map_err(PyValueError::new_err)?,
        })
    }

    #[getter]
    fn epsilon(&self) -> F {
        self.inner.epsilon()
    }
    #[getter]
    fn sigma(&self) -> F {
        self.inner.sigma()
    }
    #[getter]
    fn cutoff(&self) -> F {
        self.inner.cutoff()
    }
    #[getter]
    fn n(&self) -> i32 {
        self.inner.n()
    }
    #[getter]
    fn m(&self) -> i32 {
        self.inner.m()
    }
    #[getter]
    fn shifted(&self) -> bool {
        self.inner.shifted()
    }
    #[getter]
    fn smeared(&self) -> bool {
        self.inner.smeared()
    }

    /// Energy and force of one pair at squared distance ``r2`` along the
    /// separation ``disp`` (``PairPotential::pair_energy_force``), or
    /// ``None`` beyond the cutoff.
    fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.inner.pair_energy_force(r2, disp)
    }

    /// Energy and forces of every pair in ``neighbors``
    /// (``PairDriven::evaluate_pairs``): ``(energy, forces)``, forces of
    /// shape ``(N, 3)`` for the ``N`` rows of ``pos``.
    ///
    /// ``neighbors`` must carry its separations (built with ``disp=True``);
    /// ``factor`` is one weight per pair (``None``: all ones), a weight of
    /// exactly zero skipping the pair.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``pos`` is not ``(N, 3)``, ``neighbors`` has no ``disp`` or names a
    ///     point past ``N``, or ``factor`` is not one weight per pair.
    #[pyo3(signature = (pos, neighbors, factor = None))]
    fn evaluate_pairs<'py>(
        &self,
        py: Python<'py>,
        pos: ArrayArg2<'_, f64>,
        neighbors: &PyNeighbors,
        factor: Option<ArrayArg1<'_, f64>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
        check_nx3(&pos, "pos")?;
        let table = &neighbors.inner;
        if table.disp().is_none() {
            return Err(PyValueError::new_err(
                "evaluate_pairs needs the pair separations: build the neighbour \
                 table with disp=True",
            ));
        }
        let n = pos.as_array().nrows();
        let past = table
            .query_point_indices()
            .iter()
            .chain(table.point_indices())
            .any(|&i| i as usize >= n);
        if past {
            return Err(PyValueError::new_err(format!(
                "neighbors names a point past the {n} rows of pos"
            )));
        }
        let factor: &[F] = match factor.as_ref() {
            Some(f) => f.as_slice()?,
            None => &[],
        };
        if !factor.is_empty() && factor.len() != table.query_point_indices().len() {
            return Err(PyValueError::new_err(format!(
                "factor has {} weights for {} pairs",
                factor.len(),
                table.query_point_indices().len()
            )));
        }
        let coords = flat_coords(&pos);
        let mut acc = Accumulator::new(coords.len());
        self.inner.evaluate_pairs(&coords, table, factor, &mut acc);
        let energy = acc.energy();
        Ok((energy, forces_nx3(py, acc.into_forces())?))
    }
}

/// `pos` (``(N, 3)``) as flat `[x0, y0, z0, …]` coordinates.
fn flat_coords(pos: &ArrayArg2<'_, f64>) -> Vec<F> {
    pos.as_array().iter().copied().collect()
}

/// Flat `3N` forces as an ``(N, 3)`` array.
fn forces_nx3(py: Python<'_>, forces: Vec<F>) -> PyResult<Bound<'_, PyArray2<f64>>> {
    let n = forces.len() / 3;
    Ok(Array2::from_shape_vec((n, 3), forces)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray(py))
}

/// The kernels of a neighbour-driven force evaluation, each with the
/// special-bonds weights that scale it.
///
/// Opaque on purpose: what a caller does with this is hand it to an
/// integrator. Taking it apart in Python would mean re-deciding which member
/// is which and how its close neighbours are scaled — the two things
/// :meth:`PotentialCompiler.compile_typed` exists to decide once.
#[pyclass(name = "WeightedTerms", module = "molpy.ff.potential", subclass)]
pub struct PyWeightedTerms {
    /// Taken by the integrator that consumes it; `None` afterwards.
    pub(crate) members: Option<Vec<(ForceTerm, molrs::ff::potential::SpecialWeights)>>,
    /// The `units` the compiled force field declared; an MD run refuses
    /// another style.
    pub(crate) units: Option<String>,
}

#[pymethods]
impl PyWeightedTerms {
    /// How many kernels this carries.
    fn __len__(&self) -> usize {
        self.members.as_ref().map_or(0, |m| m.len())
    }
}

/// Compiled force-field potentials for energy and force evaluation.
///
/// Exposed to Python as `molpy.ff.potential.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = Mmff94Typifier()
/// >>> frame = typifier.typify(mol).to_frame()
/// >>> potentials = molpy.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
/// >>> energy, forces = potentials.evaluate(coords)
#[pyclass(module = "molpy.ff.potential", name = "Potentials", subclass)]
pub struct PyPotentials {
    /// The compiled kernels; `None` once moved into an integrator or another
    /// collection.
    pub(crate) inner: Option<Potentials>,
    /// Error slots of every Python-callable member (see `ErrSlot`);
    /// checked after each evaluation so a callable's exception re-raises.
    pub(crate) err_slots: Vec<ErrSlot>,
    /// The Python potentials among the members, to bind to a caller's frame.
    pub(crate) python: Vec<SharedMember>,
    /// The `units` the compiled force field declared; an MD run refuses
    /// another style. `None` for a collection assembled by hand.
    pub(crate) units: Option<String>,
}

pub(crate) fn potentials_moved_err() -> PyErr {
    PyValueError::new_err(
        "this Potentials has been moved into an integrator or another \
         Potentials; rebuild with PotentialCompiler.compile(frame)",
    )
}

impl PyPotentials {
    /// Kernels compiled by a [`PotentialCompiler`] door, ready to evaluate.
    pub(crate) fn of(potentials: Potentials, units: Option<String>) -> Self {
        Self {
            inner: Some(potentials),
            err_slots: vec![crate::ff::style_registry::kernel_err_slot()],
            python: Vec::new(),
            units,
        }
    }

    /// The compiled kernels, or the error of a moved collection.
    pub(crate) fn compiled(&self) -> PyResult<&Potentials> {
        self.inner.as_ref().ok_or_else(potentials_moved_err)
    }

    /// Evaluate energy + forces at a [`PyFrame`]'s ``atoms`` coordinates or at a
    /// flat coordinate array.
    ///
    /// A Python-callable member's exception is re-raised afterwards (see
    /// `ErrSlot`).
    fn eval_any(&self, arg: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>)> {
        let p = self.compiled()?;
        // A pair kernel over a type table reads a neighbour table and adds
        // nothing here: refuse it instead of returning its zero.
        p.require_pair_lists().map_err(ir::compile_err)?;
        let coords: Vec<f64> = if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let frame = frame.clone_core_frame()?;
            self.bind_python(&frame)?;
            frame
                .coords()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_iter()
                .collect()
        } else {
            let arr = arg.extract::<crate::array::ArrayArg1<'_, f64>>()?;
            let slice = arr.as_slice()?;
            if slice.len() % 3 != 0 {
                return Err(PyValueError::new_err(format!(
                    "coords must be flat [x0, y0, z0, x1, ...] of length 3N, got {}",
                    slice.len()
                )));
            }
            slice.to_vec()
        };
        let mut acc = Accumulator::new(coords.len());
        p.evaluate(&coords, &mut acc);
        take_err(&self.err_slots)?;
        Ok((acc.energy(), acc.into_forces()))
    }

    /// Move the compiled Rust `Potentials` (and the error slots of its
    /// Python-callable members) out, leaving this object in the moved state —
    /// the MD integrators and `Potentials.push` consume through here.
    /// Whether a Python potential sits among the members.
    pub(crate) fn holds_python_members(&self) -> bool {
        !self.python.is_empty()
    }

    /// Bind every Python member to `frame`, so it reads that frame's species
    /// and box rather than bare coordinates.
    pub(crate) fn bind_python(&self, frame: &Frame) -> PyResult<()> {
        for member in &self.python {
            member
                .lock()
                .expect("python potential state poisoned")
                .bind(frame)?;
        }
        Ok(())
    }

    pub(crate) fn take_compiled(&mut self) -> PyResult<Taken> {
        let p = self.inner.take().ok_or_else(potentials_moved_err)?;
        Ok(Taken {
            term: ForceTerm::pair(p),
            err_slots: std::mem::take(&mut self.err_slots),
            python: std::mem::take(&mut self.python),
        })
    }
}

#[pymethods]
impl PyPotentials {
    /// An empty collection; compose members with :meth:`push`.
    #[new]
    fn new() -> Self {
        Self {
            inner: Some(Potentials::new()),
            err_slots: Vec::new(),
            python: Vec::new(),
            units: None,
        }
    }

    /// Number of compiled potential kernels, or ``0`` once moved into an
    /// integrator.
    fn __len__(&self) -> usize {
        self.inner.as_ref().map_or(0, Potentials::len)
    }

    /// Move one more member into the collection: a ``PairLjCut`` nonbond term,
    /// a Python ``Potential``, or another ``Potentials``.
    ///
    /// A ``PairLjCut`` reads a neighbour table, so a collection holding one is
    /// priced by an MD evaluator (``set_evaluator``), never by
    /// :meth:`evaluate` or a minimizer, which refuse it.
    fn push(&mut self, potential: &Bound<'_, PyAny>) -> PyResult<()> {
        // Gate first so a failed push does not consume the pushed potential.
        self.compiled()?;
        let mut taken = take_potential(potential)?;
        self.inner
            .as_mut()
            .ok_or_else(potentials_moved_err)?
            .push(taken.term);
        self.err_slots.append(&mut taken.err_slots);
        self.python.append(&mut taken.python);
        Ok(())
    }

    /// Energy and forces (``-∇E``) of every member (``Potential::evaluate``
    /// into a fresh accumulator): ``(energy, forces)``, forces of shape
    /// ``(N, 3)``.
    ///
    /// ``arg`` is a :class:`Frame` (its ``atoms`` coordinates) or flat
    /// coordinates ``[x0, y0, z0, x1, ...]`` of length ``3N``.
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        let (energy, forces) = self.eval_any(arg)?;
        Ok((energy, forces_nx3(py, forces)?))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(p) => format!("Potentials(n_kernels={})", p.len()),
            None => "Potentials(<moved into integrator>)".to_string(),
        }
    }
}

pub(crate) fn check_nx3(arr: &ArrayArg2<'_, f64>, label: &str) -> PyResult<()> {
    if arr.as_array().ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The one Potential seam — Python subclasses and post-evaluation error relay.
// ---------------------------------------------------------------------------

/// Shared slot where a Python-subclass potential parks an exception raised
/// mid-evaluation — the `Potential` trait has no error channel, so the
/// evaluation returns NaNs and the Python-facing caller that drove it checks
/// the slot and re-raises the original exception.
pub(crate) type ErrSlot = Arc<Mutex<Option<PyErr>>>;

/// Re-raise the first parked exception, clearing its slot.
pub(crate) fn take_err(slots: &[ErrSlot]) -> PyResult<()> {
    for slot in slots {
        if let Some(err) = slot.lock().expect("error slot poisoned").take() {
            return Err(err);
        }
    }
    Ok(())
}

/// The members a provider will evaluate, with the weights each one takes.
///
/// A [`WeightedTerms`](PyWeightedTerms) already knows both —
/// which kernel is which and how its close neighbours are scaled — because
/// `PotentialCompiler::compile_typed` decided it. Anything else is one member
/// that scales nothing.
pub(crate) type Members = Vec<(ForceTerm, molrs::ff::potential::SpecialWeights)>;

/// Move the Rust potential out of any exposed potential class.
///
/// Arm order is a hard invariant: concrete Rust types first, duck-typed
/// fallback last. Putting the fallback first would wrap every `Potentials`
/// as a Python dispatch object.
pub(crate) fn take_members(obj: &Bound<'_, PyAny>) -> PyResult<(Members, Vec<ErrSlot>)> {
    if let Ok(typed) = obj.cast::<PyWeightedTerms>() {
        let members = typed.borrow_mut().members.take().ok_or_else(|| {
            PyValueError::new_err(
                "these WeightedTerms were already given to an integrator; \
                 build them again from the force field",
            )
        })?;
        // A Python kernel among them parks its exception here.
        return Ok((members, vec![crate::ff::style_registry::kernel_err_slot()]));
    }
    let taken = take_potential(obj)?;
    Ok((
        vec![(taken.term, molrs::ff::potential::SpecialWeights::default())],
        taken.err_slots,
    ))
}

/// One exposed potential, moved out: its term, the error slots of its Python
/// members, and those members.
pub(crate) struct Taken {
    pub(crate) term: ForceTerm,
    pub(crate) err_slots: Vec<ErrSlot>,
    pub(crate) python: Vec<SharedMember>,
}

pub(crate) fn take_potential(obj: &Bound<'_, PyAny>) -> PyResult<Taken> {
    // Each arm also settles which part the member plays. A pair kernel and an
    // aggregate of them read a neighbour table; a Python potential reads
    // coordinates and nothing else here — and, reporting no virial through
    // this door, makes the step's virial `None` rather than a number that
    // moves with the box origin.
    if let Ok(lj) = obj.cast::<PyPairLjCut>() {
        return Ok(Taken {
            term: ForceTerm::pair(lj.borrow().inner.clone()),
            err_slots: Vec::new(),
            python: Vec::new(),
        });
    }
    if let Ok(pots) = obj.cast::<PyPotentials>() {
        return pots.borrow_mut().take_compiled();
    }
    let (member, state, error) = PythonMember::new(obj)?;
    Ok(Taken {
        term: ForceTerm::plain(member),
        err_slots: vec![error],
        python: vec![state],
    })
}
