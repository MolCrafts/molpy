//! Python bindings for `molrs::ff::potential` (`molrs.ff.potential`): the
//! evaluable force terms.
//!
//! * [`PyPotentials`] — a collection of kernels evaluated together
//!   (`calc_energy_forces`); `push` moves one more member in. A force field
//!   compiles into one (or into [`PyWeightedTerms`], each kernel with its
//!   special-bonds weights) through `molrs.ff.compile`.
//! * [`PyPairLjCut`] — the one-type `lj/cut` kernel a neighbour loop feeds (the
//!   MD integrators' nonbond kernel: `energy_forces_skin`, `energy_forces_table`, `energy_forces_pairs`).
//! * `intramolecular_pairs` — the special-bonds pair list of a typed frame.
//!
//! ```text
//! pots = Potentials()
//! pots.push(ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile())
//! pots.push(ExplicitTerms("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5).compile())
//! energy, forces = pots.calc_energy_forces(pos)
//! ```
//!
//! # The Python seam
//!
//! A Python object with ``calc_energy_forces`` (a
//! `molrs.ff.potential.Potential`) is a potential like any other:
//! [`take_potential`] turns any exposed potential — `PairLjCut`, `Potentials`, or
//! a duck-typed object wrapped as [`SubclassPotential`] — into one
//! [`ForceTerm`]; [`take_members`] does the same for a whole
//! [`PyWeightedTerms`]. The `Potential` trait has no error channel, so a
//! Python exception raised mid-evaluation is parked in an [`ErrSlot`] and
//! re-raised by the caller that drove the evaluation ([`take_err`]). The MD
//! integrators and `molrs.optimize.Lbfgs` consume potentials through here.

use super::ir;
use crate::array::{ArrayArg1, ArrayArg2};
use crate::core::block::PyBlock;
use crate::core::frame::PyFrame;
use crate::core::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::ff::forcefield::PyForceField;
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::forcefield::ForceField;
use molrs::ff::potential::pair::{PairLjCut, PairPotential};
use molrs::ff::potential::{ForceTerm, Potential, Potentials};
use molrs::op::F;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, ToPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// Register `molrs.ff.potential`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPotentials>()?;
    m.add_class::<PyWeightedTerms>()?;
    m.add_class::<PyPairLjCut>()?;
    crate::add_function(
        m,
        "molrs.ff.potential",
        wrap_pyfunction!(intramolecular_pairs_py, m)?,
    )?;
    Ok(())
}

/// LAMMPS ``pair_style lj/cut``: the one-type cut Lennard-Jones / Mie kernel
/// a neighbour loop feeds pairs to (MD's nonbond kernel). A pair list with a
/// row per pair is ``ExplicitTerms("pair", "lj/cut", pairs, epsilon=…, sigma=…).compile()``.
#[pyclass(name = "PairLjCut", module = "molrs.ff.potential", subclass)]
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

    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.inner.pair_energy(r2, disp)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.inner.pair_force(r2, disp)
    }
    fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.inner.pair_energy_force(r2, disp)
    }

    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: ArrayArg2<'_, f64>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
        check_nx3(&pos, "pos")?;
        let view = pos.as_array();
        let n = view.nrows();
        // A standard-layout `(N, 3)` array *is* the flat `[x0, y0, z0, …]` a
        // kernel wants; only a strided view is copied.
        let owned: Vec<F>;
        let flat: &[F] = match view.as_slice() {
            Some(slice) => slice,
            None => {
                owned = view.iter().copied().collect();
                &owned
            }
        };
        let (energy, forces) = Potential::calc_energy_forces(&self.inner, flat);
        let arr = Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((energy, arr.into_pyarray(py)))
    }

    fn energy_forces_skin<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: ArrayArg2<'_, f64>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors.get_mut()?;
        let (e, f) = self
            .inner
            .energy_forces_skin(nl, pos.as_array())
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    fn energy_forces_table<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        neighbors: &PyNeighbors,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
        let (e, f) = self
            .inner
            .energy_forces_table(n_atoms, &neighbors.inner)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    #[pyo3(signature = (n_atoms, i, j, disp, dist_sq=None))]
    fn energy_forces_pairs<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        i: ArrayArg1<'_, u32>,
        j: ArrayArg1<'_, u32>,
        disp: ArrayArg2<'_, f64>,
        dist_sq: Option<ArrayArg1<'_, f64>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<f64>>)> {
        check_nx3(&disp, "disp")?;
        let d2 = match dist_sq.as_ref() {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let (e, f) = self
            .inner
            .energy_forces_pairs(n_atoms, i.as_slice()?, j.as_slice()?, disp.as_array(), d2)
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }
}

/// The kernels of a neighbour-driven force evaluation, each with the
/// special-bonds weights that scale it.
///
/// Opaque on purpose: what a caller does with this is hand it to an
/// integrator. Taking it apart in Python would mean re-deciding which member
/// is which and how its close neighbours are scaled — the two things
/// :meth:`PotentialCompiler.compile_typed` exists to decide once.
#[pyclass(name = "WeightedTerms", module = "molrs.ff.potential", subclass)]
pub struct PyWeightedTerms {
    /// Taken by the integrator that consumes it; `None` afterwards.
    pub(crate) members: Option<Vec<(ForceTerm, molrs::ff::potential::SpecialWeights)>>,
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
/// Exposed to Python as `molrs.ff.potential.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = Mmff94Typifier()
/// >>> frame = typifier.typify(mol).to_frame()
/// >>> frame["pairs"] = molrs.ff.potential.intramolecular_pairs(frame)
/// >>> potentials = molrs.ff.compile.PotentialCompiler(typifier.forcefield()).compile(frame)
/// >>> energy, forces = potentials.calc_energy_forces(coords)
#[pyclass(module = "molrs.ff.potential", name = "Potentials", subclass)]
pub struct PyPotentials {
    pub(crate) inner: PotBacking,
    /// Error slots of every Python-callable member (see `ErrSlot`);
    /// checked after each evaluation so a callable's exception re-raises.
    pub(crate) err_slots: Vec<ErrSlot>,
}

/// A [`PyPotentials`] is either already compiled against a molecule's topology
/// (the MMFF / pre-bound path), *deferred* (it holds the force field and binds
/// the topology lazily from the `Frame` passed to
/// ``calc_energy``/``calc_forces`` — what ``PotentialCompiler.defer()``
/// returns, matching the molpy evaluation model), or *moved*: the
/// Rust `Potentials` has been moved into an MD integrator or another
/// collection.
pub(crate) enum PotBacking {
    Compiled(Potentials),
    Deferred(ForceField),
    Moved,
}

pub(crate) fn potentials_moved_err() -> PyErr {
    PyValueError::new_err(
        "this Potentials has been moved into an integrator or another \
         Potentials; rebuild with PotentialCompiler.compile(frame)",
    )
}

impl PotBacking {
    /// The compiled potentials, or an error if this set is still deferred and
    /// no `Frame` has been supplied to bind its topology (or already moved).
    pub(crate) fn compiled(&self) -> PyResult<&Potentials> {
        match self {
            PotBacking::Compiled(p) => Ok(p),
            PotBacking::Deferred(_) => Err(PyValueError::new_err(
                "this Potentials is not bound to a molecule; \
                 call calc_energy(frame)/calc_forces(frame) with a Frame, \
                 or build it from a typifier",
            )),
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }

    /// Mutable access with the same gating as [`compiled`](Self::compiled).
    fn compiled_mut(&mut self) -> PyResult<&mut Potentials> {
        match self {
            PotBacking::Compiled(p) => Ok(p),
            PotBacking::Deferred(_) => Err(PyValueError::new_err(
                "this Potentials is not bound to a molecule; \
                 call calc_energy(frame)/calc_forces(frame) with a Frame, \
                 or build it from a typifier",
            )),
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }
}

impl PyPotentials {
    /// Evaluate energy + forces against either a [`PyFrame`] (binds topology and
    /// reads coordinates from the frame's ``atoms`` block — the molpy model) or a
    /// flat coordinate array (requires already-compiled potentials).
    ///
    /// A Python-callable member's exception is re-raised afterwards (see
    /// `ErrSlot`).
    fn eval_any(&self, arg: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<f64>)> {
        let ef = if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let core = frame.clone_core_frame()?;
            let coords: Vec<f64> = core
                .coords()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_iter()
                .collect();
            match &self.inner {
                PotBacking::Compiled(p) => p.calc_energy_forces(&coords),
                PotBacking::Deferred(ff) => PotentialCompiler::new(ff)
                    .compile(&core)
                    .map_err(ir::compile_err)?
                    .calc_energy_forces(&coords),
                PotBacking::Moved => return Err(potentials_moved_err()),
            }
        } else {
            let arr = arg.extract::<crate::array::ArrayArg1<'_, f64>>()?;
            let slice = arr.as_slice()?;
            self.inner.compiled()?.calc_energy_forces(slice)
        };
        take_err(&self.err_slots)?;
        Ok(ef)
    }

    /// Move the compiled Rust `Potentials` (and the error slots of its
    /// Python-callable members) out, leaving this object in the moved state —
    /// the MD integrators and `Potentials.push` consume through here.
    pub(crate) fn take_compiled(&mut self) -> PyResult<(Potentials, Vec<ErrSlot>)> {
        match std::mem::replace(&mut self.inner, PotBacking::Moved) {
            PotBacking::Compiled(p) => Ok((p, std::mem::take(&mut self.err_slots))),
            deferred @ PotBacking::Deferred(_) => {
                self.inner = deferred;
                Err(PyValueError::new_err(
                    "this Potentials is not bound to a molecule; \
                     compile with PotentialCompiler.compile(frame) before moving it into \
                     an integrator",
                ))
            }
            PotBacking::Moved => Err(potentials_moved_err()),
        }
    }
}

#[pymethods]
impl PyPotentials {
    /// An empty collection; compose members with :meth:`push`.
    #[new]
    fn new() -> Self {
        Self {
            inner: PotBacking::Compiled(Potentials::new()),
            err_slots: Vec::new(),
        }
    }

    /// Number of compiled potential kernels, or ``0`` while still deferred
    /// (not yet bound to a molecule) or moved into an integrator.
    fn __len__(&self) -> usize {
        match &self.inner {
            PotBacking::Compiled(p) => p.len(),
            PotBacking::Deferred(_) | PotBacking::Moved => 0,
        }
    }

    /// Move one more member into the collection: an ``PairLjCut`` nonbond term, a
    /// callable ``Potential``, or another ``Potentials``.
    fn push(&mut self, potential: &Bound<'_, PyAny>) -> PyResult<()> {
        // Gate first so a failed push does not consume the pushed potential.
        self.inner.compiled_mut()?;
        let (member, mut slots) = take_potential(potential)?;
        self.inner.compiled_mut()?.push(member);
        self.err_slots.append(&mut slots);
        Ok(())
    }

    /// Returns ``(energy, forces)``, forces shape ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<f64>>)> {
        let (energy, forces) = self.eval_any(arg)?;
        let n = forces.len() / 3;
        Ok((
            energy,
            Array2::from_shape_vec((n, 3), forces)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .to_pyarray(py),
        ))
    }

    /// Evaluate total energy (kcal/mol) against a :class:`Frame` or coordinates.
    fn calc_energy(&self, arg: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.eval_any(arg)?.0)
    }

    /// Compute forces (= -gradient) in kcal/(mol·Å), shape ``(N, 3)``.
    fn calc_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let forces = self.eval_any(arg)?.1;
        let n = forces.len() / 3;
        Ok(Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PotBacking::Compiled(p) => format!("Potentials(n_kernels={})", p.len()),
            PotBacking::Deferred(_) => "Potentials(deferred)".to_string(),
            PotBacking::Moved => "Potentials(<moved into integrator>)".to_string(),
        }
    }
}

/// Build the intramolecular non-bonded neighbour list for a typed frame.
///
/// Returns a :class:`Block` with ``atomi`` / ``atomj`` / ``is_14`` columns — the
/// exact list :meth:`PotentialCompiler.compile` consumes for the pair (van der
/// Waals + Coulomb) kernels. 1-4 pairs (from ``dihedrals``) are flagged so the
/// kernels apply the force field's 1-4 scaling.
///
/// Which 1-2 / 1-3 neighbours belong in the list is the **force field's**
/// decision, so pass it: LAMMPS ``special_bonds fene`` (``[0, 1, 1]``) keeps
/// 1-3 pairs at full strength, and a bead-spring chain without them has
/// nothing holding it open. Omitting ``forcefield`` excludes both classes —
/// what every Amber-family force field wants, and what this function always
/// did before it could be told otherwise.
///
/// Insert the result as the frame's ``"pairs"`` block before
/// :meth:`PotentialCompiler.compile` when you need the non-bonded terms — e.g. a
/// single-molecule geometry optimization where intramolecular van der Waals
/// drives chain collapse. (``compile`` silently drops the pair styles when
/// no ``"pairs"`` block is present, so bonded-only optimizations need nothing.)
///
/// Parameters
/// ----------
/// frame : Frame
///     A typed frame with ``atoms`` and the topology blocks
///     (``bonds`` / ``angles`` / ``dihedrals``) used for exclusions.
/// forcefield : ForceField, optional
///     The force field whose ``special_bonds`` decide the 1-2 / 1-3 rows.
///     Defaults to excluding both.
///
/// Returns
/// -------
/// Block
///
/// Raises
/// ------
/// ValueError
///     The force field scales 1-2 or 1-3 neighbours by a fraction, or scales
///     them differently for van der Waals and Coulomb. A list of rows cannot
///     say either; use :meth:`PotentialCompiler.compile_typed`, which carries a
///     per-pair weight.
#[pyfunction]
#[pyo3(name = "intramolecular_pairs", signature = (frame, forcefield = None))]
pub fn intramolecular_pairs_py(
    frame: &PyFrame,
    forcefield: Option<&PyForceField>,
) -> PyResult<PyBlock> {
    let owned;
    let special = match forcefield {
        Some(ff) => ff.inner.special_bonds(),
        None => {
            owned = molrs::ff::ir::SpecialBonds::default();
            &owned
        }
    };
    let block = frame
        .with_frame(|core| molrs::ff::potential::intramolecular_pairs(core, special))?
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
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

/// A Python object with `calc_energy_forces` (a `molrs.ff.potential.Potential`)
/// as the one `Potential` concept — the seam for NN / external forces. Holds a reference to the instance and
/// dispatches to its overridden ``calc_energy_forces`` under the GIL.
pub struct SubclassPotential {
    obj: Py<PyAny>,
    error: ErrSlot,
}

impl SubclassPotential {
    fn call(&self, py: Python<'_>, coords: &[F]) -> PyResult<(F, Vec<F>)> {
        let n = coords.len() / 3;
        let pos = Array2::from_shape_vec((n, 3), coords.to_vec())
            .expect("flat coords have 3N elements")
            .into_pyarray(py);
        let result = self
            .obj
            .bind(py)
            .call_method1("calc_energy_forces", (pos,))?;
        let (energy, forces): (F, ArrayArg2<'_, f64>) = result.extract().map_err(|_| {
            PyValueError::new_err(
                "Potential.calc_energy_forces must return \
                 (energy: float, forces: float64 (N, 3) ndarray)",
            )
        })?;
        let forces = forces.as_array();
        if forces.shape() != [n, 3] {
            return Err(PyValueError::new_err(format!(
                "Potential.calc_energy_forces returned forces shape {:?} for {n} atoms",
                forces.shape()
            )));
        }
        Ok((energy, forces.iter().copied().collect()))
    }
}

impl Potential for SubclassPotential {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        Python::attach(|py| match self.call(py, coords) {
            Ok(out) => out,
            Err(err) => {
                *self.error.lock().expect("error slot poisoned") = Some(err);
                (F::NAN, vec![F::NAN; coords.len()])
            }
        })
    }
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
    let (pot, slots) = take_potential(obj)?;
    Ok((
        vec![(pot, molrs::ff::potential::SpecialWeights::default())],
        slots,
    ))
}

pub(crate) fn take_potential(obj: &Bound<'_, PyAny>) -> PyResult<(ForceTerm, Vec<ErrSlot>)> {
    // Each arm also settles which part the member plays. A pair kernel and an
    // aggregate of them read a neighbour table; a duck-typed Python object has
    // only `calc_energy_forces`, so it reads coordinates and nothing else —
    // and, being unable to tally a virial over pairs, makes the step's virial
    // `None` rather than a number that moves with the box origin.
    if let Ok(lj) = obj.cast::<PyPairLjCut>() {
        return Ok((ForceTerm::pair(lj.borrow().inner.clone()), Vec::new()));
    }
    if let Ok(pots) = obj.cast::<PyPotentials>() {
        let (inner, slots) = pots.borrow_mut().take_compiled()?;
        return Ok((ForceTerm::pair(inner), slots));
    }
    if obj.hasattr("calc_energy_forces")? && obj.getattr("calc_energy_forces")?.is_callable() {
        let error: ErrSlot = Arc::default();
        return Ok((
            ForceTerm::plain(SubclassPotential {
                obj: obj.clone().unbind(),
                error: Arc::clone(&error),
            }),
            vec![error],
        ));
    }
    Err(PyTypeError::new_err(
        "expected a potential with callable calc_energy_forces (PairLjCut, Potentials, or duck-typed)",
    ))
}
