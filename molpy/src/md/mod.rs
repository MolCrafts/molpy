//! Python bindings for [`molrs::md`]: the phase driver and the components
//! it mounts. Potentials stay in `molpy.ff.potential`; a Python one is mounted
//! as a [`BoundPotential`] through [`PythonPotential`].

use crate::array::ArrayArg2;
use crate::core::frame::PyFrame;
use crate::core::neighborlist::PyNeighborList;
use crate::core::units::preset_arg;
use crate::error::ffi_error_to_pyerr;
use crate::ff::potential::{
    ErrSlot, PyPairLjCut, PyPotentials, PyWeightedTerms, take_err, take_members,
};
use crate::ff::python_potential::PythonPotential;
use molrs::core::{Frame, UnitPreset};
use molrs::md::{
    BerendsenBarostat, BoundPotential, FfEvaluator, FixLangevin, MD, MaxwellBoltzmann, MdError,
    PotentialError, PotentialResult, PotentialStep, TimestepProvider, VelocityVerlet,
};
use molrs::op::F;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A Python potential's own exception comes back unchanged; every other
/// refusal is a `ValueError`.
fn md_err(e: MdError) -> PyErr {
    match e {
        MdError::Potential { source, .. } => match source.downcast::<PyErr>() {
            Ok(err) => *err,
            Err(other) => PyValueError::new_err(other.to_string()),
        },
        other => PyValueError::new_err(other.to_string()),
    }
}

fn check_nx3(pos: &ArrayArg2<'_, f64>, what: &str) -> PyResult<()> {
    let view = pos.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{what} must have shape (N, 3), got {shape:?}"
        )));
    }
    Ok(())
}

fn mass_from(mass: &Bound<'_, PyAny>) -> PyResult<Array1<F>> {
    match mass.extract::<crate::array::ArrayArg1<f64>>() {
        Ok(arr) => {
            if arr.as_array().iter().any(|&m| !m.is_finite() || m <= 0.0) {
                return Err(PyValueError::new_err(
                    "mass must be finite and strictly positive",
                ));
            }
            Ok(arr.as_array().to_owned())
        }
        Err(_) if mass.extract::<F>().is_ok() => Err(PyTypeError::new_err(
            "mass: expected a 1-D float64 array of per-atom masses (N,), got a scalar; \
             for a homogeneous system pass np.full(n_atoms, mass)",
        )),
        Err(e) => Err(PyTypeError::new_err(format!(
            "mass: {}",
            e.value(mass.py())
        ))),
    }
}

/// LAMMPS Verlet phase names. `Phase.force` is the force accumulation point.
#[pyclass(
    eq,
    eq_int,
    frozen,
    hash,
    name = "Phase",
    module = "molpy.md",
    skip_from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyPhase {
    #[pyo3(name = "initial_integrate")]
    InitialIntegrate,
    #[pyo3(name = "post_integrate")]
    PostIntegrate,
    #[pyo3(name = "pre_exchange")]
    PreExchange,
    #[pyo3(name = "pre_neighbor")]
    PreNeighbor,
    #[pyo3(name = "neighbor")]
    Neighbor,
    #[pyo3(name = "pre_force")]
    PreForce,
    #[pyo3(name = "force")]
    Force,
    #[pyo3(name = "post_force")]
    PostForce,
    #[pyo3(name = "final_integrate")]
    FinalIntegrate,
    #[pyo3(name = "end_of_step")]
    EndOfStep,
}

/// NVE velocity Verlet. `dt` only — mass, positions, and velocities are frame columns.
#[pyclass(
    name = "VelocityVerlet",
    module = "molpy.md",
    subclass,
    skip_from_py_object
)]
pub struct PyVelocityVerlet {
    inner: VelocityVerlet,
}

#[pymethods]
impl PyVelocityVerlet {
    #[new]
    fn new(dt: F) -> PyResult<Self> {
        Ok(Self {
            inner: VelocityVerlet::new(dt).map_err(md_err)?,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
}

/// LAMMPS `fix langevin` at one temperature, mounted at `post_force`.
///
/// `temperature` is in the run's temperature unit, `damp` and `dt` in its
/// time unit; Boltzmann's constant comes from the run's units.
#[pyclass(
    name = "FixLangevin",
    module = "molpy.md",
    subclass,
    skip_from_py_object
)]
pub struct PyFixLangevin {
    inner: FixLangevin,
}

#[pymethods]
impl PyFixLangevin {
    #[new]
    #[pyo3(signature = (temperature, damp, dt, *, seed))]
    fn new(temperature: F, damp: F, dt: F, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: FixLangevin::new(temperature, damp, dt, seed).map_err(md_err)?,
        })
    }

    #[getter]
    fn temperature(&self) -> F {
        self.inner.temperature()
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
}

/// Isotropic Berendsen pressure coupling, mounted at `end_of_step`.
///
/// `pressure` and `bulk_modulus` are in the run's pressure unit (atm in
/// `real`, bar in `metal`), `tau` and `dt` in its time unit.
#[pyclass(
    name = "BerendsenBarostat",
    module = "molpy.md",
    subclass,
    skip_from_py_object
)]
pub struct PyBerendsenBarostat {
    inner: BerendsenBarostat,
}

#[pymethods]
impl PyBerendsenBarostat {
    #[new]
    #[pyo3(signature = (pressure, tau, bulk_modulus, dt))]
    fn new(pressure: F, tau: F, bulk_modulus: F, dt: F) -> PyResult<Self> {
        Ok(Self {
            inner: BerendsenBarostat::new(pressure, tau, bulk_modulus, dt).map_err(md_err)?,
        })
    }

    #[getter]
    fn pressure(&self) -> F {
        self.inner.pressure()
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
}

/// The phase driver. `run` writes the frame it is given.
#[pyclass(name = "MD", module = "molpy.md", unsendable, subclass)]
pub struct PyMD {
    inner: MD,
    /// Where a Python pair style among the evaluator's kernels parks its
    /// exception (the `Potential` trait has no error channel).
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyMD {
    /// A driver in ``units`` (a preset name such as ``"real"`` or a
    /// :class:`molpy.core.UnitPreset`): ``dt`` is in its time unit, masses,
    /// energies and forces in its units.
    #[new]
    fn new(units: &Bound<'_, PyAny>) -> PyResult<Self> {
        let preset = preset_arg(units)?;
        Ok(Self {
            inner: MD::new(preset).map_err(md_err)?,
            err_slots: Vec::new(),
        })
    }

    /// The step index of the last force evaluation; it counts across runs.
    #[getter]
    fn step(&self) -> u64 {
        self.inner.step()
    }

    /// Continue from a checkpoint: a frame written after step ``step``,
    /// forces included. The next :meth:`run` resumes the count there and
    /// takes the frame's forces as that step's instead of evaluating them
    /// again.
    fn resume(&mut self, step: u64) {
        self.inner.resume(step);
    }

    fn set_integrator(&mut self, integrator: &PyVelocityVerlet) -> PyResult<()> {
        self.inner.set_integrator(integrator.inner).map_err(md_err)
    }

    fn set_thermostat(&mut self, thermostat: &PyFixLangevin) -> PyResult<()> {
        self.inner.set_thermostat(thermostat.inner).map_err(md_err)
    }

    /// Mount ``barostat``, replacing any before it. Every force provider must
    /// give the step a virial; ``meta["pressure"]`` is written each step.
    fn set_barostat(&mut self, barostat: &PyBerendsenBarostat) -> PyResult<()> {
        self.inner.set_barostat(barostat.inner).map_err(md_err)
    }

    /// Mount ``list`` (moved in) as the run's neighbour list. It searches at
    /// its own cutoff; pairs within ``cutoff - skin`` reach the pair kernels,
    /// and it is searched again once an atom has moved half the ``skin``
    /// (``0`` searches every step). The frame box supplies the periodicity.
    #[pyo3(signature = (list, *, skin=0.0))]
    fn set_neighbor(&mut self, mut list: PyRefMut<'_, PyNeighborList>, skin: F) -> PyResult<()> {
        let list = list.take()?;
        self.inner.set_neighbor(list, skin).map_err(md_err)
    }

    /// Mount the run's evaluator, replacing any before it: native kernels (a
    /// `PairLjCut`, a compiled `Potentials` / `WeightedTerms`) or a Python
    /// :class:`molpy.ff.potential.Potential`.
    ///
    /// ``skin`` is for a Python potential: the margin its own neighbour search
    /// builds at beyond its ``cutoff``, the driver saying when to search again.
    /// Native pair kernels read the run's neighbour list (:meth:`set_neighbor`),
    /// whose skin is that list's.
    #[pyo3(signature = (potential, *, skin=0.0))]
    fn set_evaluator(&mut self, potential: &Bound<'_, PyAny>, skin: F) -> PyResult<()> {
        let native = potential.is_instance_of::<PyPairLjCut>()
            || potential.is_instance_of::<PyPotentials>()
            || potential.is_instance_of::<PyWeightedTerms>();
        if !native {
            let python = PythonPotential::new(potential)?;
            self.err_slots.clear();
            return self
                .inner
                .set_bound_potential(MountedPython(python), skin)
                .map_err(md_err);
        }
        if potential
            .cast::<PyPotentials>()
            .is_ok_and(|p| p.borrow().holds_python_members())
        {
            return Err(PyTypeError::new_err(
                "this Potentials holds a Python potential; MD mounts a Python potential \
                 itself, set_evaluator(potential), so it gets the cell, the rebuild cadence \
                 and an error that stops the run",
            ));
        }
        if skin != 0.0 {
            return Err(PyValueError::new_err(
                "skin is a Python potential's; native pair kernels read the neighbour \
                 list of set_neighbor, whose skin is that list's",
            ));
        }
        let (members, slots) = take_members(potential)?;
        self.err_slots = slots;
        self.inner
            .set_evaluator(FfEvaluator::from_members(members))
            .map_err(md_err)
    }

    /// Advance `frame` for `n_steps`. Coordinates, velocities, forces, image
    /// flags, and `meta["energy"]` / `meta["kinetic"]` are written on `frame`.
    fn run(&mut self, frame: &PyFrame, n_steps: usize) -> PyResult<()> {
        frame
            .inner
            .with_mut(|f| self.inner.run(f, n_steps))
            .map_err(ffi_error_to_pyerr)?
            .map_err(md_err)?;
        take_err(&self.err_slots)
    }
}

#[pyclass(name = "MaxwellBoltzmann", module = "molpy.md", subclass)]
pub struct PyMaxwellBoltzmann {
    inner: MaxwellBoltzmann,
}

#[pymethods]
impl PyMaxwellBoltzmann {
    #[new]
    /// Velocities at ``temperature`` in ``units`` (a preset name or a
    /// :class:`molpy.core.UnitPreset`), in its velocity unit.
    #[pyo3(signature = (temperature, *, units, seed=0, remove_com=true))]
    fn new(
        temperature: F,
        units: &Bound<'_, PyAny>,
        seed: u64,
        remove_com: bool,
    ) -> PyResult<Self> {
        let mut inner =
            MaxwellBoltzmann::new(temperature, &preset_arg(units)?, seed).map_err(md_err)?;
        if !remove_com {
            inner = inner.keep_com();
        }
        Ok(Self { inner })
    }

    #[getter]
    fn temperature(&self) -> F {
        self.inner.temperature()
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }
    #[getter]
    fn remove_com(&self) -> bool {
        self.inner.remove_com()
    }

    fn velocities<'py>(
        &self,
        py: Python<'py>,
        pos: ArrayArg2<'_, f64>,
        mass: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        check_nx3(&pos, "pos")?;
        let vel = self
            .inner
            .velocities(pos.as_array(), mass_from(&mass)?.view())
            .map_err(md_err)?;
        Ok(vel.into_pyarray(py))
    }
}

/// A Python potential as the run's [`BoundPotential`].
struct MountedPython(PythonPotential);

impl BoundPotential for MountedPython {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn units(&self) -> &UnitPreset {
        self.0.units()
    }

    fn cutoff(&self) -> Option<F> {
        self.0.cutoff()
    }

    fn bind(&mut self, frame: &Frame, skin: F) -> Result<(), PotentialError> {
        Ok(self.0.bind(frame, skin)?)
    }

    fn compute(
        &mut self,
        step: &PotentialStep<'_>,
        positions: &[F],
        forces: &mut [F],
    ) -> Result<PotentialResult, PotentialError> {
        let (energy, virial) =
            self.0
                .compute(step.step, step.rebuild, step.cell, positions, forces)?;
        Ok(PotentialResult { energy, virial })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhase>()?;
    m.add_class::<PyMD>()?;
    m.add_class::<PyVelocityVerlet>()?;
    m.add_class::<PyFixLangevin>()?;
    m.add_class::<PyBerendsenBarostat>()?;
    m.add_class::<PyMaxwellBoltzmann>()?;
    Ok(())
}
