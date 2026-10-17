//! Python bindings for [`molrs::md`]: the phase driver and the components
//! it mounts. Potentials stay in `molrs.ff.potential`.

use crate::array::ArrayArg2;
use crate::core::frame::PyFrame;
use crate::error::ffi_error_to_pyerr;
use crate::ff::potential::PyPairLjCut;
use molrs::md::{
    FfEvaluator, FixLangevin, MD, MaxwellBoltzmann, MdError, NeighborList, Phase, TimestepProvider,
    VelocityVerlet,
};
use molrs::op::F;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn md_err(e: MdError) -> PyErr {
    PyValueError::new_err(e.to_string())
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
        Ok(arr) => Ok(arr.as_array().to_owned()),
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
    module = "molrs.md",
    from_py_object
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

impl PyPhase {
    fn rust(self) -> Phase {
        match self {
            PyPhase::InitialIntegrate => Phase::InitialIntegrate,
            PyPhase::PostIntegrate => Phase::PostIntegrate,
            PyPhase::PreExchange => Phase::PreExchange,
            PyPhase::PreNeighbor => Phase::PreNeighbor,
            PyPhase::Neighbor => Phase::Neighbor,
            PyPhase::PreForce => Phase::PreForce,
            PyPhase::Force => Phase::Force,
            PyPhase::PostForce => Phase::PostForce,
            PyPhase::FinalIntegrate => Phase::FinalIntegrate,
            PyPhase::EndOfStep => Phase::EndOfStep,
        }
    }
}

/// NVE velocity Verlet. `dt` only — mass, positions, and velocities are frame columns.
#[pyclass(
    name = "VelocityVerlet",
    module = "molrs.md",
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

/// LAMMPS `fix langevin`, mounted at `post_force`.
///
/// `kb` is Boltzmann's constant in the frame's energy unit per kelvin.
/// The driver does not choose a unit system.
#[pyclass(
    name = "FixLangevin",
    module = "molrs.md",
    subclass,
    skip_from_py_object
)]
pub struct PyFixLangevin {
    inner: FixLangevin,
}

#[pymethods]
impl PyFixLangevin {
    #[new]
    #[pyo3(signature = (t_start, t_stop, damp, dt, seed, *, kb))]
    fn new(t_start: F, t_stop: F, damp: F, dt: F, seed: u64, kb: F) -> PyResult<Self> {
        Ok(Self {
            inner: FixLangevin::new(t_start, t_stop, damp, dt, seed, kb).map_err(md_err)?,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
}

/// Verlet neighbour list mounted at the `neighbor` phase.
#[pyclass(name = "NeighborList", module = "molrs.md", subclass)]
pub struct PyNeighborList {
    inner: Option<NeighborList>,
}

#[pymethods]
impl PyNeighborList {
    #[new]
    #[pyo3(signature = (cutoff, skin))]
    fn new(cutoff: F, skin: F) -> PyResult<Self> {
        Ok(Self {
            inner: Some(NeighborList::new(cutoff, skin).map_err(md_err)?),
        })
    }
}

/// The phase driver. `run` writes the frame it is given.
#[pyclass(name = "MD", module = "molrs.md", unsendable, subclass)]
pub struct PyMD {
    inner: MD,
}

#[pymethods]
impl PyMD {
    #[new]
    fn new() -> Self {
        Self { inner: MD::new() }
    }

    fn set_integrator(&mut self, integrator: &PyVelocityVerlet) -> PyResult<()> {
        self.inner.set_integrator(integrator.inner).map_err(md_err)
    }

    fn set_thermostat(&mut self, thermostat: &PyFixLangevin) -> PyResult<()> {
        self.inner.set_thermostat(thermostat.inner).map_err(md_err)
    }

    fn set_neighbor(&mut self, mut neighbor: PyRefMut<'_, PyNeighborList>) -> PyResult<()> {
        let list = neighbor
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("NeighborList was already moved into a driver"))?;
        self.inner.set_neighbor(list).map_err(md_err)
    }

    /// Mount a `PairLjCut`. It reads the neighbour list published at `neighbor`.
    fn set_evaluator(&mut self, potential: &Bound<'_, PyAny>) -> PyResult<()> {
        let lj = potential.cast::<PyPairLjCut>().map_err(|_| {
            PyTypeError::new_err(
                "set_evaluator expects a PairLjCut; a second force term uses on(Phase.force)",
            )
        })?;
        self.inner
            .set_evaluator(FfEvaluator::pair(lj.borrow().inner.clone()))
            .map_err(md_err)
    }

    /// Advance `frame` for `n_steps`. Coordinates, velocities, forces, image
    /// flags, and `meta["energy"]` / `meta["kinetic"]` are written on `frame`.
    fn run(&mut self, frame: &PyFrame, n_steps: usize) -> PyResult<()> {
        frame
            .inner
            .with_mut(|f| self.inner.run(f, n_steps))
            .map_err(ffi_error_to_pyerr)?
            .map_err(md_err)
    }
}

#[pyclass(name = "MaxwellBoltzmann", module = "molrs.md", subclass)]
pub struct PyMaxwellBoltzmann {
    inner: MaxwellBoltzmann,
}

#[pymethods]
impl PyMaxwellBoltzmann {
    #[new]
    #[pyo3(signature = (kbt, *, seed=0, remove_com=true))]
    fn new(kbt: F, seed: u64, remove_com: bool) -> PyResult<Self> {
        let mut inner = MaxwellBoltzmann::new(kbt, seed).map_err(md_err)?;
        if !remove_com {
            inner = inner.keep_com();
        }
        Ok(Self { inner })
    }

    #[getter]
    fn kbt(&self) -> F {
        self.inner.kbt()
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPhase>()?;
    m.add_class::<PyMD>()?;
    m.add_class::<PyVelocityVerlet>()?;
    m.add_class::<PyFixLangevin>()?;
    m.add_class::<PyNeighborList>()?;
    m.add_class::<PyMaxwellBoltzmann>()?;
    Ok(())
}
