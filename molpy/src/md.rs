//! Python bindings for [`molrs::md`]: the phase driver and the components
//! it mounts. Potentials stay in `molpy.ff.potential`.

use crate::array::ArrayArg2;
use crate::core::frame::PyFrame;
use crate::core::neighborlist::PyNeighborList;
use crate::error::ffi_error_to_pyerr;
use crate::ff::potential::{ErrSlot, take_err, take_members};
use molrs::md::{
    FfEvaluator, FixLangevin, MD, MaxwellBoltzmann, MdError, TimestepProvider, VelocityVerlet,
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

/// LAMMPS `fix langevin`, mounted at `post_force`.
///
/// `kb` is Boltzmann's constant in the frame's energy unit per kelvin.
/// The driver does not choose a unit system.
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

/// The phase driver. `run` writes the frame it is given.
#[pyclass(name = "MD", module = "molpy.md", unsendable, subclass)]
pub struct PyMD {
    inner: MD,
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyMD {
    #[new]
    fn new() -> Self {
        Self {
            inner: MD::new(),
            err_slots: Vec::new(),
        }
    }

    fn set_integrator(&mut self, integrator: &PyVelocityVerlet) -> PyResult<()> {
        self.inner.set_integrator(integrator.inner).map_err(md_err)
    }

    fn set_thermostat(&mut self, thermostat: &PyFixLangevin) -> PyResult<()> {
        self.inner.set_thermostat(thermostat.inner).map_err(md_err)
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

    /// Mount a `PairLjCut`, a compiled `Potentials` / `WeightedTerms`, or an
    /// object with `evaluate(pos) -> (energy, forces)`.
    fn set_evaluator(&mut self, potential: &Bound<'_, PyAny>) -> PyResult<()> {
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
    m.add_class::<PyMaxwellBoltzmann>()?;
    Ok(())
}
