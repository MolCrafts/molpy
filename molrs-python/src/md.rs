//! Python bindings for `molrs::md`: the integrators and the `MD` state.
//!
//! ```text
//! VelocityVerlet(dt, potential=lj, neighbors=nl, mass=mass)
//! VelocityVerlet(dt, potential=potentials, mass=mass)   # ff Potentials / mix
//! ```
//!
//! MD defines no potential. What it integrates is any member
//! [`take_potential`](crate::ff::potential::take_potential) accepts: `molrs.ff.potential.PairLjCut`, the force-field
//! `Potentials` collection (e.g. from `molrs.ff.compile.ExplicitTerms`), or a
//! duck-typed Python object with
//! `calc_energy_forces`. MD has no unit knowledge. Integrators own the
//! optional `VerletSkin`.

use crate::array::{ArrayArg1, ArrayArg2};
use crate::core::neighborlist::PyVerletSkin;
use crate::core::simbox::PyBox;
use crate::ff::potential::{ErrSlot, Members, PyPairLjCut, check_nx3, take_err, take_members};
use molrs::core::Virial;
use molrs::md::{
    ForceProvider, Langevin, MaxwellBoltzmann, MdError, MdState, MicPairs, SelfPairedForces,
    VelocityVerlet,
};
use molrs::op::{F, I};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

fn md_err(e: MdError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// A Python-side neighbour argument picks the minimum-image provider; the
/// ghost régime is not bound yet.
fn provider(
    members: Members,
    skin: Option<molrs::core::VerletSkin>,
) -> PyResult<Box<dyn ForceProvider>> {
    // `MicPairs` refuses a kernel whose parameters were resolved against a
    // fixed pair list — it would ignore the neighbour table and answer for the
    // list it was built from, while the driver went on rebuilding and reporting
    // that table. `PotentialCompiler.compile` builds exactly such kernels, so
    // this is the path where that mistake is made, and the error says what to
    // build instead.
    Ok(match skin {
        Some(s) => Box::new(MicPairs::from_members(members, s).map_err(md_err)?),
        None => {
            if members.len() != 1 {
                return Err(PyValueError::new_err(
                    "several members need a neighbour table to share; pass neighbors=",
                ));
            }
            let (pot, special) = members.into_iter().next().expect("checked above");
            if !special.is_empty() {
                return Err(PyValueError::new_err(
                    "special-bonds weights apply to a pair table; pass neighbors=",
                ));
            }
            Box::new(SelfPairedForces::new(pot))
        }
    })
}

/// The state an integrator advances: an [`MdState`] as is (its accumulated
/// image flags and all), or any object with `pos`, `vel`, `forces`, `energy`
/// and `images` attributes.
///
/// The image flags are part of the state — `pos + H·images` is the continuous
/// trajectory — so they are read like every other field. Rebuilding them as
/// zeros here would make each `advance` restart the count, and the flags
/// would only ever hold the crossings of the last call.
fn extract_state(state: &Bound<'_, PyAny>) -> PyResult<MdState> {
    if let Ok(state) = state.cast::<PyMdState>() {
        return Ok(state.borrow().inner.clone());
    }
    let pos: ArrayArg2<f64> = state.getattr("pos")?.extract()?;
    let vel: ArrayArg2<f64> = state.getattr("vel")?.extract()?;
    let forces: ArrayArg2<f64> = state.getattr("forces")?.extract()?;
    let images: ArrayArg2<I> = state.getattr("images")?.extract()?;
    let energy: F = state.getattr("energy")?.extract()?;
    check_nx3(&pos, "pos")?;
    check_nx3(&vel, "vel")?;
    check_nx3(&forces, "forces")?;
    check_images(images.as_array(), pos.as_array().nrows())?;
    Ok(MdState {
        images: images.as_array().to_owned(),
        pos: pos.as_array().to_owned(),
        vel: vel.as_array().to_owned(),
        forces: forces.as_array().to_owned(),
        energy,
        virial: None,
    })
}

fn check_images(images: ndarray::ArrayView2<'_, I>, n_atoms: usize) -> PyResult<()> {
    if images.ncols() != 3 || images.nrows() != n_atoms {
        return Err(PyValueError::new_err(format!(
            "images must have shape ({n_atoms}, 3) matching pos, got {:?}",
            images.shape()
        )));
    }
    Ok(())
}

/// Per-atom masses, a 1-D float64 array `(N,)`.
///
/// The integrators are built before they see a configuration, so they cannot
/// broadcast a scalar to `N` atoms; a homogeneous system passes
/// `np.full(N, m)`. A scalar is refused here rather than read as a one-atom
/// array that fails later against `N`.
fn mass_from(mass: &Bound<'_, PyAny>) -> PyResult<Array1<F>> {
    match mass.extract::<ArrayArg1<f64>>() {
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

/// Dynamical state advanced by the integrators.
///
/// Fields are settable (float64 `(N, 3)` arrays / a float energy) so hooks can
/// replace them wholesale: `state.vel = new_vel`. Getters return copies —
/// in-place slice writes (`state.vel[:] = …`) do NOT write through.
#[pyclass(name = "MdState", module = "molrs.md")]
pub struct PyMdState {
    inner: MdState,
}

#[pymethods]
impl PyMdState {
    #[new]
    fn new(
        pos: ArrayArg2<'_, f64>,
        vel: ArrayArg2<'_, f64>,
        forces: ArrayArg2<'_, f64>,
        energy: F,
    ) -> PyResult<Self> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        check_nx3(&forces, "forces")?;
        Ok(Self {
            inner: MdState {
                images: numpy::ndarray::Array2::zeros((pos.as_array().nrows(), 3)),
                pos: pos.as_array().to_owned(),
                vel: vel.as_array().to_owned(),
                forces: forces.as_array().to_owned(),
                energy,
                virial: None,
            },
        })
    }

    #[getter]
    fn pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.pos.clone().into_pyarray(py)
    }
    #[getter]
    fn vel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.vel.clone().into_pyarray(py)
    }
    #[getter]
    fn forces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.forces.clone().into_pyarray(py)
    }
    #[getter]
    fn energy(&self) -> F {
        self.inner.energy
    }

    /// Accumulated box crossings ``(N, 3)``, one signed count per lattice
    /// vector, as ``int32`` — the schema's integer type for the
    /// ``ix``/``iy``/``iz`` columns.
    ///
    /// A wrapped coordinate on its own has lost the atom's history. `pos +
    /// H·images` is the continuous position, and mean-squared displacement,
    /// diffusion and any other path-dependent quantity read that, not `pos`.
    #[getter]
    fn images<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<I>> {
        self.inner.images.clone().into_pyarray(py)
    }

    #[setter]
    fn set_images(&mut self, value: ArrayArg2<'_, I>) -> PyResult<()> {
        check_images(value.as_array(), self.inner.pos.nrows())?;
        self.inner.images = value.as_array().to_owned();
        Ok(())
    }

    /// Scalar pressure from the virial, a kinetic energy and a cell volume.
    ///
    /// ``None`` when no virial was tallied — not zero. Units are the caller's,
    /// as everywhere in MD: the result is in whatever `energy / volume` is.
    fn pressure(&self, kinetic: F, volume: F) -> Option<F> {
        self.inner.virial.map(|w| w.pressure(kinetic, volume))
    }

    /// Virial `Σ f ⊗ r` as ``(xx, yy, zz, xy, xz, yz)``, or ``None``.
    ///
    /// ``None`` means the force provider does not tally one — not that it is
    /// zero. A pressure computed from a fabricated zero is wrong and looks
    /// entirely plausible.
    #[getter]
    fn virial(&self) -> Option<[F; 6]> {
        self.inner.virial.map(|w: Virial| w.components)
    }

    #[setter]
    fn set_pos(&mut self, pos: ArrayArg2<'_, f64>) -> PyResult<()> {
        check_nx3(&pos, "pos")?;
        self.inner.pos = pos.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_vel(&mut self, vel: ArrayArg2<'_, f64>) -> PyResult<()> {
        check_nx3(&vel, "vel")?;
        self.inner.vel = vel.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_forces(&mut self, forces: ArrayArg2<'_, f64>) -> PyResult<()> {
        check_nx3(&forces, "forces")?;
        self.inner.forces = forces.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_energy(&mut self, energy: F) {
        self.inner.energy = energy;
    }

    fn __repr__(&self) -> String {
        format!(
            "MdState(n_atoms={}, energy={})",
            self.inner.pos.nrows(),
            self.inner.energy
        )
    }
}

// ---------------------------------------------------------------------------
// Integrators
// ---------------------------------------------------------------------------

#[pyclass(name = "VelocityVerlet", module = "molrs.md", subclass)]
pub struct PyVelocityVerlet {
    inner: VelocityVerlet,
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyVelocityVerlet {
    #[new]
    #[pyo3(signature = (dt, *, potential, neighbors=None, mass, r#box=None))]
    fn new(
        dt: F,
        potential: &Bound<'_, PyAny>,
        neighbors: Option<&Bound<'_, PyVerletSkin>>,
        mass: Bound<'_, PyAny>,
        r#box: Option<PyBox>,
    ) -> PyResult<Self> {
        if potential.cast::<PyPairLjCut>().is_ok() && neighbors.is_none() {
            return Err(PyValueError::new_err(
                "an PairLjCut pair kernel needs neighbors= (a VerletSkin)",
            ));
        }
        // Validate mass before moving the potential / neighbour state in.
        let mass = mass_from(&mass)?;
        let (members, err_slots) = take_members(potential)?;
        let skin = match neighbors {
            Some(nl) => Some(nl.borrow_mut().take()?),
            None => None,
        };
        Ok(Self {
            inner: VelocityVerlet::new(
                dt,
                provider(members, skin)?,
                mass.view(),
                r#box.map(|b| b.inner),
            )
            .map_err(md_err)?,
            err_slots,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }

    #[getter]
    fn removed_dof(&self) -> usize {
        self.inner.removed_dof()
    }

    /// Number of pair edges in the current list (``None`` without neighbors).
    #[getter]
    fn n_edges(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().edges
    }

    /// Neighbour-list rebuilds since construction (``None`` without neighbors).
    #[getter]
    fn n_rebuilds(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().rebuilds
    }

    /// Updates since the last rebuild (``None`` without neighbors).
    #[getter]
    fn ago(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().ago
    }

    fn initial(&mut self, pos: ArrayArg2<'_, f64>, vel: ArrayArg2<'_, f64>) -> PyResult<PyMdState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let result = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned());
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMdState> {
        let result = self.inner.advance(extract_state(state)?);
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMdState> {
        let result = self.inner.advance_n(extract_state(state)?, n_steps);
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }
}

#[pyclass(name = "Langevin", module = "molrs.md", subclass)]
pub struct PyLangevin {
    inner: Langevin,
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyLangevin {
    #[new]
    #[pyo3(signature = (dt, *, gamma, kbt, potential, neighbors=None, mass, seed=0, r#box=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dt: F,
        gamma: F,
        kbt: F,
        potential: &Bound<'_, PyAny>,
        neighbors: Option<&Bound<'_, PyVerletSkin>>,
        mass: Bound<'_, PyAny>,
        seed: u64,
        r#box: Option<PyBox>,
    ) -> PyResult<Self> {
        if potential.cast::<PyPairLjCut>().is_ok() && neighbors.is_none() {
            return Err(PyValueError::new_err(
                "an PairLjCut pair kernel needs neighbors= (a VerletSkin)",
            ));
        }
        // Validate the scheme knobs and mass before moving anything in.
        if gamma <= 0.0 {
            return Err(PyValueError::new_err(
                "Langevin requires gamma > 0; use VelocityVerlet for NVE",
            ));
        }
        if kbt <= 0.0 {
            return Err(PyValueError::new_err("Langevin requires kbt > 0"));
        }
        let mass = mass_from(&mass)?;
        let (members, err_slots) = take_members(potential)?;
        let skin = match neighbors {
            Some(nl) => Some(nl.borrow_mut().take()?),
            None => None,
        };
        Ok(Self {
            inner: Langevin::new(
                dt,
                gamma,
                kbt,
                provider(members, skin)?,
                mass.view(),
                seed,
                r#box.map(|b| b.inner),
            )
            .map_err(md_err)?,
            err_slots,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
    #[getter]
    fn gamma(&self) -> F {
        self.inner.gamma()
    }
    #[getter]
    fn c1(&self) -> F {
        self.inner.c1()
    }
    #[getter]
    fn c2(&self) -> F {
        self.inner.c2()
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner
            .sigma()
            .view()
            .insert_axis(ndarray::Axis(1))
            .to_owned()
            .into_pyarray(py)
    }
    #[getter]
    fn inv_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner
            .inv_mass()
            .view()
            .insert_axis(ndarray::Axis(1))
            .to_owned()
            .into_pyarray(py)
    }
    #[getter]
    fn removed_dof(&self) -> usize {
        self.inner.removed_dof()
    }

    /// Number of pair edges in the current list (``None`` without neighbors).
    #[getter]
    fn n_edges(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().edges
    }

    /// Neighbour-list rebuilds since construction (``None`` without neighbors).
    #[getter]
    fn n_rebuilds(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().rebuilds
    }

    /// Updates since the last rebuild (``None`` without neighbors).
    #[getter]
    fn ago(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().ago
    }

    fn initial(&mut self, pos: ArrayArg2<'_, f64>, vel: ArrayArg2<'_, f64>) -> PyResult<PyMdState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let result = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned());
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn step(&mut self, state: &Bound<'_, PyAny>, noise: ArrayArg2<'_, f64>) -> PyResult<PyMdState> {
        check_nx3(&noise, "noise")?;
        let result = self.inner.step(extract_state(state)?, noise.as_array());
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMdState> {
        let result = self.inner.advance(extract_state(state)?);
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMdState> {
        let result = self.inner.advance_n(extract_state(state)?, n_steps);
        take_err(&self.err_slots)?;
        Ok(PyMdState {
            inner: result.map_err(md_err)?,
        })
    }

    fn draw_noise<'py>(&mut self, py: Python<'py>, n_atoms: usize) -> Bound<'py, PyArray2<f64>> {
        self.inner.draw_noise(n_atoms).into_pyarray(py)
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
    m.add_class::<PyMdState>()?;
    m.add_class::<PyVelocityVerlet>()?;
    m.add_class::<PyLangevin>()?;
    m.add_class::<PyMaxwellBoltzmann>()?;
    Ok(())
}
