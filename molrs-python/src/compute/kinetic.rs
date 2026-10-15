//! Kinetic energy of one configuration (`molrs::compute::KineticEnergy`) and
//! its output `KineticEnergyResult`.

use molrs::compute::{Compute, KineticEnergy, KineticEnergyInput, KineticEnergyResult};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::py_result;
use super::spec::described;

py_result! {
    /// A kinetic energy (``molrs::compute::KineticEnergyResult``) with the
    /// Boltzmann constant of its unit preset.
    PyKineticEnergyResult(KineticEnergyResult) as "KineticEnergyResult" { energy, boltzmann }
}

#[pymethods]
impl PyKineticEnergyResult {
    /// The equipartition temperature ``2 K / (n_dof · k_B)``.
    fn temperature(&self, n_dof: usize) -> Result<f64> {
        Ok(self.inner.temperature(n_dof)?)
    }
}

/// Kinetic energy ``½ Σ m v² · mvv2e`` (``molrs::compute::KineticEnergy``).
///
/// Parameters
/// ----------
/// units : str or UnitPreset, optional
///     The unit system of masses and velocities (``"real"``, ``"metal"``, …):
///     the core ``KineticEnergy::new``.
/// mvv2e, boltzmann : float, optional
///     The ``mass · velocity²`` → energy factor and Boltzmann's constant in
///     that energy unit, both positive: the core
///     ``KineticEnergy::with_constants``. Give either ``units`` or both of
///     these.
///
/// ``compute(masses=, velocities=)`` over ``(N,)`` masses and ``(N, 3)``
/// velocities returns a ``KineticEnergyResult``.
#[pyclass(module = "molrs.compute", name = "KineticEnergy", frozen)]
pub(crate) struct PyKineticEnergy {
    inner: KineticEnergy,
}

#[pymethods]
impl PyKineticEnergy {
    #[new]
    #[pyo3(signature = (*, units = None, mvv2e = None, boltzmann = None))]
    fn new(
        units: Option<&Bound<'_, PyAny>>,
        mvv2e: Option<f64>,
        boltzmann: Option<f64>,
    ) -> Result<Self> {
        let inner = match (units, mvv2e, boltzmann) {
            (Some(units), None, None) => KineticEnergy::new(args::units(units)?)?,
            (None, Some(mvv2e), Some(boltzmann)) => {
                KineticEnergy::with_constants(mvv2e, boltzmann)?
            }
            _ => {
                return Err(PyValueError::new_err(
                    "KineticEnergy takes either units (a unit preset) or both mvv2e \
                     and boltzmann (explicit constants)",
                )
                .into());
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, masses, velocities))]
    fn compute(
        &self,
        masses: PyReadonlyArray1<'_, f64>,
        velocities: PyReadonlyArray2<'_, f64>,
    ) -> Result<PyKineticEnergyResult> {
        let out = self.inner.compute(KineticEnergyInput {
            masses: masses.as_array(),
            velocities: velocities.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyKineticEnergy, KineticEnergy);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKineticEnergyResult>()?;
    m.add_class::<PyKineticEnergy>()?;
    Ok(())
}
