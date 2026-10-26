//! Kinetic energy of one configuration (`molrs::compute::KineticEnergy`) and
//! its output `KineticEnergyResult`.

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::compute::{Compute, KineticEnergy, KineticEnergyInput, KineticEnergyResult};

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
/// units : str or UnitPreset
///     The unit system of masses and velocities (``"real"``, ``"metal"``, …):
///     the core ``KineticEnergy::new``, which takes the preset's
///     ``mass · velocity²`` → energy factor and Boltzmann constant.
/// mvv2e : float, optional
///     Replace the preset's ``mass · velocity²`` → energy factor (positive):
///     the core ``with_mvv2e``.
/// boltzmann : float, optional
///     Replace the preset's Boltzmann constant, in the energy unit
///     (positive): the core ``with_boltzmann``.
///
/// ``compute(masses=, velocities=)`` over ``(N,)`` masses and ``(N, 3)``
/// velocities returns a ``KineticEnergyResult``.
#[pyclass(module = "molpy.compute", name = "KineticEnergy", frozen)]
pub(crate) struct PyKineticEnergy {
    inner: KineticEnergy,
}

#[pymethods]
impl PyKineticEnergy {
    #[new]
    #[pyo3(signature = (units, *, mvv2e = None, boltzmann = None))]
    fn new(units: &Bound<'_, PyAny>, mvv2e: Option<f64>, boltzmann: Option<f64>) -> Result<Self> {
        let mut inner = KineticEnergy::new(args::units(units)?)?;
        if let Some(mvv2e) = mvv2e {
            inner = inner.with_mvv2e(mvv2e)?;
        }
        if let Some(boltzmann) = boltzmann {
            inner = inner.with_boltzmann(boltzmann)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, masses, velocities))]
    fn compute(
        &self,
        masses: ArrayArg1<'_, f64>,
        velocities: ArrayArg2<'_, f64>,
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
