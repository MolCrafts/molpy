//! Dielectric observables (`molrs::compute::dielectric`): `DipoleMoment`
//! (per-group dipole series) and `StaticDielectric` (ε from the dipole
//! fluctuation); their outputs `DipoleMomentResult` and
//! `StaticDielectricResult`.

use molrs::compute::{
    Compute, DipoleMoment, DipoleMomentInput, DipoleMomentResult, StaticDielectric,
    StaticDielectricInput, StaticDielectricResult,
};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::described;

py_result! {
    /// Dipole series per group (``molrs::compute::DipoleMomentResult``):
    /// ``moments[frame, group, :]`` for the group keys ``groups``.
    PyDipoleMomentResult(DipoleMomentResult) as "DipoleMomentResult" { groups, moments }
}

#[pymethods]
impl PyDipoleMomentResult {
    /// The ``(T, 3)`` dipole series at ``column`` (``0`` for the whole system).
    fn series<'py>(&self, py: Python<'py>, column: usize) -> PyResult<Bound<'py, PyAny>> {
        let n_series = self.inner.moments.shape()[1];
        if column >= n_series {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "column {column} out of range for {n_series} series"
            )));
        }
        self.inner.series(column).to_owned().to_py(py)
    }
}

py_result! {
    /// The static permittivity (``molrs::compute::StaticDielectricResult``):
    /// ``epsilon``, per-axis ``epsilon_axes``, the dipole moments behind it and
    /// the number of frames.
    PyStaticDielectricResult(StaticDielectricResult) as "StaticDielectricResult" {
        epsilon, epsilon_axes, dipole_mean, dipole_sq_mean, fluctuation, epsilon_inf, n_frames
    }
}

/// The dipole moment ``Σ q r`` of each group in each frame
/// (``molrs::compute::DipoleMoment``).
///
/// ``compute(positions=, charges=, partition=None)`` over ``(T, N, 3)``
/// positions and ``(N,)`` charges; ``partition`` (labels or a partition
/// result) splits the items into groups, else the whole system is one.
#[pyclass(module = "molrs.compute", name = "DipoleMoment", frozen)]
pub(crate) struct PyDipoleMoment {
    inner: DipoleMoment,
}

#[pymethods]
impl PyDipoleMoment {
    #[new]
    fn new() -> Self {
        Self {
            inner: DipoleMoment::new(),
        }
    }

    #[pyo3(signature = (*, positions, charges, partition = None))]
    fn compute<'py>(
        &self,
        positions: PyReadonlyArray3<'py, f64>,
        charges: PyReadonlyArray1<'py, f64>,
        partition: Option<Bound<'py, PyAny>>,
    ) -> Result<PyDipoleMomentResult> {
        let partition = partition
            .as_ref()
            .map(args::PartitionArg::extract)
            .transpose()?;
        let out = self.inner.compute(DipoleMomentInput {
            positions: positions.as_array(),
            charges: charges.as_array(),
            partition: partition.as_ref().map(args::PartitionArg::partition),
        })?;
        Ok(out.into())
    }
}

described!(PyDipoleMoment, DipoleMoment);

/// Static permittivity from the fluctuation of the total dipole
/// (``molrs::compute::StaticDielectric``).
///
/// Parameters
/// ----------
/// units : str or UnitPreset
///     The unit system of dipoles, volume and temperature.
/// epsilon_inf : float, optional
///     The high-frequency permittivity ε_∞; ``None`` keeps the core default.
///
/// ``compute(dipoles=, volume=, temperature=, boundary=)`` over a ``(T, 3)``
/// dipole series; ``boundary`` is ``"tin_foil"`` or ``"vacuum"``.
#[pyclass(module = "molrs.compute", name = "StaticDielectric", frozen)]
pub(crate) struct PyStaticDielectric {
    inner: StaticDielectric,
}

#[pymethods]
impl PyStaticDielectric {
    #[new]
    #[pyo3(signature = (units, *, epsilon_inf = None))]
    fn new(units: &Bound<'_, PyAny>, epsilon_inf: Option<f64>) -> Result<Self> {
        let mut inner = StaticDielectric::new(args::units(units)?);
        if let Some(epsilon_inf) = epsilon_inf {
            inner = inner.with_epsilon_inf(epsilon_inf);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, dipoles, volume, temperature, boundary))]
    fn compute(
        &self,
        dipoles: PyReadonlyArray2<'_, f64>,
        volume: f64,
        temperature: f64,
        boundary: &str,
    ) -> Result<PyStaticDielectricResult> {
        let out = self.inner.compute(StaticDielectricInput {
            dipoles: dipoles.as_array(),
            volume,
            temperature,
            boundary: boundary.parse()?,
        })?;
        Ok(out.into())
    }
}

described!(PyStaticDielectric, StaticDielectric);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDipoleMomentResult>()?;
    m.add_class::<PyStaticDielectricResult>()?;
    m.add_class::<PyDipoleMoment>()?;
    m.add_class::<PyStaticDielectric>()?;
    Ok(())
}
