//! Scattering (`molrs::compute::diffraction`): the static structure factor
//! by the Debye sum (`StaticStructureFactorDebye`) and by direct k-vectors
//! (`StaticStructureFactorDirect`), and the 2-D `DiffractionPattern`.

use crate::array::ArrayArg2;
use molrs::compute::{
    Compute, DiffractionPattern, DiffractionPatternInput, DiffractionPatternResult,
    StaticStructureFactorDebye, StaticStructureFactorDebyeInput, StaticStructureFactorDebyeResult,
    StaticStructureFactorDirect, StaticStructureFactorDirectInput,
    StaticStructureFactorDirectResult,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::core::simbox::PyBox;

py_result! {
    /// S(k) by the Debye sum (``molrs::compute::StaticStructureFactorDebyeResult``).
    PyStaticStructureFactorDebyeResult(StaticStructureFactorDebyeResult)
        as "StaticStructureFactorDebyeResult" { k_values, sk, n_particles }
}

py_result! {
    /// S(k) on explicit or isotropically binned k-vectors
    /// (``molrs::compute::StaticStructureFactorDirectResult``).
    PyStaticStructureFactorDirectResult(StaticStructureFactorDirectResult)
        as "StaticStructureFactorDirectResult" { k_magnitudes, sk, n_particles }
}

py_result! {
    /// A 2-D diffraction pattern and the projected density image it was
    /// transformed from (``molrs::compute::DiffractionPatternResult``).
    PyDiffractionPatternResult(DiffractionPatternResult) as "DiffractionPatternResult" {
        diffraction,
        image,
    }
}

/// Static structure factor by the Debye sum over all pairs
/// (``molrs::compute::StaticStructureFactorDebye``).
///
/// Parameters
/// ----------
/// k_values : list[float]
///     Where to evaluate S(k).
/// sinc_tolerance : float, optional
///
/// ``compute(positions=)`` over one ``(N, 3)`` snapshot.
#[pyclass(module = "molrs.compute", name = "StaticStructureFactorDebye", frozen)]
pub(crate) struct PyStaticStructureFactorDebye {
    inner: StaticStructureFactorDebye,
}

#[pymethods]
impl PyStaticStructureFactorDebye {
    #[new]
    #[pyo3(signature = (k_values, *, sinc_tolerance = None))]
    fn new(k_values: Vec<f64>, sinc_tolerance: Option<f64>) -> Result<Self> {
        let mut inner = StaticStructureFactorDebye::new(&k_values)?;
        if let Some(sinc_tolerance) = sinc_tolerance {
            inner = inner.with_sinc_tolerance(sinc_tolerance)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions))]
    fn compute(&self, positions: ArrayArg2<'_, f64>) -> Result<PyStaticStructureFactorDebyeResult> {
        let out = self.inner.compute(StaticStructureFactorDebyeInput {
            positions: positions.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyStaticStructureFactorDebye, StaticStructureFactorDebye);

/// Static structure factor from the density's Fourier components
/// (``molrs::compute::StaticStructureFactorDirect``).
///
/// Parameters
/// ----------
/// k_vecs : list of 3-vectors, optional
///     Evaluate at these k-vectors (the core ``new``).
/// k_max : float, optional
/// n_bins : int, optional
///     Bin the box's reciprocal lattice isotropically up to ``k_max`` (the
///     core ``isotropic``). Give either ``k_vecs`` or both ``k_max`` and
///     ``n_bins``.
///
/// ``compute(positions=, simbox=None)``.
#[pyclass(module = "molrs.compute", name = "StaticStructureFactorDirect", frozen)]
pub(crate) struct PyStaticStructureFactorDirect {
    inner: StaticStructureFactorDirect,
}

#[pymethods]
impl PyStaticStructureFactorDirect {
    #[new]
    #[pyo3(signature = (*, k_vecs = None, k_max = None, n_bins = None))]
    fn new(
        k_vecs: Option<Vec<[f64; 3]>>,
        k_max: Option<f64>,
        n_bins: Option<usize>,
    ) -> Result<Self> {
        let inner = match (k_vecs, k_max, n_bins) {
            (Some(k_vecs), None, None) => StaticStructureFactorDirect::new(&k_vecs)?,
            (None, Some(k_max), Some(n_bins)) => {
                StaticStructureFactorDirect::isotropic(k_max, n_bins)?
            }
            _ => {
                return Err(PyValueError::new_err(
                    "StaticStructureFactorDirect takes either k_vecs (explicit k-vectors) or \
                     both k_max and n_bins (isotropic bins)",
                )
                .into());
            }
        };
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, simbox = None))]
    fn compute(
        &self,
        positions: ArrayArg2<'_, f64>,
        simbox: Option<PyRef<'_, PyBox>>,
    ) -> Result<PyStaticStructureFactorDirectResult> {
        let out = self.inner.compute(StaticStructureFactorDirectInput {
            positions: positions.as_array(),
            simbox: simbox.as_ref().map(|simbox| &simbox.inner),
        })?;
        Ok(out.into())
    }
}

described!(PyStaticStructureFactorDirect, StaticStructureFactorDirect);

/// A 2-D diffraction pattern of one snapshot projected along an axis
/// (``molrs::compute::DiffractionPattern``).
///
/// Parameters
/// ----------
/// n_grid : int
///     Image side length in pixels.
/// sigma : float
///     The Gaussian width smearing each point.
/// axis : int, optional
///     The projection axis.
/// cutoff : float, optional
///     The cutoff of the Gaussian.
///
/// ``compute(positions=, simbox=)``.
#[pyclass(module = "molrs.compute", name = "DiffractionPattern", frozen)]
pub(crate) struct PyDiffractionPattern {
    inner: DiffractionPattern,
}

#[pymethods]
impl PyDiffractionPattern {
    #[new]
    #[pyo3(signature = (n_grid, sigma, *, axis = None, cutoff = None))]
    fn new(n_grid: usize, sigma: f64, axis: Option<usize>, cutoff: Option<f64>) -> Result<Self> {
        let mut inner = DiffractionPattern::new(n_grid, sigma)?;
        if let Some(axis) = axis {
            inner = inner.with_axis(axis)?;
        }
        if let Some(cutoff) = cutoff {
            inner = inner.with_cutoff(cutoff)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, simbox))]
    fn compute(
        &self,
        positions: ArrayArg2<'_, f64>,
        simbox: PyRef<'_, PyBox>,
    ) -> Result<PyDiffractionPatternResult> {
        let out = self.inner.compute(DiffractionPatternInput {
            positions: positions.as_array(),
            simbox: &simbox.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyDiffractionPattern, DiffractionPattern);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStaticStructureFactorDebyeResult>()?;
    m.add_class::<PyStaticStructureFactorDirectResult>()?;
    m.add_class::<PyDiffractionPatternResult>()?;
    m.add_class::<PyStaticStructureFactorDebye>()?;
    m.add_class::<PyStaticStructureFactorDirect>()?;
    m.add_class::<PyDiffractionPattern>()?;
    Ok(())
}
