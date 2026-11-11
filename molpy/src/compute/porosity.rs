//! Probe-accessible volume and the pore-size distribution
//! (`molrs::compute::porosity`): `FreeVolume`, `PoreSizeDistribution`, and
//! their outputs.

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::compute::{
    Compute, FreeVolume, FreeVolumeInput, FreeVolumeResult, PoreSizeDistribution,
    PoreSizeDistributionInput, PoreSizeDistributionResult,
};
use pyo3::prelude::*;

use super::error::Result;
use super::output::py_result;
use super::spec::described;
use crate::core::simbox::PyBox;

py_result! {
    /// Probe-accessible volume of one snapshot
    /// (``molrs::compute::FreeVolumeResult``). Lengths in Å, volumes in Å³.
    /// ``bondi_fraction`` is present only for a point probe.
    PyFreeVolumeResult(FreeVolumeResult) as "FreeVolumeResult" {
        accessible_fraction,
        accessible_volume,
        uncertainty,
        n_samples,
        n_accessible,
        box_volume,
        bondi_fraction,
    }
}

py_result! {
    /// Volume-weighted pore radii of one snapshot
    /// (``molrs::compute::PoreSizeDistributionResult``). Radii in Å.
    /// ``fraction`` and ``cumulative`` are fractions of the cell volume.
    PyPoreSizeDistributionResult(PoreSizeDistributionResult) as "PoreSizeDistributionResult" {
        radius,
        fraction,
        cumulative,
        largest_cavity_radius,
        void_fraction,
        voxel_volume,
    }
}

/// Monte Carlo accessible volume of a hard probe in a periodic cell
/// (``molrs::compute::FreeVolume``).
///
/// Parameters
/// ----------
/// probe_radius : float
///     Probe radius, Å. A sample is accessible when its clearance to every
///     van der Waals sphere is strictly larger than this. ``0`` is a point
///     probe: the complement is the occupied fraction of the union of the
///     van der Waals spheres, and the result's ``bondi_fraction`` is
///     ``1 - 1.3 * occupied``.
/// n_samples : int, optional
///     Uniform samples drawn in the cell. ``None`` keeps the core default.
/// seed : int, optional
///     Seed of the sample generator. ``None`` keeps the core default.
///
/// ``compute(positions=, radii=, simbox=)``: ``positions`` is ``N × 3`` in Å,
/// ``radii`` one van der Waals radius per atom.
#[pyclass(module = "molpy.compute", name = "FreeVolume", frozen)]
pub(crate) struct PyFreeVolume {
    inner: FreeVolume,
}

#[pymethods]
impl PyFreeVolume {
    #[new]
    #[pyo3(signature = (probe_radius, *, n_samples = None, seed = None))]
    fn new(probe_radius: f64, n_samples: Option<usize>, seed: Option<u64>) -> Result<Self> {
        let mut inner = FreeVolume::new(probe_radius)?;
        if let Some(n_samples) = n_samples {
            inner = inner.with_n_samples(n_samples)?;
        }
        if let Some(seed) = seed {
            inner = inner.with_seed(seed)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, radii, simbox))]
    fn compute(
        &self,
        py: Python<'_>,
        positions: ArrayArg2<'_, f64>,
        radii: ArrayArg1<'_, f64>,
        simbox: PyRef<'_, PyBox>,
    ) -> Result<PyFreeVolumeResult> {
        let out = {
            let input = FreeVolumeInput {
                positions: positions.as_array(),
                radii: radii.as_array(),
                simbox: &simbox.inner,
            };
            py.detach(|| self.inner.compute(input))
        }?;
        Ok(out.into())
    }
}

described!(PyFreeVolume, FreeVolume);

/// Volume-weighted distribution of the largest sphere that covers each void
/// point (``molrs::compute::PoreSizeDistribution``).
///
/// Parameters
/// ----------
/// spacing : float
///     Voxel edge, Å. The grid is ``round(L / spacing)`` per axis.
/// bin_width : float, optional
///     Histogram bin width, Å. ``None`` keeps the core default. Bin ``i``
///     holds radii in ``(i w, (i + 1) w]``. The radius is the local
///     thickness; a diameter plot is twice these values.
///
/// ``compute(positions=, radii=, simbox=)``. A triclinic box is refused.
#[pyclass(module = "molpy.compute", name = "PoreSizeDistribution", frozen)]
pub(crate) struct PyPoreSizeDistribution {
    inner: PoreSizeDistribution,
}

#[pymethods]
impl PyPoreSizeDistribution {
    #[new]
    #[pyo3(signature = (spacing, *, bin_width = None))]
    fn new(spacing: f64, bin_width: Option<f64>) -> Result<Self> {
        let mut inner = PoreSizeDistribution::new(spacing)?;
        if let Some(bin_width) = bin_width {
            inner = inner.with_bin_width(bin_width)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, positions, radii, simbox))]
    fn compute(
        &self,
        py: Python<'_>,
        positions: ArrayArg2<'_, f64>,
        radii: ArrayArg1<'_, f64>,
        simbox: PyRef<'_, PyBox>,
    ) -> Result<PyPoreSizeDistributionResult> {
        let out = {
            let input = PoreSizeDistributionInput {
                positions: positions.as_array(),
                radii: radii.as_array(),
                simbox: &simbox.inner,
            };
            py.detach(|| self.inner.compute(input))
        }?;
        Ok(out.into())
    }
}

described!(PyPoreSizeDistribution, PoreSizeDistribution);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFreeVolumeResult>()?;
    m.add_class::<PyPoreSizeDistributionResult>()?;
    m.add_class::<PyFreeVolume>()?;
    m.add_class::<PyPoreSizeDistribution>()?;
    Ok(())
}
