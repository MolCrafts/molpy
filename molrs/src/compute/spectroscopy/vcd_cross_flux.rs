//! VCD electric×magnetic dipole cross-correlation raw compute — the
//! VCD-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::Array2;

use super::{central_flux, raw_column_correlation};
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::Reduction;

/// Raw VCD cross-correlation — the VCD-spectrum raw input.
#[derive(Debug, Clone)]
pub struct VcdCrossFluxResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: ndarray::Array1<f64>,
    /// VCD cross-correlation `C(τ) = Σ_d ⟨μ̇_d(0)·ṁ_d(τ)⟩` summed over the 3
    /// Cartesian components — the (signed) ACF the
    /// [`VcdSpectrum`](super::VcdSpectrum) transform consumes.
    pub acf: ndarray::Array1<f64>,
}

impl ComputeResult for VcdCrossFluxResult {}

/// Raw VCD cross-flux compute: cross-correlation of the electric-dipole
/// derivative `μ̇` with the magnetic-dipole derivative `ṁ`.
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_VCD` branch
/// (`src/roa.cpp`): for each Cartesian component it cross-correlates
/// `m_vaDElDip` (μ̇) with `m_vaDMagDip` (ṁ) and sums the three components.
///
/// **Deviation from the spec's literal `⟨μ̇·m⟩`:** reference implementation correlates the two
/// *derivatives* (`μ̇` × `ṁ`); the two conventions differ only by a frequency-
/// domain factor and share the same peak positions and enantiomer sign law.
#[derive(Debug, Clone, Copy, Default)]
pub struct VcdCrossFlux;

/// `(electric_dipole (n,3), magnetic_dipole (n,3), dt, resolution)` for
/// [`VcdCrossFlux`]. Both series are `(n_frames, 3)`; the central-difference
/// derivatives drop the first and last frame.
pub type VcdCrossFluxArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl LegacyCompute for VcdCrossFlux {
    type Args<'a> = VcdCrossFluxArgs<'a>;
    type Output = VcdCrossFluxResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (electric, magnetic, dt, resolution) = args;
        let n_frames = electric.shape()[0];
        if electric.shape()[1] != 3 || magnetic.shape()[1] != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: electric.shape()[1].max(magnetic.shape()[1]),
                what: "VCD (electric, magnetic) dipoles (expected (n_frames, 3))",
            });
        }
        if magnetic.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: magnetic.shape()[0],
                what: "VCD electric/magnetic frame counts",
            });
        }
        if n_frames < 3 {
            return Err(ComputeError::EmptyInput);
        }
        if dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: dt.to_string(),
            });
        }
        let flux_len = n_frames - 2;
        let max_lag = resolution.min(flux_len.saturating_sub(1));
        let mu_dot = central_flux(electric, dt)?;
        let m_dot = central_flux(magnetic, dt)?;
        let raw = raw_column_correlation(&mu_dot, Some(&m_dot), max_lag, dt, Reduction::Mean)?;
        Ok(VcdCrossFluxResult {
            acf: raw.channel(0).to_owned(),
            lag_times: raw.lag_times,
        })
    }
}
