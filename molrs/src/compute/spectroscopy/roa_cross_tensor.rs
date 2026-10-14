//! ROA polarizability×optical-activity cross-correlation raw compute — the
//! ROA-spectrum raw input.

use molrs::core::FrameAccess;
use ndarray::{Array1, Array2};

use super::central_diff_series;
use super::raman_tensor::{ANISO_WEIGHTS, raman_invariant_series};
use crate::compute::Compute;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::{Normalization, TimeCorrelationSpec, cross_correlate, lag_times};

/// Raw ROA cross-correlation iso/aniso curves — the ROA-spectrum raw input.
#[derive(Debug, Clone)]
pub struct RoaCrossTensorResult {
    /// Lag times τ = i·dt, length `max_lag + 1`.
    pub lag_times: Array1<f64>,
    /// Isotropic ROA cross-correlation of `α̇` (electric polarizability
    /// derivative) with `Ġ′` (magnetic-dipole polarizability / optical-activity
    /// tensor derivative).
    pub acf_iso: Array1<f64>,
    /// Weighted anisotropic ROA cross-correlation (same diagonal/off-diagonal
    /// weighting as the Raman anisotropy).
    pub acf_aniso: Array1<f64>,
}

impl ComputeResult for RoaCrossTensorResult {}

/// Raw ROA cross-tensor compute: cross-correlation of the electric
/// polarizability derivative `α̇` with the optical-activity tensor derivative
/// `Ġ′` (and, in the full theory, the electric-quadrupole tensor `A`).
///
/// Ported from `CROAEngine::ComputeACFPair`, `ROA_SPECTRUM_ROA` branch
/// (`src/roa.cpp`): the iso part cross-correlates the polarizability trace with
/// the (negated) `G′` trace, and the anisotropic parts mirror the Raman
/// deviatoric decomposition. Both tensors are passed in Voigt form
/// `[xx, yy, zz, xy, xz, yz]`; the same diagonal (½) / off-diagonal (3) weights
/// as [`RamanTensor`](super::RamanTensor) are used so ROA shares the Raman
/// normal-mode frequencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct RoaCrossTensor;

/// `(electric_pol (n,6), g_tensor (n,6), dt, resolution)` for [`RoaCrossTensor`],
/// both in Voigt notation `[xx, yy, zz, xy, xz, yz]`.
pub type RoaCrossTensorArgs<'a> = (&'a Array2<f64>, &'a Array2<f64>, f64, usize);

impl Compute for RoaCrossTensor {
    type Args<'a> = RoaCrossTensorArgs<'a>;
    type Output = RoaCrossTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        let (el_pol, g_tensor, dt, resolution) = args;
        let n_frames = el_pol.shape()[0];
        if el_pol.shape()[1] != 6 || g_tensor.shape()[1] != 6 {
            return Err(ComputeError::DimensionMismatch {
                expected: 6,
                got: el_pol.shape()[1].max(g_tensor.shape()[1]),
                what: "ROA (electric_pol, g_tensor) (expected (n_frames, 6) Voigt)",
            });
        }
        if g_tensor.shape()[0] != n_frames {
            return Err(ComputeError::DimensionMismatch {
                expected: n_frames,
                got: g_tensor.shape()[0],
                what: "ROA electric_pol/g_tensor frame counts",
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

        // One central-diff pass each — no 12 separate column allocations.
        let a = central_diff_series(el_pol, dt);
        let g = central_diff_series(g_tensor, dt);

        let (a_iso, a_aniso) = raman_invariant_series(&a);
        let (g_iso, g_aniso) = raman_invariant_series(&g);
        let raw = TimeCorrelationSpec::new()
            .max_lag(max_lag)
            .normalization(Normalization::Raw);
        let acf_iso = cross_correlate(a_iso.view(), g_iso.view(), dt, &raw)?.values;
        let acf_aniso = cross_correlate(
            a_aniso.view(),
            g_aniso.view(),
            dt,
            &raw.component_weights(&ANISO_WEIGHTS),
        )?
        .values;

        Ok(RoaCrossTensorResult {
            lag_times: lag_times(max_lag, dt),
            acf_iso,
            acf_aniso,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::raman_tensor::{DIAG_ANISO_WEIGHT, OFFDIAG_ANISO_WEIGHT};
    use super::*;
    use crate::compute::Compute;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use rand::{RngExt, SeedableRng};
    use rustfft::FftPlanner;

    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut s = Array2::zeros((n, cols));
        for t in 0..n {
            for c in 0..cols {
                s[[t, c]] = rng.random_range(-1.0..1.0);
            }
        }
        s
    }

    /// Pin, bit for bit: central-difference fluxes, trace/3 iso cross-
    /// correlation, weighted (½ diagonal differences, 3 off-diagonals)
    /// anisotropic cross-correlation, components accumulated in that order.
    #[test]
    fn matches_manual_weighted_xcorr_path() {
        let n = 128;
        let dt = 0.5;
        let res = 40;
        let el = rng_series(n, 6, 21);
        let g = rng_series(n, 6, 22);
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let inv_2dt = 0.5 / dt;
        let diff = |s: &Array2<f64>, c: usize| -> Array1<f64> {
            (1..n - 1)
                .map(|t| (s[[t + 1, c]] - s[[t - 1, c]]) * inv_2dt)
                .collect()
        };
        let mut planner = FftPlanner::new();
        let a_iso: Array1<f64> = (0..flux_len)
            .map(|t| (diff(&el, 0)[t] + diff(&el, 1)[t] + diff(&el, 2)[t]) / 3.0)
            .collect();
        let g_iso: Array1<f64> = (0..flux_len)
            .map(|t| (diff(&g, 0)[t] + diff(&g, 1)[t] + diff(&g, 2)[t]) / 3.0)
            .collect();
        let iso = sig::xcorr_fft_with_planner(&mut planner, &a_iso, &g_iso, max_lag).unwrap();
        let mut aniso = Array1::<f64>::zeros(max_lag + 1);
        for (i, j) in [(0usize, 1usize), (1, 2), (2, 0)] {
            let av = &diff(&el, i) - &diff(&el, j);
            let gv = &diff(&g, i) - &diff(&g, j);
            let xc = sig::xcorr_fft_with_planner(&mut planner, &av, &gv, max_lag).unwrap();
            for k in 0..=max_lag {
                aniso[k] += DIAG_ANISO_WEIGHT * xc[k];
            }
        }
        for off in 3..6 {
            let xc =
                sig::xcorr_fft_with_planner(&mut planner, &diff(&el, off), &diff(&g, off), max_lag)
                    .unwrap();
            for k in 0..=max_lag {
                aniso[k] += OFFDIAG_ANISO_WEIGHT * xc[k];
            }
        }
        let raw = RoaCrossTensor
            .compute(&no_frames(), (&el, &g, dt, res))
            .unwrap();
        assert_eq!(raw.acf_iso, iso);
        assert_eq!(raw.acf_aniso, aniso);
        assert_eq!(raw.lag_times.len(), max_lag + 1);
    }

    #[test]
    fn rejects_bad_shapes() {
        let ok = rng_series(10, 6, 1);
        let bad = rng_series(10, 3, 1);
        let short = rng_series(8, 6, 1);
        assert!(
            RoaCrossTensor
                .compute(&no_frames(), (&ok, &bad, 1.0, 4))
                .is_err()
        );
        assert!(
            RoaCrossTensor
                .compute(&no_frames(), (&ok, &short, 1.0, 4))
                .is_err()
        );
        assert!(
            RoaCrossTensor
                .compute(&no_frames(), (&ok, &ok, 0.0, 4))
                .is_err()
        );
    }
}
