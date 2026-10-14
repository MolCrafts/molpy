//! Infrared absorption-spectrum transform.

use ndarray::Array1;

use super::cosine_sq_spectrum;
use super::spectra::SpectrumResult;
use crate::compute::ComputeError;
use crate::compute::Fit;

/// Infrared absorption spectrum transform of a **raw dipole-flux ACF**.
///
/// Identical window+FFT pipeline as [`PowerSpectrum`](super::PowerSpectrum);
/// the difference between IR and the power spectrum is entirely in *which* ACF
/// is supplied (dipole flux vs velocity), computed upstream by the
/// [`IrFlux`](super::IrFlux) raw compute.
#[derive(Debug, Clone, Copy, Default)]
pub struct IrSpectrum;

impl Fit for IrSpectrum {
    /// `(acf, dt_fs)` — the raw dipole-flux ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw flux ACF into an IR spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf, dt_fs) = input;
        let n = acf.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }
        let (frequencies_cm1, intensities) = cosine_sq_spectrum(acf, dt_fs)?;
        Ok(SpectrumResult {
            frequencies_cm1,
            intensities,
            resolution: n - 1,
            n_frames: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::ir_flux::IrFlux;
    use super::*;
    use crate::compute::LegacyCompute;
    use crate::compute::time_series::test_support::{assert_close, direct_lagged_sum};
    use molrs::core::Frame;
    use ndarray::Array2;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// The raw dipole-flux ACF the IrFlux compute / IrSpectrum consume,
    /// written out: central-difference flux, direct ACF, Σ over components.
    fn ir_acf(dm: &Array2<f64>, dt_fs: f64, max_lag: usize) -> Array1<f64> {
        let n_frames = dm.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Vec<f64> = (1..n_frames - 1)
                .map(|t| (dm[[t + 1, d]] - dm[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = direct_lagged_sum(&flux, &flux, max_lag);
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        acf_sum
    }

    #[test]
    fn irflux_plus_ir_transform_matches_manual_acf_path() {
        // IrFlux returns the unwindowed dipole-flux ACF the IrSpectrum
        // transform consumes; IrFlux + IrSpectrum == manual path.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let mut dm = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt;
            dm[[t, 2]] = (2.0
                * std::f64::consts::PI
                * 10.0
                * crate::core::unit_factors::THZ_TO_PER_FS.get()
                * tf)
                .sin();
        }
        let acf = ir_acf(&dm, dt, res);

        let raw = IrFlux.compute(&no_frames(), (&dm, dt, res)).unwrap();
        assert_close(
            raw.acf.as_slice().unwrap(),
            acf.as_slice().unwrap(),
            1e-12,
            "acf",
        );

        let from_raw = IrSpectrum.fit((&raw.acf, dt)).unwrap();
        let from_manual = IrSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_close(
            from_raw.intensities.as_slice().unwrap(),
            from_manual.intensities.as_slice().unwrap(),
            1e-12,
            "intensities",
        );
    }
}
