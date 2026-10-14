//! Infrared absorption-spectrum transform.

use ndarray::Array1;

use super::power_spectrum::PowerSpectrum;
use super::spectra::SpectrumResult;
use crate::compute::ComputeError;
use crate::compute::Fit;
use crate::signal::WindowType;

/// Infrared absorption spectrum transform of a **raw dipole-flux ACF**.
///
/// Identical window + FFT pipeline as [`PowerSpectrum`](super::PowerSpectrum)
/// (it delegates there); the difference between IR and the power spectrum is
/// entirely in *which* ACF is supplied (dipole flux vs velocity), computed
/// upstream by the [`IrFlux`](super::IrFlux) raw compute.
///
/// `Default` is the cosine² window with a pad factor of 4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IrSpectrum {
    /// Taper applied to the one-sided ACF before the FFT.
    pub window: WindowType,
    /// Zero-padding factor of the transform.
    pub pad_factor: usize,
}

impl Default for IrSpectrum {
    fn default() -> Self {
        let PowerSpectrum { window, pad_factor } = PowerSpectrum::default();
        Self { window, pad_factor }
    }
}

impl IrSpectrum {
    /// The default transform with another window.
    pub fn with_window(window: WindowType) -> Self {
        Self {
            window,
            ..Self::default()
        }
    }
}

impl Fit for IrSpectrum {
    /// `(acf, dt_fs)` — the raw correlation (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw correlation into a spectrum; see
    /// [`PowerSpectrum::fit`](super::PowerSpectrum) for the errors.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        PowerSpectrum {
            window: self.window,
            pad_factor: self.pad_factor,
        }
        .fit(input)
    }
}

#[cfg(test)]
mod tests {
    use super::super::ir_flux::IrFlux;
    use super::*;
    use crate::compute::Compute;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use ndarray::Array2;
    use rustfft::FftPlanner;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// Rebuild the raw dipole-flux ACF the IrFlux compute / IrSpectrum consume.
    fn ir_acf(dm: &Array2<f64>, dt_fs: f64, max_lag: usize) -> Array1<f64> {
        let n_frames = dm.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..3 {
            let flux: Array1<f64> = (1..n_frames - 1)
                .map(|t| (dm[[t + 1, d]] - dm[[t - 1, d]]) * inv_2dt)
                .collect();
            let acf = sig::acf_fft_with_planner(&mut planner, &flux, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        acf_sum
    }

    #[test]
    fn irflux_plus_ir_transform_matches_manual_acf_path() {
        // ac-003/ac-007 (IR): IrFlux returns the unwindowed dipole-flux ACF the
        // IrSpectrum transform consumes; IrFlux + IrSpectrum == manual path.
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
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let acf = ir_acf(&dm, dt, max_lag);

        let raw = IrFlux.compute(&no_frames(), (&dm, dt, res)).unwrap();
        assert_eq!(raw.acf, acf); // IrFlux returns the raw unwindowed ACF.

        let from_raw = IrSpectrum::default().fit((&raw.acf, dt)).unwrap();
        let from_manual = IrSpectrum::default().fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.intensities, from_manual.intensities);
    }

    #[test]
    fn ir_is_the_power_spectrum_transform_and_the_window_is_selectable() {
        let acf =
            Array1::from_iter((0..64).map(|k| (-(k as f64) / 10.0).exp() * (k as f64 * 0.4).cos()));
        let ir = IrSpectrum::default().fit((&acf, 0.5)).unwrap();
        let power = PowerSpectrum::default().fit((&acf, 0.5)).unwrap();
        assert_eq!(ir.intensities, power.intensities);
        let hann = IrSpectrum::with_window(WindowType::Hann)
            .fit((&acf, 0.5))
            .unwrap();
        assert_eq!(hann.frequencies_cm1, ir.frequencies_cm1);
        assert_ne!(hann.intensities, ir.intensities);
    }
}
