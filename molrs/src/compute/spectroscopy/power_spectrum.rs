//! Velocity power-spectrum (VDOS) transform.

use ndarray::Array1;
use rustfft::FftPlanner;

use super::spectra::SpectrumResult;
use super::window_and_fft;
use crate::compute::ComputeError;
use crate::compute::Fit;
use crate::signal::WindowType;

/// Velocity power spectrum (VDOS) transform of a **raw velocity ACF**.
///
/// Windows the one-sided ACF, zero-pads it to `(pad_factor · n)` rounded up
/// to a power of two, and takes the one-sided forward FFT — the `window_and_fft`
/// pipeline — on a raw, unnormalized velocity ACF, the
/// [`VacfResult`](crate::compute::VacfResult) of the
/// [`Vacf`](crate::compute::Vacf) compute. The same transform under other
/// names: [`IrSpectrum`](super::IrSpectrum) and [`VcdSpectrum`](super::VcdSpectrum)
/// delegate here.
///
/// `Default` is the cosine² window with a pad factor of 4. The cosine²
/// taper is `1` at `C(0)` and `0` at `C(max_lag)`, which keeps the zero-lag
/// value — the integral of the spectrum — intact; Hann and Blackman zero
/// `C(0)` and are for a two-sided, mean-removed signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSpectrum {
    /// Taper applied to the one-sided ACF before the FFT.
    pub window: WindowType,
    /// Zero-padding factor (the transform length is
    /// `(pad_factor · n).next_power_of_two()`); sets the frequency grid.
    pub pad_factor: usize,
}

impl Default for PowerSpectrum {
    fn default() -> Self {
        Self {
            window: WindowType::CosineSq,
            pad_factor: 4,
        }
    }
}

impl PowerSpectrum {
    /// The default transform with another window.
    pub fn with_window(window: WindowType) -> Self {
        Self {
            window,
            ..Self::default()
        }
    }
}

impl Fit for PowerSpectrum {
    /// `(acf, dt_fs)` — the raw velocity ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw ACF into a VDOS spectrum.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if the ACF is empty.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0` or `pad_factor == 0`.
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
        let mut planner = FftPlanner::new();
        let (frequencies_cm1, intensities) =
            window_and_fft(&mut planner, acf, dt_fs, self.window, self.pad_factor)?;
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
    use super::*;
    use crate::compute::Compute;
    use crate::compute::Vacf;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use ndarray::Array2;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// Rebuild the raw velocity ACF the VACF / PowerSpectrum path consumes
    /// (per-dof mean-subtract, FFT-ACF, sum, *= 1/n_dof).
    fn power_acf(velocities: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = velocities.shape()[0];
        let n_dof = velocities.shape()[1];
        let inv_n_frames = 1.0 / n_frames as f64;
        let mut planner = FftPlanner::new();
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..n_dof {
            let mut col: Array1<f64> = (0..n_frames).map(|t| velocities[[t, d]]).collect();
            let mean: f64 = col.iter().sum::<f64>() * inv_n_frames;
            for v in col.iter_mut() {
                *v -= mean;
            }
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        // Match VACF: DOF average + unbiased 1/(n-τ) time-origin normalization.
        let inv_n_dof = 1.0 / n_dof as f64;
        for k in 0..=max_lag {
            acf_sum[k] *= inv_n_dof / (n_frames - k) as f64;
        }
        acf_sum
    }

    fn sine_velocities(n: usize, dt_fs: f64, freq_thz: f64) -> Array2<f64> {
        let mut v = Array2::zeros((n, 3));
        for t in 0..n {
            let tf = t as f64 * dt_fs;
            v[[t, 0]] = (2.0
                * std::f64::consts::PI
                * freq_thz
                * crate::core::unit_factors::THZ_TO_PER_FS.get()
                * tf)
                .sin();
        }
        v
    }

    #[test]
    fn vacf_plus_vdos_matches_manual_acf_path() {
        // ac-003/ac-006: VACF returns the unbiased unwindowed ACF that
        // PowerSpectrum consumes; VACF + PowerSpectrum equals the manual path.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let v = sine_velocities(n, dt, 10.0);
        let max_lag = res.min(n - 1);
        let acf = power_acf(&v, max_lag);

        let raw = Vacf.compute(&no_frames(), (&v, dt, res)).unwrap();
        assert_eq!(raw.acf, acf); // VACF returns unbiased unwindowed ACF.

        let from_raw = PowerSpectrum::default().fit((&raw.acf, dt)).unwrap();
        let from_manual = PowerSpectrum::default().fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.intensities, from_manual.intensities);
    }

    #[test]
    fn vdos_sine_peak_at_333_cm1() {
        // ac-008 (scientific): v_x(t) = sin(2π·10 THz·t), dt = 0.5 fs.
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let n = 4096;
        let dt = 0.5;
        let v = sine_velocities(n, dt, 10.0);
        let raw = Vacf.compute(&no_frames(), (&v, dt, 200)).unwrap();
        let spec = PowerSpectrum::default().fit((&raw.acf, dt)).unwrap();
        let n_bins = spec.intensities.len();
        let search_end = n_bins.saturating_sub(3);
        let max_idx = spec
            .intensities
            .iter()
            .enumerate()
            .skip(1)
            .take(search_end.saturating_sub(1))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let peak_cm1 = spec.frequencies_cm1[max_idx];
        assert!(
            (peak_cm1 - 333.56).abs() < 20.0,
            "peak at {peak_cm1} cm⁻¹, expected ~333.56 cm⁻¹"
        );
    }

    #[test]
    fn power_rejects_empty_and_bad_dt() {
        let empty: Array1<f64> = Array1::from_vec(vec![]);
        assert!(matches!(
            PowerSpectrum::default().fit((&empty, 0.5)),
            Err(ComputeError::EmptyInput)
        ));
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            PowerSpectrum::default().fit((&acf, 0.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn default_is_the_cosine_sq_pad_4_transform_bit_for_bit() {
        // The transform every molrs spectrum used before the window became a
        // field: cos²(π k / (2 max_lag)) taper, pad to (4 n).next_power_of_two().
        let n: usize = 50;
        let dt = 0.5;
        let acf =
            Array1::from_iter((0..n).map(|k| (-(k as f64) / 12.0).exp() * (k as f64 * 0.3).cos()));
        let max_lag = n - 1;
        let denom = 2.0 * max_lag.max(1) as f64;
        let windowed: Array1<f64> = acf
            .iter()
            .enumerate()
            .map(|(k, a)| a * (std::f64::consts::PI * k as f64 / denom).cos().powi(2))
            .collect();
        let n_pad = (4 * n).next_power_of_two();
        let mut planner = FftPlanner::new();
        let (freqs, ints) = super::super::acf_to_spectrum(&mut planner, &windowed, dt, n_pad);
        let spec = PowerSpectrum::default().fit((&acf, dt)).unwrap();
        assert_eq!(spec.frequencies_cm1, freqs);
        assert_eq!(spec.intensities, ints);
        assert_eq!(
            PowerSpectrum::default(),
            PowerSpectrum {
                window: WindowType::CosineSq,
                pad_factor: 4
            }
        );
    }

    #[test]
    fn hann_blackman_and_none_differ_from_cosine_sq_and_pad_sets_the_grid() {
        let acf = Array1::from_iter((0..40).map(|k| (-(k as f64) / 8.0).exp()));
        let base = PowerSpectrum::default().fit((&acf, 1.0)).unwrap();
        for w in [WindowType::Hann, WindowType::Blackman, WindowType::None] {
            let other = PowerSpectrum::with_window(w).fit((&acf, 1.0)).unwrap();
            assert_eq!(other.frequencies_cm1, base.frequencies_cm1);
            assert_ne!(other.intensities, base.intensities, "{w:?}");
        }
        let finer = PowerSpectrum {
            window: WindowType::CosineSq,
            pad_factor: 16,
        }
        .fit((&acf, 1.0))
        .unwrap();
        assert_eq!(
            finer.frequencies_cm1.len(),
            4 * base.frequencies_cm1.len() - 3
        );
        assert!(matches!(
            PowerSpectrum {
                window: WindowType::CosineSq,
                pad_factor: 0
            }
            .fit((&acf, 1.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
