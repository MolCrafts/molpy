//! Velocity power-spectrum (VDOS) transform.

use ndarray::Array1;

use super::cosine_sq_spectrum;
use super::spectra::SpectrumResult;
use crate::compute::ComputeError;
use crate::compute::Fit;

/// Velocity power spectrum (VDOS) transform of a **raw velocity ACF**.
///
/// Applies the CosineSq window + zero-padded forward FFT (the
/// `cosine_sq_spectrum` pipeline) to a raw, unnormalized
/// velocity ACF — the [`TimeCorrelation`](crate::compute::TimeCorrelation)
/// of the [`Vacf`](crate::compute::Vacf) compute.
#[derive(Debug, Clone, Copy, Default)]
pub struct PowerSpectrum;

impl Fit for PowerSpectrum {
    /// `(acf, dt_fs)` — the raw velocity ACF (1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, f64);
    type Output = SpectrumResult;

    /// Window + FFT the raw ACF into a VDOS spectrum.
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
    use super::*;
    use crate::compute::time_series::test_support::{assert_close, direct_lagged_sum};
    use crate::compute::{Compute, Vacf, VacfInput};
    use ndarray::Array2;

    fn vacf(v: &Array2<f64>, dt: f64, max_lag: usize) -> Array1<f64> {
        Vacf::new(max_lag)
            .compute(VacfInput {
                velocities: v.view(),
                dt,
            })
            .unwrap()
            .channel(0)
            .to_owned()
    }

    /// The raw velocity ACF the VACF / PowerSpectrum path consumes, written
    /// out (per-dof mean-subtract, direct ACF, DOF average, 1/(n−τ)).
    fn power_acf(velocities: &Array2<f64>, max_lag: usize) -> Array1<f64> {
        let n_frames = velocities.shape()[0];
        let n_dof = velocities.shape()[1];
        let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
        for d in 0..n_dof {
            let col: Vec<f64> = velocities.column(d).to_vec();
            let mean = col.iter().sum::<f64>() / n_frames as f64;
            let centered: Vec<f64> = col.iter().map(|v| v - mean).collect();
            let acf = direct_lagged_sum(&centered, &centered, max_lag);
            for k in 0..=max_lag {
                acf_sum[k] += acf[k];
            }
        }
        for k in 0..=max_lag {
            acf_sum[k] /= (n_dof * (n_frames - k)) as f64;
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
        // VACF returns the unbiased unwindowed ACF that PowerSpectrum
        // consumes; VACF + PowerSpectrum equals the manual path.
        let n = 1024;
        let dt = 0.5;
        let res = 200;
        let v = sine_velocities(n, dt, 10.0);
        let acf = power_acf(&v, res);

        let raw = vacf(&v, dt, res);
        assert_close(
            raw.as_slice().unwrap(),
            acf.as_slice().unwrap(),
            1e-12,
            "acf",
        );

        let from_raw = PowerSpectrum.fit((&raw, dt)).unwrap();
        let from_manual = PowerSpectrum.fit((&acf, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_close(
            from_raw.intensities.as_slice().unwrap(),
            from_manual.intensities.as_slice().unwrap(),
            1e-12,
            "intensities",
        );
    }

    #[test]
    fn vdos_sine_peak_at_333_cm1() {
        // v_x(t) = sin(2π·10 THz·t), dt = 0.5 fs.
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let n = 4096;
        let dt = 0.5;
        let v = sine_velocities(n, dt, 10.0);
        let raw = vacf(&v, dt, 200);
        let spec = PowerSpectrum.fit((&raw, dt)).unwrap();
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
            PowerSpectrum.fit((&empty, 0.5)),
            Err(ComputeError::EmptyInput)
        ));
        let acf = Array1::from_vec(vec![1.0, 0.5, 0.25]);
        assert!(matches!(
            PowerSpectrum.fit((&acf, 0.0)),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
