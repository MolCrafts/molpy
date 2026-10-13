//! Raman spectrum transform (window + FFT + cross-section/Bose prefactors).

use ndarray::Array1;
use rustfft::FftPlanner;

use super::spectra::RamanSpectrumResult;
use super::{acf_to_intensities, acf_to_spectrum, bose_factor, windowed_acf};
use crate::compute::ComputeError;
use crate::compute::Fit;
use crate::signal::WindowType;

/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: f64 = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: f64 = 15.0;

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**.
///
/// Applies the one `window` (cosine² by default) to each ACF, FFTs both
/// (reusing the isotropic frequency grid), and applies
/// the cross-section (`(ν₀ − ν)⁴ / ν`) and Bose (`1/(1 − exp(−hcν/kT))`)
/// prefactors. The raw iso/aniso ACFs come from the
/// [`RamanTensor`](super::RamanTensor) raw compute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RamanSpectrum {
    /// Laser frequency, cm⁻¹. The cross-section correction scales as
    /// `(ν₀ − ν)⁴ / ν`. Set to `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. Set to `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular polarization spectra.
    pub averaged: bool,
    /// Taper applied to both one-sided ACFs before the FFT.
    pub window: WindowType,
    /// Zero-padding factor of the transform.
    pub pad_factor: usize,
}

impl RamanSpectrum {
    /// Prefactors as given, cosine² window, pad factor 4.
    pub fn new(incident_frequency_cm1: f64, temperature_k: f64, averaged: bool) -> Self {
        Self {
            incident_frequency_cm1,
            temperature_k,
            averaged,
            window: WindowType::CosineSq,
            pad_factor: 4,
        }
    }

    pub fn with_window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }
}

impl Fit for RamanSpectrum {
    /// `(acf_iso, acf_aniso, dt_fs)` — the raw isotropic and (weighted)
    /// anisotropic ACFs (equal-length 1D) and timestep (fs, > 0).
    type Input<'a> = (&'a Array1<f64>, &'a Array1<f64>, f64);
    type Output = RamanSpectrumResult;

    /// Window + FFT + physical prefactors on the iso/aniso ACFs.
    ///
    /// # Errors
    /// * [`ComputeError::EmptyInput`] if either ACF is empty.
    /// * [`ComputeError::DimensionMismatch`] if the two ACFs differ in length.
    /// * [`ComputeError::OutOfRange`] if `dt_fs <= 0`.
    fn fit<'a>(&self, input: Self::Input<'a>) -> Result<Self::Output, ComputeError> {
        let (acf_iso, acf_aniso, dt_fs) = input;
        let n = acf_iso.len();
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if acf_aniso.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: acf_aniso.len(),
                what: "Raman (acf_iso, acf_aniso) lengths",
            });
        }
        if dt_fs <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt_fs",
                value: dt_fs.to_string(),
            });
        }

        if self.pad_factor == 0 {
            return Err(ComputeError::OutOfRange {
                field: "pad_factor",
                value: "0".to_string(),
            });
        }

        let max_lag = n - 1;
        let win_iso = windowed_acf(acf_iso, self.window);
        let win_aniso = windowed_acf(acf_aniso, self.window);

        let mut planner = FftPlanner::new();
        let n_pad = (self.pad_factor * (max_lag + 1)).next_power_of_two();
        let (frequencies_cm1, raw_iso) = acf_to_spectrum(&mut planner, &win_iso, dt_fs, n_pad);
        let raw_aniso = acf_to_intensities(&mut planner, &win_aniso, n_pad);

        let mut iso_int = raw_iso;
        let mut aniso_int = raw_aniso;

        let apply_cross_section = self.incident_frequency_cm1 > 0.0;
        let apply_bose = self.temperature_k > 0.0;

        for j in 0..frequencies_cm1.len() {
            let nu = frequencies_cm1[j];
            let mut correction = 1.0;
            if apply_cross_section && nu > 0.0 {
                let dnu = self.incident_frequency_cm1 - nu;
                correction *= dnu.powi(4) / nu;
            }
            if apply_bose {
                correction *= bose_factor(nu, self.temperature_k);
            }
            iso_int[j] *= correction;
            aniso_int[j] *= correction;
        }

        let (parallel, perpendicular) = if self.averaged {
            let m = frequencies_cm1.len();
            let mut par = Array1::zeros(m);
            let mut perp = Array1::zeros(m);
            for j in 0..m {
                par[j] = iso_int[j] + PARALLEL_ANISO_COEFF * aniso_int[j];
                perp[j] = aniso_int[j] / PERPENDICULAR_DENOM;
            }
            (Some(par), Some(perp))
        } else {
            (None, None)
        };

        Ok(RamanSpectrumResult {
            frequencies_cm1,
            isotropic: iso_int,
            anisotropic: aniso_int,
            parallel,
            perpendicular,
            resolution: max_lag,
            n_frames: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::raman_tensor::RamanTensor;
    use super::*;
    use crate::compute::Compute;
    use molrs::core::Frame;
    use molrs::signal as sig;
    use ndarray::Array2;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// Rebuild the raw iso/aniso ACFs the RamanTensor compute / RamanSpectrum consume.
    fn raman_acfs(pol: &Array2<f64>, dt_fs: f64, max_lag: usize) -> (Array1<f64>, Array1<f64>) {
        const DIAG_W: f64 = 0.5;
        const OFFDIAG_W: f64 = 3.0;
        let n_frames = pol.shape()[0];
        let inv_2dt = 0.5 / dt_fs;
        let flux_len = n_frames - 2;
        let mut iso = Vec::with_capacity(flux_len);
        let mut comps: [Vec<f64>; 6] = Default::default();
        for t in 1..n_frames - 1 {
            let p = pol.row(t - 1);
            let q = pol.row(t + 1);
            let xx = (q[0] - p[0]) * inv_2dt;
            let yy = (q[1] - p[1]) * inv_2dt;
            let zz = (q[2] - p[2]) * inv_2dt;
            iso.push((xx + yy + zz) / 3.0);
            comps[0].push(xx - yy);
            comps[1].push(yy - zz);
            comps[2].push(zz - xx);
            comps[3].push((q[3] - p[3]) * inv_2dt);
            comps[4].push((q[4] - p[4]) * inv_2dt);
            comps[5].push((q[5] - p[5]) * inv_2dt);
        }
        let mut planner = FftPlanner::new();
        let iso_series = Array1::from_vec(iso);
        let acf_iso = sig::acf_fft_with_planner(&mut planner, &iso_series, max_lag).unwrap();
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in comps.iter_mut().enumerate() {
            let w = if c < 3 { DIAG_W } else { OFFDIAG_W };
            let col = Array1::from_vec(std::mem::take(comp));
            let acf = sig::acf_fft_with_planner(&mut planner, &col, max_lag).unwrap();
            for k in 0..=max_lag {
                acf_aniso[k] += w * acf[k];
            }
        }
        (acf_iso, acf_aniso)
    }

    #[test]
    fn ramantensor_plus_raman_transform_matches_manual_acf_path() {
        // ac-003/ac-007 (Raman): RamanTensor returns the unwindowed iso/aniso
        // ACFs; RamanTensor + RamanSpectrum == manual path (iso/aniso/par/perp).
        let n = 256;
        let dt = 0.5;
        let res = 60;
        let mut pol = Array2::zeros((n, 6));
        for t in 0..n {
            let tf = t as f64 * dt;
            let val = (2.0
                * std::f64::consts::PI
                * 30.0
                * crate::core::unit_factors::THZ_TO_PER_FS.get()
                * tf)
                .sin();
            for c in 0..6 {
                pol[[t, c]] = val * (1.0 + 0.1 * c as f64);
            }
        }
        let incident = 10000.0;
        let temp = 300.0;
        let flux_len = n - 2;
        let max_lag = res.min(flux_len - 1);
        let (acf_iso, acf_aniso) = raman_acfs(&pol, dt, max_lag);

        let raw = RamanTensor.compute(&no_frames(), (&pol, dt, res)).unwrap();
        assert_eq!(raw.acf_iso, acf_iso); // raw, unwindowed.
        assert_eq!(raw.acf_aniso, acf_aniso);

        let fit = RamanSpectrum::new(incident, temp, true);
        let from_raw = fit.fit((&raw.acf_iso, &raw.acf_aniso, dt)).unwrap();
        let from_manual = fit.fit((&acf_iso, &acf_aniso, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.isotropic, from_manual.isotropic);
        assert_eq!(from_raw.anisotropic, from_manual.anisotropic);
        assert_eq!(from_raw.parallel, from_manual.parallel);
        assert_eq!(from_raw.perpendicular, from_manual.perpendicular);
    }

    #[test]
    fn raman_rejects_mismatched_acf_lengths() {
        // ac-016: shape mismatch -> DimensionMismatch.
        let iso = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let aniso = Array1::from_vec(vec![1.0, 2.0]);
        let err = RamanSpectrum::new(0.0, 0.0, false)
            .fit((&iso, &aniso, 0.5))
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
