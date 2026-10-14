//! Raman spectrum transform (window + FFT + cross-section/Bose prefactors).

use ndarray::Array1;

use super::spectra::RamanSpectrumResult;
use super::{bose_factor, cosine_sq_spectrum};
use crate::compute::ComputeError;
use crate::compute::Fit;

/// Parallel polarization: `I_∥ = I_iso + (4/45)·I_aniso`.
const PARALLEL_ANISO_COEFF: f64 = 4.0 / 45.0;
/// Perpendicular polarization denominator: `I_⊥ = I_aniso / 15`.
const PERPENDICULAR_DENOM: f64 = 15.0;

/// Raman spectrum transform of **raw isotropic + anisotropic ACFs**.
///
/// Applies the one cos²-windowed spectrum (`cosine_sq_spectrum`) to each
/// ACF on one frequency grid, and applies
/// the cross-section (`(ν₀ − ν)⁴ / ν`) and Bose (`1/(1 − exp(−hcν/kT))`)
/// prefactors. The raw iso/aniso ACFs come from the
/// [`RamanTensor`](super::RamanTensor) raw compute.
#[derive(Debug, Clone, Copy)]
pub struct RamanSpectrum {
    /// Laser frequency, cm⁻¹. The cross-section correction scales as
    /// `(ν₀ − ν)⁴ / ν`. Set to `0.0` to skip.
    pub incident_frequency_cm1: f64,
    /// Temperature, K, for the Bose factor. Set to `0.0` to skip.
    pub temperature_k: f64,
    /// If `true`, also emit parallel / perpendicular polarization spectra.
    pub averaged: bool,
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

        let max_lag = n - 1;
        // The same cos²-windowed spectrum of both ACFs, on one grid.
        let (frequencies_cm1, raw_iso) = cosine_sq_spectrum(acf_iso, dt_fs)?;
        let (_, raw_aniso) = cosine_sq_spectrum(acf_aniso, dt_fs)?;

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
    use crate::compute::LegacyCompute;
    use crate::compute::time_series::test_support::{assert_close, direct_lagged_sum};
    use molrs::core::Frame;
    use ndarray::Array2;

    /// Empty frame slice for the series-based raw computes.
    fn no_frames() -> Vec<&'static Frame> {
        Vec::new()
    }

    /// The raw iso/aniso ACFs the RamanTensor compute / RamanSpectrum
    /// consume, written out.
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
        let acf_iso = Array1::from_vec(direct_lagged_sum(&iso, &iso, max_lag));
        let mut acf_aniso = Array1::<f64>::zeros(max_lag + 1);
        for (c, comp) in comps.iter().enumerate() {
            let w = if c < 3 { DIAG_W } else { OFFDIAG_W };
            let acf = direct_lagged_sum(comp, comp, max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += w * acf[k];
            }
        }
        (acf_iso, acf_aniso)
    }

    fn close(got: &Option<Array1<f64>>, want: &Option<Array1<f64>>, what: &str) {
        let (got, want) = (got.as_ref().unwrap(), want.as_ref().unwrap());
        assert_close(
            got.as_slice().unwrap(),
            want.as_slice().unwrap(),
            1e-12,
            what,
        );
    }

    #[test]
    fn ramantensor_plus_raman_transform_matches_manual_acf_path() {
        // RamanTensor returns the unwindowed iso/aniso ACFs; RamanTensor +
        // RamanSpectrum == manual path (iso/aniso/par/perp).
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
        let (acf_iso, acf_aniso) = raman_acfs(&pol, dt, res);

        let raw = RamanTensor.compute(&no_frames(), (&pol, dt, res)).unwrap();
        assert_close(
            raw.acf_iso.as_slice().unwrap(),
            acf_iso.as_slice().unwrap(),
            1e-12,
            "iso",
        );
        assert_close(
            raw.acf_aniso.as_slice().unwrap(),
            acf_aniso.as_slice().unwrap(),
            1e-12,
            "aniso",
        );

        let fit = RamanSpectrum {
            incident_frequency_cm1: incident,
            temperature_k: temp,
            averaged: true,
        };
        let from_raw = fit.fit((&raw.acf_iso, &raw.acf_aniso, dt)).unwrap();
        let from_manual = fit.fit((&acf_iso, &acf_aniso, dt)).unwrap();
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        for (got, want, what) in [
            (&from_raw.isotropic, &from_manual.isotropic, "isotropic"),
            (
                &from_raw.anisotropic,
                &from_manual.anisotropic,
                "anisotropic",
            ),
        ] {
            assert_close(
                got.as_slice().unwrap(),
                want.as_slice().unwrap(),
                1e-12,
                what,
            );
        }
        close(&from_raw.parallel, &from_manual.parallel, "parallel");
        close(
            &from_raw.perpendicular,
            &from_manual.perpendicular,
            "perpendicular",
        );
    }

    #[test]
    fn raman_rejects_mismatched_acf_lengths() {
        // ac-016: shape mismatch -> DimensionMismatch.
        let iso = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let aniso = Array1::from_vec(vec![1.0, 2.0]);
        let err = RamanSpectrum {
            incident_frequency_cm1: 0.0,
            temperature_k: 0.0,
            averaged: false,
        }
        .fit((&iso, &aniso, 0.5))
        .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
