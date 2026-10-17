//! Spectroscopy: Raman and the frequency-dependent permittivity, and the
//! compositions every vibrational spectrum is.
//!
//! # A spectrum is a composition
//!
//! Every vibrational spectrum is a series operation, a correlation and a
//! transform — no spectrum owns a correlation estimator or an FFT of its own:
//!
//! | Spectrum | Series | Correlation | Transform |
//! |---|---|---|---|
//! | VDOS | velocities | [`Vacf`](crate::compute::Vacf) | [`Spectrum`](crate::compute::Spectrum) |
//! | IR | [`DipoleMoment`](crate::compute::DipoleMoment) → [`Derivative`](crate::compute::Derivative) | [`CurrentAcf`](crate::compute::CurrentAcf) of `Ṁ` | [`Spectrum`](crate::compute::Spectrum) |
//! | VCD | [`Derivative`](crate::compute::Derivative) of `μ` and of `m` | [`Correlator`](crate::compute::Correlator) cross `μ̇ · ṁ` (signed) | [`Spectrum`](crate::compute::Spectrum) |
//! | Raman / resonance Raman | [`PolarizabilityParts`] → [`Derivative`](crate::compute::Derivative) | [`Correlator`](crate::compute::Correlator) auto of each part | [`RamanSpectrum`] |
//! | ROA | parts of `α̇` and of `Ġ′` | [`Correlator`](crate::compute::Correlator) cross of each part | [`RamanSpectrum`] |
//! | ε(ω) | dipole / current series | [`DipoleAcf`](crate::compute::DipoleAcf), [`CurrentAcf`](crate::compute::CurrentAcf), `Ṁ × M` cross | [`EinsteinHelfandSpectrum`] / [`GreenKuboSpectrum`] / [`DipoleAutocorrelationSpectrum`] / [`DipoleRateCrossSpectrum`] |
//!
//! Every correlation is the unbiased estimator (`1 / (T − t)` per lag) by
//! default, so a VDOS and an IR spectrum are on the same footing; the raw
//! lagged sums the former IR / VCD / Raman / ROA correlators returned are
//! [`Normalization::None`](crate::compute::Normalization::None) on
//! [`CurrentAcf`](crate::compute::CurrentAcf) or the
//! [`Correlator`](crate::compute::Correlator). The vibrational
//! intensities are in arbitrary units; the permittivity carries its physical
//! prefactor. The checks [`KramersKronig`], [`ConductivitySumRule`] and
//! [`RouteAgreement`] judge a permittivity spectrum and return a
//! [`Verdict`](crate::compute::Verdict).

mod dielectric_checks;
mod dielectric_spectrum;
mod polarizability_parts;
mod raman_spectrum;

pub use dielectric_checks::{
    ConductivitySumRule, KramersKronig, KramersKronigCheck, KramersKronigInput, PermittivityPart,
    RouteAgreement, RouteAgreementCheck, RouteAgreementInput, SumRuleCheck, SumRuleInput,
};
pub use dielectric_spectrum::{
    DielectricSpectrumInput, DielectricSpectrumResult, DipoleAutocorrelationSpectrum,
    DipoleRateCrossSpectrum, EinsteinHelfandSpectrum, GreenKuboSpectrum, PlateauValue,
};
pub use polarizability_parts::{
    PolarizabilityParts, PolarizabilityPartsInput, PolarizabilityPartsResult,
};
pub use raman_spectrum::{RamanSpectrum, RamanSpectrumInput, RamanSpectrumResult};

/// The compositions of the former raw spectroscopy computes (`IrFlux`,
/// `VcdCrossFlux`, `RamanTensor`, `RoaCrossTensor`, `PowerSpectrum`, …),
/// pinned against the definitions written out.
///
/// Behaviour change #1 (spec): the raw correlations of IR, VCD, Raman and ROA
/// were unnormalised lagged sums `Σ_τ a(τ)·b(τ+k)`; they are now the unbiased
/// estimator `1/(T−k) Σ_τ a(τ)·b(τ+k)`. The written-out references below
/// divide by `T − k` accordingly — that division is the re-pin; the spectra
/// keep their shape and change scale.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::time_series::test_support::{assert_close, direct_lagged_sum};
    use crate::compute::{
        Accumulate, Compute, Correlator, CorrelatorFrame, CorrelatorInput, CurrentAcf,
        CurrentAcfInput, Derivative, DerivativeInput, LagKernel, Normalization, Spectrum,
        SpectrumInput, Stencil, Stream, TimeCorrelation, Vacf, Window,
    };
    use molrs::op::F;
    use ndarray::{Array1, Array2, ArrayView2, Axis};

    fn central(series: ArrayView2<'_, F>, dt: F) -> Array2<F> {
        Derivative::new(Stencil::Central)
            .compute(DerivativeInput { series, dt })
            .unwrap()
    }

    /// A lagged-product correlation of `T × C` series, summed over columns.
    fn product(a: &Array2<F>, b: Option<&Array2<F>>, max_lag: usize, dt: F) -> TimeCorrelation {
        Correlator::new(LagKernel::Product, max_lag)
            .compute(CorrelatorInput {
                a: a.view().insert_axis(Axis(1)),
                b: b.map(|b| b.view().insert_axis(Axis(1))),
                dt,
                boxes: None,
            })
            .unwrap()
    }

    /// `direct_lagged_sum / (n − k)`: the unbiased reference.
    fn unbiased(a: &[F], b: &[F], max_lag: usize) -> Vec<F> {
        let n = a.len();
        direct_lagged_sum(a, b, max_lag)
            .into_iter()
            .enumerate()
            .map(|(k, s)| s / (n - k) as F)
            .collect()
    }

    fn sine(n: usize, dt: F, freq_thz: F) -> impl Fn(usize) -> F {
        move |t| {
            (2.0 * std::f64::consts::PI
                * freq_thz
                * crate::core::unit_factors::THZ_TO_PER_FS.get()
                * (t as F * dt))
                .sin()
                * if t < n { 1.0 } else { 0.0 }
        }
    }

    /// `(cm⁻¹, Re X / n_fft)` — the former `PowerSpectrum` / `IrSpectrum` /
    /// `VcdSpectrum` output, now the caller's two lines over [`Spectrum`].
    fn vibrational(acf: &TimeCorrelation) -> (Array1<F>, Array1<F>) {
        let out = Spectrum::new(Window::CosineSq, 4)
            .unwrap()
            .compute(SpectrumInput {
                signal: acf.channel(0).unwrap(),
                dt: acf.lag_times[1],
            })
            .unwrap();
        let n_fft = out.n_fft as F;
        (
            out.frequencies * RamanSpectrum::wavenumber_per_angular(),
            out.real / n_fft,
        )
    }

    // ── VDOS (former PowerSpectrum) ──

    fn sine_velocities(n: usize, dt: F, freq_thz: F) -> Array2<F> {
        let wave = sine(n, dt, freq_thz);
        Array2::from_shape_fn((n, 3), |(t, d)| if d == 0 { wave(t) } else { 0.0 })
    }

    fn vacf(v: &Array2<F>, dt: F, max_lag: usize) -> TimeCorrelation {
        Vacf::new(max_lag)
            .accumulator()
            .fold(v.outer_iter().map(|row| CorrelatorFrame {
                a: row.insert_axis(Axis(1)),
                b: None,
                dt,
                simbox: None,
            }))
            .unwrap()
    }

    #[test]
    fn vacf_plus_vdos_matches_manual_acf_path() {
        // The raw velocity ACF: per-dof mean-subtract, direct ACF, DOF
        // average, 1/(n−τ).
        let (n, dt, res) = (1024, 0.5, 200);
        let v = sine_velocities(n, dt, 10.0);
        let mut acf = Array1::<F>::zeros(res + 1);
        for d in 0..3 {
            let col: Vec<F> = v.column(d).to_vec();
            let mean = col.iter().sum::<F>() / n as F;
            let centered: Vec<F> = col.iter().map(|x| x - mean).collect();
            let c = direct_lagged_sum(&centered, &centered, res);
            for k in 0..=res {
                acf[k] += c[k];
            }
        }
        for k in 0..=res {
            acf[k] /= (3 * (n - k)) as F;
        }
        let raw = vacf(&v, dt, res);
        assert_close(
            raw.channel(0).unwrap().as_slice().unwrap(),
            acf.as_slice().unwrap(),
            1e-12,
            "acf",
        );
    }

    #[test]
    fn vdos_sine_peak_at_333_cm1() {
        // v_x(t) = sin(2π·10 THz·t), dt = 0.5 fs.
        // 10 THz = 0.01 fs⁻¹ → 333.56 cm⁻¹.
        let (n, dt) = (4096, 0.5);
        let raw = vacf(&sine_velocities(n, dt, 10.0), dt, 200);
        let (frequencies_cm1, intensities) = vibrational(&raw);
        let search_end = intensities.len().saturating_sub(3);
        let max_idx = intensities
            .iter()
            .enumerate()
            .skip(1)
            .take(search_end.saturating_sub(1))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let peak_cm1 = frequencies_cm1[max_idx];
        assert!(
            (peak_cm1 - 333.56).abs() < 20.0,
            "peak at {peak_cm1} cm⁻¹, expected ~333.56 cm⁻¹"
        );
    }

    // ── IR (former IrFlux + IrSpectrum) ──

    #[test]
    fn irflux_plus_ir_transform_matches_manual_acf_path() {
        let (n, dt, res) = (1024, 0.5, 200);
        let wave = sine(n, dt, 10.0);
        let dm = Array2::from_shape_fn((n, 3), |(t, d)| if d == 2 { wave(t) } else { 0.0 });

        // Written out: central-difference flux, unbiased ACF, Σ components.
        let inv_2dt = 0.5 / dt;
        let mut acf = Array1::<F>::zeros(res + 1);
        for d in 0..3 {
            let flux: Vec<F> = (1..n - 1)
                .map(|t| (dm[[t + 1, d]] - dm[[t - 1, d]]) * inv_2dt)
                .collect();
            let c = unbiased(&flux, &flux, res);
            for k in 0..=res {
                acf[k] += c[k];
            }
        }

        // DipoleMoment series → Derivative → CurrentAcf.
        let rate = central(dm.view(), dt);
        let raw = CurrentAcf::new(res)
            .compute(CurrentAcfInput {
                current: rate.view(),
                dt,
            })
            .unwrap();
        assert_close(
            raw.channel(0).unwrap().as_slice().unwrap(),
            acf.as_slice().unwrap(),
            1e-12,
            "acf",
        );

        // The spectrum is linear in the correlation.
        let manual = TimeCorrelation {
            lag_times: raw.lag_times.clone(),
            values: acf.insert_axis(Axis(1)),
        };
        let (f_raw, i_raw) = vibrational(&raw);
        let (f_manual, i_manual) = vibrational(&manual);
        assert_eq!(f_raw, f_manual);
        assert_close(
            i_raw.as_slice().unwrap(),
            i_manual.as_slice().unwrap(),
            1e-12,
            "intensities",
        );
    }

    /// The former convention, still selectable: the raw IR flux correlation
    /// `Σ_τ Ṁ(τ) · Ṁ(τ + k)`, unnormalised (`IrFlux`'s `sum_column_acf`).
    #[test]
    fn ir_flux_unnormalised_sum_is_selectable() {
        let (n, dt, res) = (1024, 0.5, 200);
        let wave = sine(n, dt, 10.0);
        let dm = Array2::from_shape_fn((n, 3), |(t, d)| if d == 2 { wave(t) } else { 0.0 });
        let rate = central(dm.view(), dt);
        let mut sums = vec![0.0; res + 1];
        for d in 0..3 {
            let column = rate.column(d).to_vec();
            for (k, s) in direct_lagged_sum(&column, &column, res)
                .into_iter()
                .enumerate()
            {
                sums[k] += s;
            }
        }
        let raw = CurrentAcf::new(res)
            .with_normalization(Normalization::None)
            .compute(CurrentAcfInput {
                current: rate.view(),
                dt,
            })
            .unwrap();
        assert_close(
            raw.channel(0).unwrap().as_slice().unwrap(),
            &sums,
            1e-12,
            "unnormalised acf",
        );
    }

    // ── VCD (former VcdCrossFlux + VcdSpectrum) ──

    #[test]
    fn vcd_cross_flux_matches_manual_cross_correlation() {
        use crate::compute::time_series::test_support::rng_series;
        let (n, dt, res) = (300, 0.5, 60);
        let electric = rng_series(n, 3, 21);
        let magnetic = rng_series(n, 3, 22);
        let (mu_dot, m_dot) = (central(electric.view(), dt), central(magnetic.view(), dt));
        let raw = product(&mu_dot, Some(&m_dot), res, dt);

        let mut expected = vec![0.0; res + 1];
        for d in 0..3 {
            let c = unbiased(&mu_dot.column(d).to_vec(), &m_dot.column(d).to_vec(), res);
            for k in 0..=res {
                expected[k] += c[k];
            }
        }
        assert_close(
            raw.channel(0).unwrap().as_slice().unwrap(),
            &expected,
            1e-12,
            "vcd",
        );
        // Swapping the enantiomer (m → −m) flips the sign of the spectrum.
        let flipped = product(&mu_dot, Some(&(-&m_dot)), res, dt);
        let (_, i) = vibrational(&raw);
        let (_, i_flipped) = vibrational(&flipped);
        assert_close(
            i_flipped.as_slice().unwrap(),
            &(-&i).to_vec(),
            1e-12,
            "sign",
        );
    }

    /// The composed correlations select the former unnormalised sum on the
    /// `Correlator` itself (`VcdCrossFlux`'s `sum_column_xcorr`).
    #[test]
    fn vcd_cross_flux_unnormalised_sum_is_selectable() {
        use crate::compute::time_series::test_support::rng_series;
        let (n, dt, res) = (300, 0.5, 60);
        let mu_dot = central(rng_series(n, 3, 21).view(), dt);
        let m_dot = central(rng_series(n, 3, 22).view(), dt);
        let raw = Correlator::new(LagKernel::Product, res)
            .with_normalization(Normalization::None)
            .compute(CorrelatorInput {
                a: mu_dot.view().insert_axis(Axis(1)),
                b: Some(m_dot.view().insert_axis(Axis(1))),
                dt,
                boxes: None,
            })
            .unwrap();
        let mut sums = vec![0.0; res + 1];
        for d in 0..3 {
            let c = direct_lagged_sum(&mu_dot.column(d).to_vec(), &m_dot.column(d).to_vec(), res);
            for k in 0..=res {
                sums[k] += c[k];
            }
        }
        assert_close(
            raw.channel(0).unwrap().as_slice().unwrap(),
            &sums,
            1e-12,
            "unnormalised vcd",
        );
    }

    // ── Raman (former RamanTensor + RamanSpectrum) ──

    /// The Raman iso/aniso ACFs of a Voigt series, written out.
    fn raman_acfs(
        pol: &Array2<F>,
        other: Option<&Array2<F>>,
        dt: F,
        max_lag: usize,
    ) -> (Vec<F>, Vec<F>) {
        const DIAG_W: F = 0.5;
        const OFFDIAG_W: F = 3.0;
        let parts = |p: &Array2<F>| {
            let n_frames = p.nrows();
            let inv_2dt = 0.5 / dt;
            let mut iso = Vec::new();
            let mut comps: [Vec<F>; 6] = Default::default();
            for t in 1..n_frames - 1 {
                let a = p.row(t - 1);
                let b = p.row(t + 1);
                let xx = (b[0] - a[0]) * inv_2dt;
                let yy = (b[1] - a[1]) * inv_2dt;
                let zz = (b[2] - a[2]) * inv_2dt;
                iso.push((xx + yy + zz) / 3.0);
                comps[0].push(xx - yy);
                comps[1].push(yy - zz);
                comps[2].push(zz - xx);
                comps[3].push((b[3] - a[3]) * inv_2dt);
                comps[4].push((b[4] - a[4]) * inv_2dt);
                comps[5].push((b[5] - a[5]) * inv_2dt);
            }
            (iso, comps)
        };
        let (iso_a, comps_a) = parts(pol);
        let (iso_b, comps_b) = parts(other.unwrap_or(pol));
        let acf_iso = unbiased(&iso_a, &iso_b, max_lag);
        let mut acf_aniso = vec![0.0; max_lag + 1];
        for c in 0..6 {
            let w = if c < 3 { DIAG_W } else { OFFDIAG_W };
            let acf = unbiased(&comps_a[c], &comps_b[c], max_lag);
            for k in 0..=max_lag {
                acf_aniso[k] += w * acf[k];
            }
        }
        (acf_iso, acf_aniso)
    }

    /// `PolarizabilityParts → Derivative → Correlator` for both parts.
    fn raman_correlations(
        pol: &Array2<F>,
        other: Option<&Array2<F>>,
        dt: F,
        max_lag: usize,
    ) -> (TimeCorrelation, TimeCorrelation) {
        let split = |p: &Array2<F>| {
            PolarizabilityParts::new()
                .compute(PolarizabilityPartsInput { alpha: p.view() })
                .unwrap()
        };
        let a = split(pol);
        let b = other.map(split);
        let (iso_a, aniso_a) = (
            central(a.isotropic.view(), dt),
            central(a.anisotropic.view(), dt),
        );
        let b = b.map(|b| {
            (
                central(b.isotropic.view(), dt),
                central(b.anisotropic.view(), dt),
            )
        });
        (
            product(&iso_a, b.as_ref().map(|b| &b.0), max_lag, dt),
            product(&aniso_a, b.as_ref().map(|b| &b.1), max_lag, dt),
        )
    }

    fn voigt_wave(n: usize, dt: F, freq_thz: F, phase: F) -> Array2<F> {
        let wave = sine(n, dt, freq_thz);
        Array2::from_shape_fn((n, 6), |(t, c)| {
            (wave(t) + phase * c as F) * (1.0 + 0.1 * c as F)
        })
    }

    #[test]
    fn ramantensor_plus_raman_transform_matches_manual_acf_path() {
        let (n, dt, res) = (256, 0.5, 60);
        let pol = voigt_wave(n, dt, 30.0, 0.0);
        let (acf_iso, acf_aniso) = raman_acfs(&pol, None, dt, res);
        let (iso, aniso) = raman_correlations(&pol, None, dt, res);
        assert_close(
            iso.channel(0).unwrap().as_slice().unwrap(),
            &acf_iso,
            1e-12,
            "iso",
        );
        assert_close(
            aniso.channel(0).unwrap().as_slice().unwrap(),
            &acf_aniso,
            1e-12,
            "aniso",
        );

        let fit = RamanSpectrum::default()
            .with_incident_frequency(10000.0)
            .unwrap()
            .with_averaged(true);
        let manual_iso = TimeCorrelation {
            lag_times: iso.lag_times.clone(),
            values: Array1::from_vec(acf_iso).insert_axis(Axis(1)),
        };
        let manual_aniso = TimeCorrelation {
            lag_times: iso.lag_times.clone(),
            values: Array1::from_vec(acf_aniso).insert_axis(Axis(1)),
        };
        let run = |i: &TimeCorrelation, a: &TimeCorrelation| {
            fit.compute(RamanSpectrumInput {
                isotropic: i,
                anisotropic: a,
                temperature: Some(300.0),
            })
            .unwrap()
        };
        let from_raw = run(&iso, &aniso);
        let from_manual = run(&manual_iso, &manual_aniso);
        assert_eq!(from_raw.frequencies_cm1, from_manual.frequencies_cm1);
        assert_eq!(from_raw.resolution, res);
        for (got, want, what) in [
            (&from_raw.isotropic, &from_manual.isotropic, "isotropic"),
            (
                &from_raw.anisotropic,
                &from_manual.anisotropic,
                "anisotropic",
            ),
            (
                from_raw.parallel.as_ref().unwrap(),
                from_manual.parallel.as_ref().unwrap(),
                "parallel",
            ),
            (
                from_raw.perpendicular.as_ref().unwrap(),
                from_manual.perpendicular.as_ref().unwrap(),
                "perpendicular",
            ),
        ] {
            assert_close(
                got.as_slice().unwrap(),
                want.as_slice().unwrap(),
                1e-12,
                what,
            );
        }
    }

    // ── ROA (former RoaCrossTensor + RoaSpectrum) ──

    #[test]
    fn roa_cross_tensor_matches_manual_cross_correlation() {
        let (n, dt, res) = (256, 0.5, 60);
        let alpha = voigt_wave(n, dt, 30.0, 0.0);
        let g = voigt_wave(n, dt, 30.0, 0.05);
        let (acf_iso, acf_aniso) = raman_acfs(&alpha, Some(&g), dt, res);
        let (iso, aniso) = raman_correlations(&alpha, Some(&g), dt, res);
        assert_close(
            iso.channel(0).unwrap().as_slice().unwrap(),
            &acf_iso,
            1e-12,
            "roa iso",
        );
        assert_close(
            aniso.channel(0).unwrap().as_slice().unwrap(),
            &acf_aniso,
            1e-12,
            "roa aniso",
        );
        // ROA is the Raman transform of the cross-correlations.
        let roa = RamanSpectrum::default()
            .compute(RamanSpectrumInput {
                isotropic: &iso,
                anisotropic: &aniso,
                temperature: None,
            })
            .unwrap();
        assert!(roa.isotropic.iter().all(|v| v.is_finite()));
    }
}
