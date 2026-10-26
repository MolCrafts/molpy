use super::*;
use crate::compute::fitting::curve;
use crate::compute::time_series::test_support::direct_lagged_sum;
use crate::compute::{
    CurrentAcf, CurrentAcfInput, DipoleAcf, DipoleAcfInput, StaticDielectric, StaticDielectricInput,
};
use molrs::core::constants::COULOMB_REAL as KAPPA;
use molrs::core::{FOUR_PI, FOUR_THIRDS_PI};
use ndarray::Array2;

/// The fluctuation dipole ACF (the former `DebyeRelaxation` raw curve).
fn dipole_acf(dm: &Array2<f64>, dt: f64, max_lag: usize) -> TimeCorrelation {
    DipoleAcf::new(max_lag)
        .compute(DipoleAcfInput {
            dipoles: dm.view(),
            dt,
        })
        .unwrap()
}

fn rng_dipole(n: usize, seed: u64) -> Array2<f64> {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut s = Array2::zeros((n, 3));
    for t in 0..n {
        for d in 0..3 {
            s[[t, d]] = rng.random::<f64>() - 0.5;
        }
    }
    s
}

fn state(correlation: &TimeCorrelation, volume: F, temperature: F) -> DielectricSpectrumInput<'_> {
    DielectricSpectrumInput {
        correlation,
        volume,
        temperature,
        boundary: EwaldBoundary::TinFoil,
    }
}

/// The continuous transform with 4× padding (the reference's own).
fn piecewise_linear_onesided_ft(y: &Array1<f64>, dt: f64) -> RawSpectrum {
    RouteConfig::new(UnitPreset::real())
        .continuous_transform(y, dt)
        .unwrap()
}

// ── Reference implementations ──────────────────────────────────────────
// Each spectrum written out inline in one function, an independent oracle
// the raw-compute + route composition is locked to (ac-001).

/// Einstein–Helfand spectrum, inline: fluctuation ACF, cos² taper,
/// derivative, one-sided FT, prefactor.
fn reference_einstein_helfand(
    dipole_moments: &Array2<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    max_correlation_time: usize,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let n_frames = dipole_moments.shape()[0];
    let max_lag = max_correlation_time.min(n_frames - 1);

    // ⟨|δM|²⟩ — the exact static (ω = 0) term.
    let n = n_frames as f64;
    let mut mean_m = [0.0_f64; 3];
    for t in 0..n_frames {
        for d in 0..3 {
            mean_m[d] += dipole_moments[[t, d]];
        }
    }
    for m in mean_m.iter_mut() {
        *m /= n;
    }
    let mut m_sq = 0.0;
    for t in 0..n_frames {
        for d in 0..3 {
            let dev = dipole_moments[[t, d]] - mean_m[d];
            m_sq += dev * dev;
        }
    }
    m_sq /= n;

    // Fluctuation ACF, summed over components, unbiased estimator.
    let mut acf = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..3 {
        let col: Vec<f64> = (0..n_frames)
            .map(|t| dipole_moments[[t, d]] - mean_m[d])
            .collect();
        let component = direct_lagged_sum(&col, &col, max_lag);
        for k in 0..=max_lag {
            acf[k] += component[k];
        }
    }
    for k in 0..=max_lag {
        acf[k] /= (n_frames - k) as f64;
    }

    // One-sided cos² taper.
    let denom = 2.0 * max_lag.max(1) as f64;
    for k in 0..=max_lag {
        let angle = std::f64::consts::PI * k as f64 / denom;
        acf[k] *= angle.cos().powi(2);
    }
    // Central finite-difference derivative.
    let mut deriv = Array1::<f64>::zeros(max_lag + 1);
    for k in 1..max_lag {
        deriv[k] = (acf[k + 1] - acf[k - 1]) / (2.0 * dt);
    }
    if max_lag >= 1 {
        deriv[max_lag] = (acf[max_lag] - acf[max_lag - 1]) / dt;
    }
    let (frequencies, dre, dim) = piecewise_linear_onesided_ft(&deriv, dt);

    let prefactor =
        FOUR_THIRDS_PI * KAPPA / (volume * UnitPreset::real().boltzmann() * temperature);
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    for j in 0..n_freq {
        eps_real[j] = epsilon_inf - prefactor * dre[j];
        eps_imag[j] = prefactor * dim[j];
    }
    eps_real[0] = epsilon_inf + prefactor * m_sq;
    eps_imag[0] = 0.0;
    (frequencies, eps_real, eps_imag)
}

/// The unbiased current ACF, summed directly, over the post-NaN `start=1`
/// series. Returned so a test can lock the route's transform tail to the
/// reference tail on the *same* ACF (bit-for-bit), independent of the
/// raw-compute estimator.
fn reference_gk_acf(current: &Array2<f64>, max_lag: usize) -> Array1<f64> {
    let n_frames = current.shape()[0];
    let start = 1;
    let n_eff = n_frames - start;
    let mut acf_sum = Array1::<f64>::zeros(max_lag + 1);
    for d in 0..3 {
        let col: Vec<f64> = (start..n_frames).map(|t| current[[t, d]]).collect();
        let acf = direct_lagged_sum(&col, &col, max_lag);
        for k in 0..=max_lag {
            acf_sum[k] += acf[k];
        }
    }
    for k in 0..=max_lag {
        acf_sum[k] /= (n_eff - k) as f64;
    }
    acf_sum
}

/// Green–Kubo spectrum tail on a pre-built ACF, inline: window, one-sided
/// FT, σ→ε.
fn reference_green_kubo_from_acf(
    acf: &Array1<f64>,
    dt: f64,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
    window_type: &str,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // The window, written out (symmetric N − 1 forms).
    let n = acf.len();
    let last = (n - 1) as f64;
    let pi = std::f64::consts::PI;
    let windowed_1d: Array1<f64> = acf
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let i = i as f64;
            let w = match window_type {
                "hann" => 0.5 * (1.0 - (2.0 * pi * i / last).cos()),
                "blackman" => {
                    0.42 - 0.5 * (2.0 * pi * i / last).cos() + 0.08 * (4.0 * pi * i / last).cos()
                }
                "cosine_sq" => (pi * i / (2.0 * last)).cos().powi(2),
                other => panic!("no reference window {other}"),
            };
            x * w
        })
        .collect();
    let (frequencies, spec_re, spec_im) = piecewise_linear_onesided_ft(&windowed_1d, dt);

    let sigma_prefactor = volume / (3.0 * UnitPreset::real().boltzmann() * temperature);
    let eps0_factor = FOUR_PI * KAPPA;
    let n_freq = frequencies.len();
    let mut eps_real = Array1::zeros(n_freq);
    let mut eps_imag = Array1::zeros(n_freq);
    for j in 0..n_freq {
        let omega = frequencies[j];
        if omega == 0.0 {
            eps_real[j] = epsilon_inf;
            eps_imag[j] = 0.0;
        } else {
            let sigma_re = sigma_prefactor * spec_re[j];
            let sigma_im = sigma_prefactor * spec_im[j];
            eps_real[j] = epsilon_inf + eps0_factor * sigma_im / omega;
            eps_imag[j] = eps0_factor * sigma_re / omega;
        }
    }
    (frequencies, eps_real, eps_imag)
}

/// FFT / fused-scale paths may differ by a few ULP from the hand-rolled
/// reference; keep a tight absolute tolerance.
fn assert_close_1d(got: &Array1<f64>, expected: &Array1<f64>, tol: f64, what: &str) {
    assert_eq!(got.len(), expected.len(), "{what} length");
    for k in 0..got.len() {
        let d = (got[k] - expected[k]).abs();
        assert!(
            d <= tol,
            "{what}[{k}]: |{got} − {exp}| = {d} > {tol}",
            got = got[k],
            exp = expected[k],
        );
    }
}

#[test]
fn eh_fit_matches_the_reference_bit_for_bit() {
    // ac-001: DipoleAcf raw ACF + EinsteinHelfandSpectrum matches the
    // inline reference (tight ULP tolerance).
    let n = 256;
    let dt = 0.001;
    let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
    let mct = 50;
    let dm = rng_dipole(n, 42);

    let (freq_l, re_l, im_l) = reference_einstein_helfand(&dm, dt, vol, temp, eps_inf, mct);

    let acf = dipole_acf(&dm, dt, mct);
    let fit = EinsteinHelfandSpectrum::new(UnitPreset::real())
        .with_window(Window::CosineSq)
        .with_epsilon_inf(eps_inf)
        .compute(state(&acf, vol, temp))
        .unwrap();

    assert_eq!(fit.frequencies, freq_l);
    // Shared FFT primitives may reassociate by a few ULP vs the inlined
    // reference; 1e-12 absolute is still machine-precision tight.
    assert_close_1d(&fit.eps_real, &re_l, 1e-12, "eps_real");
    assert_close_1d(&fit.eps_imag, &im_l, 1e-12, "eps_imag");
    assert_eq!(fit.epsilon_inf, eps_inf);
}

#[test]
fn gk_fit_matches_the_reference_bit_for_bit() {
    // ac-001: on the SAME raw current ACF, GreenKuboSpectrum matches the
    // reference transform tail bit-for-bit. The raw ACF is the unbiased
    // current ACF over the start=1 series.
    let n = 256;
    let dt = 0.001;
    let (vol, temp, eps_inf) = (1000.0, 300.0, 1.0);
    let mct = 50;
    // Current density with a NaN row 0 (as the old `current_density` emitted).
    let mut current = rng_dipole(n, 7);
    for d in 0..3 {
        current[[0, d]] = f64::NAN;
    }
    let start = 1;
    let effective_len = n - start;
    let max_lag = mct.min(effective_len.saturating_sub(1));
    let raw_acf = reference_gk_acf(&current, max_lag);
    let raw = curve(raw_acf.as_slice().unwrap(), dt);

    for (name, window) in [
        ("hann", Window::Hann),
        ("blackman", Window::Blackman),
        ("cosine_sq", Window::CosineSq),
    ] {
        let (freq_l, re_l, im_l) =
            reference_green_kubo_from_acf(&raw_acf, dt, vol, temp, eps_inf, name);

        let fit = GreenKuboSpectrum::new(UnitPreset::real())
            .with_window(window)
            .with_epsilon_inf(eps_inf)
            .compute(state(&raw, vol, temp))
            .unwrap();

        assert_eq!(fit.frequencies, freq_l, "window={name}");
        assert_eq!(fit.eps_real, re_l, "window={name}");
        assert_eq!(fit.eps_imag, im_l, "window={name}");
    }
}

#[test]
fn gk_raw_compute_acf_matches_the_reference_fft_acf() {
    // The CurrentAcf compute (FFT estimator) and the reference
    // direct-summation ACF agree to FP tolerance on the same series.
    let n = 256;
    let dt = 0.001;
    let mct = 50;
    let mut current = rng_dipole(n, 7);
    for d in 0..3 {
        current[[0, d]] = f64::NAN;
    }
    let start = 1;
    let effective_len = n - start;
    let max_lag = mct.min(effective_len.saturating_sub(1));
    let reference = reference_gk_acf(&current, max_lag);

    let post: Array2<f64> = current.slice(ndarray::s![start.., ..]).to_owned();
    let raw = CurrentAcf::new(max_lag)
        .compute(CurrentAcfInput {
            current: post.view(),
            dt,
        })
        .unwrap();
    assert_eq!(raw.n_lags(), reference.len());
    for k in 0..reference.len() {
        assert!((raw.values[[k, 0]] - reference[k]).abs() < 1e-9, "k={k}");
    }
}

#[test]
fn eh_dc_bin_recovers_neumann_static() {
    // ac-006: ε(ω=0) recovers the Neumann static dielectric constant.
    let n = 256;
    let dt = 0.001;
    let (vol, temp, eps_inf) = (1000.0, 300.0, 1.5);
    let dm = rng_dipole(n, 99);
    let static_eps = StaticDielectric::new(UnitPreset::real())
        .with_epsilon_inf(eps_inf)
        .compute(StaticDielectricInput {
            dipoles: dm.view(),
            volume: vol,
            temperature: temp,
            boundary: EwaldBoundary::TinFoil,
        })
        .unwrap()
        .epsilon;

    let acf = dipole_acf(&dm, dt, 50);
    let fit = EinsteinHelfandSpectrum::new(UnitPreset::real())
        .with_window(Window::CosineSq)
        .with_epsilon_inf(eps_inf)
        .compute(state(&acf, vol, temp))
        .unwrap();
    assert!((fit.eps_real[0] - static_eps).abs() < 1e-10);
    assert_eq!(fit.eps_imag[0], 0.0);
}

#[test]
fn eh_loss_finite_to_nyquist() {
    // ac-006: the loss spectrum ε″ stays finite to Nyquist (derivative-FT).
    let n = 2048;
    let dt = 1.0;
    let dm = rng_dipole(n, 11);
    let acf = dipole_acf(&dm, dt, 100);
    let fit = EinsteinHelfandSpectrum::new(UnitPreset::real())
        .with_window(Window::CosineSq)
        .compute(state(&acf, 1000.0, 300.0))
        .unwrap();
    assert!(fit.eps_imag.iter().all(|x| x.is_finite()));
    assert!(fit.eps_real.iter().all(|x| x.is_finite()));
}

#[test]
fn fits_reject_short_acf() {
    let tiny = curve(&[1.0], 1.0);
    assert!(matches!(
        EinsteinHelfandSpectrum::new(UnitPreset::real())
            .with_window(Window::CosineSq)
            .compute(state(&tiny, 1.0, 1.0)),
        Err(ComputeError::EmptyInput)
    ));
    assert!(matches!(
        GreenKuboSpectrum::new(UnitPreset::real())
            .with_window(Window::Hann)
            .compute(state(&tiny, 1.0, 1.0)),
        Err(ComputeError::EmptyInput)
    ));
}

/// The string window and its `parse_window_type` (with the `"None"` /
/// `"identity"` aliases) are gone: a window is a [`Window`], and its names
/// parse only through the canonical `FromStr`.
#[test]
fn gk_rejects_bad_window() {
    assert!(matches!(
        "nope".parse::<Window>(),
        Err(ComputeError::UnknownName { .. })
    ));
    assert!("identity".parse::<Window>().is_err());
}

/// One ε(ω) route as a closure.
type Route = dyn Fn(DielectricSpectrumInput<'_>) -> Result<DielectricSpectrumResult, ComputeError>;

/// Every route refuses a non-physical state and a vacuum boundary, and a
/// multi-channel correlation.
#[test]
fn routes_check_state_boundary_and_channels() {
    let acf = curve(&[1.0, 0.6, 0.3, 0.1], 0.5);
    let routes: [&Route; 4] = [
        &|i| {
            EinsteinHelfandSpectrum::new(UnitPreset::real())
                .with_window(Window::CosineSq)
                .compute(i)
        },
        &|i| {
            GreenKuboSpectrum::new(UnitPreset::real())
                .with_window(Window::CosineSq)
                .compute(i)
        },
        &|i| {
            DipoleRateCrossSpectrum::new(UnitPreset::real())
                .with_window(Window::None)
                .compute(i)
        },
        &|i| {
            DipoleAutocorrelationSpectrum::new(UnitPreset::real())
                .with_window(Window::None)
                .compute(i)
        },
    ];
    let two = TimeCorrelation {
        lag_times: acf.lag_times.clone(),
        values: ndarray::concatenate(Axis(1), &[acf.values.view(), acf.values.view()]).unwrap(),
    };
    for route in routes {
        assert!(route(state(&acf, 1000.0, 300.0)).is_ok());
        assert!(matches!(
            route(state(&acf, 0.0, 300.0)),
            Err(ComputeError::OutOfRange {
                field: "volume",
                ..
            })
        ));
        assert!(matches!(
            route(state(&acf, 1000.0, 0.0)),
            Err(ComputeError::OutOfRange {
                field: "temperature",
                ..
            })
        ));
        assert!(matches!(
            route(DielectricSpectrumInput {
                boundary: EwaldBoundary::Vacuum,
                ..state(&acf, 1000.0, 300.0)
            }),
            Err(ComputeError::OutOfRange {
                field: "boundary",
                ..
            })
        ));
        assert!(matches!(
            route(state(&two, 1000.0, 300.0)),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}

/// Spec §5: the auto-plateau (`|C∞/C0| > 0.1` ⇒ subtract the last sample)
/// is gone. Without a plateau nothing is subtracted, whatever the ratio;
/// with one, its windowed mean is.
#[test]
fn dacs_subtracts_only_an_explicit_plateau() {
    // A PACF that plateaus at 0.5 (ratio 0.5 > the old 0.1 trigger).
    let y: Vec<F> = (0..32)
        .map(|k| 0.5 + 0.5 * (-(k as F) / 3.0).exp())
        .collect();
    let acf = curve(&y, 0.1);
    let units = UnitPreset::real();
    let a = DielectricPrefactor::new(&units, 1000.0, 300.0)
        .unwrap()
        .fluctuation();

    let bare = DipoleAutocorrelationSpectrum::new(units.clone())
        .with_window(Window::None)
        .compute(state(&acf, 1000.0, 300.0))
        .unwrap();
    assert!((bare.eps_real[0] - (1.0 + a * y[0])).abs() < 1e-12);

    let plateau = Plateau::new(0.75, 1.0).unwrap();
    let c_inf = plateau.read(&y).unwrap().value;
    let relaxed = DipoleAutocorrelationSpectrum::new(units)
        .with_window(Window::None)
        .with_plateau(plateau)
        .compute(state(&acf, 1000.0, 300.0))
        .unwrap();
    assert!((relaxed.eps_real[0] - (1.0 + a * (y[0] - c_inf))).abs() < 1e-12);
}

/// The former forced subtraction, selectable: `C(∞)` is the curve's last
/// sample (`subtract_plateau = Some(true)` on `dev`), read as the end of a
/// plateau window ending at 1. Without a window, `End` is an error.
#[test]
fn dacs_end_value_is_the_last_sample() {
    let y: Vec<F> = (0..32)
        .map(|k| 0.5 + 0.5 * (-(k as F) / 3.0).exp())
        .collect();
    let acf = curve(&y, 0.1);
    let units = UnitPreset::real();
    let a = DielectricPrefactor::new(&units, 1000.0, 300.0)
        .unwrap()
        .fluctuation();
    let end = DipoleAutocorrelationSpectrum::new(units.clone())
        .with_window(Window::None)
        .with_plateau(Plateau::new(0.75, 1.0).unwrap())
        .with_plateau_value(PlateauValue::End)
        .compute(state(&acf, 1000.0, 300.0))
        .unwrap();
    assert!((end.eps_real[0] - (1.0 + a * (y[0] - y[31]))).abs() < 1e-12);

    assert!(matches!(
        DipoleAutocorrelationSpectrum::new(units)
            .with_plateau_value(PlateauValue::End)
            .compute(state(&acf, 1000.0, 300.0)),
        Err(ComputeError::OutOfRange {
            field: "plateau_value",
            ..
        })
    ));
}

/// ac-010: dipole series → `DipoleAcf` → `EinsteinHelfandSpectrum` →
/// `KramersKronig`, each output feeding the next input with no glue.
#[test]
fn dipole_acf_to_kramers_kronig_pipeline() {
    use crate::compute::{KramersKronig, KramersKronigInput, Verdict};
    let dm = rng_dipole(512, 3);
    let acf = dipole_acf(&dm, 1.0, 64);
    let spectrum = EinsteinHelfandSpectrum::new(UnitPreset::real())
        .with_window(Window::CosineSq)
        .compute(state(&acf, 1000.0, 300.0))
        .unwrap();
    let verdict = KramersKronig::new()
        .compute(KramersKronigInput {
            spectrum: &spectrum,
        })
        .unwrap();
    assert_eq!(verdict.recovered.len(), spectrum.frequencies.len());
    assert!(verdict.mae.is_finite());
    assert_eq!(verdict.passed(), verdict.passed);
}
