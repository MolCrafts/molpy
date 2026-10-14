//! `Correlator` with the product kernel, as the former `Acf`,
//! `signal::acf_fft` and `DipoleRateCross`; streaming specifics; names.

use super::tests::{definition, input, series};
use super::*;
use crate::compute::time_series::test_support::{assert_close, direct_lagged_sum};
use crate::compute::{Accumulate, CorrelatorFrame, Derivative, DerivativeInput, Stencil, Stream};
use ndarray::{Array3, array};

// ── Lagged product, as the former `Acf` / `signal::acf_fft` ──────────────────

/// The raw (unnormalised) autocorrelation of one scalar series.
fn raw_acf(x: &[F], max_lag: usize) -> Result<Vec<F>, ComputeError> {
    let a = Array3::from_shape_vec((x.len(), 1, 1), x.to_vec()).unwrap();
    Ok(Correlator::new(LagKernel::Product, max_lag)
        .with_normalization(Normalization::None)
        .compute(input(&a, None))?
        .values
        .into_raw_vec_and_offset()
        .0)
}

#[test]
fn constant_signal_un_normalized() {
    let result = raw_acf(&[2.0, 2.0, 2.0, 2.0], 3).unwrap();
    assert_eq!(result.len(), 4);
    let expected_lag0 = 4.0 * 2.0_f64.powi(2); // N * c^2 = 4 * 4 = 16
    assert!((result[0] - expected_lag0).abs() < 1e-10);
}

#[test]
fn constant_signal_linear_decay() {
    // Linear ACF of constant c over n samples: r[t] = (n - t) · c²
    let result = raw_acf(&[3.0, 3.0, 3.0], 2).unwrap();
    assert_eq!(result.len(), 3);
    let c_sq = 3.0_f64.powi(2);
    let n = 3.0;
    for (t, v) in result.iter().enumerate() {
        let expected = (n - t as f64) * c_sq;
        assert!(
            (v - expected).abs() < 1e-10,
            "lag {t}: got {v}, expected {expected}"
        );
    }
}

/// A one-sample series has no time axis: refused rather than read as `x²`.
#[test]
fn single_sample_is_empty_input() {
    assert!(matches!(raw_acf(&[5.0], 0), Err(ComputeError::EmptyInput)));
}

#[test]
fn max_lag_zero_is_the_sum_of_squares() {
    let result = raw_acf(&[1.0, 2.0, 3.0], 0).unwrap();
    assert_eq!(result.len(), 1);
    let expected = 1.0_f64.powi(2) + 2.0_f64.powi(2) + 3.0_f64.powi(2);
    assert!((result[0] - expected).abs() < 1e-10);
}

#[test]
fn max_lag_too_large_is_out_of_range() {
    assert!(matches!(
        raw_acf(&[1.0, 2.0], 2),
        Err(ComputeError::OutOfRange {
            field: "max_lag",
            ..
        })
    ));
}

#[test]
fn white_noise_peaks_at_zero() {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(1000);
    let data: Vec<F> = (0..1000).map(|_| rng.random()).collect();
    let result = raw_acf(&data, 10).unwrap();
    assert_eq!(result.len(), 11);
    for k in 1..result.len() {
        assert!(result[k].abs() < result[0]);
    }
}

#[test]
fn sine_wave_is_anti_correlated_at_half_period() {
    let data: Vec<F> = (0..128)
        .map(|i| (2.0 * std::f64::consts::PI * i as F / 16.0).sin())
        .collect();
    let result = raw_acf(&data, 32).unwrap();
    assert_eq!(result.len(), 33);
    assert!(result[0] > 0.0);
    assert!(result[8] < 0.0);
}

#[test]
fn cross_of_self_equals_auto_bit_for_bit() {
    // conj(A)·A and |A|² agree to the last ULP.
    let a =
        Array3::from_shape_vec((8, 1, 1), vec![1.0, -2.0, 3.0, 0.5, -1.5, 4.0, 2.0, -3.0]).unwrap();
    for mean in MeanPolicy::ALL.iter().copied() {
        let c = Correlator::new(LagKernel::Product, 5).with_mean(mean);
        assert_eq!(
            c.compute(input(&a, None)).unwrap(),
            c.compute(input(&a, Some(&a))).unwrap()
        );
    }
}

#[test]
fn cross_matches_direct_cross_sum() {
    // r_ab[t] = Σ_τ a[τ]·b[τ+t]
    let a = [1.0, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0, 0.0, -1.0, 3.0, 1.0];
    let to3 = |x: &[F]| Array3::from_shape_vec((5, 1, 1), x.to_vec()).unwrap();
    let (a3, b3) = (to3(&a), to3(&b));
    let got = Correlator::new(LagKernel::Product, 3)
        .with_normalization(Normalization::None)
        .compute(input(&a3, Some(&b3)))
        .unwrap();
    assert_close(
        got.values.as_slice().unwrap(),
        &direct_lagged_sum(&a, &b, 3),
        1e-12,
        "xcorr",
    );
}

#[test]
fn per_component_channels_sum_to_the_cartesian_curve() {
    let a = series(4, 1, 2, 0.0);
    let c = Correlator::new(LagKernel::Product, 2).with_normalization(Normalization::None);
    let total = c.compute(input(&a, None)).unwrap();
    let parts = c
        .with_reduction(Reduction::PerComponent)
        .compute(input(&a, None))
        .unwrap();
    for k in 0..=2 {
        assert!((parts.values[[k, 0]] + parts.values[[k, 1]] - total.values[[k, 0]]).abs() < 1e-14);
    }
}

#[test]
fn lag_zero_is_the_mean_square() {
    let a = series(32, 4, 3, 0.0);
    let got = Correlator::new(LagKernel::Product, 5)
        .compute(input(&a, None))
        .unwrap();
    let expected: F = a.iter().map(|v| v * v).sum::<F>() / (4 * 32) as F;
    assert!((got.values[[0, 0]] - expected).abs() < 1e-12);
}

#[test]
fn is_unbiased_not_biased() {
    // The two normalisations differ by (T−t)/T; pin which one this is.
    let a = series(40, 3, 2, 0.0);
    let c = Correlator::new(LagKernel::Product, 10);
    let got = c.compute(input(&a, None)).unwrap();
    let want = definition(&c, &a, None);
    let biased_at_10 = want[[10, 0]] * (40.0 - 10.0) / 40.0;
    assert!((got.values[[10, 0]] - want[[10, 0]]).abs() < 1e-10);
    assert!(
        (got.values[[10, 0]] - biased_at_10).abs() > 1e-12,
        "unbiased and biased coincide; the test cannot tell them apart"
    );
}

#[test]
fn a_constant_series_autocorrelates_flat() {
    let a = Array3::from_elem((16, 3, 2), 2.0);
    let got = Correlator::new(LagKernel::Product, 6)
        .compute(input(&a, None))
        .unwrap();
    // every lag: (1/(N(T-t))) * N(T-t) * D * 4 = D * 4
    for value in got.values.iter() {
        assert!((value - 8.0).abs() < 1e-10, "{value}");
    }
}

// ── The dipole-rate × dipole cross term (the former `DipoleRateCross`) ──────

/// `C_ṀM(t) = Σ_α ⟨δṀ_α(0) δM_α(t)⟩` is a composition: the NumPy-gradient
/// derivative of the dipole series, then a mean-subtracted cross-correlation.
#[test]
fn dipole_rate_cross_is_a_derivative_then_a_cross_correlation() {
    let (n, dt, max_lag) = (96, 0.02, 30);
    let dm = crate::compute::time_series::test_support::rng_series(n, 3, 7);
    let mdot = Derivative::new(Stencil::Gradient)
        .compute(DerivativeInput {
            series: dm.view(),
            dt,
        })
        .unwrap();
    let (mdot3, m3) = (mdot.insert_axis(Axis(1)), dm.clone().insert_axis(Axis(1)));
    let c = Correlator::new(LagKernel::Product, max_lag).with_mean(MeanPolicy::Subtract);
    let cross = c
        .compute(CorrelatorInput {
            a: mdot3.view(),
            b: Some(m3.view()),
            dt,
            boxes: None,
        })
        .unwrap();
    assert_eq!(cross.n_lags(), max_lag + 1);
    assert!(cross.values.iter().all(|v| v.is_finite()));
    assert!((cross.lag_times[1] - dt).abs() < 1e-15);
    assert_close(
        cross.values.as_slice().unwrap(),
        definition(&c, &mdot3, Some(&m3)).as_slice().unwrap(),
        1e-12,
        "C_ṀM",
    );
}

// ── Streaming specifics ──────────────────────────────────────────────────────

#[test]
fn streaming_refuses_inconsistent_frames() {
    let c = Correlator::new(LagKernel::Product, 1);
    let one = array![[1.0, 2.0]];
    let frame = |a, dt| CorrelatorFrame {
        a,
        b: None,
        dt,
        simbox: None,
    };
    let mut acc = c.accumulator();
    assert!(matches!(acc.finish(), Err(ComputeError::EmptyInput)));
    acc.push(frame(one.view(), 1.0)).unwrap();
    assert!(matches!(
        acc.push(frame(one.view(), 2.0)),
        Err(ComputeError::OutOfRange { field: "dt", .. })
    ));
    assert!(matches!(
        acc.push(CorrelatorFrame {
            b: Some(one.view()),
            ..frame(one.view(), 1.0)
        }),
        Err(ComputeError::BadShape { .. })
    ));
    let nan = array![[F::NAN, 2.0]];
    assert!(matches!(
        acc.push(frame(nan.view(), 1.0)),
        Err(ComputeError::NonFinite {
            where_: "a",
            index: 0
        })
    ));
    // A refused frame is not folded: the accumulator still holds one frame.
    assert_eq!(acc.n_frames(), 1);
}

#[test]
fn finish_peeks_and_the_run_continues() {
    let a = series(10, 2, 3, 0.0);
    let c = Correlator::new(LagKernel::Product, 3).with_mean(MeanPolicy::Subtract);
    let mut acc = c.accumulator();
    for (t, frame) in a.outer_iter().enumerate() {
        acc.push(CorrelatorFrame {
            a: frame,
            b: None,
            dt: 0.5,
            simbox: None,
        })
        .unwrap();
        if t == 5 {
            let head = a.slice(ndarray::s![..6, .., ..]).to_owned();
            let peek = acc.finish().unwrap();
            assert_close(
                peek.values.as_slice().unwrap(),
                c.compute(input(&head, None))
                    .unwrap()
                    .values
                    .as_slice()
                    .unwrap(),
                1e-12,
                "peek",
            );
        }
    }
    assert_close(
        acc.finish().unwrap().values.as_slice().unwrap(),
        c.compute(input(&a, None))
            .unwrap()
            .values
            .as_slice()
            .unwrap(),
        1e-12,
        "full",
    );
}

#[test]
fn kernel_names_are_canonical() {
    let kernels = [
        LagKernel::Product,
        LagKernel::SquaredDisplacement,
        LagKernel::Legendre { l: 2 },
        LagKernel::WignerD { l: 4 },
        LagKernel::Persistence,
    ];
    let names: Vec<&str> = kernels.iter().map(|k| k.name()).collect();
    assert_eq!(names, LagKernel::NAMES);
    assert_eq!(LagKernel::Legendre { l: 2 }.to_string(), "legendre(l=2)");
    assert_eq!(LagKernel::Product.to_string(), "product");
    assert_eq!(LagKernel::WignerD { l: 4 }.order(), Some(4));
    assert_eq!(LagKernel::Persistence.order(), None);
    assert_eq!("per_item".parse::<Reduction>().unwrap(), Reduction::PerItem);
    assert_eq!("first".parse::<Origins>().unwrap(), Origins::First);
    assert_eq!(
        "subtract".parse::<MeanPolicy>().unwrap(),
        MeanPolicy::Subtract
    );
    assert_eq!(
        "none".parse::<Normalization>().unwrap(),
        Normalization::None
    );
}
