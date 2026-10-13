//! Displacement correlation: mean squared displacement and its cross form
//! (Onsager), from the same FFT kernel as the auto- and cross-correlation.

use ndarray::{Array1, Array2, ArrayView3};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::engine::{Plan, TimeCorrelationSpec, fill_column, mask_origins, plan, reduce_rows};
use super::result::{CorrelationKind, Normalization, TimeCorrelation};
use crate::compute::ComputeError;
use crate::signal::LinearCorrelator;

/// Displacement correlation
/// `D_ab(t) = norm · Σ_i Σ_c w_c Σ_τ (a_ic(τ+t) − a_ic(τ)) · (b_ic(τ+t) − b_ic(τ))`
/// of `(T, N, D)` series over time origins `τ = 0, s, 2s, …`; `b = None` is
/// the mean squared displacement of `a`.
///
/// The square is expanded, `D_ab(t) = S_ab(t) − r_ab(t) − r_ba(t)`, with
/// `S_ab(t) = Σ_τ (p(τ+t) + p(τ))`, `p = Σ_c w_c a_c b_c`, by the running
/// subtraction `S(t) = S(t−1) − p(t−1) − p(T−t)` and the cross terms from
/// [`LinearCorrelator`]: the nMoldyn route (Kneller et al., *Comput. Phys.
/// Commun.* **91** (1995) 191) that `Msd` has always taken. For the
/// self form `r_ab = r_ba` is the autocorrelation taken once with weight 2.
/// With a strided origin set the two cross terms become masked
/// cross-correlations and `S` a pair of residue-class partial sums.
///
/// `values[0]` is exactly `0`. `mean_subtract` has no meaning for a
/// displacement and is rejected.
///
/// # Errors
///
/// As [`autocorrelate`](super::autocorrelate), plus
/// [`ComputeError::BadShape`] for differently shaped `a` and `b` and
/// [`ComputeError::OutOfRange`] for `mean_subtract`.
pub fn displacement_correlate(
    a: ArrayView3<'_, f64>,
    b: Option<ArrayView3<'_, f64>>,
    dt: f64,
    spec: &TimeCorrelationSpec<'_>,
) -> Result<TimeCorrelation, ComputeError> {
    if spec.mean_subtract {
        return Err(ComputeError::OutOfRange {
            field: "mean_subtract",
            value: "a displacement correlation has no mean to subtract".to_string(),
        });
    }
    if let Some(b) = b
        && b.dim() != a.dim()
    {
        return Err(ComputeError::BadShape {
            expected: format!("{:?}", a.dim()),
            got: format!("{:?}", b.dim()),
        });
    }
    let Plan {
        n_frames,
        n_entities,
        n_components,
        max_lag,
        stride,
        weights,
        counts,
    } = plan(a.dim(), dt, spec)?;

    let mut planner = FftPlanner::new();
    let corr = LinearCorrelator::new(&mut planner, n_frames);
    let mut col_a = vec![0.0_f64; n_frames];
    let mut col_b = vec![0.0_f64; n_frames];
    let mut masked = vec![0.0_f64; n_frames];
    let mut p = vec![0.0_f64; n_frames];
    let mut cross = vec![0.0_f64; max_lag + 1];
    let mut scratch_a: Vec<Complex64> = Vec::new();
    let mut scratch_b: Vec<Complex64> = Vec::new();
    let signal_err = |e: crate::signal::SignalError| ComputeError::OutOfRange {
        field: "correlation kernel",
        value: e.to_string(),
    };

    let mut aggregate = Array1::<f64>::zeros(max_lag + 1);
    let mut rows = spec
        .per_entity
        .then(|| Array2::<f64>::zeros((n_entities, max_lag + 1)));
    let count0 = counts[0] as f64;

    for entity in 0..n_entities {
        // p(τ) = Σ_c w_c a_c(τ) b_c(τ), components in order.
        p.iter_mut().for_each(|v| *v = 0.0);
        cross.iter_mut().for_each(|v| *v = 0.0);
        for component in 0..n_components {
            let weight = weights.map_or(1.0, |w| w[component]);
            fill_column(a, entity, component, false, &mut col_a);
            match b {
                None => {
                    for (pv, &av) in p.iter_mut().zip(col_a.iter()) {
                        *pv += weight * (av * av);
                    }
                    if stride == 1 {
                        corr.acf_into(&col_a, max_lag, 2.0 * weight, &mut cross, &mut scratch_a)
                            .map_err(signal_err)?;
                    } else {
                        mask_origins(&col_a, stride, &mut masked);
                        corr.xcorr_into(
                            &masked,
                            &col_a,
                            max_lag,
                            2.0 * weight,
                            &mut cross,
                            &mut scratch_a,
                            &mut scratch_b,
                        )
                        .map_err(signal_err)?;
                    }
                }
                Some(b) => {
                    fill_column(b, entity, component, false, &mut col_b);
                    for ((pv, &av), &bv) in p.iter_mut().zip(col_a.iter()).zip(col_b.iter()) {
                        *pv += weight * (av * bv);
                    }
                    if stride == 1 {
                        corr.xcorr_sym_into(
                            &col_a,
                            &col_b,
                            max_lag,
                            weight,
                            &mut cross,
                            &mut scratch_a,
                            &mut scratch_b,
                        )
                        .map_err(signal_err)?;
                    } else {
                        mask_origins(&col_a, stride, &mut masked);
                        corr.xcorr_into(
                            &masked,
                            &col_b,
                            max_lag,
                            weight,
                            &mut cross,
                            &mut scratch_a,
                            &mut scratch_b,
                        )
                        .map_err(signal_err)?;
                        mask_origins(&col_b, stride, &mut masked);
                        corr.xcorr_into(
                            &masked,
                            &col_a,
                            max_lag,
                            weight,
                            &mut cross,
                            &mut scratch_a,
                            &mut scratch_b,
                        )
                        .map_err(signal_err)?;
                    }
                }
            }
        }

        // Entity curve: S(t) − cross(t), scaled.
        let own_row = rows.is_some();
        let mut row = own_row.then(|| Array1::<f64>::zeros(max_lag + 1));
        let out: &mut [f64] = match row.as_mut() {
            Some(r) => r.as_slice_mut().expect("zeros array is contiguous"),
            None => aggregate.as_slice_mut().expect("zeros array is contiguous"),
        };
        let scale = |raw: f64, lag: usize| -> f64 {
            match spec.normalization {
                Normalization::Unbiased => raw / counts[lag] as f64,
                Normalization::Biased => raw / count0,
                Normalization::Raw => raw,
            }
        };
        if stride == 1 {
            let mut s: f64 = 2.0 * p.iter().sum::<f64>();
            for lag in 0..=max_lag {
                let value = if lag == 0 {
                    0.0
                } else {
                    scale(s - cross[lag], lag)
                };
                if own_row {
                    out[lag] = value;
                } else {
                    out[lag] += value;
                }
                if lag + 1 < n_frames {
                    s -= p[lag];
                    s -= p[n_frames - 1 - lag];
                }
            }
        } else {
            // Σ_{τ ∈ origins, τ ≤ T−1−t} p(τ): prefix sums over residue class 0.
            let mut prefix0 = vec![0.0_f64; n_frames];
            let mut acc = 0.0;
            for (k, pre) in prefix0.iter_mut().enumerate() {
                if k % stride == 0 {
                    acc += p[k];
                }
                *pre = acc;
            }
            // Σ_{τ ∈ origins, τ+t ≤ T−1} p(τ+t): suffix sums along each residue class.
            let mut suffix = vec![0.0_f64; n_frames];
            for j in (0..n_frames).rev() {
                suffix[j] = p[j]
                    + if j + stride < n_frames {
                        suffix[j + stride]
                    } else {
                        0.0
                    };
            }
            for lag in 0..=max_lag {
                let value = if lag == 0 {
                    0.0
                } else {
                    let s = prefix0[n_frames - 1 - lag] + suffix[lag];
                    scale(s - cross[lag], lag)
                };
                if own_row {
                    out[lag] = value;
                } else {
                    out[lag] += value;
                }
            }
        }
        if let (Some(rows), Some(row)) = (rows.as_mut(), row) {
            rows.row_mut(entity).assign(&row);
        }
    }

    let values = match rows.as_ref() {
        Some(rows) => reduce_rows(rows, spec.normalization),
        None => {
            if spec.normalization != Normalization::Raw {
                aggregate.mapv_inplace(|v| v / n_entities as f64);
            }
            aggregate
        }
    };

    Ok(TimeCorrelation {
        lags: Array1::from_iter(0..=max_lag),
        dt,
        values,
        counts,
        n_entities,
        normalization: spec.normalization,
        kind: if b.is_some() {
            CorrelationKind::CrossDisplacement
        } else {
            CorrelationKind::Displacement
        },
        per_entity: rows,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::super::engine::tests::series;
    use super::*;
    use ndarray::{Array3, s};

    /// The definition, written out.
    fn direct(
        a: &Array3<f64>,
        b: &Array3<f64>,
        max_lag: usize,
        stride: usize,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        let (n_frames, n_entities, n_components) = a.dim();
        (0..=max_lag)
            .map(|lag| {
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut tau = 0;
                while tau + lag < n_frames {
                    for e in 0..n_entities {
                        for c in 0..n_components {
                            let w = weights.map_or(1.0, |w| w[c]);
                            let da = a[[tau + lag, e, c]] - a[[tau, e, c]];
                            let db = b[[tau + lag, e, c]] - b[[tau, e, c]];
                            sum += w * da * db;
                        }
                    }
                    count += 1;
                    tau += stride;
                }
                sum / (n_entities * count) as f64
            })
            .collect()
    }

    /// A random-walk-like series: cumulative sums of the test series.
    fn walk(n_frames: usize, n_entities: usize, n_components: usize) -> Array3<f64> {
        let mut w = series(n_frames, n_entities, n_components);
        for t in 1..n_frames {
            for e in 0..n_entities {
                for c in 0..n_components {
                    w[[t, e, c]] += w[[t - 1, e, c]];
                }
            }
        }
        w
    }

    #[test]
    fn self_displacement_matches_the_direct_definition() {
        let a = walk(50, 4, 3);
        let got =
            displacement_correlate(a.view(), None, 1.0, &TimeCorrelationSpec::new().max_lag(20))
                .unwrap();
        let want = direct(&a, &a, 20, 1, None);
        assert_eq!(got.values[0], 0.0);
        for (k, w) in want.iter().enumerate().skip(1) {
            assert!(
                (got.values[k] - w).abs() < 1e-9 * w.abs().max(1.0),
                "k={k}: {} vs {}",
                got.values[k],
                w
            );
        }
        assert_eq!(got.kind, CorrelationKind::Displacement);
        assert_eq!(got.counts[20], 30);
    }

    #[test]
    fn cross_displacement_matches_the_direct_definition_and_self_is_the_msd() {
        let a = walk(50, 3, 3);
        let b = walk(50, 3, 3).mapv(|v| 0.7 * v.sin() + 0.2 * v);
        let spec = TimeCorrelationSpec::new().max_lag(16);
        let got = displacement_correlate(a.view(), Some(b.view()), 1.0, &spec).unwrap();
        let want = direct(&a, &b, 16, 1, None);
        for (k, w) in want.iter().enumerate() {
            assert!((got.values[k] - w).abs() < 1e-9 * w.abs().max(1.0), "k={k}");
        }
        assert_eq!(got.kind, CorrelationKind::CrossDisplacement);
        let msd = displacement_correlate(a.view(), None, 1.0, &spec).unwrap();
        let self_x = displacement_correlate(a.view(), Some(a.view()), 1.0, &spec).unwrap();
        assert_eq!(msd.values, self_x.values);
    }

    #[test]
    fn origin_stride_matches_the_strided_direct_loop() {
        let a = walk(60, 3, 3);
        let b = walk(60, 3, 3).mapv(|v| 0.5 * v * v.cos() - 0.3);
        for stride in [2usize, 3, 7] {
            let spec = TimeCorrelationSpec::new().max_lag(17).origin_stride(stride);
            let msd = displacement_correlate(a.view(), None, 1.0, &spec).unwrap();
            let want = direct(&a, &a, 17, stride, None);
            for (k, w) in want.iter().enumerate() {
                assert!(
                    (msd.values[k] - w).abs() < 1e-9 * w.abs().max(1.0),
                    "stride {stride} k={k}: {} vs {}",
                    msd.values[k],
                    w
                );
            }
            let cross = displacement_correlate(a.view(), Some(b.view()), 1.0, &spec).unwrap();
            let want = direct(&a, &b, 17, stride, None);
            for (k, w) in want.iter().enumerate() {
                assert!(
                    (cross.values[k] - w).abs() < 1e-9 * w.abs().max(1.0),
                    "stride {stride} k={k}: {} vs {}",
                    cross.values[k],
                    w
                );
            }
        }
    }

    #[test]
    fn normalizations_are_related_by_the_counts() {
        let a = walk(40, 2, 3);
        let spec = TimeCorrelationSpec::new().max_lag(10);
        let unb = displacement_correlate(a.view(), None, 1.0, &spec).unwrap();
        let bia = displacement_correlate(
            a.view(),
            None,
            1.0,
            &spec.normalization(Normalization::Biased),
        )
        .unwrap();
        let raw =
            displacement_correlate(a.view(), None, 1.0, &spec.normalization(Normalization::Raw))
                .unwrap();
        for k in 1..=10 {
            let ratio = unb.counts[k] as f64 / unb.counts[0] as f64;
            assert!(
                (bia.values[k] - unb.values[k] * ratio).abs() < 1e-9,
                "k={k}"
            );
            let rescaled = raw.values[k] / (2 * unb.counts[k]) as f64;
            assert!((rescaled - unb.values[k]).abs() < 1e-9, "k={k}");
        }
    }

    #[test]
    fn weights_and_per_entity_rows() {
        let a = walk(40, 4, 3);
        let w = [1.0, 0.5, 2.0];
        let spec = TimeCorrelationSpec::new().max_lag(9).component_weights(&w);
        let got = displacement_correlate(a.view(), None, 1.0, &spec).unwrap();
        let want = direct(&a, &a, 9, 1, Some(&w));
        for k in 0..=9 {
            assert!(
                (got.values[k] - want[k]).abs() < 1e-9 * want[k].abs().max(1.0),
                "k={k}"
            );
        }
        let per = displacement_correlate(a.view(), None, 1.0, &spec.per_entity(true)).unwrap();
        let rows = per.per_entity.as_ref().unwrap();
        assert_eq!(rows.dim(), (4, 10));
        for k in 0..=9 {
            assert_eq!(got.values[k], per.values[k], "k={k}");
        }
        for e in 0..4 {
            let one = a.slice(s![.., e..e + 1, ..]).to_owned();
            let alone = displacement_correlate(one.view(), None, 1.0, &spec).unwrap();
            for k in 0..=9 {
                assert_eq!(rows[[e, k]], alone.values[k], "e={e} k={k}");
            }
        }
    }

    #[test]
    fn linear_motion_gives_a_quadratic_msd_and_mean_subtract_is_rejected() {
        let v = [1.0, -2.0, 0.5];
        let a = Array3::from_shape_fn((30, 2, 3), |(t, _, c)| t as f64 * v[c]);
        let got =
            displacement_correlate(a.view(), None, 0.5, &TimeCorrelationSpec::new().max_lag(12))
                .unwrap();
        let v2: f64 = v.iter().map(|x| x * x).sum();
        for k in 0..=12 {
            assert!((got.values[k] - v2 * (k * k) as f64).abs() < 1e-9, "k={k}");
        }
        assert!((got.lag_times()[12] - 6.0).abs() < 1e-15);
        assert!(
            displacement_correlate(
                a.view(),
                None,
                1.0,
                &TimeCorrelationSpec::new().mean_subtract(true)
            )
            .is_err()
        );
        let other = Array3::zeros((30, 3, 3));
        assert!(
            displacement_correlate(
                a.view(),
                Some(other.view()),
                1.0,
                &TimeCorrelationSpec::new()
            )
            .is_err()
        );
    }
}
