//! `Correlator`: batch vs the definition for every kernel, streaming vs
//! batch, each option by hand, and every refusal.

use super::*;
use crate::compute::time_series::test_support::assert_close;
use crate::compute::{Accumulate, CorrelatorFrame, Stream};
use crate::core::Cell;
use ndarray::{Array2, Array3};

// ── The definition, written out ──────────────────────────────────────────────

/// Every configuration of the correlator, summed straight from its
/// definition: `O(T · L · N · D)`, no FFT, no recurrence.
pub(super) fn definition(c: &Correlator, a: &Array3<F>, b: Option<&Array3<F>>) -> Array2<F> {
    let (n_frames, n_items, n_components) = a.dim();
    let center = |x: &Array3<F>| match c.mean() {
        MeanPolicy::Keep => x.clone(),
        MeanPolicy::Subtract => x - &x.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)),
    };
    let x = center(a);
    let y = b.map(center).unwrap_or_else(|| x.clone());
    let n_channels = match c.reduction() {
        Reduction::Mean => 1,
        Reduction::PerItem => n_items,
        Reduction::PerComponent => n_components,
    };
    let channel = |i: usize, d: usize| match c.reduction() {
        Reduction::Mean => 0,
        Reduction::PerItem => i,
        Reduction::PerComponent => d,
    };
    let mut out = Array2::<F>::zeros((c.max_lag() + 1, n_channels));
    let origins_at = |lag: usize| -> Vec<usize> {
        match c.origins() {
            Origins::All => (0..n_frames - lag).step_by(c.stride()).collect(),
            Origins::First => vec![0],
        }
    };
    let count_at_zero = origins_at(0).len() as F;
    for lag in 0..=c.max_lag() {
        let origins = origins_at(lag);
        let count = origins.len() as F;
        for tau in origins {
            for i in 0..n_items {
                let early = x.slice(ndarray::s![tau, i, ..]).to_vec();
                let late = x.slice(ndarray::s![tau + lag, i, ..]).to_vec();
                match c.kernel() {
                    LagKernel::Product => {
                        for d in 0..n_components {
                            out[[lag, channel(i, d)]] += x[[tau, i, d]] * y[[tau + lag, i, d]];
                        }
                    }
                    LagKernel::SquaredDisplacement => {
                        for d in 0..n_components {
                            out[[lag, channel(i, d)]] += (x[[tau + lag, i, d]] - x[[tau, i, d]])
                                * (y[[tau + lag, i, d]] - y[[tau, i, d]]);
                        }
                    }
                    LagKernel::Legendre { l } => {
                        let norm = |v: &[F]| v.iter().map(|x| x * x).sum::<F>().sqrt();
                        let cosine = early.iter().zip(&late).map(|(p, q)| p * q).sum::<F>()
                            / (norm(&early) * norm(&late));
                        out[[lag, channel(i, 0)]] += match l {
                            0 => 1.0,
                            1 => cosine,
                            2 => 0.5 * (3.0 * cosine * cosine - 1.0),
                            3 => 0.5 * (5.0 * cosine.powi(3) - 3.0 * cosine),
                            _ => unimplemented!("reference Legendre order {l}"),
                        };
                    }
                    LagKernel::WignerD { l } => {
                        // q_rel = q_late ⊗ conj(q_early), Hamilton product.
                        let (p, q) = (&late, [early[0], -early[1], -early[2], -early[3]]);
                        let w = p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3];
                        let x = p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2];
                        let yy = p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1];
                        let z = p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0];
                        let theta = 2.0
                            * (w.abs() / (w * w + x * x + yy * yy + z * z).sqrt())
                                .min(1.0)
                                .acos();
                        let lf = l as F;
                        out[[lag, channel(i, 0)]] += if theta < 1e-12 {
                            1.0
                        } else {
                            ((lf + 0.5) * theta).sin() / (0.5 * theta).sin() / (2.0 * lf + 1.0)
                        };
                    }
                    LagKernel::Persistence => {
                        if x[[tau, i, 0]] == 1.0 && (tau..=tau + lag).all(|s| y[[s, i, 0]] == 1.0) {
                            out[[lag, channel(i, 0)]] += 1.0;
                        }
                    }
                }
            }
        }
        let per_item = match c.reduction() {
            Reduction::PerItem => 1.0,
            _ => 1.0 / n_items as F,
        };
        let per_origin = match c.normalization() {
            Normalization::Unbiased => 1.0 / count,
            Normalization::Biased => 1.0 / count_at_zero,
            Normalization::None => 1.0,
        };
        out.row_mut(lag).mapv_inplace(|v| v * per_item * per_origin);
    }
    out
}

/// A deterministic, non-periodic, mean-nonzero series: a periodic or
/// zero-mean input would hide a wrap-around or a normalisation slip.
pub(super) fn series(n_frames: usize, n_items: usize, n_components: usize, phase: F) -> Array3<F> {
    Array3::from_shape_fn((n_frames, n_items, n_components), |(t, e, c)| {
        ((t * 7 + e * 13 + c * 3) as F * 0.37 + phase).sin() + 0.25 * (e as F + 1.0)
    })
}

/// Presence of four bonds over 12 frames, written out.
fn presence() -> Array3<F> {
    let rows = [
        "111011101111",
        "000111111000",
        "101010101010",
        "111111111111",
    ];
    Array3::from_shape_fn((12, 4, 1), |(t, bond, _)| {
        if rows[bond].as_bytes()[t] == b'1' {
            1.0
        } else {
            0.0
        }
    })
}

/// The frames of [`presence`] that may start a run: a subset of its ones,
/// so the crossed persistence gates some origins away.
fn presence_starts() -> Array3<F> {
    let mut starts = presence();
    for (t, item) in [(1, 0), (5, 1), (6, 1), (8, 2), (0, 3), (4, 3), (9, 3)] {
        starts[[t, item, 0]] = 0.0;
    }
    starts
}

pub(super) fn input<'a>(a: &'a Array3<F>, b: Option<&'a Array3<F>>) -> CorrelatorInput<'a> {
    CorrelatorInput {
        a: a.view(),
        b: b.map(|b| b.view()),
        dt: 0.5,
        boxes: None,
    }
}

/// Every kernel with a fitting series and every applicable option, auto and
/// (for the linear kernels) cross.
fn configurations() -> Vec<(Correlator, Array3<F>, Option<Array3<F>>)> {
    let mut out = Vec::new();
    let max_lag = 9;
    // Every origin set: all (stride 1 and a stride that does not divide
    // the trajectory) and the first.
    let origin_sets = [(Origins::All, 1), (Origins::All, 3), (Origins::First, 1)];
    for (origins, stride) in origin_sets {
        let base = |kernel| {
            Correlator::new(kernel, max_lag)
                .with_origins(origins)
                .with_stride(stride)
                .unwrap()
        };
        for kernel in [LagKernel::Product, LagKernel::SquaredDisplacement] {
            for mean in MeanPolicy::ALL.iter().copied() {
                for norm in Normalization::ALL.iter().copied() {
                    for reduce in Reduction::ALL.iter().copied() {
                        let c = base(kernel)
                            .with_mean(mean)
                            .with_normalization(norm)
                            .with_reduction(reduce);
                        out.push((c, series(16, 3, 2, 0.0), None));
                        out.push((c, series(16, 3, 2, 0.0), Some(series(16, 3, 2, 1.1))));
                    }
                }
            }
        }
        for reduce in [Reduction::Mean, Reduction::PerItem] {
            for (kernel, a) in [
                (LagKernel::Legendre { l: 0 }, series(16, 3, 3, 0.6)),
                (LagKernel::Legendre { l: 1 }, series(16, 3, 3, 0.0)),
                (LagKernel::Legendre { l: 2 }, series(16, 3, 3, 0.4)),
                (LagKernel::Legendre { l: 2 }, series(16, 3, 2, 0.8)),
                (LagKernel::Legendre { l: 3 }, series(16, 3, 3, 0.9)),
                (LagKernel::WignerD { l: 2 }, series(16, 3, 4, 0.2)),
                (LagKernel::WignerD { l: 4 }, series(16, 3, 4, 0.7)),
                (LagKernel::Persistence, presence()),
            ] {
                for norm in Normalization::ALL.iter().copied() {
                    let c = base(kernel).with_normalization(norm).with_reduction(reduce);
                    out.push((c, a.clone(), None));
                }
            }
            let crossed = base(LagKernel::Persistence).with_reduction(reduce);
            out.push((crossed, presence_starts(), Some(presence())));
        }
    }
    out
}

#[test]
fn batch_matches_the_definition_for_every_kernel_and_option() {
    for (c, a, b) in configurations() {
        let got = c.compute(input(&a, b.as_ref())).unwrap();
        let want = definition(&c, &a, b.as_ref());
        assert_eq!(got.values.dim(), want.dim(), "{c:?}");
        assert_close(
            got.values.as_slice().unwrap(),
            want.as_slice().unwrap(),
            1e-12,
            &format!("{c:?} cross={}", b.is_some()),
        );
        assert_eq!(
            got.lag_times.to_vec(),
            (0..=c.max_lag()).map(|k| k as F * 0.5).collect::<Vec<_>>()
        );
    }
}

#[test]
fn streaming_equals_batch_at_every_lag() {
    for (c, a, b) in configurations() {
        let batch = c.compute(input(&a, b.as_ref())).unwrap();
        let mut acc = c.accumulator();
        for (t, frame) in a.outer_iter().enumerate() {
            acc.push(CorrelatorFrame {
                a: frame,
                b: b.as_ref().map(|b| b.index_axis(Axis(0), t)),
                dt: 0.5,
                simbox: None,
            })
            .unwrap();
        }
        let streamed = acc.finish().unwrap();
        assert_eq!(streamed.lag_times, batch.lag_times, "{c:?}");
        assert_close(
            streamed.values.as_slice().unwrap(),
            batch.values.as_slice().unwrap(),
            1e-12,
            &format!("{c:?} cross={}", b.is_some()),
        );
    }
}

/// More items than one FFT work unit: the chunked parallel reduction still
/// matches the definition.
#[test]
fn many_items_reduce_across_work_units() {
    let a = series(24, 3 * ITEMS_PER_CHUNK + 5, 2, 0.3);
    for kernel in [LagKernel::Product, LagKernel::SquaredDisplacement] {
        for reduce in Reduction::ALL.iter().copied() {
            let c = Correlator::new(kernel, 11).with_reduction(reduce);
            let got = c.compute(input(&a, None)).unwrap();
            assert_close(
                got.values.as_slice().unwrap(),
                definition(&c, &a, None).as_slice().unwrap(),
                1e-12,
                &format!("{c:?}"),
            );
        }
    }
}

// ── Each option on a 3-sample series, by hand ────────────────────────────────

/// One item, one component: x = [1, 2, 4].
fn x124() -> Array3<F> {
    Array3::from_shape_vec((3, 1, 1), vec![1.0, 2.0, 4.0]).unwrap()
}

fn curve(c: Correlator, a: &Array3<F>) -> Vec<F> {
    c.compute(input(a, None))
        .unwrap()
        .values
        .into_raw_vec_and_offset()
        .0
}

fn close(got: Vec<F>, want: &[F]) {
    assert_close(&got, want, 1e-14, "hand-computed");
}

#[test]
fn origins_by_hand() {
    let product = Correlator::new(LagKernel::Product, 2);
    // All: (1+4+16)/3, (1·2+2·4)/2, 1·4.
    close(curve(product, &x124()), &[7.0, 5.0, 4.0]);
    // First: 1·1, 1·2, 1·4.
    close(
        curve(product.with_origins(Origins::First), &x124()),
        &[1.0, 2.0, 4.0],
    );
    // Displacement, all: 0, (1² + 2²)/2, 3².
    close(
        curve(Correlator::new(LagKernel::SquaredDisplacement, 2), &x124()),
        &[0.0, 2.5, 9.0],
    );
    // Displacement, first: 0, 1², 3².
    close(
        curve(
            Correlator::new(LagKernel::SquaredDisplacement, 2).with_origins(Origins::First),
            &x124(),
        ),
        &[0.0, 1.0, 9.0],
    );
}

#[test]
fn mean_policy_by_hand() {
    // mean 7/3, δ = [−4/3, −1/3, 5/3]:
    // C0 = (16 + 1 + 25)/9/3, C1 = (4/9 − 5/9)/2, C2 = −20/9.
    let c = Correlator::new(LagKernel::Product, 2).with_mean(MeanPolicy::Subtract);
    close(curve(c, &x124()), &[14.0 / 9.0, -1.0 / 18.0, -20.0 / 9.0]);
}

#[test]
fn normalization_by_hand() {
    let c = Correlator::new(LagKernel::Product, 2).with_normalization(Normalization::None);
    close(curve(c, &x124()), &[21.0, 10.0, 4.0]);
}

#[test]
fn reduction_by_hand() {
    // item 0: [1, 2, 4] and [0, 1, 0]; item 1: [2, 0, 1] and [1, 1, 1].
    let a = Array3::from_shape_vec(
        (3, 2, 2),
        vec![1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 0.0, 1.0, 4.0, 0.0, 1.0, 1.0],
    )
    .unwrap();
    let c = Correlator::new(LagKernel::Product, 1);
    // lag 0 per (item, component): 7, 1/3 | 5/3, 1.  lag 1: 5, 0 | 0, 1.
    close(curve(c, &a), &[5.0, 3.0]);
    close(
        curve(c.with_reduction(Reduction::PerItem), &a),
        &[22.0 / 3.0, 8.0 / 3.0, 5.0, 1.0],
    );
    close(
        curve(c.with_reduction(Reduction::PerComponent), &a),
        &[13.0 / 3.0, 2.0 / 3.0, 2.5, 0.5],
    );
}

#[test]
fn stride_and_biased_normalization_by_hand() {
    // x = [1, 2, 4, 3]; stride 2 keeps origins 0 and 2.
    let x = Array3::from_shape_vec((4, 1, 1), vec![1.0, 2.0, 4.0, 3.0]).unwrap();
    let c = Correlator::new(LagKernel::Product, 2)
        .with_stride(2)
        .unwrap();
    // lag 0: (1 + 16)/2; lag 1: (1·2 + 4·3)/2; lag 2: 1·4 / 1.
    close(curve(c, &x), &[8.5, 7.0, 4.0]);
    // Biased: every lag over n(0) = 2 origins.
    close(
        curve(c.with_normalization(Normalization::Biased), &x),
        &[8.5, 7.0, 2.0],
    );
    // Unstrided biased: (1+4+16+9)/4, (2+8+12)/4, (4+6)/4.
    close(
        curve(
            Correlator::new(LagKernel::Product, 2).with_normalization(Normalization::Biased),
            &x,
        ),
        &[7.5, 5.5, 2.5],
    );
    // Squared displacement over origins 0, 2: lag 1 (1 + 1)/2, lag 2 9/1.
    let sd = Correlator::new(LagKernel::SquaredDisplacement, 2)
        .with_stride(2)
        .unwrap();
    close(curve(sd, &x), &[0.0, 1.0, 9.0]);
}

/// `P_2` of quarter turns over every origin is `1, −½, 1, −½`, on the FFT
/// path, and the per-item curves of two counter-rotating vectors agree.
#[test]
fn legendre_fft_route_by_hand() {
    let turns = Array3::from_shape_fn((8, 2, 3), |(t, i, d)| {
        let angle = t as F * std::f64::consts::FRAC_PI_2 * if i == 0 { 1.0 } else { -1.0 };
        [angle.cos(), angle.sin(), 0.0][d]
    });
    for reduce in [Reduction::Mean, Reduction::PerItem] {
        let c = Correlator::new(LagKernel::Legendre { l: 2 }, 3).with_reduction(reduce);
        let got = c.compute(input(&turns, None)).unwrap();
        for channel in 0..got.n_channels() {
            close(
                got.channel(channel).unwrap().to_vec(),
                &[1.0, -0.5, 1.0, -0.5],
            );
        }
    }
}

// ── Refusals ─────────────────────────────────────────────────────────────────

fn refusal(c: Correlator, input: CorrelatorInput<'_>) -> ComputeError {
    c.compute(input).unwrap_err()
}

#[test]
fn a_zero_stride_and_a_strided_first_origin_are_refused() {
    assert!(matches!(
        Correlator::new(LagKernel::Product, 2).with_stride(0),
        Err(ComputeError::OutOfRange {
            field: "stride",
            ..
        })
    ));
    let a = series(6, 1, 2, 0.0);
    let first = Correlator::new(LagKernel::Product, 2)
        .with_origins(Origins::First)
        .with_stride(2)
        .unwrap();
    assert!(matches!(
        refusal(first, input(&a, None)),
        ComputeError::OutOfRange { field, .. } if field.starts_with("stride")
    ));
    let mut acc = first.accumulator();
    assert!(matches!(
        acc.push(CorrelatorFrame {
            a: a.index_axis(Axis(0), 0),
            b: None,
            dt: 0.5,
            simbox: None,
        }),
        Err(ComputeError::OutOfRange { field, .. }) if field.starts_with("stride")
    ));
}

#[test]
fn a_lag_without_origins_is_out_of_range() {
    let a = series(8, 2, 3, 0.0);
    let err = refusal(Correlator::new(LagKernel::Product, 8), input(&a, None));
    assert!(matches!(
        err,
        ComputeError::OutOfRange {
            field: "max_lag",
            ..
        }
    ));
    assert_eq!(
        Correlator::new(LagKernel::Product, 7)
            .compute(input(&a, None))
            .unwrap()
            .n_lags(),
        8
    );
}

#[test]
fn too_few_frames_or_nothing_to_correlate_is_empty_input() {
    let product = Correlator::new(LagKernel::Product, 0);
    for shape in [(1, 2, 3), (0, 2, 3), (4, 0, 3), (4, 2, 0)] {
        let a = Array3::<F>::ones(shape);
        assert!(
            matches!(refusal(product, input(&a, None)), ComputeError::EmptyInput),
            "{shape:?}"
        );
    }
}

#[test]
fn non_finite_values_and_bad_dt_are_refused() {
    let mut a = series(6, 1, 2, 0.0);
    a[[3, 0, 1]] = F::NAN;
    let err = refusal(Correlator::new(LagKernel::Product, 2), input(&a, None));
    assert!(matches!(
        err,
        ComputeError::NonFinite {
            where_: "a",
            index: 7
        }
    ));

    let a = series(6, 1, 2, 0.0);
    for dt in [0.0, -1.0, F::INFINITY, F::NAN] {
        let err = refusal(
            Correlator::new(LagKernel::Product, 2),
            CorrelatorInput {
                dt,
                ..input(&a, None)
            },
        );
        assert!(matches!(err, ComputeError::OutOfRange { field: "dt", .. }));
    }
}

#[test]
fn shapes_must_agree() {
    let a = series(6, 2, 3, 0.0);
    let b = series(6, 3, 3, 0.0);
    assert!(matches!(
        refusal(Correlator::new(LagKernel::Product, 2), input(&a, Some(&b))),
        ComputeError::BadShape { .. }
    ));
    assert!(matches!(
        refusal(
            Correlator::new(LagKernel::Product, 2).with_dimension(2),
            input(&a, None)
        ),
        ComputeError::DimensionMismatch {
            expected: 2,
            got: 3,
            ..
        }
    ));
    assert!(matches!(
        refusal(
            Correlator::new(LagKernel::WignerD { l: 2 }, 2),
            input(&a, None)
        ),
        ComputeError::DimensionMismatch {
            expected: 4,
            got: 3,
            ..
        }
    ));
}

#[test]
fn orientation_and_presence_kernels_refuse_what_they_cannot_mean() {
    let v = series(6, 2, 3, 0.0);
    let legendre = Correlator::new(LagKernel::Legendre { l: 2 }, 2);
    let field = |err| match err {
        ComputeError::OutOfRange { field, .. } => field,
        other => panic!("expected OutOfRange, got {other:?}"),
    };
    assert_eq!(field(refusal(legendre, input(&v, Some(&v)))), "b");
    assert_eq!(
        field(refusal(
            legendre.with_reduction(Reduction::PerComponent),
            input(&v, None)
        )),
        "reduce"
    );
    assert_eq!(
        field(refusal(
            legendre.with_mean(MeanPolicy::Subtract),
            input(&v, None)
        )),
        "mean"
    );
    let boxes = vec![SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap(); 6];
    assert_eq!(
        field(refusal(
            Correlator::new(LagKernel::Product, 2),
            CorrelatorInput {
                boxes: Some(&boxes),
                ..input(&v, None)
            }
        )),
        "boxes"
    );
    let not_presence = Array3::from_elem((4, 1, 1), 0.5);
    assert_eq!(
        field(refusal(
            Correlator::new(LagKernel::Persistence, 1),
            input(&not_presence, None)
        )),
        "a"
    );
    let present = Array3::from_elem((4, 1, 1), 1.0);
    assert_eq!(
        field(refusal(
            Correlator::new(LagKernel::Persistence, 1),
            input(&present, Some(&not_presence))
        )),
        "b"
    );
}

// ── Wrapped trajectories ─────────────────────────────────────────────────────

#[test]
fn a_wrapped_step_is_refused_and_an_unwrapped_one_is_not() {
    let cube = SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
    let boxes = vec![cube; 4];
    // Item 1 steps 4.9 Å (allowed, < 5 Å), then 5.1 Å (refused) at frame 3.
    let mut a = Array3::<F>::zeros((4, 2, 3));
    a[[1, 1, 0]] = 4.9;
    a[[2, 1, 0]] = 4.9;
    a[[3, 1, 0]] = 10.0;
    let sd = Correlator::new(LagKernel::SquaredDisplacement, 2);
    fn with_boxes<'a>(a: ArrayView3<'a, F>, boxes: &'a [SimBox]) -> CorrelatorInput<'a> {
        CorrelatorInput {
            a,
            b: None,
            dt: 1.0,
            boxes: Some(boxes),
        }
    }
    assert!(matches!(
        sd.compute(with_boxes(a.view(), &boxes)).unwrap_err(),
        ComputeError::Wrapped { frame: 3, item: 1 }
    ));
    a[[3, 1, 0]] = 9.0;
    assert!(sd.compute(with_boxes(a.view(), &boxes)).is_ok());

    // The cross partner is checked too.
    let mut b = a.clone();
    b[[2, 0, 2]] = -6.0;
    assert!(matches!(
        sd.compute(CorrelatorInput {
            b: Some(b.view()),
            ..with_boxes(a.view(), &boxes)
        })
        .unwrap_err(),
        ComputeError::Wrapped { frame: 2, item: 0 }
    ));

    // A box with no periodic axis bounds nothing.
    let free = vec![SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [false; 3]).unwrap(); 4];
    a[[3, 1, 0]] = 100.0;
    assert!(
        sd.compute(CorrelatorInput {
            boxes: Some(&free),
            ..with_boxes(a.view(), &boxes)
        })
        .is_ok()
    );

    // One box per frame.
    assert!(matches!(
        sd.compute(CorrelatorInput {
            boxes: Some(&boxes[..3]),
            ..with_boxes(a.view(), &boxes)
        })
        .unwrap_err(),
        ComputeError::DimensionMismatch {
            expected: 4,
            got: 3,
            ..
        }
    ));
}
