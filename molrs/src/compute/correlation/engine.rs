//! The correlation engine: auto- and cross-correlation of `(T, N, D)` series
//! over time origins, column by column through one
//! [`LinearCorrelator`](crate::signal::LinearCorrelator).

use ndarray::{Array1, Array2, ArrayView3, s};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

use super::result::{CorrelationKind, Normalization, TimeCorrelation, origin_count};
use crate::compute::ComputeError;
use crate::signal::LinearCorrelator;

/// Estimator settings of a time correlation.
///
/// `Default` is the setting every molrs correlator used before the engine:
/// all lags (`max_lag` clamped to `T − 1`), no mean subtraction, the
/// unbiased normalization, every time origin, unit component weights, no
/// per-entity rows.
#[derive(Debug, Clone, Copy)]
pub struct TimeCorrelationSpec<'w> {
    /// Longest lag in frames; clamped to `T − 1`.
    pub max_lag: usize,
    /// Subtract each `(entity, component)` column's mean over all `T`
    /// frames before correlating.
    pub mean_subtract: bool,
    /// How the lag sums are scaled.
    pub normalization: Normalization,
    /// Time origins every `origin_stride` frames, starting at frame 0.
    pub origin_stride: usize,
    /// Per-component weights `w_c` in `Σ_c w_c a_c(τ)·b_c(τ+t)`; `None` is
    /// all ones. Must have length `D`.
    pub component_weights: Option<&'w [f64]>,
    /// Also return one curve per entity.
    pub per_entity: bool,
}

impl Default for TimeCorrelationSpec<'_> {
    fn default() -> Self {
        Self {
            max_lag: usize::MAX,
            mean_subtract: false,
            normalization: Normalization::Unbiased,
            origin_stride: 1,
            component_weights: None,
            per_entity: false,
        }
    }
}

impl<'w> TimeCorrelationSpec<'w> {
    /// All lags, default settings.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    pub fn mean_subtract(mut self, yes: bool) -> Self {
        self.mean_subtract = yes;
        self
    }

    pub fn normalization(mut self, normalization: Normalization) -> Self {
        self.normalization = normalization;
        self
    }

    pub fn origin_stride(mut self, stride: usize) -> Self {
        self.origin_stride = stride;
        self
    }

    pub fn component_weights(mut self, weights: &'w [f64]) -> Self {
        self.component_weights = Some(weights);
        self
    }

    pub fn per_entity(mut self, yes: bool) -> Self {
        self.per_entity = yes;
        self
    }
}

/// Validated, shared view of one engine call.
pub(super) struct Plan<'a> {
    pub n_frames: usize,
    pub n_entities: usize,
    pub n_components: usize,
    pub max_lag: usize,
    pub stride: usize,
    pub weights: Option<&'a [f64]>,
    pub counts: Array1<usize>,
}

pub(super) fn plan<'a>(
    dims: (usize, usize, usize),
    dt: f64,
    spec: &TimeCorrelationSpec<'a>,
) -> Result<Plan<'a>, ComputeError> {
    let (n_frames, n_entities, n_components) = dims;
    if n_frames < 2 || n_entities == 0 || n_components == 0 {
        return Err(ComputeError::EmptyInput);
    }
    if dt <= 0.0 || !dt.is_finite() {
        return Err(ComputeError::OutOfRange {
            field: "dt",
            value: dt.to_string(),
        });
    }
    if spec.origin_stride == 0 {
        return Err(ComputeError::OutOfRange {
            field: "origin_stride",
            value: "0".to_string(),
        });
    }
    if let Some(w) = spec.component_weights
        && w.len() != n_components
    {
        return Err(ComputeError::DimensionMismatch {
            expected: n_components,
            got: w.len(),
            what: "component_weights",
        });
    }
    let max_lag = spec.max_lag.min(n_frames - 1);
    let stride = spec.origin_stride;
    let counts = Array1::from_iter((0..=max_lag).map(|t| origin_count(n_frames, t, stride)));
    Ok(Plan {
        n_frames,
        n_entities,
        n_components,
        max_lag,
        stride,
        weights: spec.component_weights,
        counts,
    })
}

/// Copy column `(entity, component)` of `series` into `out`, minus its mean
/// over all frames when asked (the mean is `Σ · (1/T)`, as every molrs
/// correlator has formed it).
pub(super) fn fill_column(
    series: ArrayView3<'_, f64>,
    entity: usize,
    component: usize,
    mean_subtract: bool,
    out: &mut [f64],
) {
    let lane = series.slice(s![.., entity, component]);
    let mean = if mean_subtract {
        lane.iter().sum::<f64>() * (1.0 / lane.len() as f64)
    } else {
        0.0
    };
    for (slot, value) in out.iter_mut().zip(lane.iter()) {
        *slot = value - mean;
    }
}

/// Zero every sample whose frame index is not a multiple of `stride`.
pub(super) fn mask_origins(column: &[f64], stride: usize, out: &mut [f64]) {
    for (t, (slot, &v)) in out.iter_mut().zip(column.iter()).enumerate() {
        *slot = if t % stride == 0 { v } else { 0.0 };
    }
}

/// Scale raw lag sums in place as `normalization` says.
pub(super) fn apply_normalization(
    values: &mut [f64],
    counts: &[usize],
    n_entities: usize,
    normalization: Normalization,
) {
    match normalization {
        Normalization::Raw => {}
        Normalization::Unbiased => {
            let scale = 1.0 / n_entities as f64;
            for (value, &count) in values.iter_mut().zip(counts) {
                *value *= scale / count as f64;
            }
        }
        Normalization::Biased => {
            let scale = 1.0 / n_entities as f64;
            let count0 = counts[0] as f64;
            for value in values.iter_mut() {
                *value *= scale / count0;
            }
        }
    }
}

/// Entity reduction of per-entity rows into `values`: the mean for the
/// averaged normalizations, the sum for `Raw`.
pub(super) fn reduce_rows(rows: &Array2<f64>, normalization: Normalization) -> Array1<f64> {
    let n_entities = rows.nrows() as f64;
    let mut values = Array1::<f64>::zeros(rows.ncols());
    for row in rows.rows() {
        for (v, r) in values.iter_mut().zip(row.iter()) {
            *v += r;
        }
    }
    if normalization != Normalization::Raw {
        values.mapv_inplace(|v| v / n_entities);
    }
    values
}

/// Correlate `a` with `b` (`b = None` is the autocorrelation) under `spec`.
fn correlate(
    a: ArrayView3<'_, f64>,
    b: Option<ArrayView3<'_, f64>>,
    dt: f64,
    spec: &TimeCorrelationSpec<'_>,
) -> Result<TimeCorrelation, ComputeError> {
    if let Some(b) = b
        && b.dim() != a.dim()
    {
        return Err(ComputeError::BadShape {
            expected: format!("{:?}", a.dim()),
            got: format!("{:?}", b.dim()),
        });
    }
    let plan = plan(a.dim(), dt, spec)?;
    let Plan {
        n_frames,
        n_entities,
        n_components,
        max_lag,
        stride,
        weights,
        counts,
    } = plan;

    let mut planner = FftPlanner::new();
    let corr = LinearCorrelator::new(&mut planner, n_frames);
    let mut col_a = vec![0.0_f64; n_frames];
    let mut col_b = vec![0.0_f64; n_frames];
    let mut masked = vec![0.0_f64; n_frames];
    let mut scratch_a: Vec<Complex64> = Vec::new();
    let mut scratch_b: Vec<Complex64> = Vec::new();
    let signal_err = |e: crate::signal::SignalError| ComputeError::OutOfRange {
        field: "correlation kernel",
        value: e.to_string(),
    };

    // One accumulator for the aggregate curve, or one row per entity.
    let mut aggregate = Array1::<f64>::zeros(max_lag + 1);
    let mut rows = spec
        .per_entity
        .then(|| Array2::<f64>::zeros((n_entities, max_lag + 1)));

    for entity in 0..n_entities {
        let mut row_buf = rows.is_some().then(|| Array1::<f64>::zeros(max_lag + 1));
        for component in 0..n_components {
            let weight = weights.map_or(1.0, |w| w[component]);
            fill_column(a, entity, component, spec.mean_subtract, &mut col_a);
            let out: &mut [f64] = match row_buf.as_mut() {
                Some(r) => r.as_slice_mut().expect("zeros array is contiguous"),
                None => aggregate.as_slice_mut().expect("zeros array is contiguous"),
            };
            match (b, stride) {
                (None, 1) => corr
                    .acf_into(&col_a, max_lag, weight, out, &mut scratch_a)
                    .map_err(signal_err)?,
                (None, _) => {
                    mask_origins(&col_a, stride, &mut masked);
                    corr.xcorr_into(
                        &masked,
                        &col_a,
                        max_lag,
                        weight,
                        out,
                        &mut scratch_a,
                        &mut scratch_b,
                    )
                    .map_err(signal_err)?
                }
                (Some(b), 1) => {
                    fill_column(b, entity, component, spec.mean_subtract, &mut col_b);
                    corr.xcorr_into(
                        &col_a,
                        &col_b,
                        max_lag,
                        weight,
                        out,
                        &mut scratch_a,
                        &mut scratch_b,
                    )
                    .map_err(signal_err)?
                }
                (Some(b), _) => {
                    fill_column(b, entity, component, spec.mean_subtract, &mut col_b);
                    mask_origins(&col_a, stride, &mut masked);
                    corr.xcorr_into(
                        &masked,
                        &col_b,
                        max_lag,
                        weight,
                        out,
                        &mut scratch_a,
                        &mut scratch_b,
                    )
                    .map_err(signal_err)?
                }
            }
        }
        if let (Some(rows), Some(row)) = (rows.as_mut(), row_buf) {
            rows.row_mut(entity).assign(&row);
        }
    }

    let counts_slice = counts.as_slice().expect("counts is contiguous");
    let values = match rows.as_mut() {
        Some(rows) => {
            // Each row is one entity: normalize it as a one-entity curve, then
            // reduce across entities.
            for mut row in rows.rows_mut() {
                let row = row
                    .as_slice_mut()
                    .expect("row of a zeros array is contiguous");
                apply_normalization(row, counts_slice, 1, spec.normalization);
            }
            reduce_rows(rows, spec.normalization)
        }
        None => {
            apply_normalization(
                aggregate.as_slice_mut().expect("zeros array is contiguous"),
                counts_slice,
                n_entities,
                spec.normalization,
            );
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
            CorrelationKind::Cross
        } else {
            CorrelationKind::Auto
        },
        per_entity: rows,
    })
}

/// Autocorrelation `C(t) = norm · Σ_i Σ_c w_c Σ_τ a_ic(τ) · a_ic(τ+t)` of an
/// `(n_frames T, n_entities N, n_components D)` series over time origins
/// `τ = 0, s, 2s, …` (`s` the origin stride).
///
/// Every `(entity, component)` column is correlated by the zero-padded
/// Wiener–Khinchin route of [`LinearCorrelator`] with one shared plan and
/// scratch buffer; the lag sums are then scaled as the
/// [`Normalization`] says, the unbiased default being the fused
/// `(1/N)/counts[t]` multiply. With `mean_subtract`, each column loses its
/// own time mean first (`δa = a − ⟨a⟩`).
///
/// This is the estimator of Allen & Tildesley, *Computer Simulation of
/// Liquids* §8.4, by the FFT route of Press et al., *Numerical Recipes*
/// §13.2 (zero-pad to `≥ 2T`, forward FFT, `|X|²`, inverse FFT), the same
/// route nMoldyn takes (Kneller et al., *Comput. Phys. Commun.* **91**
/// (1995) 191). A strided origin set is the masked cross-correlation
/// `Σ_τ m_s(τ) a(τ) · a(τ+t)` with `m_s` the stride mask, so it stays on the
/// FFT path.
///
/// A `(T, D)` Cartesian series is the `N = 1` case
/// (`series.insert_axis(Axis(1))`); a `(T, n_dof)` series averaged over
/// degrees of freedom is the `D = 1` case (`insert_axis(Axis(2))`). Both are
/// views, no copy.
///
/// # Errors
///
/// [`ComputeError::EmptyInput`] for fewer than two frames, no entity or no
/// component; [`ComputeError::OutOfRange`] for a non-positive `dt` or a zero
/// stride; [`ComputeError::DimensionMismatch`] for weights of the wrong
/// length.
pub fn autocorrelate(
    series: ArrayView3<'_, f64>,
    dt: f64,
    spec: &TimeCorrelationSpec<'_>,
) -> Result<TimeCorrelation, ComputeError> {
    correlate(series, None, dt, spec)
}

/// Cross-correlation `C_ab(t) = norm · Σ_i Σ_c w_c Σ_τ a_ic(τ) · b_ic(τ+t)`
/// of two equally shaped `(T, N, D)` series, with everything else as
/// [`autocorrelate`]. `cross_correlate(a, a, …)` reproduces
/// `autocorrelate(a, …)` bit for bit at stride 1.
///
/// # Errors
///
/// As [`autocorrelate`], plus [`ComputeError::BadShape`] when the shapes
/// differ.
pub fn cross_correlate(
    a: ArrayView3<'_, f64>,
    b: ArrayView3<'_, f64>,
    dt: f64,
    spec: &TimeCorrelationSpec<'_>,
) -> Result<TimeCorrelation, ComputeError> {
    correlate(a, Some(b), dt, spec)
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use ndarray::Array3;

    /// Deterministic, non-periodic, mean-nonzero series.
    pub(crate) fn series(n_frames: usize, n_entities: usize, n_components: usize) -> Array3<f64> {
        Array3::from_shape_fn((n_frames, n_entities, n_components), |(t, e, c)| {
            ((t * 7 + e * 13 + c * 3) as f64 * 0.37).sin() + 0.25 * (e as f64 + 1.0)
        })
    }

    /// The definition, written out: strided origins, optional mean
    /// subtraction, weights, unbiased.
    pub(crate) fn direct(
        a: &Array3<f64>,
        b: &Array3<f64>,
        max_lag: usize,
        stride: usize,
        mean_subtract: bool,
        weights: Option<&[f64]>,
    ) -> Vec<f64> {
        let (n_frames, n_entities, n_components) = a.dim();
        let mean = |s: &Array3<f64>, e: usize, c: usize| -> f64 {
            if mean_subtract {
                (0..n_frames).map(|t| s[[t, e, c]]).sum::<f64>() / n_frames as f64
            } else {
                0.0
            }
        };
        (0..=max_lag)
            .map(|lag| {
                let mut sum = 0.0;
                let mut count = 0usize;
                let mut tau = 0;
                while tau + lag < n_frames {
                    for e in 0..n_entities {
                        for c in 0..n_components {
                            let w = weights.map_or(1.0, |w| w[c]);
                            sum += w
                                * (a[[tau, e, c]] - mean(a, e, c))
                                * (b[[tau + lag, e, c]] - mean(b, e, c));
                        }
                    }
                    count += 1;
                    tau += stride;
                }
                sum / (n_entities * count) as f64
            })
            .collect()
    }

    #[test]
    fn fft_route_matches_the_direct_double_loop() {
        let s = series(64, 5, 3);
        let got = autocorrelate(s.view(), 1.0, &TimeCorrelationSpec::new().max_lag(20)).unwrap();
        let want = direct(&s, &s, 20, 1, false, None);
        for (lag, (g, w)) in got.values.iter().zip(want.iter()).enumerate() {
            assert!((g - w).abs() < 1e-10, "lag {lag}: {g} vs {w}");
        }
        assert_eq!(got.counts[0], 64);
        assert_eq!(got.counts[20], 44);
        assert_eq!(got.kind, CorrelationKind::Auto);
    }

    #[test]
    fn lag_zero_is_the_mean_square() {
        let s = series(32, 4, 3);
        let got = autocorrelate(s.view(), 1.0, &TimeCorrelationSpec::new().max_lag(5)).unwrap();
        let expected: f64 = s.iter().map(|v| v * v).sum::<f64>() / (4 * 32) as f64;
        assert!((got.values[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn unbiased_biased_and_raw_are_related_by_the_counts() {
        let s = series(40, 3, 2);
        let spec = TimeCorrelationSpec::new().max_lag(10);
        let unb = autocorrelate(s.view(), 1.0, &spec).unwrap();
        let bia = autocorrelate(s.view(), 1.0, &spec.normalization(Normalization::Biased)).unwrap();
        let raw = autocorrelate(s.view(), 1.0, &spec.normalization(Normalization::Raw)).unwrap();
        let want = direct(&s, &s, 10, 1, false, None);
        for k in 0..=10 {
            assert!((unb.values[k] - want[k]).abs() < 1e-10);
            let ratio = unb.counts[k] as f64 / unb.counts[0] as f64;
            assert!(
                (bia.values[k] - unb.values[k] * ratio).abs() < 1e-12,
                "k={k}"
            );
            let rescaled = raw.values[k] / (3 * unb.counts[k]) as f64;
            assert!((rescaled - unb.values[k]).abs() < 1e-12, "k={k}");
        }
        assert!(
            (unb.values[10] - bia.values[10]).abs() > 1e-12,
            "the normalizations must differ"
        );
    }

    #[test]
    fn max_lag_is_clamped_to_the_available_origins() {
        let s = series(8, 2, 3);
        let got = autocorrelate(s.view(), 1.0, &TimeCorrelationSpec::new().max_lag(999)).unwrap();
        assert_eq!(got.values.len(), 8);
        assert_eq!(got.lags[7], 7);
        assert_eq!(got.counts[7], 1);
    }

    #[test]
    fn a_constant_series_autocorrelates_flat() {
        let s = Array3::from_elem((16, 3, 2), 2.0);
        let got = autocorrelate(s.view(), 1.0, &TimeCorrelationSpec::new().max_lag(6)).unwrap();
        for value in got.values.iter() {
            assert!((value - 8.0).abs() < 1e-10, "{value}");
        }
        let centred = autocorrelate(
            s.view(),
            1.0,
            &TimeCorrelationSpec::new().max_lag(6).mean_subtract(true),
        )
        .unwrap();
        for value in centred.values.iter() {
            assert!(value.abs() < 1e-12, "{value}");
        }
    }

    #[test]
    fn too_few_frames_bad_dt_zero_stride_and_bad_weights_are_errors() {
        let one = series(1, 2, 3);
        assert!(autocorrelate(one.view(), 1.0, &TimeCorrelationSpec::new()).is_err());
        let s = series(8, 2, 3);
        assert!(autocorrelate(s.view(), 0.0, &TimeCorrelationSpec::new()).is_err());
        assert!(
            autocorrelate(s.view(), 1.0, &TimeCorrelationSpec::new().origin_stride(0)).is_err()
        );
        assert!(
            autocorrelate(
                s.view(),
                1.0,
                &TimeCorrelationSpec::new().component_weights(&[1.0, 2.0])
            )
            .is_err()
        );
        let other = series(8, 3, 3);
        assert!(cross_correlate(s.view(), other.view(), 1.0, &TimeCorrelationSpec::new()).is_err());
    }

    #[test]
    fn mean_subtraction_matches_the_direct_definition() {
        let s = series(48, 3, 3);
        let got = autocorrelate(
            s.view(),
            1.0,
            &TimeCorrelationSpec::new().max_lag(15).mean_subtract(true),
        )
        .unwrap();
        let want = direct(&s, &s, 15, 1, true, None);
        for k in 0..=15 {
            assert!((got.values[k] - want[k]).abs() < 1e-10, "k={k}");
        }
    }

    #[test]
    fn cross_correlation_matches_the_direct_definition_and_self_is_the_acf() {
        let a = series(48, 3, 3);
        let b = series(48, 3, 3).mapv(|v| v.cos() + 0.1);
        let spec = TimeCorrelationSpec::new().max_lag(12).mean_subtract(true);
        let got = cross_correlate(a.view(), b.view(), 1.0, &spec).unwrap();
        let want = direct(&a, &b, 12, 1, true, None);
        for k in 0..=12 {
            assert!((got.values[k] - want[k]).abs() < 1e-10, "k={k}");
        }
        assert_eq!(got.kind, CorrelationKind::Cross);
        let acf = autocorrelate(a.view(), 1.0, &spec).unwrap();
        let self_x = cross_correlate(a.view(), a.view(), 1.0, &spec).unwrap();
        assert_eq!(acf.values, self_x.values);
    }

    #[test]
    fn origin_stride_matches_the_strided_direct_loop() {
        let a = series(50, 4, 3);
        let b = series(50, 4, 3).mapv(|v| 0.5 * v * v - 0.3);
        for stride in [2usize, 3, 7] {
            let spec = TimeCorrelationSpec::new().max_lag(14).origin_stride(stride);
            let auto = autocorrelate(a.view(), 1.0, &spec).unwrap();
            let want = direct(&a, &a, 14, stride, false, None);
            for k in 0..=14 {
                assert!(
                    (auto.values[k] - want[k]).abs() < 1e-10,
                    "stride {stride} k={k}"
                );
                assert_eq!(auto.counts[k], origin_count(50, k, stride));
            }
            let cross =
                cross_correlate(a.view(), b.view(), 1.0, &spec.mean_subtract(true)).unwrap();
            let want = direct(&a, &b, 14, stride, true, None);
            for k in 0..=14 {
                assert!(
                    (cross.values[k] - want[k]).abs() < 1e-10,
                    "stride {stride} k={k}"
                );
            }
        }
    }

    #[test]
    fn component_weights_weight_the_dot_product() {
        let s = series(40, 2, 3);
        let w = [1.0, 2.0, 0.5];
        let got = autocorrelate(
            s.view(),
            1.0,
            &TimeCorrelationSpec::new().max_lag(9).component_weights(&w),
        )
        .unwrap();
        let want = direct(&s, &s, 9, 1, false, Some(&w));
        for k in 0..=9 {
            assert!((got.values[k] - want[k]).abs() < 1e-10, "k={k}");
        }
    }

    #[test]
    fn per_entity_rows_average_to_the_aggregate_curve() {
        let s = series(40, 5, 3);
        let spec = TimeCorrelationSpec::new().max_lag(9).mean_subtract(true);
        let agg = autocorrelate(s.view(), 1.0, &spec).unwrap();
        let per = autocorrelate(s.view(), 1.0, &spec.per_entity(true)).unwrap();
        let rows = per.per_entity.as_ref().unwrap();
        assert_eq!(rows.dim(), (5, 10));
        for k in 0..=9 {
            assert!((agg.values[k] - per.values[k]).abs() < 1e-12, "k={k}");
            let mean = rows.column(k).sum() / 5.0;
            assert!((mean - per.values[k]).abs() < 1e-12, "k={k}");
        }
        // Each row is that entity alone.
        for e in 0..5 {
            let one = s.slice(s![.., e..e + 1, ..]).to_owned();
            let alone = autocorrelate(one.view(), 1.0, &spec).unwrap();
            for k in 0..=9 {
                assert!(
                    (rows[[e, k]] - alone.values[k]).abs() < 1e-12,
                    "e={e} k={k}"
                );
            }
        }
    }

    #[test]
    fn lag_times_follow_dt() {
        let s = series(10, 1, 1);
        let got = autocorrelate(s.view(), 0.5, &TimeCorrelationSpec::new().max_lag(4)).unwrap();
        let lt = got.lag_times();
        assert_eq!(lt.len(), 5);
        assert!((lt[4] - 2.0).abs() < 1e-15);
        assert_eq!(got.max_lag(), 4);
    }
}
