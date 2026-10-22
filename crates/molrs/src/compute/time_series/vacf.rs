//! [`Vacf`] — the velocity autocorrelation function.

use super::{Correlator, CorrelatorAccumulator, LagKernel, MeanPolicy, Normalization};
use crate::compute::{ComputeError, Stream};

/// Velocity autocorrelation function, averaged over degrees of freedom:
///
/// ```text
/// C(t) = (1/n_dof) Σ_d  1/(T − t) Σ_τ δv_d(τ) · δv_d(τ + t),   δv_d = v_d − ⟨v_d⟩_T
/// ```
///
/// A [`Correlator`] with the [`LagKernel::Product`] kernel over every
/// origin, each degree of freedom mean-subtracted by default, unbiased,
/// averaged over degrees of freedom. `C(0)` is the mean square fluctuation of
/// one velocity component.
///
/// The mean policy is a convention ([`with_mean`](Self::with_mean)).
/// Subtracting each velocity's time mean removes any drift, but a finite run's
/// sample mean is not the ensemble mean, and the Green–Kubo integral up to
/// `τ_max` comes out low by a relative amount of order `2 τ_max / T`. [`MeanPolicy::Keep`] correlates the
/// velocities as given (the ensemble mean velocity is zero), which is the
/// unbiased Green–Kubo integrand for a run without centre-of-mass drift. The same curve feeds the vibrational density of states (a
/// [`Spectrum`](super::Spectrum)) and Green–Kubo self-diffusion,
/// `D = (1/d) ∫ C(t) dt` (a [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)).
///
/// `Vacf` is a [`Stream`] only: [`accumulator`](Stream::accumulator) hands
/// out a [`CorrelatorAccumulator`]; push one
/// [`CorrelatorFrame`](super::CorrelatorFrame) per frame whose `a` is that
/// frame's velocities as `n_dof × 1` (one degree of freedom per row, `b =
/// None`, no box) and [`finish`](crate::compute::Accumulate::finish) reads
/// the curve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vacf {
    correlator: Correlator,
}

impl Vacf {
    /// The VACF over lags `0 … max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Product, max_lag).with_mean(Self::DEFAULT_MEAN),
        }
    }

    /// The mean policy when not set: each velocity component's time mean is
    /// subtracted.
    pub const DEFAULT_MEAN: MeanPolicy = MeanPolicy::Subtract;

    /// Subtract each column's time mean first, or correlate the values as
    /// given (default [`DEFAULT_MEAN`](Self::DEFAULT_MEAN)).
    pub fn with_mean(self, mean: MeanPolicy) -> Self {
        Self {
            correlator: self.correlator.with_mean(mean),
        }
    }

    /// Scale the lagged sums by `norm` (default [`Normalization::Unbiased`];
    /// `Biased` for a VDOS that must stay non-negative).
    pub fn with_normalization(self, norm: Normalization) -> Self {
        Self {
            correlator: self.correlator.with_normalization(norm),
        }
    }

    /// Take every `stride`-th frame as a time origin (default
    /// [`Correlator::DEFAULT_STRIDE`]; see [`Correlator::with_stride`]).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0`.
    pub fn with_stride(self, stride: usize) -> Result<Self, ComputeError> {
        Ok(Self {
            correlator: self.correlator.with_stride(stride)?,
        })
    }
}

/// Streams one frame of velocities at a time as an `n_dof × 1`
/// [`CorrelatorFrame`](super::CorrelatorFrame).
impl Stream for Vacf {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    Vacf as "Vacf", Frames;
    params |s| {
        max_lag: Int;
        normalization: Choice(Normalization::NAMES)
            = Normalization::default() => s.correlator.normalization();
        stride: Int = Correlator::DEFAULT_STRIDE => s.correlator.stride();
        mean: Choice(MeanPolicy::NAMES) = Vacf::DEFAULT_MEAN => s.correlator.mean();
    }
    inputs {
        a: Scalars;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::rng_series;
    use super::super::{CorrelatorInput, TimeCorrelation};
    use super::*;
    use crate::compute::{
        Accumulate, Compute, CorrelatorFrame, CumulativeTrapezoid, CumulativeTrapezoidInput,
    };
    use molrs::op::F;
    use ndarray::{Array1, Array2, Axis};

    /// Push every row of `v` (`T × n_dof`) into `acc` as an `n_dof × 1`
    /// frame, then finish.
    fn fold(
        acc: CorrelatorAccumulator,
        v: &Array2<F>,
        dt: F,
    ) -> Result<TimeCorrelation, ComputeError> {
        acc.fold(v.outer_iter().map(|row| CorrelatorFrame {
            a: row.insert_axis(Axis(1)),
            b: None,
            dt,
            simbox: None,
        }))
    }

    fn vacf(v: &Array2<F>, dt: F, max_lag: usize) -> Result<TimeCorrelation, ComputeError> {
        fold(Vacf::new(max_lag).accumulator(), v, dt)
    }

    /// Fold `v` frame by frame through the VACF accumulator.
    fn stream(v: &Array2<F>, max_lag: usize) -> CorrelatorAccumulator {
        let mut acc = Vacf::new(max_lag).accumulator();
        for row in v.outer_iter() {
            acc.push(CorrelatorFrame {
                a: row.insert_axis(Axis(1)),
                b: None,
                dt: 1.0,
                simbox: None,
            })
            .unwrap();
        }
        acc
    }

    #[test]
    fn surface_is_its_correlator() {
        let v = rng_series(40, 6, 2);
        let explicit = Correlator::new(LagKernel::Product, 12).with_mean(MeanPolicy::Subtract);
        assert_eq!(Vacf::new(12).correlator, explicit);
        let via_correlator = fold(explicit.accumulator(), &v, 0.25).unwrap();
        assert_eq!(vacf(&v, 0.25, 12).unwrap(), via_correlator);
    }

    /// `Keep` is the product correlator with the mean kept, and differs from
    /// the default on a series with a nonzero mean.
    #[test]
    fn keep_is_the_product_correlator_with_the_mean_kept() {
        let v = rng_series(64, 6, 3) + 0.4;
        let keep = Vacf::new(12).with_mean(MeanPolicy::Keep);
        let explicit = Correlator::new(LagKernel::Product, 12).with_mean(MeanPolicy::Keep);
        assert_eq!(keep.correlator, explicit);
        let got = fold(keep.accumulator(), &v, 0.5).unwrap();
        let via_correlator = fold(explicit.accumulator(), &v, 0.5).unwrap();
        assert_eq!(got, via_correlator);
        assert_ne!(got, vacf(&v, 0.5, 12).unwrap());
        assert_eq!(Vacf::new(12).correlator.mean(), Vacf::DEFAULT_MEAN);
        assert_eq!(Vacf::DEFAULT_MEAN, MeanPolicy::Subtract);

        // The kept-mean fold equals the correlator's FFT evaluation.
        let fft = explicit
            .compute(CorrelatorInput {
                a: v.view().insert_axis(Axis(2)),
                b: None,
                dt: 0.5,
                boxes: None,
            })
            .unwrap();
        for k in 0..=12 {
            let (s, b) = (got.values[[k, 0]], fft.values[[k, 0]]);
            assert!(
                (s - b).abs() < 1e-12 * b.abs().max(1.0),
                "lag {k}: {s} vs {b}"
            );
        }
    }

    #[test]
    fn vacf_equals_unbiased_dof_averaged_acf() {
        // The per-DOF mean-subtract + ACF + DOF-average curve, written out.
        let n = 512;
        let dt = 0.5;
        let max_lag = 100;
        let v = rng_series(n, 9, 7);

        let mut expected = Array1::<F>::zeros(max_lag + 1);
        for d in 0..9 {
            let col: Vec<F> = (0..n).map(|t| v[[t, d]]).collect();
            let mean = col.iter().sum::<F>() / n as F;
            for k in 0..=max_lag {
                expected[k] += (0..n - k)
                    .map(|t| (col[t] - mean) * (col[t + k] - mean))
                    .sum::<F>();
            }
        }
        for k in 0..=max_lag {
            expected[k] *= (1.0 / 9.0) / (n - k) as F;
        }

        let raw = vacf(&v, dt, max_lag).unwrap();
        assert_eq!(raw.n_lags(), expected.len());
        for k in 0..raw.n_lags() {
            assert!((raw.values[[k, 0]] - expected[k]).abs() < 1e-12, "k={k}");
        }
        assert_eq!(raw.n_lags(), max_lag + 1);
    }

    /// `max_lag` beyond the trajectory used to be clamped to `n_frames − 1`;
    /// a lag with no time origin is now an error.
    #[test]
    fn raw_max_lag_exceeds_length_is_out_of_range() {
        let v = rng_series(8, 3, 1);
        assert!(matches!(
            vacf(&v, 1.0, 1000),
            Err(ComputeError::OutOfRange {
                field: "max_lag",
                ..
            })
        ));
        assert_eq!(vacf(&v, 1.0, 7).unwrap().n_lags(), 8);
    }

    #[test]
    fn vacf_is_cartesian_acf_over_ndof() {
        let n = 64;
        let n_dof = 6;
        let max_lag = 20;
        let v = rng_series(n, n_dof, 99);
        let raw = vacf(&v, 0.5, max_lag).unwrap();
        // One item carrying all n_dof components: the Cartesian sum.
        let cart = Correlator::new(LagKernel::Product, max_lag)
            .with_mean(MeanPolicy::Subtract)
            .compute(CorrelatorInput {
                a: v.view().insert_axis(Axis(1)),
                b: None,
                dt: 0.5,
                boxes: None,
            })
            .unwrap();
        for k in 0..raw.n_lags() {
            assert!(
                (raw.values[[k, 0]] - cart.values[[k, 0]] / n_dof as f64).abs() < 1e-12,
                "k={k}"
            );
        }
    }

    #[test]
    fn velocities_to_diffusion_through_cumulative_trapezoid() {
        // velocity array → Vacf → CumulativeTrapezoid, no glue in between.
        let v = rng_series(200, 3, 5);
        let dt = 0.5;
        let curve = vacf(&v, dt, 40).unwrap();
        let integral = CumulativeTrapezoid::new()
            .compute(CumulativeTrapezoidInput {
                correlation: &curve,
                channel: 0,
            })
            .unwrap();
        let manual: F = (1..curve.n_lags())
            .map(|k| 0.5 * (curve.values[[k - 1, 0]] + curve.values[[k, 0]]) * dt)
            .sum();
        assert!((integral.values[[integral.n_lags() - 1, 0]] - manual).abs() < 1e-12);
    }

    // ── Streaming (the former `VacfAccumulator`) ──

    fn shifted_series(n: usize, cols: usize, seed: u64) -> Array2<F> {
        // Nonzero mean so the mean-subtraction correction is exercised.
        rng_series(n, cols, seed) + 0.3
    }

    /// The streamed, mean-subtracted VACF equals the correlator's FFT
    /// evaluation of the same series.
    #[test]
    fn streaming_matches_the_correlator_fft() {
        let (n, dof, res) = (128, 9, 20);
        let v = shifted_series(n, dof, 11);
        let batch = Vacf::new(res)
            .correlator
            .compute(CorrelatorInput {
                a: v.view().insert_axis(Axis(2)),
                b: None,
                dt: 1.0,
                boxes: None,
            })
            .unwrap();
        let streamed = stream(&v, res).finish().unwrap();

        assert_eq!(streamed.n_lags(), batch.n_lags());
        let scale = batch.values[[0, 0]].abs().max(1.0);
        for k in 0..=res {
            let s = streamed.values[[k, 0]];
            assert!(
                (s - batch.values[[k, 0]]).abs() / scale < 1e-12,
                "lag {k}: streamed {s} vs batch {}",
                batch.values[[k, 0]]
            );
        }
        assert_eq!(streamed.lag_times, batch.lag_times);
    }

    /// A lag with no origin is refused at `finish`.
    #[test]
    fn short_trajectory_is_out_of_range() {
        let (n, dof, res) = (6, 3, 20); // T − 1 < max_lag
        let v = shifted_series(n, dof, 5);
        assert!(matches!(
            stream(&v, res).finish(),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn dof_mismatch_and_too_few_frames_are_errors() {
        let mut acc = Vacf::new(4).accumulator();
        let frame = |a| CorrelatorFrame {
            a,
            b: None,
            dt: 1.0,
            simbox: None,
        };
        let two = ndarray::array![[1.0], [2.0]];
        let one = ndarray::array![[1.0]];
        acc.push(frame(two.view())).unwrap();
        assert!(matches!(
            acc.push(frame(one.view())).unwrap_err(),
            ComputeError::DimensionMismatch { .. }
        ));
        assert!(matches!(
            acc.finish().unwrap_err(),
            ComputeError::EmptyInput
        ));
    }
}
