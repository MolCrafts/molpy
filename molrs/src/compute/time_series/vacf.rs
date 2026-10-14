//! [`Vacf`] — the velocity autocorrelation function.

use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{
    Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, MeanPolicy, TimeCorrelation,
};
use crate::compute::{Compute, ComputeError, Stream};

/// Velocity autocorrelation function, averaged over degrees of freedom:
///
/// ```text
/// C(t) = (1/n_dof) Σ_d  1/(T − t) Σ_τ δv_d(τ) · δv_d(τ + t),   δv_d = v_d − ⟨v_d⟩_T
/// ```
///
/// A [`Correlator`] with the [`LagKernel::Product`] kernel over every
/// origin, each degree of freedom mean-subtracted, unbiased, averaged over
/// degrees of freedom. `C(0)` is the mean square fluctuation of one velocity
/// component. The same curve feeds the vibrational density of states (a
/// [`Spectrum`](super::Spectrum)) and Green–Kubo self-diffusion,
/// `D = (1/d) ∫ C(t) dt` (a [`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vacf {
    correlator: Correlator,
}

/// The data [`Vacf`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct VacfInput<'a> {
    /// Velocities, `T × n_dof` (e.g. `N × 3` per frame, flattened).
    pub velocities: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl Vacf {
    /// The VACF over lags `0 … max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Product, max_lag)
                .with_mean(MeanPolicy::Subtract),
        }
    }

    /// The correlator this VACF is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for Vacf {
    type Input<'a> = VacfInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: VacfInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.velocities.insert_axis(Axis(2)),
            b: None,
            dt: input.dt,
            boxes: None,
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
    Vacf as "Vacf";
    params |s| {
        max_lag: Int;
    }
    inputs {
        velocities: Series;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::rng_series;
    use super::*;
    use crate::compute::{
        Accumulate, CorrelatorFrame, CumulativeTrapezoid, CumulativeTrapezoidInput,
    };
    use ndarray::{Array1, Array2};

    fn vacf(v: &Array2<F>, dt: F, max_lag: usize) -> Result<TimeCorrelation, ComputeError> {
        Vacf::new(max_lag).compute(VacfInput {
            velocities: v.view(),
            dt,
        })
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
        assert_eq!(*Vacf::new(12).correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: v.view().insert_axis(Axis(2)),
                b: None,
                dt: 0.25,
                boxes: None,
            })
            .unwrap();
        assert_eq!(vacf(&v, 0.25, 12).unwrap(), via_correlator);
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

    #[test]
    fn streaming_matches_batch_fft_vacf() {
        let (n, dof, res) = (128, 9, 20);
        let v = shifted_series(n, dof, 11);
        let batch = vacf(&v, 1.0, res).unwrap();
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

    /// The streaming form refuses a lag with no origin just like the batch.
    #[test]
    fn short_trajectory_is_out_of_range() {
        let (n, dof, res) = (6, 3, 20); // T − 1 < max_lag
        let v = shifted_series(n, dof, 5);
        assert!(matches!(
            vacf(&v, 1.0, res),
            Err(ComputeError::OutOfRange { .. })
        ));
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
