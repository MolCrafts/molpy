//! [`DipoleAcf`] — the fluctuation autocorrelation of the total dipole.

use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{
    Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, MeanPolicy, Normalization,
    TimeCorrelation,
};
use crate::compute::{Compute, ComputeError, Stream};

/// Autocorrelation of the dipole fluctuation, summed over the Cartesian
/// components:
///
/// ```text
/// C(t) = 1/(T − t) Σ_τ δM(τ) · δM(τ + t),   δM = M − ⟨M⟩_T
/// ```
///
/// A [`Correlator`] with the [`LagKernel::Product`] kernel on one `T × 3`
/// series, mean-subtracted and unbiased. `C(0) = ⟨|δM|²⟩` is the variance the
/// Debye amplitude and the static permittivity are scaled by; the normalised
/// decay gives the Debye relaxation time
/// ([`DebyeFit`](crate::compute::DebyeFit)). The volume, temperature and Ewald
/// boundary that turn it into ε(ω) belong to the spectrum route, not here.
///
/// The estimator is unbiased by default; the raw lagged sum
/// ([`Normalization::None`], through
/// [`with_normalization`](Self::with_normalization)) is selectable like on
/// every other product correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DipoleAcf {
    correlator: Correlator,
}

/// The data [`DipoleAcf`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct DipoleAcfInput<'a> {
    /// Total dipole moment per frame, `T × 3`, unwrapped.
    pub dipoles: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl DipoleAcf {
    /// The dipole ACF over lags `0 … max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Product, max_lag)
                .with_mean(MeanPolicy::Subtract)
                .with_dimension(3),
        }
    }

    /// Scale the lagged sums by `norm` (default [`Normalization::Unbiased`]).
    pub fn with_normalization(mut self, norm: Normalization) -> Self {
        self.correlator = self.correlator.with_normalization(norm);
        self
    }

    /// The correlator this ACF is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for DipoleAcf {
    type Input<'a> = DipoleAcfInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: DipoleAcfInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.dipoles.insert_axis(Axis(1)),
            b: None,
            dt: input.dt,
            boxes: None,
        })
    }
}

impl Stream for DipoleAcf {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    DipoleAcf as "DipoleAcf";
    params |s| {
        max_lag: Int;
        normalization: Choice(Normalization::NAMES)
            = Normalization::default() => s.correlator.normalization();
    }
    inputs {
        dipoles: Series;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::rng_series;
    use super::*;
    use ndarray::Array2;

    fn dipole_acf(dm: &Array2<F>, dt: F, max_lag: usize) -> Result<TimeCorrelation, ComputeError> {
        DipoleAcf::new(max_lag).compute(DipoleAcfInput {
            dipoles: dm.view(),
            dt,
        })
    }

    #[test]
    fn surface_is_its_correlator() {
        let dm = rng_series(50, 3, 4);
        let explicit = Correlator::new(LagKernel::Product, 15)
            .with_mean(MeanPolicy::Subtract)
            .with_dimension(3);
        assert_eq!(*DipoleAcf::new(15).correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: dm.view().insert_axis(Axis(1)),
                b: None,
                dt: 0.1,
                boxes: None,
            })
            .unwrap();
        assert_eq!(dipole_acf(&dm, 0.1, 15).unwrap(), via_correlator);
    }

    #[test]
    fn cartesian_acf_matches_direct_sum() {
        let n = 64;
        let max_lag = 20;
        let s = rng_series(n, 3, 11);
        let got = dipole_acf(&s, 1.0, max_lag).unwrap();
        let mut mean = [0.0_f64; 3];
        for t in 0..n {
            for d in 0..3 {
                mean[d] += s[[t, d]];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        for tau in 0..=max_lag {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                for d in 0..3 {
                    let a = s[[t, d]] - mean[d];
                    let b = s[[t + tau, d]] - mean[d];
                    acc += a * b;
                }
            }
            let expected = acc / count as f64;
            assert!(
                (got.values[[tau, 0]] - expected).abs() < 1e-10,
                "tau={tau}: {} vs {expected}",
                got.values[[tau, 0]]
            );
        }
    }

    /// The former `DebyeRelaxation` raw result: an unnormalised ACF whose
    /// zero lag is the dipole variance.
    #[test]
    fn zero_lag_is_the_unnormalised_dipole_variance() {
        let dipole = rng_series(128, 3, 9);
        let res = dipole_acf(&dipole, 0.5, 40).unwrap();
        let mean = dipole.mean_axis(Axis(0)).unwrap();
        let variance: F = dipole
            .outer_iter()
            .map(|m| (&m - &mean).mapv(|x| x * x).sum())
            .sum::<F>()
            / 128.0;
        assert!(res.values[[0, 0]] > 0.0);
        assert!((res.values[[0, 0]] - variance).abs() < 1e-12);
    }

    /// `Normalization::None` leaves the raw lagged sum of the fluctuations.
    #[test]
    fn the_raw_lagged_sum_is_selectable() {
        let dm = rng_series(40, 3, 6);
        let unbiased = dipole_acf(&dm, 1.0, 10).unwrap();
        let raw = DipoleAcf::new(10)
            .with_normalization(Normalization::None)
            .compute(DipoleAcfInput {
                dipoles: dm.view(),
                dt: 1.0,
            })
            .unwrap();
        for k in 0..=10 {
            let expected = unbiased.values[[k, 0]] * (40 - k) as F;
            assert!((raw.values[[k, 0]] - expected).abs() < 1e-12, "k={k}");
        }
    }

    #[test]
    fn a_non_cartesian_series_is_rejected() {
        let bad = rng_series(10, 2, 1);
        assert!(matches!(
            dipole_acf(&bad, 1.0, 5),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
