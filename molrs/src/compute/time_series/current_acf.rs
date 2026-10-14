//! [`CurrentAcf`] — the autocorrelation of a collective current.

use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, TimeCorrelation};
use crate::compute::{Compute, ComputeError, Stream};

/// Autocorrelation of a collective current, summed over the Cartesian
/// components:
///
/// ```text
/// C(t) = 1/(T − t) Σ_τ J(τ) · J(τ + t)
/// ```
///
/// A [`Correlator`] with the [`LagKernel::Product`] kernel on one `T × 3`
/// series, unbiased, with no mean subtraction (a current is already a flux).
/// With `J = Σ_a q_a v_a` it is the Green–Kubo conductivity integrand,
/// `σ = 1/(3 V k_B T) ∫ C(t) dt`
/// ([`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)); with the
/// dipole rate `J = Ṁ` (a [`Derivative`](super::Derivative) of the dipole
/// series) it is the IR flux correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurrentAcf {
    correlator: Correlator,
}

/// The data [`CurrentAcf`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct CurrentAcfInput<'a> {
    /// Collective current per frame, `T × 3`.
    pub current: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl CurrentAcf {
    /// The current ACF over lags `0 … max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::Product, max_lag).with_dimension(3),
        }
    }

    /// The correlator this ACF is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for CurrentAcf {
    type Input<'a> = CurrentAcfInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: CurrentAcfInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.current.insert_axis(Axis(1)),
            b: None,
            dt: input.dt,
            boxes: None,
        })
    }
}

impl Stream for CurrentAcf {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    CurrentAcf as "CurrentAcf";
    params |s| {
        max_lag: Int;
    }
    inputs {
        current: Series;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::rng_series;
    use super::*;
    use crate::compute::{CumulativeTrapezoid, CumulativeTrapezoidInput};
    use ndarray::{Array1, Array2};

    fn current_acf(j: &Array2<F>, dt: F, max_lag: usize) -> Result<TimeCorrelation, ComputeError> {
        CurrentAcf::new(max_lag).compute(CurrentAcfInput {
            current: j.view(),
            dt,
        })
    }

    #[test]
    fn surface_is_its_correlator() {
        let j = rng_series(50, 3, 8);
        let explicit = Correlator::new(LagKernel::Product, 15).with_dimension(3);
        assert_eq!(*CurrentAcf::new(15).correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: j.view().insert_axis(Axis(1)),
                b: None,
                dt: 0.1,
                boxes: None,
            })
            .unwrap();
        assert_eq!(current_acf(&j, 0.1, 15).unwrap(), via_correlator);
    }

    #[test]
    fn green_kubo_raw_jacf_matches_direct_acf() {
        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let current = rng_series(n, 3, 5);
        let mut expected = Array1::<f64>::zeros(mct + 1);
        for tau in 0..=mct {
            let count = n - tau;
            let mut acc = 0.0;
            for t in 0..count {
                let mut s = 0.0;
                for d in 0..3 {
                    s += current[[t, d]] * current[[t + tau, d]];
                }
                acc += s;
            }
            expected[tau] = acc / count as f64;
        }
        let raw = current_acf(&current, dt, mct).unwrap();
        assert_eq!(raw.n_lags(), expected.len());
        for k in 0..raw.n_lags() {
            assert!((raw.values[[k, 0]] - expected[k]).abs() < 1e-12, "k={k}");
            assert!((raw.lag_times[k] - k as f64 * dt).abs() < 1e-12);
        }
    }

    #[test]
    fn green_kubo_raw_plus_cumulative_trapezoid_matches_manual_trapezoid() {
        use molrs::core::constants::{BOLTZMANN as K_B_SI, ELEMENTARY_CHARGE as E_C};
        let angstrom_m = crate::core::unit_factors::ANGSTROM_TO_M.get();
        let femtosecond_s = crate::core::unit_factors::FS_TO_S.get();

        let n = 256;
        let dt = 0.5;
        let mct = 80;
        let (volume, temperature) = (1000.0, 300.0);
        let current = rng_series(n, 3, 19);

        let raw = current_acf(&current, dt, mct).unwrap();
        let jacf = raw.channel(0).to_owned();
        let integ = CumulativeTrapezoid::new()
            .compute(CumulativeTrapezoidInput {
                correlation: &raw,
                channel: 0,
            })
            .unwrap()
            .channel(0)
            .to_owned();

        let mut manual = 0.0;
        for tau in 1..jacf.len() {
            manual += 0.5 * (jacf[tau - 1] + jacf[tau]) * dt;
        }
        let last = integ.len() - 1;
        assert!((integ[last] - manual).abs() < 1e-12);

        // Green–Kubo 1/3 prefactor.
        let prefactor = (E_C * E_C * angstrom_m * angstrom_m / femtosecond_s)
            / (3.0 * angstrom_m.powi(3) * K_B_SI);
        let sigma = prefactor * integ[last] / (volume * temperature);
        assert!(sigma.is_finite());
    }

    #[test]
    fn series_computes_reject_bad_shape() {
        let bad = rng_series(10, 2, 1);
        assert!(matches!(
            current_acf(&bad, 1.0, 5),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
