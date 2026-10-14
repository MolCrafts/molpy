//! [`OnsagerCorrelation`] — collective displacement cross-correlation.

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, TimeCorrelation};
use crate::compute::{Compute, ComputeError, Stream};

/// Onsager displacement correlation of two species,
///
/// ```text
/// L_ij(t) = 1/(T − t) Σ_τ ΔP_i(τ, t) · ΔP_j(τ, t),   ΔP(τ, t) = P(τ + t) − P(τ),
/// ```
///
/// of their **collective** coordinates `P_s = Σ_{a ∈ s} r_a` (unwrapped) — a
/// [`Correlator`] with the [`LagKernel::SquaredDisplacement`] kernel as a
/// cross-correlation of two `T × 3` series. `L_ii` is the collective MSD of
/// species `i`; the off-diagonal term carries the correlated drift that the
/// Nernst–Einstein sum misses. The transport coefficient is the long-time
/// slope, `L_ij(t) ≈ 2 d D_ij t`, a [`LinearFit`](crate::compute::LinearFit).
///
/// Unit-agnostic: positions in Å give Å². No volume normalisation is
/// applied. Ported from the `onsager` recipe of the *tame* library.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnsagerCorrelation {
    correlator: Correlator,
}

/// The data [`OnsagerCorrelation`] correlates.
#[derive(Debug, Clone, Copy)]
pub struct OnsagerInput<'a> {
    /// Collective coordinate of species `i`, `T × 3`, unwrapped.
    pub a: ArrayView2<'a, F>,
    /// Collective coordinate of species `j`, `T × 3`, unwrapped.
    pub b: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
    /// One box per frame to reject a wrapped series; `None` to skip.
    pub boxes: Option<&'a [SimBox]>,
}

impl OnsagerCorrelation {
    /// The correlation over lags `0 … max_lag` frames.
    pub fn new(max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::SquaredDisplacement, max_lag).with_dimension(3),
        }
    }

    /// The correlator this correlation is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for OnsagerCorrelation {
    type Input<'a> = OnsagerInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: OnsagerInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.a.insert_axis(Axis(1)),
            b: Some(input.b.insert_axis(Axis(1))),
            dt: input.dt,
            boxes: input.boxes,
        })
    }
}

impl Stream for OnsagerCorrelation {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::rng_series;
    use super::*;
    use ndarray::{Array2, array};

    fn onsager(
        a: &Array2<F>,
        b: &Array2<F>,
        dt: F,
        max_lag: usize,
    ) -> Result<TimeCorrelation, ComputeError> {
        OnsagerCorrelation::new(max_lag).compute(OnsagerInput {
            a: a.view(),
            b: b.view(),
            dt,
            boxes: None,
        })
    }

    #[test]
    fn surface_is_its_correlator() {
        let (a, b) = (rng_series(30, 3, 1), rng_series(30, 3, 2));
        let explicit = Correlator::new(LagKernel::SquaredDisplacement, 10).with_dimension(3);
        assert_eq!(*OnsagerCorrelation::new(10).correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: a.view().insert_axis(Axis(1)),
                b: Some(b.view().insert_axis(Axis(1))),
                dt: 1.0,
                boxes: None,
            })
            .unwrap();
        assert_eq!(onsager(&a, &b, 1.0, 10).unwrap(), via_correlator);
    }

    #[test]
    fn self_correlation_is_collective_msd() {
        // A single collective coordinate drifting at constant velocity v=(1,0,0):
        // P(t) = (t, 0, 0). ΔP(τ) = (τ, 0, 0). L(τ) = τ² for every origin.
        let p = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let r = onsager(&p, &p, 1.0, 4).unwrap();
        for tau in 0..=4 {
            let expected = (tau as f64) * (tau as f64);
            assert!(
                (r.values[[tau, 0]] - expected).abs() < 1e-12,
                "lag {tau}: got {}, expected {expected}",
                r.values[[tau, 0]]
            );
        }
        assert!((r.lag_times[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn anticorrelated_species_give_negative_offdiagonal() {
        // P_i drifts +x, P_j drifts −x. ΔP_i·ΔP_j = (τ)(−τ) = −τ².
        let pi = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let pj = array![[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]];
        let r = onsager(&pi, &pj, 0.5, 2).unwrap();
        assert!((r.values[[1, 0]] + 1.0).abs() < 1e-12);
        assert!((r.values[[2, 0]] + 4.0).abs() < 1e-12);
        assert!((r.lag_times[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn time_origin_average_matches_manual() {
        // τ=1 origins: (t0=0) Δ=(1,0,0) → 1; (t0=1) Δ=(0,2,0) → 4. mean = 2.5
        let p = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let r = onsager(&p, &p, 1.0, 2).unwrap();
        assert!((r.values[[1, 0]] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_shape_and_mismatch() {
        let p3 = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let p_short = array![[0.0, 0.0, 0.0]];
        assert!(onsager(&p3, &p_short, 1.0, 1).is_err());
        let bad = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(matches!(
            onsager(&bad, &bad, 1.0, 1),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(onsager(&p3, &p3, 0.0, 1).is_err());
        let nan = array![[0.0, 0.0, 0.0], [F::NAN, 0.0, 0.0]];
        assert!(matches!(
            onsager(&p3, &nan, 1.0, 1),
            Err(ComputeError::NonFinite { where_: "b", .. })
        ));
    }
}
