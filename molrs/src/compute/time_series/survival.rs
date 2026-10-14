//! [`Survival`] — survival correlation of 0/1 presence series.

use molrs::op::F;
use ndarray::{ArrayView2, Axis};

use super::{Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, TimeCorrelation};
use crate::compute::named_enum::named_enum;
use crate::compute::{Compute, ComputeError, Stream};

named_enum! {
    /// When a bond (pair, contact, …) counts as surviving a lag.
    pub enum SurvivalKind {
        /// Present at every frame of `[τ, τ + t]` — the continuous `S(t)`.
        Continuous => "continuous",
        /// Present at `τ` and at `τ + t`, gaps allowed — the intermittent
        /// `C(t) = ⟨h(0) h(t)⟩`.
        Intermittent => "intermittent",
    }
}

/// Survival time-correlation of presence series `h_b(t) ∈ {0, 1}`, one per
/// bond (Luzar & Chandler, *Phys. Rev. Lett.* **76**, 928 (1996)):
///
/// ```text
/// continuous:    S(t) = ⟨ h(τ) Π_{s ≤ t} h(τ + s) ⟩
/// intermittent:  C(t) = ⟨ h(τ) h(τ + t) ⟩
/// ```
///
/// averaged over bonds and every origin `τ`; `S(0) = C(0) = ⟨h⟩`. The
/// series is a [`Presence`](crate::compute::Presence)'s `values`, from
/// [`PairPresence`](crate::compute::PairPresence) (distance criterion, with
/// hysteresis for the stable-states picture) or
/// [`HBondsResult::presence`](crate::compute::HBondsResult::presence). Divide by
/// the value at lag 0 for the normalised curves, and integrate
/// ([`CumulativeTrapezoid`](crate::compute::CumulativeTrapezoid)) for a
/// lifetime. A [`Correlator`] with the [`LagKernel::Persistence`] kernel
/// (continuous) or the [`LagKernel::Product`] kernel (intermittent),
/// unbiased, averaged over bonds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Survival {
    correlator: Correlator,
}

/// The data [`Survival`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalInput<'a> {
    /// Presence per frame and bond, `T × n_bonds`, every value `0` or `1`.
    pub presence: ArrayView2<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl Survival {
    /// The `kind` survival correlation over lags `0 … max_lag` frames.
    pub fn new(kind: SurvivalKind, max_lag: usize) -> Self {
        let kernel = match kind {
            SurvivalKind::Continuous => LagKernel::Persistence,
            SurvivalKind::Intermittent => LagKernel::Product,
        };
        Self {
            correlator: Correlator::new(kernel, max_lag),
        }
    }

    /// The correlator this survival correlation is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for Survival {
    type Input<'a> = SurvivalInput<'a>;
    type Output = TimeCorrelation;

    fn compute(&self, input: SurvivalInput<'_>) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.presence.insert_axis(Axis(2)),
            b: None,
            dt: input.dt,
            boxes: None,
        })
    }
}

impl Stream for Survival {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    Survival as "Survival";
    params |s| {
        kind: Choice(SurvivalKind::NAMES);
        max_lag: Int;
    }
    inputs {
        presence: Series;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// One bond: present, present, absent, present, present, present.
    fn one_bond() -> Array2<F> {
        Array2::from_shape_vec((6, 1), vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0]).unwrap()
    }

    fn survival(kind: SurvivalKind, h: &Array2<F>, max_lag: usize) -> TimeCorrelation {
        Survival::new(kind, max_lag)
            .compute(SurvivalInput {
                presence: h.view(),
                dt: 1.0,
            })
            .unwrap()
    }

    #[test]
    fn surface_is_its_correlator() {
        let h = one_bond();
        for (kind, kernel) in [
            (SurvivalKind::Continuous, LagKernel::Persistence),
            (SurvivalKind::Intermittent, LagKernel::Product),
        ] {
            let explicit = Correlator::new(kernel, 3);
            assert_eq!(*Survival::new(kind, 3).correlator(), explicit);
            let via_correlator = explicit
                .compute(CorrelatorInput {
                    a: h.view().insert_axis(Axis(2)),
                    b: None,
                    dt: 1.0,
                    boxes: None,
                })
                .unwrap();
            assert_eq!(survival(kind, &h, 3), via_correlator);
        }
    }

    /// Counted by hand on `1 1 0 1 1 1`.
    #[test]
    fn continuous_and_intermittent_by_hand() {
        let h = one_bond();
        let s = survival(SurvivalKind::Continuous, &h, 3);
        let c = survival(SurvivalKind::Intermittent, &h, 3);
        // lag 0: 5 present of 6 origins, for both.
        // lag 1: unbroken pairs (0,1) (3,4) (4,5): 3 of 5 — also 3 of 5 gapped.
        // lag 2: unbroken run 3..=5 only: 1 of 4; gapped (1,3) (3,5): 2 of 4.
        // lag 3: no run of 4: 0 of 3; gapped (0,3) (1,4): 2 of 3.
        let continuous = [5.0 / 6.0, 3.0 / 5.0, 1.0 / 4.0, 0.0];
        let intermittent = [5.0 / 6.0, 3.0 / 5.0, 2.0 / 4.0, 2.0 / 3.0];
        for lag in 0..=3 {
            assert!(
                (s.values[[lag, 0]] - continuous[lag]).abs() < 1e-12,
                "S({lag})"
            );
            assert!(
                (c.values[[lag, 0]] - intermittent[lag]).abs() < 1e-12,
                "C({lag})"
            );
        }
    }

    #[test]
    fn presence_must_be_zero_or_one() {
        let mut h = one_bond();
        h[[2, 0]] = 0.5;
        let err = Survival::new(SurvivalKind::Continuous, 2)
            .compute(SurvivalInput {
                presence: h.view(),
                dt: 1.0,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { field: "a", .. }));
    }
}
