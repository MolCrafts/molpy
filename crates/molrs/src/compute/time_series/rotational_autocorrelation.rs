//! [`RotationalAutocorrelation`] — the Wigner-D rotational autocorrelation.

use molrs::op::F;
use ndarray::ArrayView3;

use super::{
    Correlator, CorrelatorAccumulator, CorrelatorInput, LagKernel, Origins, Reduction,
    TimeCorrelation,
};
use crate::compute::{Compute, ComputeError, Stream};

/// Rotational autocorrelation of rigid-body orientations, in the sense of
/// `freud.order.RotationalAutocorrelation`.
///
/// For each particle, the rotation from its orientation in frame 0 to that
/// in frame `t` is `q_rel = q(t) · q(0)*`, of angle `θ`; the normalised
/// Wigner-D character
///
/// ```text
/// Ψ_l(t) = χ_l(θ) / (2l + 1),   χ_l(θ) = sin((l + ½) θ) / sin(θ / 2),   χ_l(0) = 2l + 1
/// ```
///
/// is 1 for an unrotated particle and decays toward 0 as the ensemble
/// decorrelates. A [`Correlator`] with the [`LagKernel::WignerD`] kernel and
/// a single origin ([`Origins::First`], freud's reference frame); `q` and
/// `−q` are the same rotation. [`Reduction::Mean`] (default) gives the
/// system-wide order parameter, [`Reduction::PerItem`] one curve per
/// particle. A zero quaternion has no rotation and is an error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotationalAutocorrelation {
    correlator: Correlator,
}

/// The data [`RotationalAutocorrelation`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct RotationalAutocorrelationInput<'a> {
    /// Orientation quaternions `(w, x, y, z)`, `T × N × 4`; frame 0 is the
    /// reference.
    pub orientations: ArrayView3<'a, F>,
    /// Frame spacing.
    pub dt: F,
}

impl RotationalAutocorrelation {
    /// `Ψ_l` over lags `0 … max_lag` frames from frame 0, averaged over
    /// particles.
    pub fn new(l: u32, max_lag: usize) -> Self {
        Self {
            correlator: Correlator::new(LagKernel::WignerD { l }, max_lag)
                .with_origins(Origins::First),
        }
    }

    /// Average over particles (`Mean`) or keep one curve per particle
    /// (`PerItem`).
    pub fn with_reduction(self, reduce: Reduction) -> Self {
        Self {
            correlator: self.correlator.with_reduction(reduce),
        }
    }

    /// The correlator this autocorrelation is.
    pub fn correlator(&self) -> &Correlator {
        &self.correlator
    }
}

impl Compute for RotationalAutocorrelation {
    type Input<'a> = RotationalAutocorrelationInput<'a>;
    type Output = TimeCorrelation;

    fn compute(
        &self,
        input: RotationalAutocorrelationInput<'_>,
    ) -> Result<TimeCorrelation, ComputeError> {
        self.correlator.compute(CorrelatorInput {
            a: input.orientations,
            b: None,
            dt: input.dt,
            boxes: None,
        })
    }
}

impl Stream for RotationalAutocorrelation {
    type Accumulator = CorrelatorAccumulator;

    fn accumulator(&self) -> CorrelatorAccumulator {
        self.correlator.accumulator()
    }
}

crate::compute::spec::compute_spec! {
    RotationalAutocorrelation as "RotationalAutocorrelation";
    params |s| {
        l: Int;
        max_lag: Int;
        reduction: Choice(Reduction::NAMES) = Reduction::default() => s.correlator.reduction();
    }
    inputs {
        orientations: Trajectory;
        dt: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::f64::consts::FRAC_PI_4;

    const TOL: F = 1e-12;

    /// A two-frame series: `reference` then `current`, one quaternion each
    /// per particle.
    fn pair(reference: &[[F; 4]], current: &[[F; 4]]) -> Array3<F> {
        let n = reference.len();
        Array3::from_shape_fn((2, n, 4), |(t, i, c)| {
            if t == 0 {
                reference[i][c]
            } else {
                current[i][c]
            }
        })
    }

    fn psi(l: u32, series: &Array3<F>, reduce: Reduction) -> TimeCorrelation {
        RotationalAutocorrelation::new(l, 1)
            .with_reduction(reduce)
            .compute(RotationalAutocorrelationInput {
                orientations: series.view(),
                dt: 1.0,
            })
            .unwrap()
    }

    #[test]
    fn surface_is_its_correlator() {
        let q = [FRAC_PI_4.cos(), 0.0, 0.0, FRAC_PI_4.sin()];
        let series = pair(&[[1.0, 0.0, 0.0, 0.0]; 3], &[q; 3]);
        let explicit = Correlator::new(LagKernel::WignerD { l: 2 }, 1)
            .with_origins(Origins::First)
            .with_reduction(Reduction::PerItem);
        let surface = RotationalAutocorrelation::new(2, 1).with_reduction(Reduction::PerItem);
        assert_eq!(*surface.correlator(), explicit);
        let via_correlator = explicit
            .compute(CorrelatorInput {
                a: series.view(),
                b: None,
                dt: 1.0,
                boxes: None,
            })
            .unwrap();
        assert_eq!(psi(2, &series, Reduction::PerItem), via_correlator);
    }

    #[test]
    fn unrotated_gives_psi_one() {
        let q = [[1.0_f64, 0.0, 0.0, 0.0]; 5];
        let series = pair(&q, &q);
        let per_item = psi(4, &series, Reduction::PerItem);
        for p in per_item.values.row(1) {
            assert!((p - 1.0).abs() < TOL);
        }
        assert!((psi(4, &series, Reduction::Mean).values[[1, 0]] - 1.0).abs() < TOL);
    }

    #[test]
    fn antipodal_quaternion_is_same_rotation() {
        // q and -q represent the same rotation → Ψ_ℓ = 1.
        let series = pair(&[[1.0, 0.0, 0.0, 0.0]], &[[-1.0, 0.0, 0.0, 0.0]]);
        assert!((psi(2, &series, Reduction::PerItem).values[[1, 0]] - 1.0).abs() < TOL);
    }

    #[test]
    fn ninety_degree_rotation_known_character() {
        // 90° rotation about z: quaternion (cos 45°, 0, 0, sin 45°).
        // χ^2(π/2) = sin(2.5 · π/2) / sin(π/4) = sin(5π/4) / sin(π/4)
        //         = (-√2/2) / (√2/2) = -1
        // Normalised: −1 / 5
        let q = [FRAC_PI_4.cos(), 0.0, 0.0, FRAC_PI_4.sin()];
        let series = pair(&[[1.0, 0.0, 0.0, 0.0]], &[q]);
        assert!((psi(2, &series, Reduction::PerItem).values[[1, 0]] - (-1.0 / 5.0)).abs() < 1e-12);
    }

    /// A zero quaternion used to read as the identity; it is now an error.
    #[test]
    fn a_zero_quaternion_is_rejected() {
        let series = pair(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0; 4]]);
        let err = RotationalAutocorrelation::new(2, 1)
            .compute(RotationalAutocorrelationInput {
                orientations: series.view(),
                dt: 1.0,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { field: "a", .. }));
    }

    #[test]
    fn orientations_must_be_quaternions() {
        let series = Array3::<F>::ones((2, 1, 3));
        let err = RotationalAutocorrelation::new(2, 1)
            .compute(RotationalAutocorrelationInput {
                orientations: series.view(),
                dt: 1.0,
            })
            .unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 4,
                got: 3,
                ..
            }
        ));
    }
}
