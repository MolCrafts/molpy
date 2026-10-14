//! Rotational autocorrelation for time-series of orientations.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::op::Quat;
use crate::op::{quat_conj, quat_mul, quat_norm};

/// Rotational autocorrelation calculator.
///
/// Mirrors `freud.order.RotationalAutocorrelation`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/RotationalAutocorrelation.cc)).
///
/// For each particle with reference orientation `q_ref` and current
/// orientation `q_t` the relative rotation is `q_rel = q_t · q_ref⁻¹`.
/// The Wigner-D character at angular order ℓ depends only on the rotation
/// angle `θ_rel`:
///
/// ```text
///   χ^ℓ(θ) = sin((ℓ + ½) θ) / sin(θ/2)        (for θ ≠ 0)
///   χ^ℓ(0) = 2ℓ + 1
/// ```
///
/// and is the trace of the (2ℓ+1)×(2ℓ+1) D-matrix. The output is the
/// **normalised** rotational autocorrelation per particle:
///
/// ```text
///   Ψ_ℓ(i) = χ^ℓ(θ_rel(i)) / (2ℓ + 1)
/// ```
///
/// so that `Ψ_ℓ = 1` for unrotated particles and decays toward 0 as the
/// ensemble decorrelates. The system-wide order parameter is the mean of
/// `Ψ_ℓ(i)` across particles.
#[derive(Debug, Clone, Copy)]
pub struct RotationalAutocorrelation {
    l: u32,
}

impl RotationalAutocorrelation {
    /// Hyperspherical correlation order `l`.
    pub fn new(l: u32) -> Self {
        Self { l }
    }

    pub fn l(&self) -> u32 {
        self.l
    }
}

#[inline]
fn rotation_angle(q: Quat) -> F {
    let n = quat_norm(q);
    if n == 0.0 {
        return 0.0;
    }
    // The w-component of a unit quaternion is cos(θ/2). Use |w| to fold
    // antipodal pairs onto the same rotation.
    2.0 * (q[0].abs() / n).clamp(0.0, 1.0).acos()
}

/// `χ^ℓ(θ) / (2ℓ+1)`. Continuous at `θ = 0`.
fn normalised_character(l: u32, theta: F) -> F {
    let two_lp1 = 2.0 * l as F + 1.0;
    if theta.abs() < 1e-12 {
        return 1.0;
    }
    let num = ((l as F + 0.5) * theta).sin();
    let den = (0.5 * theta).sin();
    if den == 0.0 { 1.0 } else { num / den / two_lp1 }
}

/// `Args` for [`RotationalAutocorrelation`]: per-particle reference and
/// current orientations. Both slices must have the same length.
pub struct RotationalAutocorrelationArgs<'a> {
    pub ref_orientations: &'a [Quat],
    pub orientations: &'a [Quat],
}

impl LegacyCompute for RotationalAutocorrelation {
    type Args<'a> = RotationalAutocorrelationArgs<'a>;
    type Output = Vec<RotationalAutocorrelationResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: RotationalAutocorrelationArgs<'a>,
    ) -> Result<Vec<RotationalAutocorrelationResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.ref_orientations.len() != args.orientations.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: args.ref_orientations.len(),
                got: args.orientations.len(),
                what: "RotationalAutocorrelation orientations",
            });
        }
        let n = args.orientations.len();
        let mut out = Vec::with_capacity(frames.len());
        for _ in frames {
            let mut psi = Vec::with_capacity(n);
            let mut sum: F = 0.0;
            for k in 0..n {
                let q_rel = quat_mul(args.orientations[k], quat_conj(args.ref_orientations[k]));
                let theta = rotation_angle(q_rel);
                let v = normalised_character(self.l, theta);
                psi.push(v);
                sum += v;
            }
            let mean = if n > 0 { sum / n as F } else { 0.0 };
            out.push(RotationalAutocorrelationResult {
                l: self.l,
                psi,
                mean,
            });
        }
        Ok(out)
    }
}

/// Per-frame rotational-autocorrelation result.
#[derive(Debug, Clone, Default)]
pub struct RotationalAutocorrelationResult {
    /// Angular momentum order.
    pub l: u32,
    /// Per-particle Ψ_ℓ values (length `N`).
    pub psi: Vec<F>,
    /// System-averaged Ψ_ℓ.
    pub mean: F,
}

impl ComputeResult for RotationalAutocorrelationResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;

    fn frame() -> Frame {
        Frame::new()
    }

    const TOL: F = 1e-12;

    #[test]
    fn unrotated_gives_psi_one() {
        let q = [[1.0_f64, 0.0, 0.0, 0.0]; 5];
        let r = &RotationalAutocorrelation::new(4)
            .compute(
                &[&frame()],
                RotationalAutocorrelationArgs {
                    ref_orientations: &q,
                    orientations: &q,
                },
            )
            .unwrap()[0];
        for p in &r.psi {
            assert!((p - 1.0).abs() < TOL);
        }
        assert!((r.mean - 1.0).abs() < TOL);
    }

    #[test]
    fn antipodal_quaternion_is_same_rotation() {
        // q and -q represent the same rotation → Ψ_ℓ = 1.
        let q = [1.0_f64, 0.0, 0.0, 0.0];
        let neg_q = [-1.0_f64, 0.0, 0.0, 0.0];
        let r = &RotationalAutocorrelation::new(2)
            .compute(
                &[&frame()],
                RotationalAutocorrelationArgs {
                    ref_orientations: &[q],
                    orientations: &[neg_q],
                },
            )
            .unwrap()[0];
        assert!((r.psi[0] - 1.0).abs() < TOL);
    }

    #[test]
    fn ninety_degree_rotation_known_character() {
        // 90° rotation about z: quaternion (cos 45°, 0, 0, sin 45°).
        // χ^2(π/2) = sin(2.5 · π/2) / sin(π/4) = sin(5π/4) / sin(π/4)
        //         = (-√2/2) / (√2/2) = -1
        // Normalised: −1 / 5
        let q = [
            std::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_4.sin(),
        ];
        let identity = [1.0_f64, 0.0, 0.0, 0.0];
        let r = &RotationalAutocorrelation::new(2)
            .compute(
                &[&frame()],
                RotationalAutocorrelationArgs {
                    ref_orientations: &[identity],
                    orientations: &[q],
                },
            )
            .unwrap()[0];
        assert!((r.psi[0] - (-1.0 / 5.0)).abs() < 1e-12);
    }

    #[test]
    fn mismatched_orientation_lengths_error() {
        let err = RotationalAutocorrelation::new(2)
            .compute(
                &[&frame()],
                RotationalAutocorrelationArgs {
                    ref_orientations: &[[1.0, 0.0, 0.0, 0.0]; 2],
                    orientations: &[[1.0, 0.0, 0.0, 0.0]; 3],
                },
            )
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
