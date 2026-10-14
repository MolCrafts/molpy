//! The `(x, y, θ)` coordinate system: the bond in the query particle's 2-D
//! frame plus the relative orientation.

use std::f64::consts::TAU;

use molrs::core::SimBox;
use molrs::op::{F, Vec3};
use ndarray::{ArrayView1, Ix3};

use super::{PairCoordinate, Pmft, angle_axis, ortho_lengths, symmetric_axis, wrap_angle};
use crate::compute::{Axis, ComputeError, require_angles};

/// For a 2-D system of oriented particles: the bond `r_j − r_i` rotated
/// into the query particle's frame, `(x, y) = R(−θ_i) · (dx, dy)` (Å), and
/// the relative orientation `θ = wrap(θ_j − θ_i) ∈ [0, 2π)`, binned on
/// `[−x_max, x_max) × [−y_max, y_max) × [0, 2π)`; `dz` is ignored.
///
/// Mirrors `freud.pmft.PmftXyt`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXyt.cc)).
/// Orientations are planar angles (radians) and are required. The reference
/// measure is the box area times `2π` (orthorhombic boxes).
#[derive(Debug, Clone)]
pub struct Xyt {
    axes: [Axis; 3],
}

impl Pmft<Xyt> {
    /// Body-frame window `±x_max × ±y_max` (Å); `n_x × n_y × n_t` (x, y, θ)
    /// bins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite range
    /// or a zero bin count.
    pub fn new(
        x_max: F,
        y_max: F,
        n_x: usize,
        n_y: usize,
        n_t: usize,
    ) -> Result<Self, ComputeError> {
        Ok(Self {
            coordinate: Xyt {
                axes: [
                    symmetric_axis("Pmft<Xyt>::x_max", x_max, n_x)?,
                    symmetric_axis("Pmft<Xyt>::y_max", y_max, n_y)?,
                    angle_axis(n_t)?,
                ],
            },
        })
    }
}

impl PairCoordinate for Xyt {
    type Orientation = F;
    type Orientations<'a> = ArrayView1<'a, F>;
    type Dim = Ix3;
    const DIM: usize = 3;
    const LAB_FRAME: Option<F> = None;

    fn orientations(angles: ArrayView1<'_, F>) -> Result<Vec<F>, ComputeError> {
        require_angles(angles, "orientation angles")
    }

    fn axes(&self) -> &[Axis] {
        &self.axes
    }

    fn coords(&self, disp: Vec3, oi: F, oj: F, out: &mut [F]) -> bool {
        let (s, c) = oi.sin_cos();
        out[0] = c * disp[0] + s * disp[1];
        out[1] = -s * disp[0] + c * disp[1];
        out[2] = wrap_angle(oj - oi);
        true
    }

    fn ref_measure(&self, simbox: &SimBox) -> Result<F, ComputeError> {
        let [lx, ly, _] = ortho_lengths(simbox)?;
        Ok(lx * ly * TAU)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use crate::compute::pmft::PmftInput;
    use crate::compute::pmft::tests::table;
    use ndarray::array;

    #[test]
    fn zero_orientations_match_lab_frame() {
        // Two particles along +x, both orientations = 0 → local-frame bond
        // is (+1, 0); t = 0. The reverse contributes (−1, 0, 0).
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let orientations = array![0.0_f64, 0.0];
        let r = &Pmft::<Xyt>::new(2.0, 2.0, 8, 8, 8)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn rotated_query_rotates_local_bond() {
        // Particle 0 oriented at +π/2 (looking in +y direction), particle 1
        // at lab (1, 0, 0). In particle 0's frame, the bond runs along its
        // -y axis (i.e. local (x, y) = (0, -1)).
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let orientations = array![std::f64::consts::FRAC_PI_2, 0.0];
        let r = &Pmft::<Xyt>::new(2.0, 2.0, 8, 8, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        // The bond from particle 0 lands in a bin with y < 0 (negative side).
        // Find the bin holding particle-0's contribution.
        let mut found = false;
        for ix in 0..8 {
            for iy in 0..4 {
                // y < 0 → iy < 4
                for it in 0..4 {
                    if r.raw_counts[[ix, iy, it]] > 0 {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "rotated-frame bond should land in y < 0 bins");
    }

    /// Two-particle golden: one pair in a 10 × 10 box gives `ρ_ref = 2 /
    /// (100 · 2π)`; the `+x` bond with both particles at angle 0 lands at
    /// `(x, y, θ) = (1, 0, 0)`, in a bin of measure `1 · 1 · π/2`, so
    /// `PMF = −ln((2/π) / (1/(100π))) = −ln(200)`.
    #[test]
    fn two_particle_pmf_golden() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let orientations = array![0.0_f64, 0.0];
        let r = Pmft::<Xyt>::new(2.0, 2.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        assert_eq!(r.raw_counts[[3, 2, 0]], 1);
        assert_eq!(r.raw_counts[[1, 2, 0]], 1);
        assert!((r.pmf[[3, 2, 0]] + 200.0_f64.ln()).abs() < 1e-12);
    }

    /// The relative orientation is undefined without orientations.
    #[test]
    fn missing_orientations_are_bad_shape() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let err = Pmft::<Xyt>::new(2.0, 2.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }

    #[test]
    fn invalid_args_error() {
        assert!(Pmft::<Xyt>::new(0.0, 1.0, 4, 4, 4).is_err());
        assert!(Pmft::<Xyt>::new(1.0, 1.0, 0, 4, 4).is_err());
        assert!(Pmft::<Xyt>::new(1.0, 1.0, 4, 4, 0).is_err());
    }
}
