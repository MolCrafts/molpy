//! The `(r, θ₁, θ₂)` coordinate system: a 2-D bond as its length and its
//! angle in each particle's frame.

use std::f64::consts::{PI, TAU};

use molrs::core::SimBox;
use molrs::op::{F, Vec3};
use ndarray::{ArrayView1, Ix3};

use super::{PairCoordinate, Pmft, angle_axis, ortho_lengths, wrap_angle};
use crate::compute::{Axis, ComputeError, require_angles};

/// For a neighbor pair `(i, j)` with lab-frame bond `r_ij = r_j − r_i`:
///
/// - `r  = |r_ij|` (Å, in-plane)
/// - `θ₁ = wrap(θ_lab − θ_i)`, the bond in `i`'s frame
/// - `θ₂ = wrap(θ_lab + π − θ_j)`, the reverse bond in `j`'s frame
///
/// with `θ_lab = atan2(r_ij.y, r_ij.x)`, binned on
/// `[0, r_max) × [0, 2π) × [0, 2π)`; `r_ij.z` is ignored and a zero-length
/// bond (no angle) is not binned.
///
/// Mirrors `freud.pmft.PmftR12`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftR12.cc)).
/// Orientations are planar angles (radians) and are required. A bin's
/// measure is the polar area element times the two angle widths,
/// `2π r_c · dr · dθ₁ · dθ₂`; the reference measure is the box area times
/// `(2π)²` (orthorhombic boxes), so an ideal gas has `PMF = 0`.
#[derive(Debug, Clone)]
pub struct R12 {
    axes: [Axis; 3],
}

impl Pmft<R12> {
    /// Radial range `r_max` (Å); `n_r × n_t1 × n_t2` (r, θ₁, θ₂) bins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite `r_max`
    /// or a zero bin count.
    pub fn new(r_max: F, n_r: usize, n_t1: usize, n_t2: usize) -> Result<Self, ComputeError> {
        if !(r_max.is_finite() && r_max > 0.0) {
            return Err(ComputeError::OutOfRange {
                field: "Pmft<R12>::r_max",
                value: r_max.to_string(),
            });
        }
        Ok(Self {
            coordinate: R12 {
                axes: [
                    Axis::new(0.0, r_max, n_r)?,
                    angle_axis(n_t1)?,
                    angle_axis(n_t2)?,
                ],
            },
        })
    }
}

impl PairCoordinate for R12 {
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
        let r = (disp[0] * disp[0] + disp[1] * disp[1]).sqrt();
        if r == 0.0 {
            return false;
        }
        let theta_lab = disp[1].atan2(disp[0]);
        out[0] = r;
        out[1] = wrap_angle(theta_lab - oi);
        out[2] = wrap_angle(theta_lab + PI - oj);
        true
    }

    fn ref_measure(&self, simbox: &SimBox) -> Result<F, ComputeError> {
        let [lx, ly, _] = ortho_lengths(simbox)?;
        Ok(lx * ly * TAU * TAU)
    }

    fn bin_measure(&self, center: &[F]) -> F {
        let widths: F = self.axes.iter().map(Axis::width).product();
        TAU * center[0] * widths
    }
}

crate::compute::spec::compute_spec! {
    Pmft<R12> as "PmftR12";
    params |s| {
        r_max: Float;
        n_r: Int;
        n_t1: Int;
        n_t2: Int;
    }
    inputs {
        pairs: Neighbors;
        orientations: Angles, optional;
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
    fn two_particles_land_in_one_bin() {
        // Two particles along +x; both orientations = 0. Bond from i to j
        // has θ_lab = 0 → t1 = 0, t2 = π. The symmetric reverse contributes
        // a second count at t1 = π, t2 = 0.
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let orientations = array![0.0_f64, 0.0];
        let r = &Pmft::<R12>::new(2.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    /// Two-particle golden: the bond `(r, θ₁, θ₂) = (1, 0, π)` sits in the
    /// bin centred at `r_c = 1.25` of measure `2π · 1.25 · 0.5 · (π/2)²`;
    /// one pair in a 10 × 10 box gives `ρ_ref = 2 / (100 · 4π²)`, so
    /// `PMF = −ln(640/π)`.
    #[test]
    fn two_particle_pmf_golden() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let orientations = array![0.0_f64, 0.0];
        let r = Pmft::<R12>::new(2.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        assert_eq!(r.raw_counts[[2, 0, 2]], 1);
        assert_eq!(r.raw_counts[[2, 2, 0]], 1);
        assert!((r.pmf[[2, 0, 2]] + (640.0 / PI).ln()).abs() < 1e-12);
    }

    /// The bin measures tile the coordinate space: they sum to the disc
    /// area `π r_max²` times `(2π)²` — the Jacobian the reference measure
    /// assumes (the old per-bin measure lacked the `dr` factor).
    #[test]
    fn bin_measures_tile_the_disc() {
        let p = Pmft::<R12>::new(2.0, 8, 3, 5).unwrap();
        let c = p.coordinate();
        let total: F = c.axes()[0]
            .centers()
            .iter()
            .map(|&r| c.bin_measure(&[r, 0.0, 0.0]) * 15.0)
            .sum();
        assert!((total - PI * 4.0 * TAU * TAU).abs() < 1e-9);
    }

    #[test]
    fn out_of_range_dropped() {
        let nl = table(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 10.0, 1.0, 5.0);
        let orientations = array![0.0_f64, 0.0];
        let r = &Pmft::<R12>::new(1.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orientations.view()),
            })
            .unwrap();
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(Pmft::<R12>::new(0.0, 4, 4, 4).is_err());
        assert!(Pmft::<R12>::new(1.0, 0, 4, 4).is_err());
    }

    /// The body-frame angles are undefined without orientations.
    #[test]
    fn missing_orientations_are_bad_shape() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.0, 1.5);
        let err = Pmft::<R12>::new(2.0, 4, 4, 4)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }
}
