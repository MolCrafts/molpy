//! The `(x, y, z)` coordinate system: the bond in the query particle's 3-D
//! frame.

use molrs::core::SimBox;
use molrs::op::{F, Quat, Vec3, quat_conj, rotate_by_quat};
use ndarray::{ArrayView2, Ix3};

use super::{PairCoordinate, Pmft, ortho_lengths, symmetric_axis};
use crate::compute::{Axis, ComputeError, require_quaternions};

/// The bond `r_j − r_i` rotated into the query particle's frame,
/// `q_i* · r · q_i` (Å), binned on
/// `[−x_max, x_max) × [−y_max, y_max) × [−z_max, z_max)`.
///
/// Mirrors `freud.pmft.PmftXyz`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXyz.cc)).
/// Orientations are quaternions `(w, x, y, z)`, `N × 4`, normalised (a zero
/// quaternion is an error); without them the bond is binned in the lab
/// frame. The reference measure is the box volume (orthorhombic boxes).
#[derive(Debug, Clone)]
pub struct Xyz {
    axes: [Axis; 3],
}

impl Pmft<Xyz> {
    /// Body-frame window `±x_max × ±y_max × ±z_max` (Å); `n_x × n_y × n_z`
    /// bins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite range
    /// or a zero bin count.
    pub fn new(
        x_max: F,
        y_max: F,
        z_max: F,
        n_x: usize,
        n_y: usize,
        n_z: usize,
    ) -> Result<Self, ComputeError> {
        Ok(Self {
            coordinate: Xyz {
                axes: [
                    symmetric_axis("Pmft<Xyz>::x_max", x_max, n_x)?,
                    symmetric_axis("Pmft<Xyz>::y_max", y_max, n_y)?,
                    symmetric_axis("Pmft<Xyz>::z_max", z_max, n_z)?,
                ],
            },
        })
    }
}

impl PairCoordinate for Xyz {
    type Orientation = Quat;
    type Orientations<'a> = ArrayView2<'a, F>;
    type Dim = Ix3;
    const DIM: usize = 3;
    const LAB_FRAME: Option<Quat> = Some([1.0, 0.0, 0.0, 0.0]);

    fn orientations(quaternions: ArrayView2<'_, F>) -> Result<Vec<Quat>, ComputeError> {
        require_quaternions(quaternions, "orientation quaternions")
    }

    fn axes(&self) -> &[Axis] {
        &self.axes
    }

    fn coords(&self, disp: Vec3, oi: Quat, _oj: Quat, out: &mut [F]) -> bool {
        // q⁻¹ · v · q, with q⁻¹ = q* for a unit quaternion.
        out.copy_from_slice(&rotate_by_quat(quat_conj(oi), disp));
        true
    }

    fn ref_measure(&self, simbox: &SimBox) -> Result<F, ComputeError> {
        let [lx, ly, lz] = ortho_lengths(simbox)?;
        Ok(lx * ly * lz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;
    use crate::compute::pmft::PmftInput;
    use crate::compute::pmft::tests::table;
    use ndarray::array;

    fn lab(p: Pmft<Xyz>, nl: &molrs::core::Neighbors) -> super::super::PmftResult<Xyz> {
        p.compute(PmftInput {
            pairs: nl,
            orientations: None,
        })
        .unwrap()
    }

    #[test]
    fn antiparallel_bonds_populate_symmetric_bins() {
        let nl = table(&[[0.0, 0.0, 0.0], [0.5, 0.4, 0.3]], 10.0, 10.0, 1.5);
        let r = &lab(Pmft::<Xyz>::new(1.0, 1.0, 1.0, 8, 8, 8).unwrap(), &nl);
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn out_of_range_dropped() {
        let nl = table(&[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], 10.0, 10.0, 5.0);
        let r = &lab(Pmft::<Xyz>::new(1.0, 1.0, 1.0, 4, 4, 4).unwrap(), &nl);
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    /// Two-particle golden: one pair in a 10³ box gives `ρ_ref = 2 / 1000`;
    /// the bond `(0.5, 0.4, 0.3)` lands in the `0.5³` bin `(3, 2, 2)` of a
    /// `4³` grid over `±1` (its reverse in `(1, 1, 1)`), so
    /// `PMF = −ln(8 / 0.002) = −ln(4000)`.
    #[test]
    fn two_particle_pmf_golden() {
        let nl = table(&[[0.0, 0.0, 0.0], [0.5, 0.4, 0.3]], 10.0, 10.0, 1.5);
        let r = lab(Pmft::<Xyz>::new(1.0, 1.0, 1.0, 4, 4, 4).unwrap(), &nl);
        assert_eq!(r.raw_counts[[3, 2, 2]], 1);
        assert_eq!(r.raw_counts[[1, 1, 1]], 1);
        assert!((r.pmf[[3, 2, 2]] + 4000.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn invalid_args_error() {
        assert!(Pmft::<Xyz>::new(0.0, 1.0, 1.0, 4, 4, 4).is_err());
        assert!(Pmft::<Xyz>::new(1.0, 1.0, 1.0, 0, 4, 4).is_err());
    }

    #[test]
    fn quaternion_orientations_rotate_bond_into_local_frame() {
        // Particle 0 at origin oriented at 90° about +z
        // (quaternion (cos 45°, 0, 0, sin 45°)). Particle 1 at lab
        // (1, 0, 0). In particle 0's local frame the bond runs along
        // its −y axis: rotating the +x lab vector by q⁻¹ gives (0, -1, 0).
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let h = std::f64::consts::FRAC_PI_4;
        let orient = array![[h.cos(), 0.0, 0.0, h.sin()], [1.0, 0.0, 0.0, 0.0]];

        let r = &Pmft::<Xyz>::new(1.5, 1.5, 1.5, 6, 6, 6)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orient.view()),
            })
            .unwrap();

        // Particle 0's contribution rotates into y < 0. Confirm the
        // overall y-binning shifted into the negative half.
        let mut found = false;
        for ix in 0..6 {
            for iy in 0..3 {
                for iz in 0..6 {
                    if r.raw_counts[[ix, iy, iz]] > 0 {
                        found = true;
                    }
                }
            }
        }
        assert!(found, "quaternion-rotated bond should land in y < 0 bins");
    }

    /// A zero quaternion used to read as the identity; it is no rotation.
    #[test]
    fn zero_quaternion_is_out_of_range() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let orient = array![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let err = Pmft::<Xyz>::new(1.5, 1.5, 1.5, 6, 6, 6)
            .unwrap()
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orient.view()),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }
}
