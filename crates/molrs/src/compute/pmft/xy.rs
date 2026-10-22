//! The `(x, y)` coordinate system: the bond in the query particle's 2-D frame.

use molrs::core::SimBox;
use molrs::op::{F, Vec3};
use ndarray::{ArrayView1, Ix2};

use super::{PairCoordinate, Pmft, ortho_lengths, symmetric_axis};
use crate::compute::{Axis, ComputeError, require_angles};

/// The bond `r_j − r_i` rotated into the query particle's planar frame,
/// `(x, y) = R(−θ_i) · (dx, dy)` (Å), binned on
/// `[−x_max, x_max) × [−y_max, y_max)`; `dz` is ignored.
///
/// Mirrors `freud.pmft.PmftXy`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/pmft/PmftXy.cc)).
/// Orientations are planar angles (radians); without them the bond is
/// binned in the lab frame. The reference measure is the box area
/// `L_x · L_y` (orthorhombic boxes).
#[derive(Debug, Clone)]
pub struct Xy {
    axes: [Axis; 2],
}

impl Pmft<Xy> {
    /// Body-frame window `±x_max × ±y_max` (Å); `n_x × n_y` bins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite range
    /// or a zero bin count.
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize) -> Result<Self, ComputeError> {
        Ok(Self {
            coordinate: Xy {
                axes: [
                    symmetric_axis("Pmft<Xy>::x_max", x_max, n_x)?,
                    symmetric_axis("Pmft<Xy>::y_max", y_max, n_y)?,
                ],
            },
        })
    }
}

impl PairCoordinate for Xy {
    type Orientation = F;
    type Orientations<'a> = ArrayView1<'a, F>;
    type Dim = Ix2;
    const DIM: usize = 2;
    const LAB_FRAME: Option<F> = Some(0.0);

    fn orientations(angles: ArrayView1<'_, F>) -> Result<Vec<F>, ComputeError> {
        require_angles(angles, "orientation angles")
    }

    fn axes(&self) -> &[Axis] {
        &self.axes
    }

    fn coords(&self, disp: Vec3, oi: F, _oj: F, out: &mut [F]) -> bool {
        let (s, c) = oi.sin_cos();
        out[0] = c * disp[0] + s * disp[1];
        out[1] = -s * disp[0] + c * disp[1];
        true
    }

    fn ref_measure(&self, simbox: &SimBox) -> Result<F, ComputeError> {
        let [lx, ly, _] = ortho_lengths(simbox)?;
        Ok(lx * ly)
    }
}

crate::compute::spec::compute_spec! {
    Pmft<Xy> as "PmftXy";
    params |s| {
        x_max: Float;
        y_max: Float;
        n_x: Int;
        n_y: Int;
    }
    inputs {
        pairs: Neighbors;
        orientations: Angles, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::pmft::PmftInput;
    use crate::compute::pmft::tests::table;
    use crate::compute::{Batch, Compute};
    use ndarray::array;

    fn pmft(n: usize) -> Pmft<Xy> {
        Pmft::<Xy>::new(2.0, 2.0, n, n).unwrap()
    }

    #[test]
    fn two_particles_land_in_symmetric_bins() {
        // Two particles separated by (1, 0). In a self-query NL the pair
        // is visited once but accumulated symmetrically, so we expect
        // count = 1 in (+1, 0) bin and count = 1 in (−1, 0) bin.
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let r = &pmft(8)
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap();
        let total: u64 = r.raw_counts.iter().copied().sum();
        assert_eq!(total, 2);
        // Bin for (+1, 0): dx = 0.5 → bx = ((1 + 2)/0.5) = 6.
        let bx_pos = ((1.0_f64 + 2.0) / 0.5) as usize;
        let by_zero = ((0.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_pos, by_zero]], 1);
        let bx_neg = ((-1.0_f64 + 2.0) / 0.5) as usize;
        assert_eq!(r.raw_counts[[bx_neg, by_zero]], 1);
    }

    #[test]
    fn pmf_is_finite_only_in_occupied_bins() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let r = &pmft(4)
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap();
        let mut finite_count = 0;
        for v in r.pmf.iter() {
            if v.is_finite() {
                finite_count += 1;
            }
        }
        assert_eq!(finite_count, 2);
    }

    /// Two-particle golden: one pair in a 10 × 10 box gives `ρ_ref = 2 /
    /// 100` and, in its 1 Å² bin, `ρ = 1`, so `PMF = −ln(50)`.
    #[test]
    fn two_particle_pmf_golden() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let r = pmft(4)
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap();
        assert_eq!(r.density[[3, 2]], 1.0);
        assert!((r.pmf[[3, 2]] + 50.0_f64.ln()).abs() < 1e-12);
        assert_eq!(r.edges.len(), 2);
        assert_eq!(r.edges[0].to_vec(), vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn out_of_range_pairs_dropped() {
        // Bond length 5 → outside the (x_max=2, y_max=2) box → no count.
        let nl = table(&[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], 10.0, 10.0, 6.0);
        let r = &pmft(8)
            .compute(PmftInput {
                pairs: &nl,
                orientations: None,
            })
            .unwrap();
        assert_eq!(r.raw_counts.iter().copied().sum::<u64>(), 0);
    }

    #[test]
    fn invalid_args_error() {
        assert!(Pmft::<Xy>::new(0.0, 2.0, 4, 4).is_err());
        assert!(Pmft::<Xy>::new(2.0, 2.0, 0, 4).is_err());
    }

    #[test]
    fn orientations_rotate_bond_into_local_frame() {
        // Particle 0 at origin, oriented at +π/2 (looking in +y). Particle 1
        // at (1, 0, 0). In particle 0's local frame the bond is (0, -1, 0),
        // so it lands in a `y < 0` bin. Without orientations the same bond
        // would land in `x > 0`.
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let orient = array![std::f64::consts::FRAC_PI_2, 0.0];

        let with_orient = &pmft(8)
            .compute(PmftInput {
                pairs: &nl,
                orientations: Some(orient.view()),
            })
            .unwrap();

        // The i-side contribution from particle 0 must be in a y < 0 bin
        // when orientations are applied.
        let mut found_y_negative = false;
        for ix in 0..8 {
            for iy in 0..4 {
                if with_orient.raw_counts[[ix, iy]] > 0 {
                    found_y_negative = true;
                }
            }
        }
        assert!(
            found_y_negative,
            "rotated-frame bond should land in y < 0 bins"
        );
    }

    #[test]
    fn orientations_must_be_one_per_particle_and_finite() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let short = array![0.0];
        assert!(matches!(
            pmft(4).compute(PmftInput {
                pairs: &nl,
                orientations: Some(short.view()),
            }),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        let nan = array![0.0, F::NAN];
        assert!(matches!(
            pmft(4).compute(PmftInput {
                pairs: &nl,
                orientations: Some(nan.view()),
            }),
            Err(ComputeError::NonFinite { .. })
        ));
    }

    /// [`Batch`] must match one call per snapshot exactly, in input order.
    #[test]
    fn parallel_matches_serial() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 10.0, 1.5);
        let p = pmft(8);
        let input = PmftInput {
            pairs: &nl,
            orientations: None,
        };
        let solo = p.compute(input).unwrap();
        let par = Batch(p).compute(vec![input; 2]).unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].raw_counts, solo.raw_counts);
        assert_eq!(par[1].raw_counts, solo.raw_counts);
    }
}
