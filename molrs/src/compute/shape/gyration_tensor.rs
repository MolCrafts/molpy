//! Per-group gyration tensor and radius of gyration.

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array1, Array3, ArrayView1, ArrayView2};

use super::group_reduce::GroupReduce;
use crate::compute::{Compute, ComputeError, Partition};

/// The weighted gyration tensor of every group of a [`Partition`],
///
/// `G_ab = Σᵢ wᵢ sᵢ,a sᵢ,b / Σᵢ wᵢ`, with `sᵢ = rᵢ − r_c` the minimum-image
/// unwrapped displacement from the group's weighted centre
/// `r_c = Σᵢ wᵢ rᵢ / Σᵢ wᵢ`.
///
/// Configuration: none. Data: one snapshot's `N × 3` positions, its box (or
/// `None`), the partition and one weight per item. The weighting is the
/// caller's explicit choice: masses give the mass-weighted tensor (freud's
/// `ClusterProperties`), `Array1::ones(n)` the number-averaged one. Each group
/// is unwrapped about its first member, as in
/// [`CenterOfMass`](crate::compute::CenterOfMass). The radius of gyration is
/// [`GyrationTensorResult::radius_of_gyration`], `√tr G` of the same
/// weighting. Many snapshots go through [`Batch`](crate::compute::Batch).
///
/// # References
///
/// Theodorou & Suter, *Macromolecules* **18**, 1206 (1985).
#[derive(Debug, Clone, Copy, Default)]
pub struct GyrationTensor;

/// The data a [`GyrationTensor`] reduces.
#[derive(Clone, Copy)]
pub struct GyrationTensorInput<'a> {
    /// Positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box whose minimum image unwraps each group; `None` = no wrapping.
    pub simbox: Option<&'a SimBox>,
    /// Which group each item belongs to.
    pub partition: &'a (dyn Partition + Sync),
    /// One non-negative weight per item (masses, or ones).
    pub weights: ArrayView1<'a, F>,
}

/// Per-group gyration tensors of one snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct GyrationTensorResult {
    /// The group key of each tensor, ascending.
    pub groups: Vec<u64>,
    /// The tensors, `G × 3 × 3`, in the square of the position unit.
    pub tensors: Array3<F>,
}

impl GyrationTensorResult {
    /// The radius of gyration of each group, `R_g = √tr G` (position unit).
    pub fn radius_of_gyration(&self) -> Array1<F> {
        Array1::from_iter(
            self.tensors
                .outer_iter()
                .map(|g| (g[[0, 0]] + g[[1, 1]] + g[[2, 2]]).sqrt()),
        )
    }
}

impl GyrationTensor {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for GyrationTensor {
    type Input<'a> = GyrationTensorInput<'a>;
    type Output = GyrationTensorResult;

    /// # Errors
    /// As [`CenterOfMass`](crate::compute::CenterOfMass), for the weights.
    fn compute(
        &self,
        input: GyrationTensorInput<'_>,
    ) -> Result<GyrationTensorResult, ComputeError> {
        let reduce = GroupReduce::new(input.positions, input.simbox, input.partition)?;
        let centroids = reduce.centroids(input.weights, "weights")?;
        let moments = reduce.second_moments(input.weights, &centroids);
        let tensors = Array3::from_shape_fn((moments.len(), 3, 3), |(g, a, b)| {
            moments[g][a][b] / centroids.totals[g]
        });
        Ok(GyrationTensorResult {
            groups: reduce.keys().to_vec(),
            tensors,
        })
    }
}

crate::compute::spec::compute_spec! {
    GyrationTensor as "GyrationTensor";
    params |s| {}
    inputs {
        positions: Positions;
        simbox: SimBox, optional;
        partition: Partition;
        weights: Scalars;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{InertiaTensor, InertiaTensorInput};
    use ndarray::{Array2, arr1, arr2, array};

    fn cube(len: F) -> SimBox {
        SimBox::cube(len, array![0.0, 0.0, 0.0], [false; 3]).unwrap()
    }

    fn gyration(rows: &[[F; 3]], box_len: F, weights: &[F]) -> GyrationTensorResult {
        let positions = Array2::from_shape_fn((rows.len(), 3), |(i, d)| rows[i][d]);
        let one_group: Vec<u8> = vec![0; rows.len()];
        let b = cube(box_len);
        GyrationTensor::new()
            .compute(GyrationTensorInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &one_group,
                weights: arr1(weights).view(),
            })
            .unwrap()
    }

    /// Number-averaged (uniform weights) — the former unweighted
    /// `GyrationTensor` about the geometric centre.
    fn uniform(rows: &[[F; 3]], box_len: F) -> GyrationTensorResult {
        gyration(rows, box_len, &vec![1.0; rows.len()])
    }

    #[test]
    fn zero_for_coincident() {
        let g = uniform(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 6.0);
        assert!(g.tensors.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn off_center_cluster() {
        let g = uniform(&[[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]], 6.0);
        let expected = [
            [0.0025, 0.0025, 0.0],
            [0.0025, 0.0025, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for (a, row) in expected.iter().enumerate() {
            for (b, &e) in row.iter().enumerate() {
                assert!(
                    (g.tensors[[0, a, b]] - e as F).abs() < 1e-5,
                    "G[{a}][{b}] = {}, expected {e}",
                    g.tensors[[0, a, b]],
                );
            }
        }
    }

    #[test]
    fn symmetric() {
        let g = uniform(&[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]], 10.0);
        for a in 0..3 {
            for b in 0..3 {
                assert!((g.tensors[[0, a, b]] - g.tensors[[0, b, a]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_particles_along_x() {
        let g = uniform(&[[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]], 10.0);
        assert!((g.tensors[[0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!(g.tensors[[0, 1, 1]].abs() < 1e-5);
        assert!(g.tensors[[0, 0, 1]].abs() < 1e-5);
    }

    #[test]
    fn single_particle_zero() {
        let g = uniform(&[[5.0, 5.0, 5.0]], 10.0);
        assert!(g.tensors.iter().all(|v| v.abs() < 1e-10));
    }

    // --- the former `RadiusOfGyration`: √tr G of the same weighting ---

    #[test]
    fn rg_zero_for_coincident() {
        let g = gyration(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 6.0, &[2.0, 5.0]);
        assert!(g.radius_of_gyration()[0].abs() < 1e-10);
    }

    #[test]
    fn rg_single_particle_zero() {
        let g = uniform(&[[5.0, 5.0, 5.0]], 10.0);
        assert!(g.radius_of_gyration()[0].abs() < 1e-10);
    }

    #[test]
    fn rg_weighted_off_center() {
        let g = gyration(&[[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]], 6.0, &[3.0, 4.0]);
        let rg = g.radius_of_gyration()[0];
        assert!(
            (rg - 0.0699854212).abs() < 1e-4,
            "rg = {rg}, expected ~0.0700"
        );
    }

    #[test]
    fn rg_two_equal_mass_along_x() {
        let g = uniform(&[[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]], 10.0);
        assert!((g.radius_of_gyration()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rg_equals_trace_of_inertia_over_2m() {
        let positions = arr2(&[[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]]);
        let masses = arr1(&[3.0, 4.0]);
        let one_group: Vec<u8> = vec![0, 0];
        let b = cube(6.0);

        let rg = GyrationTensor::new()
            .compute(GyrationTensorInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &one_group,
                weights: masses.view(),
            })
            .unwrap()
            .radius_of_gyration();
        let inertia = InertiaTensor::new()
            .compute(InertiaTensorInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &one_group,
                masses: masses.view(),
            })
            .unwrap();

        let trace =
            inertia.tensors[[0, 0, 0]] + inertia.tensors[[0, 1, 1]] + inertia.tensors[[0, 2, 2]];
        let rg_from_trace = (trace / (2.0 * masses.sum())).sqrt();
        assert!(
            (rg[0] - rg_from_trace).abs() < 1e-5,
            "rg={}, from_trace={}",
            rg[0],
            rg_from_trace
        );
    }

    /// Behaviour change 2: the weighting is explicit. Masses 3 and 1 at
    /// x = 0 and 4 put the centre at 1, so `G_xx = (3·1 + 1·9)/4 = 3`; the
    /// number average about x = 2 is `4`.
    #[test]
    fn the_weighting_is_the_callers() {
        let rows = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]];
        assert_eq!(gyration(&rows, 10.0, &[3.0, 1.0]).tensors[[0, 0, 0]], 3.0);
        assert_eq!(uniform(&rows, 10.0).tensors[[0, 0, 0]], 4.0);
    }
}
