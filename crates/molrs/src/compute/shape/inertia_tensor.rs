//! Per-group moment of inertia tensor.

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array3, ArrayView1, ArrayView2};

use super::group_reduce::GroupReduce;
use crate::compute::{Compute, ComputeError, Partition};

/// The moment of inertia tensor of every group of a [`Partition`] about its
/// centre of mass,
///
/// `I_ab = Σᵢ mᵢ (|sᵢ|² δ_ab − sᵢ,a sᵢ,b) = tr(S) δ_ab − S_ab`, with
/// `S = Σᵢ mᵢ sᵢ sᵢᵀ` the mass-weighted second moment and `sᵢ` the
/// minimum-image unwrapped displacement from the centre of mass.
///
/// Configuration: none. Data: one snapshot's `N × 3` positions, its box (or
/// `None`), the partition and one mass per item; the centre of mass is the
/// one [`CenterOfMass`](crate::compute::CenterOfMass) reports. Many snapshots
/// go through [`Batch`](crate::compute::Batch).
#[derive(Debug, Clone, Copy, Default)]
pub struct InertiaTensor;

/// The data an [`InertiaTensor`] reduces.
#[derive(Clone, Copy)]
pub struct InertiaTensorInput<'a> {
    /// Positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box whose minimum image unwraps each group; `None` = no wrapping.
    pub simbox: Option<&'a SimBox>,
    /// Which group each item belongs to.
    pub partition: &'a (dyn Partition + Sync),
    /// One non-negative mass per item, e.g. amu.
    pub masses: ArrayView1<'a, F>,
}

/// Per-group inertia tensors of one snapshot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct InertiaTensorResult {
    /// The group key of each tensor, ascending.
    pub groups: Vec<u64>,
    /// The tensors, `G × 3 × 3`, in mass × position² (amu·Å² for amu, Å).
    pub tensors: Array3<F>,
}

impl InertiaTensor {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for InertiaTensor {
    type Input<'a> = InertiaTensorInput<'a>;
    type Output = InertiaTensorResult;

    /// # Errors
    /// As [`CenterOfMass`](crate::compute::CenterOfMass).
    fn compute(&self, input: InertiaTensorInput<'_>) -> Result<InertiaTensorResult, ComputeError> {
        let reduce = GroupReduce::new(input.positions, input.simbox, input.partition)?;
        let centroids = reduce.centroids(input.masses, "masses")?;
        let moments = reduce.second_moments(input.masses, &centroids);
        let tensors = Array3::from_shape_fn((moments.len(), 3, 3), |(g, a, b)| {
            let s = &moments[g];
            let trace = if a == b {
                s[0][0] + s[1][1] + s[2][2]
            } else {
                0.0
            };
            trace - s[a][b]
        });
        Ok(InertiaTensorResult {
            groups: reduce.keys().to_vec(),
            tensors,
        })
    }
}

crate::compute::spec::compute_spec! {
    InertiaTensor as "InertiaTensor";
    params |s| {}
    inputs {
        positions: Positions;
        simbox: SimBox, optional;
        partition: Partition;
        masses: Scalars;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::{Array2, arr1};

    fn inertia(rows: &[[F; 3]], box_len: F, masses: &[F]) -> InertiaTensorResult {
        let positions = Array2::from_shape_fn((rows.len(), 3), |(i, d)| rows[i][d]);
        let one_group: Vec<u8> = vec![0; rows.len()];
        let b = SimBox::new(Cell::Lengths([box_len; 3]), [0.0, 0.0, 0.0], [false; 3]).unwrap();
        InertiaTensor::new()
            .compute(InertiaTensorInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &one_group,
                masses: arr1(masses).view(),
            })
            .unwrap()
    }

    #[test]
    fn zero_for_coincident() {
        let t = inertia(&[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], 6.0, &[2.0, 5.0]);
        assert!(t.tensors.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn single_particle_zero() {
        let t = inertia(&[[5.0, 5.0, 5.0]], 10.0, &[3.0]);
        assert!(t.tensors.iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn weighted_off_center() {
        let t = inertia(&[[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]], 6.0, &[3.0, 4.0]);

        let expected: [[f64; 3]; 3] = [
            [0.0171429, -0.0171429, 0.0],
            [-0.0171429, 0.0171429, 0.0],
            [0.0, 0.0, 0.0342857],
        ];
        for (a, row) in expected.iter().enumerate() {
            for (b, &e) in row.iter().enumerate() {
                assert!(
                    (t.tensors[[0, a, b]] as f64 - e).abs() < 1e-4,
                    "I[{a}][{b}] = {}, expected {e}",
                    t.tensors[[0, a, b]]
                );
            }
        }
    }

    #[test]
    fn symmetric() {
        let t = inertia(
            &[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]],
            10.0,
            &[1.0, 1.0, 1.0],
        );
        for a in 0..3 {
            for b in 0..3 {
                assert!((t.tensors[[0, a, b]] - t.tensors[[0, b, a]]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_along_x() {
        let t = inertia(&[[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]], 10.0, &[1.0, 1.0]);
        assert!(t.tensors[[0, 0, 0]].abs() < 1e-5);
        assert!((t.tensors[[0, 1, 1]] - 2.0).abs() < 1e-5);
        assert!((t.tensors[[0, 2, 2]] - 2.0).abs() < 1e-5);
    }
}
