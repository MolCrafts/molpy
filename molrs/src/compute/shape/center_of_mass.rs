//! Per-group centre of mass under the minimum-image convention.

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::group_reduce::GroupReduce;
use crate::compute::{Compute, ComputeError, Partition};

/// The mass-weighted centre `Σᵢ mᵢ rᵢ / Σᵢ mᵢ` of every group of a
/// [`Partition`].
///
/// Configuration: none. Data: one snapshot's `N × 3` positions, its box (or
/// `None` for open boundaries), the partition and one mass per item. Each
/// group is unwrapped about its first member under the box's minimum image
/// before averaging, so a group that straddles a periodic boundary has its
/// centre where the group is, not in the middle of the box; the centre is
/// not wrapped back into the box.
///
/// It is also the one entry for the related centres:
/// - the **geometric centre** of each cluster is this with uniform masses
///   (`Array1::ones(n)`);
/// - the **centre-of-mass velocity** is this on the `N × 3` velocities with
///   no box and a single-group partition.
///
/// Many snapshots go through [`Batch`](crate::compute::Batch).
///
/// ```
/// use molrs::compute::{CenterOfMass, CenterOfMassInput, Compute};
/// use ndarray::array;
///
/// let positions = array![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
/// let masses = array![3.0, 1.0, 2.0];
/// let mol_id: Vec<u32> = vec![0, 0, 1];
/// let com = CenterOfMass::new().compute(CenterOfMassInput {
///     positions: positions.view(),
///     simbox: None,
///     partition: &mol_id,
///     masses: masses.view(),
/// })?;
/// assert_eq!(com.groups, vec![0, 1]);
/// assert_eq!(com.centers[[0, 0]], 1.5);
/// assert_eq!(com.total_masses.to_vec(), vec![4.0, 2.0]);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct CenterOfMass;

/// The data a [`CenterOfMass`] reduces.
#[derive(Clone, Copy)]
pub struct CenterOfMassInput<'a> {
    /// Positions (or velocities), `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box whose minimum image unwraps each group; `None` = no wrapping.
    pub simbox: Option<&'a SimBox>,
    /// Which group each item belongs to.
    pub partition: &'a (dyn Partition + Sync),
    /// One non-negative mass per item (any unit; only ratios move the centre).
    pub masses: ArrayView1<'a, F>,
}

/// Per-group centres of mass of one snapshot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CenterOfMassResult {
    /// The group key of each row, ascending.
    pub groups: Vec<u64>,
    /// The centres, `G × 3`, in the unit of the positions.
    pub centers: Array2<F>,
    /// The total mass of each group.
    pub total_masses: Array1<F>,
}

impl CenterOfMass {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for CenterOfMass {
    type Input<'a> = CenterOfMassInput<'a>;
    type Output = CenterOfMassResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] when the positions are not `N × 3`
    /// or the partition or masses do not have one entry per item;
    /// [`ComputeError::NonFinite`] for a non-finite position or mass;
    /// [`ComputeError::OutOfRange`] for a negative mass or a group of total
    /// mass zero.
    fn compute(&self, input: CenterOfMassInput<'_>) -> Result<CenterOfMassResult, ComputeError> {
        let reduce = GroupReduce::new(input.positions, input.simbox, input.partition)?;
        let centroids = reduce.centroids(input.masses, "masses")?;
        Ok(CenterOfMassResult {
            groups: reduce.keys().to_vec(),
            centers: centroids.centers,
            total_masses: centroids.totals,
        })
    }
}

crate::compute::spec::compute_spec! {
    CenterOfMass as "CenterOfMass";
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
    use crate::compute::fixtures::neighbors_of;
    use crate::compute::{Cluster, ClusterInput, ClusterResult};
    use ndarray::{Array1, arr1, arr2, array};

    fn cube(len: F, pbc: [bool; 3]) -> SimBox {
        SimBox::cube(len, array![0.0, 0.0, 0.0], pbc).unwrap()
    }

    fn manual_clusters(idx: &[i64]) -> ClusterResult {
        let nc = (*idx.iter().max().unwrap_or(&-1) + 1).max(0) as usize;
        let mut sizes = vec![0usize; nc];
        for &c in idx {
            if c >= 0 {
                sizes[c as usize] += 1;
            }
        }
        ClusterResult {
            cluster_idx: Array1::from_vec(idx.to_vec()),
            n_clusters: nc,
            cluster_sizes: sizes,
            cluster_keys: vec![],
        }
    }

    fn com(
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
        partition: &(dyn Partition + Sync),
        masses: ArrayView1<'_, F>,
    ) -> Result<CenterOfMassResult, ComputeError> {
        CenterOfMass::new().compute(CenterOfMassInput {
            positions,
            simbox,
            partition,
            masses,
        })
    }

    /// Uniform masses: the geometric centre (the former `ClusterCenters`).
    fn centers(
        positions: ArrayView2<'_, F>,
        simbox: Option<&SimBox>,
        partition: &(dyn Partition + Sync),
    ) -> CenterOfMassResult {
        let ones = Array1::<F>::ones(positions.nrows());
        com(positions, simbox, partition, ones.view()).unwrap()
    }

    #[test]
    fn uniform_mass_equals_geometric_center() {
        let pos = arr2(&[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]);
        let b = cube(10.0, [false; 3]);
        let r = com(
            pos.view(),
            Some(&b),
            &manual_clusters(&[0, 0]),
            arr1(&[1.0, 1.0]).view(),
        )
        .unwrap();
        assert!((r.centers[[0, 0]] - 2.0).abs() < 1e-5);
        assert!((r.total_masses[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_shifts_toward_heavy() {
        let pos = arr2(&[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]);
        let b = cube(10.0, [false; 3]);
        let r = com(
            pos.view(),
            Some(&b),
            &manual_clusters(&[0, 0]),
            arr1(&[3.0, 1.0]).view(),
        )
        .unwrap();
        assert!((r.centers[[0, 0]] - 1.5).abs() < 1e-5);
        assert!((r.total_masses[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn two_clusters_weighted() {
        let pos = arr2(&[
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ]);
        let b = cube(6.0, [false; 3]);
        let r = com(
            pos.view(),
            Some(&b),
            &manual_clusters(&[0, 0, 1, 1]),
            arr1(&[1.0, 2.0, 3.0, 4.0]).view(),
        )
        .unwrap();
        assert!((r.centers[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((r.total_masses[0] - 3.0).abs() < 1e-5);
        assert!((r.centers[[1, 0]] - 6.6 / 7.0).abs() < 1e-4);
        assert!((r.total_masses[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn periodic_wrapping() {
        let pos = arr2(&[[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]]);
        let b = cube(10.0, [true; 3]);
        let r = com(
            pos.view(),
            Some(&b),
            &manual_clusters(&[0, 0]),
            arr1(&[1.0, 3.0]).view(),
        )
        .unwrap();
        let cx = r.centers[[0, 0]];
        assert!(
            (cx - 9.5).abs() < 0.5 || (cx + 0.5).abs() < 0.5,
            "COM should be near 9.5 (or -0.5), got {cx}"
        );
    }

    #[test]
    fn masses_dimension_mismatch() {
        let pos = arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let err = com(
            pos.view(),
            None,
            &manual_clusters(&[0, 0]),
            arr1(&[1.0]).view(),
        )
        .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    // --- the former `ClusterCenters`: uniform masses ---

    #[test]
    fn coincident_and_offset_clusters() {
        let rows = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ];
        let b = cube(6.0, [false; 3]);
        let pairs = neighbors_of(&rows, &b, 0.5);
        let cl = Cluster::new()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: None,
            })
            .unwrap();
        assert_eq!(cl.n_clusters, 2);

        let (ca, cb) = if cl.cluster_idx[0] == 0 {
            (0, 1)
        } else {
            (1, 0)
        };
        let c = centers(arr2(&rows).view(), Some(&b), &cl);

        assert!((c.centers[[ca, 0]] - 1.0).abs() < 1e-5);
        assert!((c.centers[[ca, 1]] - 1.0).abs() < 1e-5);
        assert!((c.centers[[cb, 0]] - 0.95).abs() < 1e-5);
        assert!((c.centers[[cb, 1]] - 2.95).abs() < 1e-5);
    }

    #[test]
    fn mic_wrapping_across_boundary() {
        let rows = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let b = cube(10.0, [true; 3]);
        let pairs = neighbors_of(&rows, &b, 2.0);
        let cl = Cluster::new()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: None,
            })
            .unwrap();
        assert_eq!(cl.n_clusters, 1);

        let c = centers(arr2(&rows).view(), Some(&b), &cl);
        let cx = c.centers[[0, 0]];
        assert!(
            !(1.0..=9.0).contains(&cx),
            "center should wrap near boundary, got {cx}"
        );
    }

    #[test]
    fn single_particle() {
        let b = cube(10.0, [false; 3]);
        let c = centers(
            arr2(&[[3.0, 4.0, 5.0]]).view(),
            Some(&b),
            &manual_clusters(&[0]),
        );
        assert!((c.centers[[0, 0]] - 3.0).abs() < 1e-5);
        assert!((c.centers[[0, 1]] - 4.0).abs() < 1e-5);
        assert!((c.centers[[0, 2]] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn empty_cluster_set() {
        let b = cube(10.0, [false; 3]);
        let c = centers(
            Array2::<F>::zeros((0, 3)).view(),
            Some(&b),
            &manual_clusters(&[]),
        );
        assert!(c.centers.is_empty());
        assert!(c.groups.is_empty());
    }

    #[test]
    fn free_boundary() {
        let c = centers(
            arr2(&[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]).view(),
            None,
            &manual_clusters(&[0, 0]),
        );
        assert!((c.centers[[0, 0]] - 2.0).abs() < 1e-5);
    }

    // --- the former `center_of_mass_velocity`: velocities, no box, one group ---

    #[test]
    fn center_of_mass_velocity_is_mass_weighted() {
        let vel = array![[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]];
        let one_group: Vec<u8> = vec![0; 2];
        let v = com(vel.view(), None, &one_group, array![3.0, 1.0].view()).unwrap();
        assert_eq!(v.centers.row(0).to_vec(), vec![0.0, 1.0, 0.0]);
    }

    /// Spec §5: a zero total mass used to return a zero velocity.
    #[test]
    fn a_massless_set_has_no_centre_of_mass_velocity() {
        let vel = array![[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]];
        let one_group: Vec<u8> = vec![0; 2];
        assert!(matches!(
            com(vel.view(), None, &one_group, array![0.0, 0.0].view()),
            Err(ComputeError::OutOfRange {
                field: "masses",
                ..
            })
        ));
    }

    /// Spec §5: masses are required data; a negative one is an error rather
    /// than a silently shifted centre.
    #[test]
    fn a_negative_mass_is_out_of_range() {
        let labels: Vec<u32> = vec![0, 0];
        assert!(matches!(
            com(
                arr2(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).view(),
                None,
                &labels,
                arr1(&[1.0, -2.0]).view()
            ),
            Err(ComputeError::OutOfRange {
                field: "masses",
                ..
            })
        ));
    }
}
