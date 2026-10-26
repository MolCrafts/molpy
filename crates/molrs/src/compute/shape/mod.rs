//! Shape and mass-distribution descriptors of groups (clusters, molecules):
//! centre of mass, gyration tensor (and radius of gyration), inertia tensor.
//!
//! Each takes one snapshot's positions, its box, a [`Partition`] — a label
//! array such as a `mol_id` column, or an upstream
//! [`ClusterResult`](crate::compute::ClusterResult) — and one weight per item,
//! and reports one row per group in ascending key order. They share one
//! reduction (anchor + minimum-image unwrap, weighted centroid, weighted
//! second moment), so a group straddling a periodic boundary is handled the
//! same way by all three.
//!
//! | Method | Weights | Output (per group) |
//! |--------|---------|--------------------|
//! | [`CenterOfMass`] | masses (ones: geometric centre) | centre, total mass |
//! | [`GyrationTensor`] | the caller's weighting | `G`; `R_g = √tr G` via [`GyrationTensorResult::radius_of_gyration`] |
//! | [`InertiaTensor`] | masses | `I` about the centre of mass |
//!
//! freud's `ClusterProperties` is this module composed: [`Partition::sizes`],
//! [`CenterOfMass`] with masses and with ones, and [`GyrationTensor`].
//!
//! [`Partition`]: crate::compute::Partition
//! [`Partition::sizes`]: crate::compute::Partition::sizes

mod center_of_mass;
/// Crate-internal: the one group reduction, also for the Voronoi per-molecule
/// reference (link 08).
pub(crate) mod group_reduce;
mod gyration_tensor;
mod inertia_tensor;

pub use center_of_mass::{CenterOfMass, CenterOfMassInput, CenterOfMassResult};
pub use gyration_tensor::{GyrationTensor, GyrationTensorInput, GyrationTensorResult};
pub use inertia_tensor::{InertiaTensor, InertiaTensorInput, InertiaTensorResult};

/// The former `ClusterProperties`, as the composition that replaces it:
/// sizes from the [`Partition`](crate::compute::Partition), centres from
/// [`CenterOfMass`], tensors and radii from [`GyrationTensor`].
#[cfg(test)]
mod cluster_properties {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use crate::compute::{Cluster, ClusterInput, ClusterResult, Compute, ComputeError, Partition};
    use crate::core::Cell;
    use molrs::core::SimBox;
    use molrs::op::F;
    use ndarray::{Array1, Array2, arr1};

    fn cube(len: F, pbc: [bool; 3]) -> SimBox {
        SimBox::new(Cell::Lengths([len; 3]), [0.0, 0.0, 0.0], pbc).unwrap()
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

    /// Everything `ClusterPropertiesResult` held, from the replacing computes.
    struct Properties {
        sizes: Vec<usize>,
        centers: Array2<F>,
        centers_of_mass: Array2<F>,
        cluster_masses: Array1<F>,
        gyration: GyrationTensorResult,
    }

    fn properties(
        rows: &[[F; 3]],
        simbox: &SimBox,
        clusters: &ClusterResult,
        masses: Option<&[F]>,
    ) -> Properties {
        let positions = Array2::from_shape_fn((rows.len(), 3), |(i, d)| rows[i][d]);
        let masses = masses.map_or_else(|| Array1::ones(rows.len()), arr1);
        let ones = Array1::<F>::ones(rows.len());
        let com = |m: &Array1<F>| {
            CenterOfMass::new()
                .compute(CenterOfMassInput {
                    positions: positions.view(),
                    simbox: Some(simbox),
                    partition: clusters,
                    masses: m.view(),
                })
                .unwrap()
        };
        let weighted = com(&masses);
        Properties {
            sizes: clusters.sizes().into_values().collect(),
            centers: com(&ones).centers,
            centers_of_mass: weighted.centers,
            cluster_masses: weighted.total_masses,
            gyration: GyrationTensor::new()
                .compute(GyrationTensorInput {
                    positions: positions.view(),
                    simbox: Some(simbox),
                    partition: clusters,
                    weights: masses.view(),
                })
                .unwrap(),
        }
    }

    #[test]
    fn two_particle_uniform_mass() {
        // (0,0,0) and (2,0,0) → center = (1,0,0), Rg = 1, gyration[0][0]=1
        let b = cube(10.0, [false; 3]);
        let r = properties(
            &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            &b,
            &manual_clusters(&[0, 0]),
            None,
        );

        assert_eq!(r.sizes, vec![2]);
        assert!((r.centers[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((r.centers_of_mass[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((r.cluster_masses[0] - 2.0).abs() < 1e-10);
        // Gyration tensor: G_xx = (1/M) Σ m_i (x_i - x_c)² = (1+1)/2 = 1
        assert!((r.gyration.tensors[[0, 0, 0]] - 1.0).abs() < 1e-10);
        // Rg = sqrt(trace) = sqrt(1) = 1
        assert!((r.gyration.radius_of_gyration()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_center_shifts_to_heavy_atom() {
        // Masses 3, 1 at (1,0,0) and (3,0,0).
        // CoM = (3·1 + 1·3) / (3+1) = 6/4 = 1.5
        let b = cube(10.0, [false; 3]);
        let r = properties(
            &[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            &b,
            &manual_clusters(&[0, 0]),
            Some(&[3.0, 1.0]),
        );
        assert!((r.centers_of_mass[[0, 0]] - 1.5).abs() < 1e-10);
        // Unweighted geometric center is still 2.0.
        assert!((r.centers[[0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn two_clusters_independent() {
        let b = cube(20.0, [false; 3]);
        let r = properties(
            &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [5.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
            ],
            &b,
            &manual_clusters(&[0, 0, 1, 1]),
            None,
        );
        let rg = r.gyration.radius_of_gyration();
        assert_eq!(r.sizes, vec![2, 2]);
        // Cluster 0 (two coincident particles): center = (1,1,1), Rg = 0
        assert!((r.centers[[0, 0]] - 1.0).abs() < 1e-10);
        assert!(rg[0].abs() < 1e-10);
        // Cluster 1: center = (6, 0, 0), Rg = 1
        assert!((r.centers[[1, 0]] - 6.0).abs() < 1e-10);
        assert!((rg[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn filtered_atoms_ignored() {
        let b = cube(200.0, [false; 3]);
        let r = properties(
            &[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 100.0, 100.0]],
            &b,
            &manual_clusters(&[0, 0, -1]), // third atom filtered
            None,
        );
        assert_eq!(r.sizes, vec![2]);
        assert!((r.centers[[0, 0]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pbc_wrapped_cluster() {
        // Atoms at x=0.5 and x=9.5 in a box of length 10 with PBC.
        // True separation is 1.0 (via wrap), unwrapped center is 0 (or 10).
        let b = cube(10.0, [true, true, true]);
        let r = properties(
            &[[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]],
            &b,
            &manual_clusters(&[0, 0]),
            None,
        );
        // Gyration tensor x-component should reflect 1.0 separation, NOT 9.0.
        // G_xx = (0.5)² for each particle around their unwrapped center → 0.25.
        assert!(
            (r.gyration.tensors[[0, 0, 0]] - 0.25).abs() < 1e-9,
            "PBC-aware G_xx = {} (expected 0.25)",
            r.gyration.tensors[[0, 0, 0]]
        );
        assert!((r.gyration.radius_of_gyration()[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn masses_dimension_mismatch_errors() {
        let positions = Array2::<F>::zeros((2, 3));
        let err = GyrationTensor::new()
            .compute(GyrationTensorInput {
                positions: positions.view(),
                simbox: None,
                partition: &manual_clusters(&[0, 0]),
                weights: arr1(&[1.0]).view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    /// Acceptance ac-010: `Cluster` → `CenterOfMass` / `GyrationTensor`
    /// through `Partition`, with no glue — the cluster result is the
    /// partition.
    #[test]
    fn cluster_feeds_the_shape_computes_as_a_partition() {
        let rows = [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [9.5, 5.0, 5.0],
            [0.5, 5.0, 5.0],
        ];
        let b = cube(10.0, [true; 3]);
        let pairs = neighbors_of(&rows, &b, 1.5);
        let clusters = Cluster::new()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: None,
            })
            .unwrap();
        assert_eq!(clusters.n_clusters, 2);

        let positions = Array2::from_shape_fn((4, 3), |(i, d)| rows[i][d]);
        let masses = Array1::<F>::ones(4);
        let com = CenterOfMass::new()
            .compute(CenterOfMassInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &clusters,
                masses: masses.view(),
            })
            .unwrap();
        let gyration = GyrationTensor::new()
            .compute(GyrationTensorInput {
                positions: positions.view(),
                simbox: Some(&b),
                partition: &clusters,
                weights: masses.view(),
            })
            .unwrap();

        assert_eq!(com.groups, vec![0, 1]);
        assert!((com.centers[[0, 0]] - 1.5).abs() < 1e-12);
        // Cluster 1 straddles x = 0: anchored at 9.5, its centre is 10.0.
        assert!((com.centers[[1, 0]] - 10.0).abs() < 1e-12);
        let rg = gyration.radius_of_gyration();
        assert!((rg[0] - 0.5).abs() < 1e-12);
        assert!((rg[1] - 0.5).abs() < 1e-12);
    }
}
