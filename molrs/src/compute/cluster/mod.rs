//! Distance-based cluster detection: connected components of a neighbor
//! table, optionally merged by membership keys.

mod result;

pub use result::ClusterResult;

use std::collections::{BTreeSet, HashMap};

use molrs::core::{Neighbors, Topology};
use ndarray::Array1;

use crate::compute::{Compute, ComputeError, Partition, require_self_query};

/// Cluster analysis: two particles belong to the same cluster when they are
/// connected, directly or transitively, by a pair of the neighbor table — or
/// share a group of the optional `merge` partition.
///
/// Mirrors `freud.cluster.Cluster`; `merge` is freud's `keys`.
///
/// Configuration: the minimum cluster size. Data: one snapshot's half-shell
/// self-query [`Neighbors`] table (any column policy — only the pair indices
/// are read; the particle count is the table's `SelfQuery { n_points }`) and,
/// optionally, a [`Partition`] whose groups are joined whole (a `mol_id`
/// column keeps every molecule in one piece even when a bond is longer than
/// the cutoff). Clustering purely by key is `merge` over a table with no
/// pairs. Many snapshots go through [`Batch`](crate::compute::Batch).
///
/// The components are the connected components of
/// [`Topology::connected_components`], numbered in order of their lowest
/// particle. The result is a [`Partition`], so it feeds the shape computes
/// ([`CenterOfMass`](crate::compute::CenterOfMass),
/// [`GyrationTensor`](crate::compute::GyrationTensor), …) directly.
#[derive(Debug, Clone, Copy)]
pub struct Cluster {
    min_cluster_size: usize,
}

/// The data a [`Cluster`] analyses.
#[derive(Clone, Copy)]
pub struct ClusterInput<'a> {
    /// One snapshot's self-query table; its pairs are the bonds.
    pub pairs: &'a Neighbors,
    /// Groups joined whole (freud's `keys`), one entry per particle; `None`
    /// joins nothing beyond the pairs.
    pub merge: Option<&'a (dyn Partition + Sync)>,
}

impl Default for Cluster {
    fn default() -> Self {
        Self::new()
    }
}

impl Cluster {
    /// Every cluster is kept (minimum size 1).
    pub fn new() -> Self {
        Self {
            min_cluster_size: 1,
        }
    }

    /// Drop clusters with fewer than `n` particles; their particles are in no
    /// cluster.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0.
    pub fn with_min_cluster_size(mut self, n: usize) -> Result<Self, ComputeError> {
        if n == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Cluster::min_cluster_size",
                value: "0".into(),
            });
        }
        self.min_cluster_size = n;
        Ok(self)
    }

    /// The minimum cluster size.
    pub fn min_cluster_size(&self) -> usize {
        self.min_cluster_size
    }
}

impl Compute for Cluster {
    type Input<'a> = ClusterInput<'a>;
    type Output = ClusterResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a cross-query table;
    /// [`ComputeError::OutOfRange`] for a pair index past the table's point
    /// count; [`ComputeError::DimensionMismatch`] when `merge` does not have
    /// one entry per particle.
    fn compute(&self, input: ClusterInput<'_>) -> Result<ClusterResult, ComputeError> {
        let n = require_self_query(input.pairs)?;
        let mut bonds: Vec<[usize; 2]> = Vec::with_capacity(input.pairs.n_pairs());
        for (&i, &j) in input
            .pairs
            .query_point_indices()
            .iter()
            .zip(input.pairs.point_indices())
        {
            let (i, j) = (i as usize, j as usize);
            if i >= n || j >= n {
                return Err(ComputeError::OutOfRange {
                    field: "Neighbors pair index",
                    value: format!("({i}, {j}) with {n} points"),
                });
            }
            bonds.push([i, j]);
        }

        if let Some(merge) = input.merge {
            if merge.n_items() != n {
                return Err(ComputeError::DimensionMismatch {
                    expected: n,
                    got: merge.n_items(),
                    what: "merge partition items against the table's points",
                });
            }
            // Join each group as a star on its first member.
            let mut first: HashMap<u64, usize> = HashMap::new();
            for item in 0..n {
                if let Some(key) = merge.group_of(item) {
                    let root = *first.entry(key).or_insert(item);
                    if root != item {
                        bonds.push([root, item]);
                    }
                }
            }
        }

        let mut graph = Topology::with_atoms(n);
        graph.add_bonds(&bonds);
        let components = graph.connected_components();
        let n_components = components.iter().max().map_or(0, |&c| c as usize + 1);

        let mut sizes = vec![0usize; n_components];
        for &c in &components {
            sizes[c as usize] += 1;
        }
        // Components below the minimum size are dropped; the rest keep their
        // order (lowest particle first) and are renumbered densely.
        let mut remap = vec![-1_i64; n_components];
        let mut cluster_sizes = Vec::new();
        for (c, &size) in sizes.iter().enumerate() {
            if size >= self.min_cluster_size {
                remap[c] = cluster_sizes.len() as i64;
                cluster_sizes.push(size);
            }
        }
        let cluster_idx: Array1<i64> = components.iter().map(|&c| remap[c as usize]).collect();

        let cluster_keys = match input.merge {
            None => Vec::new(),
            Some(merge) => {
                let mut keys = vec![BTreeSet::new(); cluster_sizes.len()];
                for (item, &c) in cluster_idx.iter().enumerate() {
                    if c >= 0
                        && let Some(key) = merge.group_of(item)
                    {
                        keys[c as usize].insert(key);
                    }
                }
                keys.into_iter().map(|k| k.into_iter().collect()).collect()
            }
        };

        Ok(ClusterResult {
            cluster_idx,
            n_clusters: cluster_sizes.len(),
            cluster_sizes,
            cluster_keys,
        })
    }
}

crate::compute::spec::compute_spec! {
    Cluster as "Cluster";
    params |s| {
        min_cluster_size: Int = Cluster::new().min_cluster_size => s.min_cluster_size;
    }
    inputs {
        pairs: Neighbors;
        merge: Partition, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::{NeighborColumns, NeighborPair, QueryMode, SimBox};
    use molrs::op::F;
    use ndarray::array;

    fn cube(len: F, pbc: [bool; 3]) -> SimBox {
        SimBox::cube(len, array![0.0, 0.0, 0.0], pbc).unwrap()
    }

    fn cluster(positions: &[[F; 3]], box_len: F, cutoff: F, min: usize) -> ClusterResult {
        let pairs = neighbors_of(positions, &cube(box_len, [false; 3]), cutoff);
        Cluster::new()
            .with_min_cluster_size(min)
            .unwrap()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: None,
            })
            .unwrap()
    }

    /// A table over `n` points with no pairs: clustering by `merge` alone.
    fn no_pairs(n: usize) -> Neighbors {
        Neighbors::from_pairs(
            std::iter::empty::<NeighborPair>(),
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: n },
        )
    }

    fn keyed(n: usize, keys: &[u64], min: usize) -> Result<ClusterResult, ComputeError> {
        let keys = keys.to_vec();
        Cluster::new()
            .with_min_cluster_size(min)?
            .compute(ClusterInput {
                pairs: &no_pairs(n),
                merge: Some(&keys),
            })
    }

    #[test]
    fn two_separated_groups() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [8.0, 8.0, 8.0],
            [8.5, 8.0, 8.0],
            [8.0, 8.5, 8.0],
        ];
        let result = cluster(&positions, 20.0, 2.0, 1);

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.cluster_idx[0], result.cluster_idx[1]);
        assert_eq!(result.cluster_idx[0], result.cluster_idx[2]);
        assert_eq!(result.cluster_idx[3], result.cluster_idx[4]);
        assert_eq!(result.cluster_idx[3], result.cluster_idx[5]);
        assert_ne!(result.cluster_idx[0], result.cluster_idx[3]);
    }

    #[test]
    fn min_cluster_size_filters_small() {
        let positions = [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [8.0, 8.0, 8.0]];
        let result = cluster(&positions, 20.0, 2.0, 2);

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_idx[2], -1);
        assert!(result.cluster_idx[0] >= 0);
        assert_eq!(result.group_of(2), None);
    }

    #[test]
    fn single_cluster() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [1.5, 1.5, 1.0],
        ];
        let result = cluster(&positions, 20.0, 2.0, 1);

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn collinear_four_particles() {
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ];
        let result = cluster(&positions, 10.0, 2.01, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn all_isolated() {
        let positions = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let result = cluster(&positions, 20.0, 0.5, 1);
        assert_eq!(result.n_clusters, 3);
        for &s in &result.cluster_sizes {
            assert_eq!(s, 1);
        }
        assert_ne!(result.cluster_idx[0], result.cluster_idx[1]);
        assert_ne!(result.cluster_idx[1], result.cluster_idx[2]);
    }

    #[test]
    fn coincident_particles() {
        let positions = [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]];
        let result = cluster(&positions, 10.0, 0.5, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 3);
    }

    #[test]
    fn empty_frame() {
        let result = cluster(&[], 10.0, 1.0, 1);
        assert_eq!(result.n_clusters, 0);
        assert!(result.cluster_idx.is_empty());
    }

    #[test]
    fn single_particle() {
        let result = cluster(&[[5.0, 5.0, 5.0]], 10.0, 1.0, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_idx[0], 0);
    }

    #[test]
    fn transitive_chain() {
        let positions = [
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
        ];
        let result = cluster(&positions, 10.0, 1.5, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn pbc_wrapping_cluster() {
        let positions = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let pairs = neighbors_of(&positions, &cube(10.0, [true; 3]), 2.0);
        let result = Cluster::new()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: None,
            })
            .unwrap();
        assert_eq!(result.n_clusters, 1);
    }

    #[test]
    fn multi_frame_runs_per_frame() {
        let b = cube(10.0, [false; 3]);
        let n1 = neighbors_of(&[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]], &b, 1.0);
        let n2 = neighbors_of(&[[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]], &b, 1.0);
        let out = Batch(Cluster::new())
            .compute(vec![
                ClusterInput {
                    pairs: &n1,
                    merge: None,
                },
                ClusterInput {
                    pairs: &n2,
                    merge: None,
                },
            ])
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].n_clusters, 1);
        assert_eq!(out[1].n_clusters, 2);
    }

    // --- `merge` (freud's keys; the former `compute_keyed`) ---

    #[test]
    fn keyed_groups_by_key_ignoring_geometry() {
        // Six atoms all packed within one spatial cutoff (a single spatial
        // cluster), but split into two molecules by key. Keyed clustering must
        // recover the two molecules regardless of proximity.
        let positions = [
            [1.0, 1.0, 1.0],
            [1.2, 1.0, 1.0],
            [1.4, 1.0, 1.0],
            [1.1, 1.1, 1.0],
            [1.3, 1.1, 1.0],
            [1.5, 1.1, 1.0],
        ];
        // Spatial clustering merges all six into one.
        let spatial = cluster(&positions, 20.0, 2.0, 1);
        assert_eq!(spatial.n_clusters, 1);

        // Keyed clustering by mol-id splits into two clusters of three.
        let r = keyed(6, &[0, 0, 0, 1, 1, 1], 1).unwrap();
        assert_eq!(r.n_clusters, 2);
        assert_eq!(r.cluster_sizes, vec![3, 3]);
        assert_eq!(r.cluster_idx[0], r.cluster_idx[1]);
        assert_eq!(r.cluster_idx[0], r.cluster_idx[2]);
        assert_eq!(r.cluster_idx[3], r.cluster_idx[5]);
        assert_ne!(r.cluster_idx[0], r.cluster_idx[3]);
        // freud-style cluster_keys: one key per cluster.
        assert_eq!(r.cluster_keys, vec![vec![0u64], vec![1u64]]);
    }

    #[test]
    fn keyed_respects_first_appearance_order() {
        let r = keyed(4, &[7, 3, 7, 3], 1).unwrap();
        assert_eq!(r.n_clusters, 2);
        // key 7 seen first -> cluster 0, key 3 -> cluster 1
        assert_eq!(r.cluster_idx.to_vec(), vec![0, 1, 0, 1]);
        assert_eq!(r.cluster_keys, vec![vec![7u64], vec![3u64]]);
    }

    #[test]
    fn keyed_min_cluster_size_filters() {
        // key 0 has 3 members, key 1 has 1 member.
        let r = keyed(4, &[0, 0, 0, 1], 2).unwrap();
        assert_eq!(r.n_clusters, 1);
        assert_eq!(r.cluster_sizes, vec![3]);
        assert_eq!(r.cluster_idx[3], -1);
        assert_eq!(r.cluster_keys, vec![vec![0u64]]);
    }

    #[test]
    fn keyed_length_mismatch_is_error() {
        let err = keyed(2, &[0], 1).unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    /// `merge` joins groups on top of the pairs (freud's semantics): two
    /// spatial clusters sharing a key are one cluster carrying both keys.
    #[test]
    fn merge_joins_spatial_clusters_that_share_a_key() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [8.0, 8.0, 8.0],
            [8.5, 8.0, 8.0],
            [15.0, 15.0, 15.0],
        ];
        let pairs = neighbors_of(&positions, &cube(20.0, [false; 3]), 1.0);
        let keys: Vec<u32> = vec![4, 9, 9, 2, 6];
        let r = Cluster::new()
            .compute(ClusterInput {
                pairs: &pairs,
                merge: Some(&keys),
            })
            .unwrap();
        assert_eq!(r.n_clusters, 2);
        assert_eq!(r.cluster_idx.to_vec(), vec![0, 0, 0, 0, 1]);
        assert_eq!(r.cluster_keys, vec![vec![2, 4, 9], vec![6]]);
    }

    #[test]
    fn min_cluster_size_zero_is_out_of_range() {
        assert!(matches!(
            Cluster::new().with_min_cluster_size(0),
            Err(ComputeError::OutOfRange {
                field: "Cluster::min_cluster_size",
                ..
            })
        ));
    }

    #[test]
    fn a_cross_query_table_is_bad_shape() {
        let table = Neighbors::from_pairs(
            std::iter::empty::<NeighborPair>(),
            NeighborColumns::INDICES_ONLY,
            QueryMode::CrossQuery {
                n_query_points: 2,
                n_points: 2,
            },
        );
        assert!(matches!(
            Cluster::new().compute(ClusterInput {
                pairs: &table,
                merge: None
            }),
            Err(ComputeError::BadShape { .. })
        ));
    }

    /// A hand-built table naming a point past its own count used to panic in
    /// the adjacency build.
    #[test]
    fn a_pair_past_the_point_count_is_out_of_range() {
        let table = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 5,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: 2 },
        );
        assert!(matches!(
            Cluster::new().compute(ClusterInput {
                pairs: &table,
                merge: None
            }),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
