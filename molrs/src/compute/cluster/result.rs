//! Result type for [`Cluster`](super::Cluster): per-particle cluster IDs,
//! cluster count, sizes, and per-cluster membership keys.

use ndarray::Array1;

use crate::compute::Partition;

/// One snapshot's clusters; a [`Partition`] of the particles (a filtered
/// particle is in no group).
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterResult {
    /// Particle → cluster ID (`0..n_clusters`), clusters numbered in order of
    /// their lowest particle; `-1` for a particle whose cluster was dropped
    /// by the minimum size.
    pub cluster_idx: Array1<i64>,
    /// Number of clusters kept.
    pub n_clusters: usize,
    /// Size (particle count) of each cluster, indexed by cluster ID.
    pub cluster_sizes: Vec<usize>,
    /// The `merge` keys present in each cluster, ascending, indexed by cluster
    /// ID (freud's `cluster_keys`). Empty when clustering without `merge`.
    pub cluster_keys: Vec<Vec<u64>>,
}

impl Partition for ClusterResult {
    fn n_items(&self) -> usize {
        self.cluster_idx.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        u64::try_from(self.cluster_idx[item]).ok()
    }
}
