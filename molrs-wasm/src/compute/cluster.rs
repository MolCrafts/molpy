//! Distance-based cluster analysis — WASM face of `molrs::compute::Cluster`.

use crate::core::frame::Frame;
use crate::core::neighbors::Neighbors;
use molrs::compute::LegacyCompute;
use molrs::compute::{Cluster as RsCluster, ClusterResult as RsClusterResult};
use wasm_bindgen::prelude::*;

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Particles that are connected (directly or transitively) through
/// neighbor-list pairs are grouped into clusters. Clusters smaller
/// than `minClusterSize` are filtered out (their particles get
/// cluster ID = -1).
///
/// # Example (JavaScript)
///
/// ```js
/// const nl = new NeighborList(2.0);
/// nl.build(frame);
/// const nlist = nl.neighbors();
///
/// const cluster = new Cluster(5); // min 5 particles per cluster
/// const result = cluster.compute(frame, nlist);
///
/// console.log(result.nClusters);     // number of valid clusters
/// console.log(result.clusterIdx());    // Int32Array, per-particle IDs
/// console.log(result.clusterSizes());  // Uint32Array, size of each cluster
/// ```
#[wasm_bindgen(js_name = Cluster)]
pub struct Cluster {
    inner: RsCluster,
}

#[wasm_bindgen(js_class = Cluster)]
impl Cluster {
    /// Create a cluster analysis with a minimum cluster size filter.
    ///
    /// # Arguments
    ///
    /// * `min_cluster_size` - Minimum number of particles for a cluster
    ///   to be considered valid. Clusters with fewer particles are
    ///   discarded (their particles get cluster ID = -1).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const cluster = new Cluster(5); // ignore clusters < 5 particles
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(min_cluster_size: usize) -> Self {
        Self {
            inner: RsCluster::new(min_cluster_size),
        }
    }

    /// Run cluster analysis on a frame with pre-built neighbor pairs.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with atom positions
    /// * `neighbors` - Pre-built [`Neighbors`] defining connectivity
    ///
    /// # Returns
    ///
    /// A [`ClusterResult`] with per-particle cluster IDs and cluster sizes.
    ///
    /// # Errors
    ///
    /// Throws if the frame cannot be cloned or the analysis fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const result = cluster.compute(frame, nlist);
    /// ```
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<ClusterResult, JsValue> {
        frame.with_frame(|rs_frame| {
            let nlist_vec = std::slice::from_ref(&neighbors.inner);
            let mut results = self
                .inner
                .compute(&[rs_frame], nlist_vec)
                .map_err(|e| JsValue::from_str(&format!("Cluster compute: {e}")))?;
            let first = results
                .pop()
                .ok_or_else(|| JsValue::from_str("Cluster compute: empty result"))?;
            Ok(ClusterResult { inner: first })
        })
    }
}

/// Result of a distance-based cluster analysis.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = cluster.compute(frame, nlist);
/// console.log(result.nClusters);       // number
///
/// const ids   = result.clusterIdx();     // Int32Array (per-particle)
/// const sizes = result.clusterSizes();   // Uint32Array (per-cluster)
///
/// // Particles in filtered-out clusters have id = -1
/// for (let i = 0; i < ids.length; i++) {
///   if (ids[i] === -1) console.log(`Particle ${i} not in any valid cluster`);
/// }
/// ```
#[wasm_bindgen(js_name = ClusterResult)]
pub struct ClusterResult {
    pub(super) inner: RsClusterResult,
}

#[wasm_bindgen(js_class = ClusterResult)]
impl ClusterResult {
    /// Number of valid clusters found (after min-size filtering).
    #[wasm_bindgen(getter, js_name = nClusters)]
    pub fn n_clusters(&self) -> usize {
        self.inner.n_clusters
    }

    /// Per-particle cluster ID assignment as `Int32Array`.
    ///
    /// `clusterIdx()[i]` is the cluster ID for particle `i`.
    /// Particles in clusters smaller than `minClusterSize` are
    /// assigned ID = -1 (filtered out).
    ///
    /// Cluster IDs are zero-based and contiguous: `0, 1, ..., nClusters-1`.
    #[wasm_bindgen(js_name = clusterIdx)]
    pub fn cluster_idx(&self) -> Vec<i32> {
        self.inner.cluster_idx.iter().map(|&id| id as i32).collect()
    }

    /// Size (particle count) of each valid cluster as `Uint32Array`.
    ///
    /// `clusterSizes()[c]` is the number of particles in cluster `c`.
    /// Length equals `nClusters`.
    #[wasm_bindgen(js_name = clusterSizes)]
    pub fn cluster_sizes(&self) -> Vec<u32> {
        self.inner.cluster_sizes.iter().map(|&s| s as u32).collect()
    }
}
