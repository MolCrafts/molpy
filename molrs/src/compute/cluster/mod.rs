//! Distance-based cluster detection and per-cluster properties.

mod properties;
mod result;

pub use properties::{ClusterProperties, ClusterPropertiesResult};
pub use result::ClusterResult;

use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::Idx;
use ndarray::Array1;
use std::collections::HashMap;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

/// Distance-based cluster analysis using BFS on the neighbor graph.
///
/// Two particles belong to the same cluster if they are connected (directly
/// or transitively) within the neighbor cutoff. Uses CSR adjacency for
/// cache-friendly traversal. One [`ClusterResult`] per input frame: the
/// per-particle cluster IDs and the cluster sizes. Ported from
/// `freud.cluster.Cluster`; feed its result to [`ClusterProperties`] for the
/// per-cluster centers, gyration tensors and radii of gyration.
#[derive(Debug, Clone)]
pub struct Cluster {
    min_cluster_size: usize,
}

impl Cluster {
    /// Keep only clusters with at least `min_cluster_size` particles.
    pub fn new(min_cluster_size: usize) -> Self {
        Self { min_cluster_size }
    }

    fn cluster_one<FA: FrameAccess>(
        &self,
        frame: &FA,
        neighbors: &Neighbors,
    ) -> Result<ClusterResult, ComputeError> {
        let n = frame
            .visit_block("atoms", |b| b.n_rows().unwrap_or(0))
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;

        if n == 0 {
            return Ok(ClusterResult {
                cluster_idx: Array1::zeros(0),
                n_clusters: 0,
                cluster_sizes: vec![],
                cluster_keys: vec![],
            });
        }

        let n_pairs = neighbors.n_pairs();
        let query_indices = neighbors.query_point_indices();
        let point_indices = neighbors.point_indices();

        let mut degree = vec![0u32; n];
        for k in 0..n_pairs {
            degree[query_indices[k] as usize] += 1;
            degree[point_indices[k] as usize] += 1;
        }

        let mut offsets = vec![0usize; n + 1];
        for i in 0..n {
            offsets[i + 1] = offsets[i] + degree[i] as usize;
        }

        let mut flat_adj = vec![0u32; 2 * n_pairs];
        let mut cursor = offsets[..n].to_vec();
        for k in 0..n_pairs {
            let i = query_indices[k] as usize;
            let j = point_indices[k] as usize;
            flat_adj[cursor[i]] = j as u32;
            cursor[i] += 1;
            flat_adj[cursor[j]] = i as u32;
            cursor[j] += 1;
        }

        let mut cluster_idx = vec![-1_i64; n];
        let mut current_id: i64 = 0;
        let mut cluster_sizes: Vec<usize> = Vec::new();
        let mut queue: Vec<usize> = Vec::new();

        for start in 0..n {
            if cluster_idx[start] >= 0 {
                continue;
            }

            queue.clear();
            queue.push(start);
            cluster_idx[start] = current_id;
            let mut size = 0;
            let mut head = 0;

            while head < queue.len() {
                let node = queue[head];
                head += 1;
                size += 1;

                for &nbr in &flat_adj[offsets[node]..offsets[node + 1]] {
                    let neighbor = nbr as usize;
                    if cluster_idx[neighbor] < 0 {
                        cluster_idx[neighbor] = current_id;
                        queue.push(neighbor);
                    }
                }
            }

            cluster_sizes.push(size);
            current_id += 1;
        }

        if self.min_cluster_size > 1 {
            let mut remap = vec![-1_i64; cluster_sizes.len()];
            let mut new_id: i64 = 0;
            let mut new_sizes = Vec::new();

            for (old_id, &size) in cluster_sizes.iter().enumerate() {
                if size >= self.min_cluster_size {
                    remap[old_id] = new_id;
                    new_sizes.push(size);
                    new_id += 1;
                }
            }

            for cid in cluster_idx.iter_mut() {
                if *cid >= 0 {
                    *cid = remap[*cid as usize];
                }
            }

            cluster_sizes = new_sizes;
        }

        let n_clusters = cluster_sizes.len();

        Ok(ClusterResult {
            cluster_idx: Array1::from_vec(cluster_idx),
            n_clusters,
            cluster_sizes,
            cluster_keys: vec![],
        })
    }

    /// Cluster a single frame by membership key (one cluster per distinct key).
    fn keyed_one<FA: FrameAccess>(
        &self,
        frame: &FA,
        keys: &[Idx],
    ) -> Result<ClusterResult, ComputeError> {
        let n = frame
            .visit_block("atoms", |b| b.n_rows().unwrap_or(0))
            .ok_or(ComputeError::MissingBlock { name: "atoms" })?;

        if keys.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: keys.len(),
                what: "membership-key count",
            });
        }

        if n == 0 {
            return Ok(ClusterResult {
                cluster_idx: Array1::zeros(0),
                n_clusters: 0,
                cluster_sizes: vec![],
                cluster_keys: vec![],
            });
        }

        // Group by key, assigning cluster ids in order of first appearance.
        let mut key_to_id: HashMap<Idx, usize> = HashMap::new();
        let mut order_keys: Vec<Idx> = Vec::new();
        let mut sizes: Vec<usize> = Vec::new();
        let mut raw_idx = vec![0usize; n];
        for (i, &k) in keys.iter().enumerate() {
            let id = *key_to_id.entry(k).or_insert_with(|| {
                order_keys.push(k);
                sizes.push(0);
                order_keys.len() - 1
            });
            raw_idx[i] = id;
            sizes[id] += 1;
        }

        let mut cluster_idx = vec![-1_i64; n];

        // Drop groups below min_cluster_size, mirroring the spatial path.
        if self.min_cluster_size > 1 {
            let mut remap = vec![-1_i64; sizes.len()];
            let mut new_id: i64 = 0;
            let mut new_sizes = Vec::new();
            let mut new_keys = Vec::new();
            for (old, &sz) in sizes.iter().enumerate() {
                if sz >= self.min_cluster_size {
                    remap[old] = new_id;
                    new_sizes.push(sz);
                    new_keys.push(vec![order_keys[old]]);
                    new_id += 1;
                }
            }
            for (i, &raw) in raw_idx.iter().enumerate() {
                cluster_idx[i] = remap[raw];
            }
            return Ok(ClusterResult {
                cluster_idx: Array1::from_vec(cluster_idx),
                n_clusters: new_sizes.len(),
                cluster_sizes: new_sizes,
                cluster_keys: new_keys,
            });
        }

        for (i, &raw) in raw_idx.iter().enumerate() {
            cluster_idx[i] = raw as i64;
        }
        Ok(ClusterResult {
            cluster_idx: Array1::from_vec(cluster_idx),
            n_clusters: sizes.len(),
            cluster_sizes: sizes,
            cluster_keys: order_keys.into_iter().map(|k| vec![k]).collect(),
        })
    }

    /// Cluster every frame by membership key: all particles sharing a key
    /// (e.g. a molecule id) form one cluster, independent of geometry. Unlike
    /// [`Cluster::compute`], this needs no neighbor list and is robust for
    /// per-molecule properties (e.g. per-chain radius of gyration) even when
    /// molecules overlap in space or a bond exceeds any spatial cutoff
    /// (freud's `Cluster.compute(..., keys=...)`). The same `keys` apply to
    /// every frame; their length must equal each frame's atom count.
    pub fn compute_keyed<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        keys: &[Idx],
    ) -> Result<Vec<ClusterResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let mut out = Vec::with_capacity(frames.len());
        for frame in frames {
            out.push(self.keyed_one(*frame, keys)?);
        }
        Ok(out)
    }
}

impl LegacyCompute for Cluster {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<ClusterResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        neighbors: &'a [Neighbors],
    ) -> Result<Vec<ClusterResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if neighbors.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: neighbors.len(),
                what: "neighbor-list count",
            });
        }
        // Cluster has heavy per-frame work (CSR build + BFS, ~100 µs per
        // 5k-atom frame), so rayon pays off from 2 frames onward.
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(neighbors.par_iter())
                .map(|(frame, nlist)| self.cluster_one(*frame, nlist))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (frame, nlist) in frames.iter().zip(neighbors.iter()) {
            out.push(self.cluster_one(*frame, nlist)?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use molrs::op::F;
    use ndarray::{Array1 as A1, array};

    fn make_frame_with_positions(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));

        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();

        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox = Some(
            SimBox::cube(
                box_len,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );
        frame
    }

    fn build_neighbors(frame: &Frame, cutoff: F) -> Neighbors {
        nlist_from_frame(frame, cutoff)
    }

    fn cluster_single(frame: &Frame, nlist: Neighbors, min: usize) -> ClusterResult {
        let out = Cluster::new(min).compute(&[frame], &[nlist]).unwrap();
        assert_eq!(out.len(), 1);
        out.into_iter().next().unwrap()
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
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);

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
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 2);

        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_idx[2], -1);
        assert!(result.cluster_idx[0] >= 0);
    }

    #[test]
    fn single_cluster() {
        let positions = [
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [1.0, 1.5, 1.0],
            [1.5, 1.5, 1.0],
        ];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);

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
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 2.01);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn all_isolated() {
        let positions = [[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let frame = make_frame_with_positions(&positions, 20.0);
        let nbrs = build_neighbors(&frame, 0.5);
        let result = cluster_single(&frame, nbrs, 1);
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
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 0.5);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 3);
    }

    #[test]
    fn empty_frame() {
        let frame = make_frame_with_positions(&[], 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.n_clusters, 0);
        assert!(result.cluster_idx.is_empty());
    }

    #[test]
    fn single_particle() {
        let positions = [[5.0, 5.0, 5.0]];
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.0);
        let result = cluster_single(&frame, nbrs, 1);
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
        let frame = make_frame_with_positions(&positions, 10.0);
        let nbrs = build_neighbors(&frame, 1.5);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.n_clusters, 1);
        assert_eq!(result.cluster_sizes[0], 4);
    }

    #[test]
    fn pbc_wrapping_cluster() {
        let positions = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox = Some(
            SimBox::cube(
                10.0,
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [true, true, true],
            )
            .unwrap(),
        );
        let nbrs = build_neighbors(&frame, 2.0);
        let result = cluster_single(&frame, nbrs, 1);
        assert_eq!(result.n_clusters, 1);
    }

    #[test]
    fn multi_frame_runs_per_frame() {
        let f1 = make_frame_with_positions(&[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]], 10.0);
        let f2 = make_frame_with_positions(&[[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0);
        let n1 = build_neighbors(&f1, 1.0);
        let n2 = build_neighbors(&f2, 1.0);
        let out = Cluster::new(1).compute(&[&f1, &f2], &[n1, n2]).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].n_clusters, 1);
        assert_eq!(out[1].n_clusters, 2);
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Cluster::new(1)
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

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
        let frame = make_frame_with_positions(&positions, 20.0);
        // Spatial clustering merges all six into one.
        let nbrs = build_neighbors(&frame, 2.0);
        let spatial = cluster_single(&frame, nbrs, 1);
        assert_eq!(spatial.n_clusters, 1);

        // Keyed clustering by mol-id splits into two clusters of three.
        let keys = [0u64, 0, 0, 1, 1, 1];
        let out = Cluster::new(1).compute_keyed(&[&frame], &keys).unwrap();
        assert_eq!(out.len(), 1);
        let r = &out[0];
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
        let positions = [[0.0, 0.0, 0.0]; 4];
        let frame = make_frame_with_positions(&positions, 20.0);
        let keys = [7u64, 3, 7, 3];
        let out = Cluster::new(1).compute_keyed(&[&frame], &keys).unwrap();
        let r = &out[0];
        assert_eq!(r.n_clusters, 2);
        // key 7 seen first -> cluster 0, key 3 -> cluster 1
        assert_eq!(r.cluster_idx.to_vec(), vec![0, 1, 0, 1]);
        assert_eq!(r.cluster_keys, vec![vec![7u64], vec![3u64]]);
    }

    #[test]
    fn keyed_min_cluster_size_filters() {
        let positions = [[0.0, 0.0, 0.0]; 4];
        let frame = make_frame_with_positions(&positions, 20.0);
        // key 0 has 3 members, key 1 has 1 member.
        let keys = [0u64, 0, 0, 1];
        let out = Cluster::new(2).compute_keyed(&[&frame], &keys).unwrap();
        let r = &out[0];
        assert_eq!(r.n_clusters, 1);
        assert_eq!(r.cluster_sizes, vec![3]);
        assert_eq!(r.cluster_idx[3], -1);
        assert_eq!(r.cluster_keys, vec![vec![0u64]]);
    }

    #[test]
    fn keyed_length_mismatch_is_error() {
        let frame = make_frame_with_positions(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0);
        let err = Cluster::new(1)
            .compute_keyed(&[&frame], &[0u64])
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn keyed_empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Cluster::new(1).compute_keyed(&frames, &[]).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
