//! Gyration tensor computation for clusters.

#![allow(clippy::needless_range_loop)]

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ClusterCentersResult;
use crate::compute::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Gyration tensor per cluster, per frame.
///
/// `G_k[a][b] = (1/N_k) * SUM_i s_i[a] * s_i[b]`
/// where `s_i = shortest_vector(center_k, r_i)` is the MIC displacement from
/// the geometric center. Geometric centers come from the
/// [`ClusterCentersResult`] arg — this Compute does **not** recompute them.
///
/// Reads `atoms.{x,y,z}` (Å); `Args` = per-frame
/// ([`ClusterResult`],
/// [`ClusterCentersResult`])
/// pairs — run [`Cluster`](crate::compute::Cluster) and
/// [`ClusterCenters`](crate::compute::ClusterCenters) first. Output:
/// per-cluster 3×3 gyration tensors (Å²).
#[derive(Debug, Clone, Default)]
pub struct GyrationTensor;

impl GyrationTensor {
    /// No parameters — the tensor is unweighted (number-averaged).
    pub fn new() -> Self {
        Self
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
        centers: &ClusterCentersResult,
    ) -> Result<GyrationTensorResult, ComputeError> {
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;

        if centers.centers.len() != nc {
            return Err(ComputeError::DimensionMismatch {
                expected: nc,
                got: centers.centers.len(),
                what: "ClusterCentersResult cluster count",
            });
        }

        let mut tensors = vec![[[0.0 as F; 3]; 3]; nc];
        let mut counts = vec![0usize; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let s = mic.apply(sub(pos, centers.centers[c]));

            // Fully unrolled 3x3 rank-1 update — compiler emits straight-line code.
            let t = &mut tensors[c];
            t[0][0] += s[0] * s[0];
            t[0][1] += s[0] * s[1];
            t[0][2] += s[0] * s[2];
            t[1][0] += s[1] * s[0];
            t[1][1] += s[1] * s[1];
            t[1][2] += s[1] * s[2];
            t[2][0] += s[2] * s[0];
            t[2][1] += s[2] * s[1];
            t[2][2] += s[2] * s[2];
            counts[c] += 1;
        }

        for (tensor, &count) in tensors.iter_mut().zip(counts.iter()) {
            if count > 0 {
                let n = count as F;
                for row in tensor.iter_mut() {
                    for val in row.iter_mut() {
                        *val /= n;
                    }
                }
            }
        }

        Ok(GyrationTensorResult(tensors))
    }
}

impl LegacyCompute for GyrationTensor {
    type Args<'a> = (&'a Vec<ClusterResult>, &'a Vec<ClusterCentersResult>);
    type Output = Vec<GyrationTensorResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        (clusters, centers): Self::Args<'a>,
    ) -> Result<Vec<GyrationTensorResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if clusters.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: clusters.len(),
                what: "ClusterResult count",
            });
        }
        if centers.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: centers.len(),
                what: "ClusterCentersResult count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 4;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(clusters.par_iter())
                .zip(centers.par_iter())
                .map(|((frame, cl), cc)| self.one_frame(*frame, cl, cc))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for ((frame, cl), cc) in frames.iter().zip(clusters.iter()).zip(centers.iter()) {
            out.push(self.one_frame(*frame, cl, cc)?);
        }
        Ok(out)
    }
}

/// Per-cluster gyration tensors for one frame.
///
/// `self.0[c][a][b]` is the `(a, b)` component of the gyration tensor of
/// cluster `c`, in **Å²**. The tensor is particle-count-normalized:
/// `G[a][b] = (1/N) Σ s_a s_b` where `s` is the MIC displacement from the
/// cluster's geometric center.
///
/// # References
///
/// Theodorou & Suter, *Macromolecules* **18**, 1206 (1985).
#[derive(Debug, Clone, Default)]
pub struct GyrationTensorResult(pub Vec<[[F; 3]; 3]>);

impl ComputeResult for GyrationTensorResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::ClusterCenters;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
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

    fn manual_clusters(idx: &[i64]) -> ClusterResult {
        let nc = (*idx.iter().max().unwrap_or(&-1) + 1).max(0) as usize;
        let mut sizes = vec![0usize; nc];
        for &c in idx {
            if c >= 0 {
                sizes[c as usize] += 1;
            }
        }
        ClusterResult {
            cluster_idx: ndarray::Array1::from_vec(idx.to_vec()),
            n_clusters: nc,
            cluster_sizes: sizes,
            cluster_keys: vec![],
        }
    }

    fn gyration_single(frame: &Frame, cl: ClusterResult) -> GyrationTensorResult {
        let centers = ClusterCenters::new()
            .compute(&[frame], &vec![cl.clone()])
            .unwrap();
        let out = GyrationTensor::new()
            .compute(&[frame], (&vec![cl], &centers))
            .unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let g = gyration_single(&frame, cl);
        for a in 0..3 {
            for b in 0..3 {
                assert!(g.0[0][a][b].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn off_center_cluster() {
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let g = gyration_single(&frame, cl);
        let expected = [
            [0.0025, 0.0025, 0.0],
            [0.0025, 0.0025, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (g.0[0][a][b] - expected[a][b] as F).abs() < 1e-5,
                    "G[{a}][{b}] = {}, expected {}",
                    g.0[0][a][b],
                    expected[a][b]
                );
            }
        }
    }

    #[test]
    fn symmetric() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0, 0]);
        let g = gyration_single(&frame, cl);
        for a in 0..3 {
            for b in 0..3 {
                assert!((g.0[0][a][b] - g.0[0][b][a]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_particles_along_x() {
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let g = gyration_single(&frame, cl);
        assert!((g.0[0][0][0] - 1.0).abs() < 1e-5);
        assert!(g.0[0][1][1].abs() < 1e-5);
        assert!(g.0[0][0][1].abs() < 1e-5);
    }

    #[test]
    fn single_particle_zero() {
        let pos = [[5.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0]);
        let g = gyration_single(&frame, cl);
        for a in 0..3 {
            for b in 0..3 {
                assert!(g.0[0][a][b].abs() < 1e-10);
            }
        }
    }
}
