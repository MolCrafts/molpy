//! Geometric cluster centers computed with minimum image convention.

use crate::compute::{ComputeResult, DescriptorRow};
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Computes the geometric center of each cluster per frame using the minimum
/// image convention (MIC).
///
/// Algorithm: for each cluster, pick the first particle as reference,
/// accumulate MIC-corrected displacements, then average.
///
/// The unweighted counterpart of
/// [`CenterOfMass`](crate::compute::CenterOfMass). Reads
/// `atoms.{x,y,z}` (Å); takes one
/// [`ClusterResult`] per frame as
/// `Args`. Output: per-cluster geometric centers (Å), one
/// [`ClusterCentersResult`] per frame.
#[derive(Debug, Clone, Default)]
pub struct ClusterCenters;

impl ClusterCenters {
    /// No parameters — geometric (unweighted) centers.
    pub fn new() -> Self {
        Self
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<ClusterCentersResult, ComputeError> {
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;

        let mut ref_pos = vec![[0.0 as F; 3]; nc];
        let mut sum_delta = vec![[0.0 as F; 3]; nc];
        let mut counts = vec![0usize; nc];
        let mut has_ref = vec![false; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];

            if !has_ref[c] {
                ref_pos[c] = pos;
                has_ref[c] = true;
            }

            let d = mic.apply(sub(pos, ref_pos[c]));
            sum_delta[c][0] += d[0];
            sum_delta[c][1] += d[1];
            sum_delta[c][2] += d[2];
            counts[c] += 1;
        }

        let mut centers = vec![[0.0 as F; 3]; nc];
        for c in 0..nc {
            if counts[c] > 0 {
                let n = counts[c] as F;
                centers[c][0] = ref_pos[c][0] + sum_delta[c][0] / n;
                centers[c][1] = ref_pos[c][1] + sum_delta[c][1] / n;
                centers[c][2] = ref_pos[c][2] + sum_delta[c][2] / n;
            }
        }

        Ok(ClusterCentersResult { centers })
    }
}

impl LegacyCompute for ClusterCenters {
    type Args<'a> = &'a Vec<ClusterResult>;
    type Output = Vec<ClusterCentersResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        clusters: &'a Vec<ClusterResult>,
    ) -> Result<Vec<ClusterCentersResult>, ComputeError> {
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
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 4;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(clusters.par_iter())
                .map(|(frame, cl)| self.one_frame(*frame, cl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (frame, cl) in frames.iter().zip(clusters.iter()) {
            out.push(self.one_frame(*frame, cl)?);
        }
        Ok(out)
    }
}

/// Per-cluster geometric center for one frame.
#[derive(Debug, Clone, Default)]
pub struct ClusterCentersResult {
    pub centers: Vec<[F; 3]>,
}

impl ClusterCentersResult {
    /// Wrap precomputed centers (Å) (one entry per cluster).
    pub fn new(centers: Vec<[F; 3]>) -> Self {
        Self { centers }
    }

    pub fn len(&self) -> usize {
        self.centers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }
}

impl ComputeResult for ClusterCentersResult {}

impl DescriptorRow for ClusterCentersResult {
    fn as_row(&self) -> &[F] {
        // Flatten [[x,y,z]; n] → &[F; 3n]. Safe: nested arrays are layout-
        // compatible with a flat [F; 3n] array.
        // SAFETY: `[F; 3]` has the same memory layout as three consecutive Fs.
        let len = self.centers.len() * 3;
        let ptr = self.centers.as_ptr() as *const F;
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Cluster;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    fn clusters_via_nlist(frame: &Frame, cutoff: F) -> ClusterResult {
        let nlist = nlist_from_frame(frame, cutoff);
        let out = Cluster::new(1).compute(&[frame], &[nlist]).unwrap();
        out.into_iter().next().unwrap()
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

    fn centers_single(frame: &Frame, cl: ClusterResult) -> ClusterCentersResult {
        let out = ClusterCenters::new().compute(&[frame], &vec![cl]).unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn coincident_and_offset_clusters() {
        let pos = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ];
        let frame = frame_with(&pos, 6.0, [false, false, false]);
        let cl = clusters_via_nlist(&frame, 0.5);
        assert_eq!(cl.n_clusters, 2);

        let (ca, cb) = if cl.cluster_idx[0] == 0 {
            (0, 1)
        } else {
            (1, 0)
        };
        let centers = centers_single(&frame, cl);

        assert!((centers.centers[ca][0] - 1.0).abs() < 1e-5);
        assert!((centers.centers[ca][1] - 1.0).abs() < 1e-5);
        assert!((centers.centers[cb][0] - 0.95).abs() < 1e-5);
        assert!((centers.centers[cb][1] - 2.95).abs() < 1e-5);
    }

    #[test]
    fn mic_wrapping_across_boundary() {
        let pos = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [true, true, true]);
        let cl = clusters_via_nlist(&frame, 2.0);
        assert_eq!(cl.n_clusters, 1);

        let centers = centers_single(&frame, cl);
        let cx = centers.centers[0][0];
        assert!(
            !(1.0..=9.0).contains(&cx),
            "center should wrap near boundary, got {cx}"
        );
    }

    #[test]
    fn single_particle() {
        let pos = [[3.0, 4.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0]);
        let centers = centers_single(&frame, cl);
        assert!((centers.centers[0][0] - 3.0).abs() < 1e-5);
        assert!((centers.centers[0][1] - 4.0).abs() < 1e-5);
        assert!((centers.centers[0][2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn empty_cluster_set() {
        let pos: Vec<[F; 3]> = vec![];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[]);
        let centers = centers_single(&frame, cl);
        assert!(centers.centers.is_empty());
    }

    #[test]
    fn free_boundary() {
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let mut frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        frame.simbox = None;
        let centers = centers_single(&frame, cl);
        assert!((centers.centers[0][0] - 2.0).abs() < 1e-5);
    }
}
