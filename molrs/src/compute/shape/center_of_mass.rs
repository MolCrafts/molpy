//! Mass-weighted cluster centers (center of mass) with MIC.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Computes the center of mass of each cluster per frame using MIC.
///
/// Masses are optional — defaults to 1.0 for all particles (uniform).
///
/// Reads `atoms.{x,y,z}` (Å) from each frame; takes one
/// [`ClusterResult`] per frame as
/// `Args` (run [`Cluster`](crate::compute::Cluster) first). Output:
/// one [`CenterOfMassResult`] per frame — per-cluster COM (Å) + total mass.
#[derive(Debug, Clone, Default)]
pub struct CenterOfMass {
    masses: Option<Vec<F>>,
}

impl CenterOfMass {
    /// Uniform masses (every particle weighs 1).
    pub fn new() -> Self {
        Self { masses: None }
    }

    /// Per-particle masses (length = particle count; any mass unit — only
    /// ratios matter for the center).
    pub fn with_masses(self, masses: &[F]) -> Self {
        Self {
            masses: Some(masses.to_vec()),
        }
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<CenterOfMassResult, ComputeError> {
        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();
        let n = xs.len();

        if let Some(ref ms) = self.masses
            && ms.len() != n
        {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: ms.len(),
                what: "CenterOfMass::masses",
            });
        }

        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;

        let mut ref_pos = vec![[0.0 as F; 3]; nc];
        let mut sum_m_delta = vec![[0.0 as F; 3]; nc];
        let mut total_mass = vec![0.0 as F; nc];
        let mut has_ref = vec![false; nc];

        let masses_ref = self.masses.as_deref();
        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = masses_ref.map_or(1.0 as F, |ms| ms[i]);

            if !has_ref[c] {
                ref_pos[c] = pos;
                has_ref[c] = true;
            }

            let d = mic.apply(sub(pos, ref_pos[c]));
            sum_m_delta[c][0] += m * d[0];
            sum_m_delta[c][1] += m * d[1];
            sum_m_delta[c][2] += m * d[2];
            total_mass[c] += m;
        }

        let mut centers_of_mass = vec![[0.0 as F; 3]; nc];
        for c in 0..nc {
            if total_mass[c] > 0.0 {
                let m = total_mass[c];
                centers_of_mass[c][0] = ref_pos[c][0] + sum_m_delta[c][0] / m;
                centers_of_mass[c][1] = ref_pos[c][1] + sum_m_delta[c][1] / m;
                centers_of_mass[c][2] = ref_pos[c][2] + sum_m_delta[c][2] / m;
            }
        }

        Ok(CenterOfMassResult {
            centers_of_mass,
            cluster_masses: total_mass,
        })
    }
}

impl LegacyCompute for CenterOfMass {
    type Args<'a> = &'a Vec<ClusterResult>;
    type Output = Vec<CenterOfMassResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        clusters: &'a Vec<ClusterResult>,
    ) -> Result<Vec<CenterOfMassResult>, ComputeError> {
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

/// Per-cluster center of mass and total mass for one frame.
#[derive(Debug, Clone, Default)]
pub struct CenterOfMassResult {
    /// Mass-weighted center per cluster.
    pub centers_of_mass: Vec<[F; 3]>,
    /// Total mass per cluster.
    pub cluster_masses: Vec<F>,
}

impl ComputeResult for CenterOfMassResult {}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn com_single(frame: &Frame, cl: ClusterResult, com: CenterOfMass) -> CenterOfMassResult {
        let out = com.compute(&[frame], &vec![cl]).unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn uniform_mass_equals_geometric_center() {
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let r = com_single(&frame, cl, CenterOfMass::new());
        assert!((r.centers_of_mass[0][0] - 2.0).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn weighted_shifts_toward_heavy() {
        let pos = [[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let r = com_single(&frame, cl, CenterOfMass::new().with_masses(&[3.0, 1.0]));
        assert!((r.centers_of_mass[0][0] - 1.5).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn two_clusters_weighted() {
        let pos = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [0.9, 2.9, 1.0],
        ];
        let masses: Vec<F> = vec![1.0, 2.0, 3.0, 4.0];
        let frame = frame_with(&pos, 6.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0, 1, 1]);
        let r = com_single(&frame, cl, CenterOfMass::new().with_masses(&masses));
        assert!((r.centers_of_mass[0][0] - 1.0).abs() < 1e-5);
        assert!((r.cluster_masses[0] - 3.0).abs() < 1e-5);
        assert!((r.centers_of_mass[1][0] - 6.6 / 7.0).abs() < 1e-4);
        assert!((r.cluster_masses[1] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn periodic_wrapping() {
        let pos = [[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [true, true, true]);
        let cl = manual_clusters(&[0, 0]);
        let r = com_single(&frame, cl, CenterOfMass::new().with_masses(&[1.0, 3.0]));
        let cx = r.centers_of_mass[0][0];
        assert!(
            (cx - 9.5).abs() < 0.5 || (cx + 0.5).abs() < 0.5,
            "COM should be near 9.5 (or -0.5), got {cx}"
        );
    }

    #[test]
    fn masses_dimension_mismatch() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let frame = frame_with(&pos, 10.0, [false, false, false]);
        let cl = manual_clusters(&[0, 0]);
        let err = CenterOfMass::new()
            .with_masses(&[1.0])
            .compute(&[&frame], &vec![cl])
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
