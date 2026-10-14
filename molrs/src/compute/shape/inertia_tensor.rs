//! Moment of inertia tensor computation for clusters.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::CenterOfMassResult;
use crate::compute::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Moment of inertia tensor per cluster, per frame.
///
/// `I_k[a][b] = SUM_i m_i * (|s_i|^2 * delta_ab - s_i[a] * s_i[b])`
/// where `s_i = shortest_vector(com_k, r_i)` is the MIC displacement from the
/// center of mass. Centers of mass come from the [`CenterOfMassResult`] arg — this
/// Compute does **not** recompute them.
///
/// Reads `atoms.{x,y,z}` (Å); `Args` = per-frame
/// ([`ClusterResult`],
/// [`CenterOfMassResult`]) pairs — run
/// [`Cluster`](crate::compute::Cluster) and
/// [`CenterOfMass`](crate::compute::CenterOfMass) first. Output:
/// per-cluster 3×3 inertia tensors (mass·Å²) (mass unit = whatever
/// `with_masses` supplies; 1 per particle by default).
#[derive(Debug, Clone, Default)]
pub struct InertiaTensor {
    masses: Option<Vec<F>>,
}

impl InertiaTensor {
    /// Uniform masses (every particle weighs 1).
    pub fn new() -> Self {
        Self { masses: None }
    }

    /// Per-particle masses (length = particle count), e.g. amu.
    pub fn with_masses(self, masses: &[F]) -> Self {
        Self {
            masses: Some(masses.to_vec()),
        }
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
        com: &CenterOfMassResult,
    ) -> Result<InertiaTensorResult, ComputeError> {
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
                what: "InertiaTensor::masses",
            });
        }

        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;

        if com.centers_of_mass.len() != nc {
            return Err(ComputeError::DimensionMismatch {
                expected: nc,
                got: com.centers_of_mass.len(),
                what: "CenterOfMassResult cluster count",
            });
        }

        let mut tensors = vec![[[0.0 as F; 3]; 3]; nc];
        let masses_ref = self.masses.as_deref();

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = masses_ref.map_or(1.0 as F, |ms| ms[i]);
            let s = mic.apply(sub(pos, com.centers_of_mass[c]));
            let sx = s[0];
            let sy = s[1];
            let sz = s[2];
            let s_sq = sx * sx + sy * sy + sz * sz;
            let t = &mut tensors[c];
            t[0][0] += m * (s_sq - sx * sx);
            t[0][1] += m * (-sx * sy);
            t[0][2] += m * (-sx * sz);
            t[1][0] += m * (-sy * sx);
            t[1][1] += m * (s_sq - sy * sy);
            t[1][2] += m * (-sy * sz);
            t[2][0] += m * (-sz * sx);
            t[2][1] += m * (-sz * sy);
            t[2][2] += m * (s_sq - sz * sz);
        }

        Ok(InertiaTensorResult(tensors))
    }
}

impl LegacyCompute for InertiaTensor {
    type Args<'a> = (&'a Vec<ClusterResult>, &'a Vec<CenterOfMassResult>);
    type Output = Vec<InertiaTensorResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        (clusters, com): Self::Args<'a>,
    ) -> Result<Vec<InertiaTensorResult>, ComputeError> {
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
        if com.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: com.len(),
                what: "CenterOfMassResult count",
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
                .zip(com.par_iter())
                .map(|((frame, cl), com_i)| self.one_frame(*frame, cl, com_i))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for ((frame, cl), com_i) in frames.iter().zip(clusters.iter()).zip(com.iter()) {
            out.push(self.one_frame(*frame, cl, com_i)?);
        }
        Ok(out)
    }
}

/// Per-cluster moment of inertia tensors for one frame.
///
/// `self.0[c][a][b]` is the `(a, b)` component of the inertia tensor of
/// cluster `c`, in **amu·Å²** when masses are in amu. If `with_masses` is
/// not set, masses default to 1.0, so the result is in **Å²** units of the
/// "inertia-like" sum `Σ (|s|² δ_ab − s_a s_b)`.
#[derive(Debug, Clone, Default)]
pub struct InertiaTensorResult(pub Vec<[[F; 3]; 3]>);

impl ComputeResult for InertiaTensorResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::CenterOfMass;
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

    fn inertia_single(
        frame: &Frame,
        cl: ClusterResult,
        inertia: InertiaTensor,
    ) -> InertiaTensorResult {
        let com_calc = match inertia.masses.as_ref() {
            Some(ms) => CenterOfMass::new().with_masses(ms),
            None => CenterOfMass::new(),
        };
        let com = com_calc.compute(&[frame], &vec![cl.clone()]).unwrap();
        let out = inertia.compute(&[frame], (&vec![cl], &com)).unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let t = inertia_single(&frame, cl, InertiaTensor::new().with_masses(&[2.0, 5.0]));
        for a in 0..3 {
            for b in 0..3 {
                assert!(t.0[0][a][b].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn single_particle_zero() {
        let pos = [[5.0, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0]);
        let t = inertia_single(&frame, cl, InertiaTensor::new().with_masses(&[3.0]));
        for a in 0..3 {
            for b in 0..3 {
                assert!(t.0[0][a][b].abs() < 1e-10);
            }
        }
    }

    #[test]
    fn weighted_off_center() {
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let masses: Vec<F> = vec![3.0, 4.0];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let t = inertia_single(&frame, cl, InertiaTensor::new().with_masses(&masses));

        let expected: [[f64; 3]; 3] = [
            [0.0171429, -0.0171429, 0.0],
            [-0.0171429, 0.0171429, 0.0],
            [0.0, 0.0, 0.0342857],
        ];
        for (a, row) in expected.iter().enumerate() {
            for (b, &e) in row.iter().enumerate() {
                assert!(
                    (t.0[0][a][b] as f64 - e).abs() < 1e-4,
                    "I[{a}][{b}] = {}, expected {e}",
                    t.0[0][a][b]
                );
            }
        }
    }

    #[test]
    fn symmetric() {
        let pos = [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 1.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0, 0]);
        let t = inertia_single(&frame, cl, InertiaTensor::new());
        for a in 0..3 {
            for b in 0..3 {
                assert!((t.0[0][a][b] - t.0[0][b][a]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn two_along_x() {
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let t = inertia_single(&frame, cl, InertiaTensor::new());
        assert!(t.0[0][0][0].abs() < 1e-5);
        assert!((t.0[0][1][1] - 2.0).abs() < 1e-5);
        assert!((t.0[0][2][2] - 2.0).abs() < 1e-5);
    }
}
