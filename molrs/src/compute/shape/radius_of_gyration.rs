//! Radius of gyration computation for clusters.

use crate::compute::{ComputeResult, DescriptorRow};
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::CenterOfMassResult;
use crate::compute::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Computes the radius of gyration for each cluster per frame.
///
/// `R_g_k = sqrt( (1/M_k) * SUM_i m_i * |s_i|^2 )`
/// where `s_i = shortest_vector(com_k, r_i)` is the MIC displacement from the
/// center of mass. Centers of mass come from the [`CenterOfMassResult`] arg — this
/// Compute does **not** recompute them.
///
/// Reads `atoms.{x,y,z}` (Å); `Args` = per-frame
/// ([`ClusterResult`],
/// [`CenterOfMassResult`]) pairs — run
/// [`Cluster`](crate::compute::Cluster) and
/// [`CenterOfMass`](crate::compute::CenterOfMass) first. Output:
/// per-cluster `R_g` (Å).
#[derive(Debug, Clone, Default)]
pub struct RadiusOfGyration {
    masses: Option<Vec<F>>,
}

impl RadiusOfGyration {
    /// Uniform masses (every particle weighs 1).
    pub fn new() -> Self {
        Self { masses: None }
    }

    /// Per-particle masses (length = particle count; only ratios matter).
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
    ) -> Result<RadiusOfGyrationResult, ComputeError> {
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
                what: "RadiusOfGyration::masses",
            });
        }

        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;

        if com.centers_of_mass.len() != nc || com.cluster_masses.len() != nc {
            return Err(ComputeError::DimensionMismatch {
                expected: nc,
                got: com.centers_of_mass.len(),
                what: "CenterOfMassResult cluster count",
            });
        }

        let mut rg_sum = vec![0.0 as F; nc];
        let masses_ref = self.masses.as_deref();

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = masses_ref.map_or(1.0 as F, |ms| ms[i]);
            let s = mic.apply(sub(pos, com.centers_of_mass[c]));
            let s_sq = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];
            rg_sum[c] += m * s_sq;
        }

        let mut radii = vec![0.0 as F; nc];
        for c in 0..nc {
            if com.cluster_masses[c] > 0.0 {
                radii[c] = (rg_sum[c] / com.cluster_masses[c]).sqrt();
            }
        }

        Ok(RadiusOfGyrationResult(radii))
    }
}

impl LegacyCompute for RadiusOfGyration {
    type Args<'a> = (&'a Vec<ClusterResult>, &'a Vec<CenterOfMassResult>);
    type Output = Vec<RadiusOfGyrationResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        (clusters, com): Self::Args<'a>,
    ) -> Result<Vec<RadiusOfGyrationResult>, ComputeError> {
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

/// Per-cluster radii of gyration for one frame.
///
/// `self.0[c]` is the radius of gyration of cluster `c`, in **Å**.
#[derive(Debug, Clone, Default)]
pub struct RadiusOfGyrationResult(pub Vec<F>);

impl ComputeResult for RadiusOfGyrationResult {}

impl DescriptorRow for RadiusOfGyrationResult {
    fn as_row(&self) -> &[F] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::CenterOfMass;
    use crate::compute::InertiaTensor;
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

    fn rg_single(frame: &Frame, cl: ClusterResult, rg: RadiusOfGyration) -> RadiusOfGyrationResult {
        let masses: Option<Vec<F>> = rg.masses.clone();
        let com_calc = match masses {
            Some(ref ms) => CenterOfMass::new().with_masses(ms),
            None => CenterOfMass::new(),
        };
        let com = com_calc.compute(&[frame], &vec![cl.clone()]).unwrap();
        let out = rg.compute(&[frame], (&vec![cl], &com)).unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn zero_for_coincident() {
        let pos = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = rg_single(&frame, cl, RadiusOfGyration::new().with_masses(&[2.0, 5.0]));
        assert!(rg.0[0].abs() < 1e-10);
    }

    #[test]
    fn single_particle_zero() {
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0);
        let cl = manual_clusters(&[0]);
        let rg = rg_single(&frame, cl, RadiusOfGyration::new());
        assert!(rg.0[0].abs() < 1e-10);
    }

    #[test]
    fn weighted_off_center() {
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = rg_single(&frame, cl, RadiusOfGyration::new().with_masses(&[3.0, 4.0]));
        assert!(
            (rg.0[0] - 0.0699854212).abs() < 1e-4,
            "rg = {}, expected ~0.0700",
            rg.0[0]
        );
    }

    #[test]
    fn two_equal_mass_along_x() {
        let pos = [[2.0, 3.0, 3.0], [4.0, 3.0, 3.0]];
        let frame = frame_with(&pos, 10.0);
        let cl = manual_clusters(&[0, 0]);
        let rg = rg_single(&frame, cl, RadiusOfGyration::new());
        assert!((rg.0[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rg_equals_trace_of_inertia_over_2m() {
        let pos = [[1.0, 3.0, 1.0], [0.9, 2.9, 1.0]];
        let masses: Vec<F> = vec![3.0, 4.0];
        let frame = frame_with(&pos, 6.0);
        let cl = manual_clusters(&[0, 0]);

        let com = CenterOfMass::new()
            .with_masses(&masses)
            .compute(&[&frame], &vec![cl.clone()])
            .unwrap();
        let rg = RadiusOfGyration::new()
            .with_masses(&masses)
            .compute(&[&frame], (&vec![cl.clone()], &com))
            .unwrap();
        let inertia = InertiaTensor::new()
            .with_masses(&masses)
            .compute(&[&frame], (&vec![cl], &com))
            .unwrap();

        let trace = inertia[0].0[0][0][0] + inertia[0].0[0][1][1] + inertia[0].0[0][2][2];
        let rg_from_trace = (trace / (2.0 * com[0].cluster_masses[0])).sqrt();
        assert!(
            (rg[0].0[0] - rg_from_trace).abs() < 1e-5,
            "rg={}, from_trace={}",
            rg[0].0[0],
            rg_from_trace
        );
    }
}
