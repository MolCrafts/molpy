// Tensor accumulation in fixed 3×3 windows reads more clearly with
// double-indexed loops than with iterator zips.
#![allow(clippy::needless_range_loop)]

//! Per-cluster scalar / tensor properties, freud-compatible aggregator.

use molrs::core::FrameAccess;
use molrs::op::F;

use super::ClusterResult;
use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::op::vec3::sub;
use molrs::core::{Mic, SimBox};

/// Per-frame bundle of cluster scalars and tensors.
#[derive(Debug, Clone, Default)]
pub struct ClusterPropertiesResult {
    /// Particle count per cluster.
    pub sizes: Vec<usize>,
    /// Unweighted geometric center per cluster.
    pub centers: Vec<[F; 3]>,
    /// Mass-weighted center per cluster.
    pub centers_of_mass: Vec<[F; 3]>,
    /// Total mass per cluster.
    pub cluster_masses: Vec<F>,
    /// Mass-weighted gyration tensors, row-major 3×3 per cluster.
    pub gyration_tensors: Vec<[[F; 3]; 3]>,
    /// Radius of gyration per cluster = sqrt(trace G).
    pub radii_of_gyration: Vec<F>,
}

impl ComputeResult for ClusterPropertiesResult {}

/// freud-style `ClusterProperties`: bundles size, center, COM, gyration
/// tensor, and RG into a single per-cluster pass.
///
/// Run [`Cluster`](crate::compute::Cluster) first and pass its
/// [`ClusterResult`](crate::compute::ClusterResult) as `Args`.
///
/// Mirrors `freud.cluster.ClusterProperties`: for each cluster in a frame,
/// reports its size, geometric center, mass-weighted center, the (mass-
/// weighted) gyration tensor, and the scalar radius of gyration. All
/// quantities are PBC-aware via [`Mic`](molrs::core::Mic): the first atom assigned to
/// each cluster is used as the local reference and subsequent atom positions
/// are accumulated through minimum-image displacements, so a cluster that
/// wraps across the box boundary is handled correctly.
///
/// # Conventions (matching `freud.cluster.ClusterProperties`)
///
/// - `center`             unweighted mean position
/// - `center_of_mass`     mass-weighted mean position (equal to `center`
///   when no masses are supplied)
/// - `gyration_tensors`   `G_ab = (1/M) Σ_i m_i (r_i − r_com)_a (r_i − r_com)_b`
/// - `radii_of_gyration`  `√(trace G)`
/// - `sizes`              particle counts per cluster
///
/// Atoms with `cluster_idx < 0` (filtered by `min_cluster_size`) are ignored.
#[derive(Debug, Clone, Default)]
pub struct ClusterProperties {
    masses: Option<Vec<F>>,
}

impl ClusterProperties {
    /// No parameters.
    pub fn new() -> Self {
        Self { masses: None }
    }

    /// Optional per-particle masses (length must equal atom count).
    pub fn with_masses(self, masses: &[F]) -> Self {
        Self {
            masses: Some(masses.to_vec()),
        }
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        clusters: &ClusterResult,
    ) -> Result<ClusterPropertiesResult, ComputeError> {
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
                what: "ClusterProperties::masses",
            });
        }

        let mic = frame.simbox_ref().map_or(Mic::Free, SimBox::mic);
        let nc = clusters.n_clusters;
        let masses_ref = self.masses.as_deref();

        // First pass: per-cluster reference atom + accumulated displacement sums.
        let mut ref_pos = vec![[0.0_f64; 3]; nc];
        let mut has_ref = vec![false; nc];
        let mut sum_d = vec![[0.0_f64; 3]; nc]; // unweighted
        let mut sum_m_d = vec![[0.0_f64; 3]; nc]; // mass-weighted
        let mut counts = vec![0_usize; nc];
        let mut total_mass = vec![0.0_f64; nc];

        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = masses_ref.map_or(1.0, |ms| ms[i]);

            if !has_ref[c] {
                ref_pos[c] = pos;
                has_ref[c] = true;
            }
            let d = mic.apply(sub(pos, ref_pos[c]));
            sum_d[c][0] += d[0];
            sum_d[c][1] += d[1];
            sum_d[c][2] += d[2];
            sum_m_d[c][0] += m * d[0];
            sum_m_d[c][1] += m * d[1];
            sum_m_d[c][2] += m * d[2];
            counts[c] += 1;
            total_mass[c] += m;
        }

        let mut centers = vec![[0.0_f64; 3]; nc];
        let mut centers_of_mass = vec![[0.0_f64; 3]; nc];
        for c in 0..nc {
            let count = counts[c] as F;
            if count > 0.0 {
                centers[c][0] = ref_pos[c][0] + sum_d[c][0] / count;
                centers[c][1] = ref_pos[c][1] + sum_d[c][1] / count;
                centers[c][2] = ref_pos[c][2] + sum_d[c][2] / count;
            }
            if total_mass[c] > 0.0 {
                let m = total_mass[c];
                centers_of_mass[c][0] = ref_pos[c][0] + sum_m_d[c][0] / m;
                centers_of_mass[c][1] = ref_pos[c][1] + sum_m_d[c][1] / m;
                centers_of_mass[c][2] = ref_pos[c][2] + sum_m_d[c][2] / m;
            }
        }

        // Second pass: gyration tensor about the (mass-weighted) center of mass.
        let mut gyration = vec![[[0.0_f64; 3]; 3]; nc];
        for (i, &cid) in clusters.cluster_idx.iter().enumerate() {
            if cid < 0 {
                continue;
            }
            let c = cid as usize;
            let pos = [xs[i], ys[i], zs[i]];
            let m = masses_ref.map_or(1.0, |ms| ms[i]);
            let d = mic.apply(sub(pos, centers_of_mass[c]));
            for a in 0..3 {
                for b in 0..3 {
                    gyration[c][a][b] += m * d[a] * d[b];
                }
            }
        }
        // Normalise by total mass.
        for c in 0..nc {
            if total_mass[c] > 0.0 {
                let inv = 1.0 / total_mass[c];
                for a in 0..3 {
                    for b in 0..3 {
                        gyration[c][a][b] *= inv;
                    }
                }
            }
        }

        let radii_of_gyration: Vec<F> = (0..nc)
            .map(|c| {
                let tr = gyration[c][0][0] + gyration[c][1][1] + gyration[c][2][2];
                tr.max(0.0).sqrt()
            })
            .collect();

        Ok(ClusterPropertiesResult {
            sizes: counts,
            centers,
            centers_of_mass,
            cluster_masses: total_mass,
            gyration_tensors: gyration,
            radii_of_gyration,
        })
    }
}

impl LegacyCompute for ClusterProperties {
    type Args<'a> = &'a Vec<ClusterResult>;
    type Output = Vec<ClusterPropertiesResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        clusters: &'a Vec<ClusterResult>,
    ) -> Result<Vec<ClusterPropertiesResult>, ComputeError> {
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

    fn single(frame: &Frame, cl: ClusterResult, cp: ClusterProperties) -> ClusterPropertiesResult {
        let out = cp.compute(&[frame], &vec![cl]).unwrap();
        out.into_iter().next().unwrap()
    }

    #[test]
    fn two_particle_uniform_mass() {
        // (0,0,0) and (2,0,0) → center = (1,0,0), Rg = 1, gyration[0][0]=1
        let frame = frame_with(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], 10.0, [false; 3]);
        let cl = manual_clusters(&[0, 0]);
        let r = single(&frame, cl, ClusterProperties::new());

        assert_eq!(r.sizes, vec![2]);
        assert!((r.centers[0][0] - 1.0).abs() < 1e-10);
        assert!((r.centers_of_mass[0][0] - 1.0).abs() < 1e-10);
        assert!((r.cluster_masses[0] - 2.0).abs() < 1e-10);
        // Gyration tensor: G_xx = (1/M) Σ m_i (x_i - x_c)² = (1+1)/2 = 1
        assert!((r.gyration_tensors[0][0][0] - 1.0).abs() < 1e-10);
        // Rg = sqrt(trace) = sqrt(1) = 1
        assert!((r.radii_of_gyration[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn weighted_center_shifts_to_heavy_atom() {
        // Masses 3, 1 at (1,0,0) and (3,0,0).
        // CoM = (3·1 + 1·3) / (3+1) = 6/4 = 1.5
        let frame = frame_with(&[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], 10.0, [false; 3]);
        let cl = manual_clusters(&[0, 0]);
        let r = single(
            &frame,
            cl,
            ClusterProperties::new().with_masses(&[3.0, 1.0]),
        );
        assert!((r.centers_of_mass[0][0] - 1.5).abs() < 1e-10);
        // Unweighted geometric center is still 2.0.
        assert!((r.centers[0][0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn two_clusters_independent() {
        let pos = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [5.0, 0.0, 0.0],
            [7.0, 0.0, 0.0],
        ];
        let frame = frame_with(&pos, 20.0, [false; 3]);
        let cl = manual_clusters(&[0, 0, 1, 1]);
        let r = single(&frame, cl, ClusterProperties::new());
        assert_eq!(r.sizes, vec![2, 2]);
        // Cluster 0 (two coincident particles): center = (1,1,1), Rg = 0
        assert!((r.centers[0][0] - 1.0).abs() < 1e-10);
        assert!(r.radii_of_gyration[0].abs() < 1e-10);
        // Cluster 1: center = (6, 0, 0), Rg = 1
        assert!((r.centers[1][0] - 6.0).abs() < 1e-10);
        assert!((r.radii_of_gyration[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn filtered_atoms_ignored() {
        let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 100.0, 100.0]];
        let frame = frame_with(&pos, 200.0, [false; 3]);
        let cl = manual_clusters(&[0, 0, -1]); // third atom filtered
        let r = single(&frame, cl, ClusterProperties::new());
        assert_eq!(r.sizes, vec![2]);
        assert!((r.centers[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pbc_wrapped_cluster() {
        // Atoms at x=0.5 and x=9.5 in a box of length 10 with PBC.
        // True separation is 1.0 (via wrap), unwrapped center is 0 (or 10).
        let pos = [[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]];
        let frame = frame_with(&pos, 10.0, [true, true, true]);
        let cl = manual_clusters(&[0, 0]);
        let r = single(&frame, cl, ClusterProperties::new());
        // Gyration tensor x-component should reflect 1.0 separation, NOT 9.0.
        // G_xx = (0.5)² for each particle around their unwrapped center → 0.25.
        assert!(
            (r.gyration_tensors[0][0][0] - 0.25).abs() < 1e-9,
            "PBC-aware G_xx = {} (expected 0.25)",
            r.gyration_tensors[0][0][0]
        );
        assert!((r.radii_of_gyration[0] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn masses_dimension_mismatch_errors() {
        let frame = frame_with(&[[0.0, 0.0, 0.0]; 2], 10.0, [false; 3]);
        let cl = manual_clusters(&[0, 0]);
        let err = ClusterProperties::new()
            .with_masses(&[1.0])
            .compute(&[&frame], &vec![cl])
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
