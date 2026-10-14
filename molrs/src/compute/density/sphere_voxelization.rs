//! Boolean voxel rasterisation of point particles as hard spheres.

use super::wrap_index;
use crate::compute::ComputeResult;
use ndarray::Array3;

use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// Sphere-voxelisation calculator.
///
/// Mirrors `freud.density.SphereVoxelization`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/SphereVoxelization.cc)).
///
/// For each particle of radius `r_max`, every voxel whose centre lies inside
/// that sphere is set to `1`. The output is the boolean overlap mask
/// (stored as a `u32` grid so callers can also use it for "count of
/// overlapping particles per voxel" by reading the `raw_counts` field).
///
/// Like [`GaussianDensity`](super::gaussian_density::GaussianDensity), this
/// is orthorhombic-box only and PBC-aware via wrap-around grid indexing.
#[derive(Debug, Clone, Copy)]
pub struct SphereVoxelization {
    nx: usize,
    ny: usize,
    nz: usize,
    r_max: F,
}

impl SphereVoxelization {
    /// `nx × ny × nz` voxel grid; particles rasterised as spheres of radius `r_max` (Å).
    pub fn new(nx: usize, ny: usize, nz: usize, r_max: F) -> Result<Self, ComputeError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(ComputeError::OutOfRange {
                field: "SphereVoxelization::width",
                value: format!("({nx}, {ny}, {nz})"),
            });
        }
        if r_max.is_nan() || r_max <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "SphereVoxelization::r_max",
                value: r_max.to_string(),
            });
        }
        Ok(Self { nx, ny, nz, r_max })
    }

    pub fn width(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
    pub fn r_max(&self) -> F {
        self.r_max
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
    ) -> Result<SphereVoxelizationResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly, lz) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "SphereVoxelization::simbox",
                    value: "triclinic boxes are not supported".into(),
                });
            }
        };
        let origin = simbox.origin_view();
        let ox = origin[0];
        let oy = origin[1];
        let oz = origin[2];
        let pbc = simbox.pbc();

        let dx = lx / self.nx as F;
        let dy = ly / self.ny as F;
        let dz = lz / self.nz as F;

        let (xs_p, ys_p, zs_p) = get_positions_ref(frame)?;
        let xs = xs_p.slice();
        let ys = ys_p.slice();
        let zs = zs_p.slice();

        let mut voxels = Array3::<u8>::zeros((self.nx, self.ny, self.nz));
        let mut counts = Array3::<u32>::zeros((self.nx, self.ny, self.nz));
        let r_max_sq = self.r_max * self.r_max;
        let half_kx = (self.r_max / dx).ceil() as isize;
        let half_ky = (self.r_max / dy).ceil() as isize;
        let half_kz = (self.r_max / dz).ceil() as isize;

        for p in 0..xs.len() {
            let px = xs[p];
            let py = ys[p];
            let pz = zs[p];
            let cx = ((px - ox) / dx).floor() as isize;
            let cy = ((py - oy) / dy).floor() as isize;
            let cz = ((pz - oz) / dz).floor() as isize;

            for ix in (cx - half_kx)..=(cx + half_kx) {
                let (wx, gx) = wrap_index(ix, self.nx as isize, pbc[0]);
                if !wx {
                    continue;
                }
                let vx = ox + (ix as F + 0.5) * dx - px;
                for iy in (cy - half_ky)..=(cy + half_ky) {
                    let (wy, gy) = wrap_index(iy, self.ny as isize, pbc[1]);
                    if !wy {
                        continue;
                    }
                    let vy = oy + (iy as F + 0.5) * dy - py;
                    for iz in (cz - half_kz)..=(cz + half_kz) {
                        let (wz, gz) = wrap_index(iz, self.nz as isize, pbc[2]);
                        if !wz {
                            continue;
                        }
                        let vz = oz + (iz as F + 0.5) * dz - pz;
                        let r2 = vx * vx + vy * vy + vz * vz;
                        if r2 <= r_max_sq {
                            voxels[[gx, gy, gz]] = 1;
                            counts[[gx, gy, gz]] += 1;
                        }
                    }
                }
            }
        }

        Ok(SphereVoxelizationResult {
            voxels,
            raw_counts: counts,
        })
    }
}

impl LegacyCompute for SphereVoxelization {
    type Args<'a> = ();
    type Output = Vec<SphereVoxelizationResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _: (),
    ) -> Result<Vec<SphereVoxelizationResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames.par_iter().map(|f| self.one_frame(*f)).collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            out.push(self.one_frame(*f)?);
        }
        Ok(out)
    }
}

/// Per-frame sphere-voxelisation result.
#[derive(Debug, Clone, Default)]
pub struct SphereVoxelizationResult {
    /// Boolean voxel grid: `1` if any particle's sphere covers the voxel
    /// centre, `0` otherwise.
    pub voxels: Array3<u8>,
    /// Optional accumulating count of overlapping particles per voxel.
    pub raw_counts: Array3<u32>,
}

impl ComputeResult for SphereVoxelizationResult {}

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

    #[test]
    fn single_particle_voxels_form_a_sphere() {
        // Particle at the box centre, radius 1.5 → voxelised sphere with
        // volume ≈ (4/3) π (1.5)³ = 14.137.
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let sv = SphereVoxelization::new(50, 50, 50, 1.5).unwrap();
        let r = &sv.compute(&[&frame], ()).unwrap()[0];
        let voxel_vol = (10.0_f64 / 50.0).powi(3);
        let filled: u64 = r.voxels.iter().map(|&v| v as u64).sum();
        let measured_vol = filled as F * voxel_vol;
        let analytic = crate::core::FOUR_THIRDS_PI * 1.5_f64.powi(3);
        // Voxelisation error: a couple of percent at 50³ grid.
        assert!(
            ((measured_vol - analytic) / analytic).abs() < 0.05,
            "voxel sphere = {measured_vol:.3}, analytic = {analytic:.3}"
        );
    }

    #[test]
    fn raw_counts_doubled_when_two_spheres_overlap() {
        // Two coincident particles → every covered voxel has count = 2.
        let frame = frame_with(&[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let sv = SphereVoxelization::new(20, 20, 20, 1.0).unwrap();
        let r = &sv.compute(&[&frame], ()).unwrap()[0];
        let max_count = r.raw_counts.iter().max().copied().unwrap();
        assert_eq!(max_count, 2);
    }

    #[test]
    fn pbc_wraps_voxel_mask_across_boundary() {
        let frame = frame_with(&[[0.1, 5.0, 5.0]], 10.0, [true; 3]);
        let sv = SphereVoxelization::new(40, 40, 40, 1.5).unwrap();
        let r = &sv.compute(&[&frame], ()).unwrap()[0];
        // Some voxel near the right edge should be set.
        let right_band: u64 = (0..40)
            .flat_map(|iy| {
                (0..40).map(move |iz| r.voxels[[38_usize, iy as usize, iz as usize]] as u64)
            })
            .sum();
        assert!(right_band > 0, "PBC wrap should set voxels near x ≈ 9.5");
    }

    #[test]
    fn invalid_args_error() {
        assert!(SphereVoxelization::new(0, 10, 10, 1.0).is_err());
        assert!(SphereVoxelization::new(10, 10, 10, 0.0).is_err());
        assert!(SphereVoxelization::new(10, 10, 10, -1.0).is_err());
    }
}
