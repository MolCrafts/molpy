//! Grid-smeared 3-D Gaussian density.

use crate::compute::ComputeResult;
use ndarray::Array3;

use molrs::core::BoxKind;
use molrs::core::FrameAccess;
use molrs::op::F;

use super::wrap_index;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// Gaussian-density calculator.
///
/// Mirrors `freud.density.GaussianDensity`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/GaussianDensity.cc)).
///
/// Discretises the simulation box into a regular `(nx, ny, nz)` voxel grid
/// and accumulates a Gaussian of width `σ` centred on each particle:
///
/// ```text
///   ρ(r) = Σ_i (2π σ²)^{-3/2} · exp( −|r − r_i|² / (2 σ²) )
/// ```
///
/// Only voxels within `r_max` of each particle are touched (`r_max = 3 σ`
/// by default, which captures ≳ 99% of the Gaussian mass). The total
/// integral `Σ_v ρ_v · ΔV = N_particles` to within the truncation.
///
/// # Conventions
///
/// - Voxel centres at `origin[d] + (i + 0.5) · Lx[d] / nx[d]`, matching
///   freud's `GaussianDensity::compute`.
/// - Currently orthorhombic-box only; triclinic returns
///   [`ComputeError::OutOfRange`]. (freud's GaussianDensity has the same
///   ortho-only restriction in its `period` calculation.)
/// - PBC is honoured per-axis via wrap-around grid indexing when the
///   corresponding `pbc` flag is true.
#[derive(Debug, Clone, Copy)]
pub struct GaussianDensity {
    nx: usize,
    ny: usize,
    nz: usize,
    sigma: F,
    r_max: F,
}

impl GaussianDensity {
    /// New analyzer with `(nx, ny, nz)` grid points per axis, Gaussian width
    /// `sigma`, and a default truncation radius of `3 σ`.
    pub fn new(nx: usize, ny: usize, nz: usize, sigma: F) -> Result<Self, ComputeError> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(ComputeError::OutOfRange {
                field: "GaussianDensity::width",
                value: format!("({nx}, {ny}, {nz})"),
            });
        }
        if sigma.is_nan() || sigma <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "GaussianDensity::sigma",
                value: sigma.to_string(),
            });
        }
        Ok(Self {
            nx,
            ny,
            nz,
            sigma,
            r_max: 3.0 * sigma,
        })
    }

    /// Override the Gaussian truncation radius (default: `3 σ`).
    pub fn with_r_max(mut self, r_max: F) -> Self {
        debug_assert!(r_max > 0.0);
        self.r_max = r_max;
        self
    }

    pub fn width(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }
    pub fn r_max(&self) -> F {
        self.r_max
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
    ) -> Result<GaussianDensityResult, ComputeError> {
        let simbox = frame.simbox_ref().ok_or(ComputeError::MissingSimBox)?;
        let (lx, ly, lz) = match simbox.kind() {
            BoxKind::Ortho { len, .. } => (len[0], len[1], len[2]),
            BoxKind::Triclinic => {
                return Err(ComputeError::OutOfRange {
                    field: "GaussianDensity::simbox",
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
        let n = xs.len();

        let mut density = Array3::<F>::zeros((self.nx, self.ny, self.nz));
        let two_sigma_sq = 2.0 * self.sigma * self.sigma;
        let pref = (two_sigma_sq * std::f64::consts::PI).powf(-1.5);
        let r_max_sq = self.r_max * self.r_max;

        // Number of voxels covered by r_max along each axis.
        //
        // Use `floor` (not `ceil`) to match freud's `GaussianDensity::compute`,
        // which iterates `ix ∈ [cx - floor(r_max/dx), cx + floor(r_max/dx)]`.
        // `ceil` over-iterates by one shell of voxels at radial distance just
        // inside r_max (their centres satisfy r < r_max but axial offset is
        // r_max − dx/2 ≤ |ddx| < r_max), accumulating a uniform ≈5e-3 offset
        // per bulk voxel at N ≳ 10⁴ versus freud's grid.
        let half_kx = (self.r_max / dx).floor() as isize;
        let half_ky = (self.r_max / dy).floor() as isize;
        let half_kz = (self.r_max / dz).floor() as isize;

        for p in 0..n {
            let px = xs[p];
            let py = ys[p];
            let pz = zs[p];
            // Voxel index containing the particle.
            let cx = ((px - ox) / dx).floor() as isize;
            let cy = ((py - oy) / dy).floor() as isize;
            let cz = ((pz - oz) / dz).floor() as isize;

            for ix in (cx - half_kx)..=(cx + half_kx) {
                let (wx, gx_idx) = wrap_index(ix, self.nx as isize, pbc[0]);
                if !wx {
                    continue;
                }
                let vx = ox + (ix as F + 0.5) * dx;
                let ddx = vx - px;
                if ddx.abs() > self.r_max {
                    continue;
                }
                for iy in (cy - half_ky)..=(cy + half_ky) {
                    let (wy, gy_idx) = wrap_index(iy, self.ny as isize, pbc[1]);
                    if !wy {
                        continue;
                    }
                    let vy = oy + (iy as F + 0.5) * dy;
                    let ddy = vy - py;
                    if ddy.abs() > self.r_max {
                        continue;
                    }
                    for iz in (cz - half_kz)..=(cz + half_kz) {
                        let (wz, gz_idx) = wrap_index(iz, self.nz as isize, pbc[2]);
                        if !wz {
                            continue;
                        }
                        let vz = oz + (iz as F + 0.5) * dz;
                        let ddz = vz - pz;
                        let r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                        if r2 > r_max_sq {
                            continue;
                        }
                        let v = pref * (-r2 / two_sigma_sq).exp();
                        density[[gx_idx, gy_idx, gz_idx]] += v;
                    }
                }
            }
        }

        Ok(GaussianDensityResult { density })
    }
}

impl LegacyCompute for GaussianDensity {
    type Args<'a> = ();
    type Output = Vec<GaussianDensityResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _: (),
    ) -> Result<Vec<GaussianDensityResult>, ComputeError> {
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

/// Per-frame Gaussian-density grid.
#[derive(Debug, Clone, Default)]
pub struct GaussianDensityResult {
    /// Voxel grid `(nx, ny, nz)` of density values.
    pub density: Array3<F>,
}

impl ComputeResult for GaussianDensityResult {}

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
    fn integral_over_grid_recovers_particle_count() {
        // One particle at box centre. With r_max = 4σ, ≥99.99% of the
        // Gaussian mass should land inside the grid → integral ≈ 1.
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new(40, 40, 40, 0.5)
            .unwrap()
            .with_r_max(2.0);
        let r = &gd.compute(&[&frame], ()).unwrap()[0];
        let voxel_volume = (10.0_f64 / 40.0).powi(3);
        let integral: F = r.density.iter().copied().sum::<F>() * voxel_volume;
        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral = {integral} (expected ≈ 1.0)"
        );
    }

    #[test]
    fn two_particles_integrate_to_two() {
        // r_max = 5σ captures > 99.99% of each Gaussian's volume; the
        // remaining mismatch with the analytic integral comes from voxel
        // discretisation (~1% at 40×40×40).
        let frame = frame_with(&[[3.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new(40, 40, 40, 0.4)
            .unwrap()
            .with_r_max(2.0);
        let r = &gd.compute(&[&frame], ()).unwrap()[0];
        let voxel_volume = (10.0_f64 / 40.0).powi(3);
        let integral: F = r.density.iter().copied().sum::<F>() * voxel_volume;
        assert!(
            (integral - 2.0).abs() < 0.02,
            "integral = {integral} (expected ≈ 2.0)"
        );
    }

    #[test]
    fn peak_is_near_particle_position() {
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new(20, 20, 20, 0.5).unwrap();
        let r = &gd.compute(&[&frame], ()).unwrap()[0];
        // Peak voxel should be at (10, 10, 10) — i.e. centre of grid.
        let mut max_val = F::MIN;
        let mut max_idx = (0, 0, 0);
        for (idx, &v) in r.density.indexed_iter() {
            if v > max_val {
                max_val = v;
                max_idx = idx;
            }
        }
        assert!(
            (max_idx.0 as isize - 10).abs() <= 1
                && (max_idx.1 as isize - 10).abs() <= 1
                && (max_idx.2 as isize - 10).abs() <= 1,
            "peak at {max_idx:?}, expected near (10, 10, 10)"
        );
    }

    #[test]
    fn pbc_wraps_density_across_boundary() {
        // Particle at the left edge of the box. With PBC, density should
        // appear at both the left edge and the right edge of the grid.
        let frame = frame_with(&[[0.1, 5.0, 5.0]], 10.0, [true; 3]);
        let gd = GaussianDensity::new(40, 40, 40, 0.4)
            .unwrap()
            .with_r_max(2.0);
        let r = &gd.compute(&[&frame], ()).unwrap()[0];
        // Density at x ≈ 9.9 should be non-zero (wrapped across the boundary).
        let cx_right = ((9.9_f64 / 10.0) * 40.0) as usize;
        let cy = ((5.0_f64 / 10.0) * 40.0) as usize;
        let cz = ((5.0_f64 / 10.0) * 40.0) as usize;
        assert!(
            r.density[[cx_right, cy, cz]] > 1e-3,
            "PBC wrap should give non-zero density at x ≈ 9.9 ({} found)",
            r.density[[cx_right, cy, cz]]
        );
    }

    #[test]
    fn invalid_width_or_sigma_errors() {
        assert!(GaussianDensity::new(0, 10, 10, 1.0).is_err());
        assert!(GaussianDensity::new(10, 10, 10, -1.0).is_err());
        assert!(GaussianDensity::new(10, 10, 10, 0.0).is_err());
    }

    #[test]
    fn missing_simbox_is_error() {
        let mut frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        frame.simbox = None;
        let err = GaussianDensity::new(10, 10, 10, 0.5)
            .unwrap()
            .compute(&[&frame], ())
            .unwrap_err();
        assert!(matches!(err, ComputeError::MissingSimBox));
    }

    /// The `frames.len() >= PAR_THRESHOLD` rayon branch must match the
    /// serial path exactly, in frame order.
    #[test]
    fn parallel_matches_serial() {
        let frame = frame_with(&[[3.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new(16, 16, 16, 0.5)
            .unwrap()
            .with_r_max(2.0);
        let solo = gd.compute(&[&frame], ()).unwrap();
        let par = gd.compute(&[&frame, &frame], ()).unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].density, solo[0].density);
        assert_eq!(par[1].density, solo[0].density);
    }
}
