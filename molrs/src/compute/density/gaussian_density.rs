//! Grid-smeared 3-D Gaussian density.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;
use ndarray::{Array3, ArrayView2};

use crate::compute::{Compute, ComputeError, Gaussian, RegularGrid};

/// Gaussian-density calculator.
///
/// Mirrors `freud.density.GaussianDensity`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/GaussianDensity.cc)).
///
/// Configuration: the voxel counts per axis and the [`Gaussian`] kernel
/// (width `σ`, truncated at `r_max = 3σ` unless set). Data: one snapshot's
/// positions and its box. The box is divided into a
/// [`RegularGrid`] ([`RegularGrid::over_box`]) and each particle is spread
/// over it:
///
/// ```text
///   ρ(r) = Σ_i (2π σ²)^{-3/2} · exp( −|r − r_i|² / (2 σ²) )
/// ```
///
/// Every voxel whose centre lies within `r_max` of a particle (boundary
/// included) receives its share; `r_max = 3σ` captures ≳ 99% of the mass, so
/// `Σ_v ρ_v · ΔV = N_particles` to within the truncation.
///
/// # Conventions
///
/// - Voxel centres at `origin[d] + (i + 0.5) · L[d] / n[d]`, matching
///   freud's `GaussianDensity::compute`.
/// - Orthorhombic boxes only; a triclinic box is
///   [`ComputeError::OutOfRange`] (freud's GaussianDensity has the same
///   restriction).
/// - A periodic box axis wraps the spread; an open one drops what falls
///   outside the box.
///
/// For many snapshots, wrap in [`Batch`](crate::compute::Batch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianDensity {
    dims: [usize; 3],
    kernel: Gaussian,
}

/// The data a [`GaussianDensity`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct GaussianDensityInput<'a> {
    /// Positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box the grid divides.
    pub simbox: &'a SimBox,
}

impl GaussianDensity {
    /// `dims` voxels per axis, Gaussian width `sigma`, truncated at `3 σ`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a zero voxel count or a non-positive
    /// `sigma`.
    pub fn new(dims: [usize; 3], sigma: F) -> Result<Self, ComputeError> {
        if dims.contains(&0) {
            return Err(ComputeError::OutOfRange {
                field: "GaussianDensity::dims",
                value: format!("{dims:?}"),
            });
        }
        Ok(Self {
            dims,
            kernel: Gaussian::new(sigma)?,
        })
    }

    /// Truncate the Gaussian at `r_max` (Å) instead of `3 σ`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite `r_max`.
    pub fn with_r_max(mut self, r_max: F) -> Result<Self, ComputeError> {
        self.kernel = self.kernel.with_cutoff(r_max)?;
        Ok(self)
    }

    /// Voxels per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// The spreading kernel.
    pub fn kernel(&self) -> Gaussian {
        self.kernel
    }
}

impl Compute for GaussianDensity {
    type Input<'a> = GaussianDensityInput<'a>;
    type Output = GaussianDensityResult;

    /// # Errors
    /// [`ComputeError::OutOfRange`] for a triclinic box;
    /// [`ComputeError::BadShape`] for positions without 3 columns;
    /// [`ComputeError::NonFinite`] for a non-finite position.
    fn compute(
        &self,
        input: GaussianDensityInput<'_>,
    ) -> Result<GaussianDensityResult, ComputeError> {
        if matches!(input.simbox.kind(), BoxKind::Triclinic) {
            return Err(ComputeError::OutOfRange {
                field: "GaussianDensity::simbox",
                value: "triclinic boxes are not supported".into(),
            });
        }
        if input.positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("N × {}", input.positions.ncols()),
            });
        }
        let grid = RegularGrid::over_box(input.simbox, self.dims)?;
        let mut density = Array3::<F>::zeros(self.dims);
        for p in input.positions.outer_iter() {
            grid.spread(&self.kernel, [p[0], p[1], p[2]], |[i, j, k], w| {
                density[[i, j, k]] += w;
            })?;
        }
        Ok(GaussianDensityResult { density, grid })
    }
}

/// One snapshot's Gaussian-density grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianDensityResult {
    /// Density per voxel, shape `dims`, in particles per unit volume.
    pub density: Array3<F>,
    /// The grid the density lives on.
    pub grid: RegularGrid<3>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use ndarray::{Array2, array};

    fn setup(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> (Array2<F>, SimBox) {
        let points = Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
        let simbox = SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap();
        (points, simbox)
    }

    fn run(gd: &GaussianDensity, points: &Array2<F>, simbox: &SimBox) -> GaussianDensityResult {
        gd.compute(GaussianDensityInput {
            positions: points.view(),
            simbox,
        })
        .unwrap()
    }

    #[test]
    fn integral_over_grid_recovers_particle_count() {
        // One particle at box centre. With r_max = 4σ, ≥99.99% of the
        // Gaussian mass should land inside the grid → integral ≈ 1.
        let (points, simbox) = setup(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new([40, 40, 40], 0.5)
            .unwrap()
            .with_r_max(2.0)
            .unwrap();
        let r = run(&gd, &points, &simbox);
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
        let (points, simbox) = setup(&[[3.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new([40, 40, 40], 0.4)
            .unwrap()
            .with_r_max(2.0)
            .unwrap();
        let r = run(&gd, &points, &simbox);
        let voxel_volume = (10.0_f64 / 40.0).powi(3);
        let integral: F = r.density.iter().copied().sum::<F>() * voxel_volume;
        assert!(
            (integral - 2.0).abs() < 0.02,
            "integral = {integral} (expected ≈ 2.0)"
        );
    }

    #[test]
    fn peak_is_near_particle_position() {
        let (points, simbox) = setup(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new([20, 20, 20], 0.5).unwrap();
        let r = run(&gd, &points, &simbox);
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
        let (points, simbox) = setup(&[[0.1, 5.0, 5.0]], 10.0, [true; 3]);
        let gd = GaussianDensity::new([40, 40, 40], 0.4)
            .unwrap()
            .with_r_max(2.0)
            .unwrap();
        let r = run(&gd, &points, &simbox);
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

    /// Without PBC nothing wraps: the far edge stays empty.
    #[test]
    fn open_boundary_does_not_wrap() {
        let (points, simbox) = setup(&[[0.1, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new([40, 40, 40], 0.4)
            .unwrap()
            .with_r_max(2.0)
            .unwrap();
        let r = run(&gd, &points, &simbox);
        assert_eq!(r.density[[39, 20, 20]], 0.0);
    }

    #[test]
    fn invalid_width_or_sigma_errors() {
        assert!(GaussianDensity::new([0, 10, 10], 1.0).is_err());
        assert!(GaussianDensity::new([10, 10, 10], -1.0).is_err());
        assert!(GaussianDensity::new([10, 10, 10], 0.0).is_err());
        // `with_r_max` used to `debug_assert`; a bad radius is an error.
        assert!(matches!(
            GaussianDensity::new([10, 10, 10], 1.0)
                .unwrap()
                .with_r_max(-1.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// Many snapshots go through `Batch`, in order.
    #[test]
    fn batch_matches_single_snapshots() {
        let (points, simbox) = setup(&[[3.0, 5.0, 5.0], [7.0, 5.0, 5.0]], 10.0, [false; 3]);
        let gd = GaussianDensity::new([16, 16, 16], 0.5)
            .unwrap()
            .with_r_max(2.0)
            .unwrap();
        let solo = run(&gd, &points, &simbox);
        let input = GaussianDensityInput {
            positions: points.view(),
            simbox: &simbox,
        };
        let many = Batch(gd).compute(vec![input, input]).unwrap();
        assert_eq!(many, vec![solo.clone(), solo]);
    }
}
