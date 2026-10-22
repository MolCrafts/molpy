//! Grid-smeared 3-D Gaussian density.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;
use ndarray::{Array3, ArrayView2};

use crate::compute::{Compute, ComputeError, Gaussian, RegularGrid, StencilReach};

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
/// `Σ_v ρ_v · ΔV = N_particles` to within the truncation. freud's stencil,
/// which reaches only `⌊r_max / Δ⌋` voxels from the particle's own and so
/// skips the outermost shell of those centres, is
/// [`with_reach`](Self::with_reach)`(`[`StencilReach::Floor`]`)`.
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
    reach: StencilReach,
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
            reach: StencilReach::default(),
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

    /// Spread over every voxel centre within `r_max` (default
    /// [`StencilReach::Ceil`]) or over freud's narrower stencil
    /// ([`StencilReach::Floor`]).
    pub fn with_reach(mut self, reach: StencilReach) -> Self {
        self.reach = reach;
        self
    }

    /// The stencil reach.
    pub fn reach(&self) -> StencilReach {
        self.reach
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
            grid.spread(
                &self.kernel,
                [p[0], p[1], p[2]],
                self.reach,
                |[i, j, k], w| {
                    density[[i, j, k]] += w;
                },
            )?;
        }
        Ok(GaussianDensityResult { density, grid })
    }
}

/// One snapshot's Gaussian-density grid.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct GaussianDensityResult {
    /// Density per voxel, shape `dims`, in particles per unit volume.
    pub density: Array3<F>,
    /// The grid the density lives on.
    pub grid: RegularGrid<3>,
}

crate::compute::spec::compute_spec! {
    GaussianDensity as "GaussianDensity";
    params |s| {
        dims: IntList;
        sigma: Float;
        r_max: Float
            = crate::compute::ParamDefault::Scaled {
                of: "sigma",
                factor: Gaussian::DEFAULT_CUTOFF_SIGMAS,
            }
            => crate::compute::SpreadKernel::cutoff(&s.kernel);
        reach: Choice(StencilReach::NAMES) = StencilReach::default() => s.reach;
    }
    inputs {
        positions: Positions;
        simbox: SimBox;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use crate::core::Cell;
    use ndarray::Array2;

    fn setup(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> (Array2<F>, SimBox) {
        let points = Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
        let simbox = SimBox::new(
            Cell::Lengths([box_len; 3]),
            [0.0 as F, 0.0 as F, 0.0 as F],
            pbc,
        )
        .unwrap();
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

    /// The former freud-parity stencil, selectable: `Floor` is the `dev`
    /// `GaussianDensity` loop written out — offsets up to `⌊r_max / Δ⌋`
    /// voxels from the particle's own, then the `r ≤ r_max` test — on every
    /// voxel. The voxel 1.51 Å away on x (inside `r_max = 1.6`, two voxels
    /// out) is what `Ceil` adds.
    #[test]
    fn floor_reach_is_the_former_freud_stencil() {
        let (points, simbox) = setup(&[[5.99, 5.5, 5.5], [0.3, 9.6, 2.2]], 10.0, [true; 3]);
        let (sigma, r_max, n) = (0.5, 1.6, 10usize);
        let gd = GaussianDensity::new([n; 3], sigma)
            .unwrap()
            .with_r_max(r_max)
            .unwrap();
        let floor = run(&gd.with_reach(StencilReach::Floor), &points, &simbox).density;
        let ceil = run(&gd, &points, &simbox).density;

        let dx = 10.0 / n as F;
        let half = (r_max / dx).floor() as isize;
        let two_sigma_sq = 2.0 * sigma * sigma;
        let pref = (two_sigma_sq * std::f64::consts::PI).powf(-1.5);
        let mut want = Array3::<F>::zeros([n; 3]);
        for p in points.outer_iter() {
            let home = [0, 1, 2].map(|d| (p[d] / dx).floor() as isize);
            for ix in home[0] - half..=home[0] + half {
                for iy in home[1] - half..=home[1] + half {
                    for iz in home[2] - half..=home[2] + half {
                        let offset = [ix, iy, iz];
                        let r2: F = (0..3)
                            .map(|d| ((offset[d] as F + 0.5) * dx - p[d]).powi(2))
                            .sum();
                        if r2 <= r_max * r_max {
                            let wrap = offset.map(|i| i.rem_euclid(n as isize) as usize);
                            want[wrap] += pref * (-r2 / two_sigma_sq).exp();
                        }
                    }
                }
            }
        }
        for (got, want) in floor.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert_eq!(floor[[7, 5, 5]], 0.0);
        let r2 = (7.5 - 5.99_f64).powi(2);
        assert!((ceil[[7, 5, 5]] - pref * (-r2 / two_sigma_sq).exp()).abs() < 1e-12);
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
