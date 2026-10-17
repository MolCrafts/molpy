//! Boolean voxel rasterisation of point particles as hard spheres.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;
use ndarray::{Array3, ArrayView2};

use crate::compute::{Compute, ComputeError, Indicator, RegularGrid, SpreadKernel, StencilReach};

/// Sphere-voxelisation calculator.
///
/// Mirrors `freud.density.SphereVoxelization`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/SphereVoxelization.cc)).
///
/// Configuration: the voxel counts per axis and the sphere radius (an
/// [`Indicator`] kernel). Data: one snapshot's positions and its box, which
/// the [`RegularGrid`] divides. Every voxel whose centre lies inside some
/// particle's sphere (boundary included) is set to `1`; `raw_counts` holds how
/// many spheres cover each voxel.
///
/// Like [`GaussianDensity`](super::GaussianDensity), orthorhombic boxes only;
/// a periodic axis wraps.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereVoxelization {
    dims: [usize; 3],
    sphere: Indicator,
}

/// The data a [`SphereVoxelization`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct SphereVoxelizationInput<'a> {
    /// Positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box the grid divides.
    pub simbox: &'a SimBox,
}

impl SphereVoxelization {
    /// `dims` voxels per axis; particles rasterised as spheres of radius
    /// `r_max` (Å).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a zero voxel count or a non-positive
    /// radius.
    pub fn new(dims: [usize; 3], r_max: F) -> Result<Self, ComputeError> {
        if dims.contains(&0) {
            return Err(ComputeError::OutOfRange {
                field: "SphereVoxelization::dims",
                value: format!("{dims:?}"),
            });
        }
        Ok(Self {
            dims,
            sphere: Indicator::new(r_max)?,
        })
    }

    /// Voxels per axis.
    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Sphere radius, Å.
    pub fn r_max(&self) -> F {
        self.sphere.cutoff()
    }
}

impl Compute for SphereVoxelization {
    type Input<'a> = SphereVoxelizationInput<'a>;
    type Output = SphereVoxelizationResult;

    /// # Errors
    /// [`ComputeError::OutOfRange`] for a triclinic box;
    /// [`ComputeError::BadShape`] for positions without 3 columns;
    /// [`ComputeError::NonFinite`] for a non-finite position.
    fn compute(
        &self,
        input: SphereVoxelizationInput<'_>,
    ) -> Result<SphereVoxelizationResult, ComputeError> {
        if matches!(input.simbox.kind(), BoxKind::Triclinic) {
            return Err(ComputeError::OutOfRange {
                field: "SphereVoxelization::simbox",
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
        let mut voxels = Array3::<u8>::zeros(self.dims);
        let mut counts = Array3::<u32>::zeros(self.dims);
        for p in input.positions.outer_iter() {
            let center = [p[0], p[1], p[2]];
            grid.spread(&self.sphere, center, StencilReach::Ceil, |[i, j, k], _| {
                voxels[[i, j, k]] = 1;
                counts[[i, j, k]] += 1;
            })?;
        }
        Ok(SphereVoxelizationResult {
            voxels,
            raw_counts: counts,
            grid,
        })
    }
}

/// One snapshot's sphere voxelisation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SphereVoxelizationResult {
    /// Boolean voxel grid: `1` if any particle's sphere covers the voxel
    /// centre, `0` otherwise.
    pub voxels: Array3<u8>,
    /// Number of spheres covering each voxel centre.
    pub raw_counts: Array3<u32>,
    /// The grid the voxels live on.
    pub grid: RegularGrid<3>,
}

crate::compute::spec::compute_spec! {
    SphereVoxelization as "SphereVoxelization";
    params |s| {
        dims: IntList;
        r_max: Float;
    }
    inputs {
        positions: Positions;
        simbox: SimBox;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::Array2;

    fn run(
        positions: &[[F; 3]],
        box_len: F,
        pbc: [bool; 3],
        sv: &SphereVoxelization,
    ) -> SphereVoxelizationResult {
        let points = Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
        let simbox = SimBox::new(
            Cell::Lengths([box_len; 3]),
            [0.0 as F, 0.0 as F, 0.0 as F],
            pbc,
        )
        .unwrap();
        sv.compute(SphereVoxelizationInput {
            positions: points.view(),
            simbox: &simbox,
        })
        .unwrap()
    }

    #[test]
    fn single_particle_voxels_form_a_sphere() {
        // Particle at the box centre, radius 1.5 → voxelised sphere with
        // volume ≈ (4/3) π (1.5)³ = 14.137.
        let sv = SphereVoxelization::new([50, 50, 50], 1.5).unwrap();
        let r = run(&[[5.0, 5.0, 5.0]], 10.0, [false; 3], &sv);
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
        let sv = SphereVoxelization::new([20, 20, 20], 1.0).unwrap();
        let r = run(&[[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], 10.0, [false; 3], &sv);
        let max_count = r.raw_counts.iter().max().copied().unwrap();
        assert_eq!(max_count, 2);
    }

    #[test]
    fn pbc_wraps_voxel_mask_across_boundary() {
        let sv = SphereVoxelization::new([40, 40, 40], 1.5).unwrap();
        let r = &run(&[[0.1, 5.0, 5.0]], 10.0, [true; 3], &sv);
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
        assert!(SphereVoxelization::new([0, 10, 10], 1.0).is_err());
        assert!(SphereVoxelization::new([10, 10, 10], 0.0).is_err());
        assert!(SphereVoxelization::new([10, 10, 10], -1.0).is_err());
    }
}
