//! Pore-size distribution from the local thickness of the void.

use molrs::op::F;
use ndarray::{ArrayView1, ArrayView2};

use super::geometry::{self, SurfaceIndex};
use crate::compute::{Compute, ComputeError};
use crate::core::SimBox;

const DEFAULT_BIN_WIDTH: F = 0.1;
/// A finer grid than this is refused rather than allocated.
const MAX_VOXELS: usize = 8_000_000;

/// Volume-weighted pore-size distribution of a hard-sphere solid.
///
/// Configuration: the voxel spacing and the histogram width. Data: one
/// radius per atom, the positions and the box.
///
/// Each voxel centre is assigned the radius of the largest sphere that
/// covers it without intersecting an atom (the local thickness of Gelb and
/// Gubbins, *Langmuir* **2006**, 22, 7726; the distribution Poreblazer and
/// Zeo++ report). The sphere may be centred anywhere. Solid voxel centres,
/// those inside an atom, are left out of the histogram and counted in
/// [`PoreSizeDistributionResult::void_fraction`]'s complement.
///
/// The grid is orthogonal. A triclinic cell is refused: the voxel lattice
/// used here is the box's own lattice only when that lattice is orthogonal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoreSizeDistribution {
    spacing: F,
    bin_width: F,
}

/// The data a [`PoreSizeDistribution`] analyses. Lengths in Å.
#[derive(Clone, Copy)]
pub struct PoreSizeDistributionInput<'a> {
    /// Atom positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// Van der Waals radius of each atom, length `N`.
    pub radii: ArrayView1<'a, F>,
    /// An orthogonal cell.
    pub simbox: &'a SimBox,
}

/// One histogram of pore radii.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PoreSizeDistributionResult {
    /// Bin centres, Å. Bin `i` holds radii in `(i w, (i + 1) w]`.
    pub radius: Vec<F>,
    /// Fraction of the cell volume in each bin.
    pub fraction: Vec<F>,
    /// Fraction of the cell volume whose pore radius exceeds the bin's lower
    /// edge: the volume a probe of that radius can cover.
    pub cumulative: Vec<F>,
    /// Largest sphere that fits in the void, Å.
    pub largest_cavity_radius: F,
    /// Fraction of voxel centres that lie outside every atom.
    pub void_fraction: F,
    /// Volume of one voxel, Å³.
    pub voxel_volume: F,
}

impl PoreSizeDistribution {
    /// Voxels of edge about `spacing` (Å), histogram bins of
    /// [`DEFAULT`](Self::default_bin_width) Å.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite spacing.
    pub fn new(spacing: F) -> Result<Self, ComputeError> {
        if !spacing.is_finite() || spacing <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PoreSizeDistribution::spacing",
                value: spacing.to_string(),
            });
        }
        Ok(Self {
            spacing,
            bin_width: DEFAULT_BIN_WIDTH,
        })
    }

    /// Histogram bin width, Å.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a non-positive or non-finite width.
    pub fn with_bin_width(mut self, bin_width: F) -> Result<Self, ComputeError> {
        if !bin_width.is_finite() || bin_width <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "PoreSizeDistribution::bin_width",
                value: bin_width.to_string(),
            });
        }
        self.bin_width = bin_width;
        Ok(self)
    }
}

impl Compute for PoreSizeDistribution {
    type Input<'a> = PoreSizeDistributionInput<'a>;
    type Output = PoreSizeDistributionResult;

    /// # Errors
    /// The position and radius errors of [`geometry`](super::geometry), plus
    /// [`ComputeError::OutOfRange`] for a triclinic cell or a grid above
    /// [`MAX_VOXELS`](self).
    fn compute(
        &self,
        input: PoreSizeDistributionInput<'_>,
    ) -> Result<PoreSizeDistributionResult, ComputeError> {
        let spheres = geometry::read_spheres(input.positions, input.radii)?;
        let cell = geometry::read_cell(input.simbox)?;
        if !cell.orthogonal {
            return Err(ComputeError::OutOfRange {
                field: "PoreSizeDistribution::simbox",
                value: "triclinic boxes are not supported".into(),
            });
        }
        let dims = grid_dims(&cell, self.spacing)?;
        let index = SurfaceIndex::build(spheres, cell);
        let volume = index.cell.volume;
        let thickness = local_thickness(&index, dims);
        Ok(histogram(&thickness, volume, self.bin_width))
    }
}

fn grid_dims(cell: &geometry::CellGeom, spacing: F) -> Result<[usize; 3], ComputeError> {
    let dims = std::array::from_fn(|k| (cell.lengths[k] / spacing).round().max(1.0) as usize);
    let n = dims[0] * dims[1] * dims[2];
    if n > MAX_VOXELS {
        return Err(ComputeError::OutOfRange {
            field: "PoreSizeDistribution::spacing",
            value: format!("{n} voxels exceeds {MAX_VOXELS}"),
        });
    }
    Ok(dims)
}

fn local_thickness(index: &SurfaceIndex, dims: [usize; 3]) -> Vec<F> {
    let n = dims[0] * dims[1] * dims[2];
    let mut clear = vec![0.0; n];
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let point = voxel_center(&index.cell, dims, [i, j, k]);
                let c = index.clearance(point);
                clear[(i * dims[1] + j) * dims[2] + k] = c.max(0.0);
            }
        }
    }
    let mut thick = clear.clone();
    paint(index, dims, &clear, &mut thick);
    thick
}

fn voxel_center(cell: &geometry::CellGeom, dims: [usize; 3], idx: [usize; 3]) -> [F; 3] {
    let frac = std::array::from_fn(|axis| (idx[axis] as F + 0.5) / dims[axis] as F);
    geometry::cartesian(cell, frac)
}

fn paint(index: &SurfaceIndex, dims: [usize; 3], clear: &[F], thick: &mut [F]) {
    let lengths = index.cell.lengths;
    let spacing = std::array::from_fn(|k| lengths[k] / dims[k] as F);
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                let radius = clear[(i * dims[1] + j) * dims[2] + k];
                if radius <= 0.0 {
                    continue;
                }
                cover(index, dims, spacing, [i, j, k], radius, thick);
            }
        }
    }
}

fn cover(
    index: &SurfaceIndex,
    dims: [usize; 3],
    spacing: [F; 3],
    centre: [usize; 3],
    radius: F,
    thick: &mut [F],
) {
    let reach: [isize; 3] = std::array::from_fn(|k| (radius / spacing[k]).ceil() as isize);
    let origin = voxel_center(&index.cell, dims, centre);
    for di in -reach[0]..=reach[0] {
        for dj in -reach[1]..=reach[1] {
            for dk in -reach[2]..=reach[2] {
                let Some(idx) = wrap(dims, index.cell.pbc, centre, [di, dj, dk]) else {
                    continue;
                };
                let point = voxel_center(&index.cell, dims, idx);
                if farther_than(&index.cell, origin, point, radius) {
                    continue;
                }
                let slot = (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2];
                if radius > thick[slot] {
                    thick[slot] = radius;
                }
            }
        }
    }
}

fn farther_than(cell: &geometry::CellGeom, a: [F; 3], b: [F; 3], radius: F) -> bool {
    let d = cell.mic.apply([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > radius * radius
}

fn wrap(
    dims: [usize; 3],
    pbc: [bool; 3],
    centre: [usize; 3],
    delta: [isize; 3],
) -> Option<[usize; 3]> {
    let mut idx = [0; 3];
    for k in 0..3 {
        let n = dims[k] as isize;
        let i = centre[k] as isize + delta[k];
        if pbc[k] {
            idx[k] = i.rem_euclid(n) as usize;
        } else if (0..n).contains(&i) {
            idx[k] = i as usize;
        } else {
            return None;
        }
    }
    Some(idx)
}

fn histogram(thickness: &[F], box_volume: F, bin_width: F) -> PoreSizeDistributionResult {
    let n = thickness.len();
    let voxel_volume = box_volume / n as F;
    let largest = thickness.iter().copied().fold(0.0, F::max);
    let void = thickness.iter().filter(|t| **t > 0.0).count();
    let n_bins = if largest > 0.0 {
        (largest / bin_width).ceil() as usize
    } else {
        0
    };
    let mut counts = vec![0usize; n_bins];
    for &t in thickness {
        if t <= 0.0 {
            continue;
        }
        let bin = ((t / bin_width).ceil() as usize).saturating_sub(1);
        counts[bin.min(n_bins - 1)] += 1;
    }
    let fraction: Vec<F> = counts.iter().map(|c| *c as F / n as F).collect();
    let mut cumulative = vec![0.0; n_bins];
    let mut running = 0.0;
    for i in (0..n_bins).rev() {
        running += fraction[i];
        cumulative[i] = running;
    }
    PoreSizeDistributionResult {
        radius: (0..n_bins).map(|i| (i as F + 0.5) * bin_width).collect(),
        fraction,
        cumulative,
        largest_cavity_radius: largest,
        void_fraction: void as F / n as F,
        voxel_volume,
    }
}

crate::compute::spec::compute_spec! {
    PoreSizeDistribution as "PoreSizeDistribution";
    params |s| {
        spacing: Float;
        bin_width: Float = DEFAULT_BIN_WIDTH => s.bin_width;
    }
    inputs {
        positions: Positions;
        radii: Scalars;
        simbox: SimBox;
    }
}
