//! Grid primitives: [`RegularGrid`] (a lattice of voxels with per-axis
//! periodicity) and the [`SpreadKernel`]s that spread a point over it,
//! [`Gaussian`] and [`Indicator`]. Every density grid, voxelisation and
//! projected image in `compute` is built on them.

mod kernel;
mod regular_grid;

pub use kernel::{Gaussian, Indicator, SpreadKernel};
pub use regular_grid::RegularGrid;
