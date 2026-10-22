//! CMAP (correction map) kernels: a five-atom crossterm priced from an
//! energy grid over the two consecutive dihedrals it spans.

pub(crate) mod charmm;

pub use charmm::{CmapCharmm, CmapGrid, cmap_charmm_constructor};

#[cfg(test)]
mod lammps_check;
