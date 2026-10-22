//! Dihedral potential kernels.

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod multi_harmonic;
pub(crate) mod opls;
pub(crate) mod periodic;
pub(crate) mod uff;

pub use charmm::{DihedralCharmm, dihedral_charmm_constructor};
pub use class2::{DihedralClass2, dihedral_class2_constructor};
pub use harmonic::dihedral_harmonic_constructor;
pub use mmff::{DihedralMmff, dihedral_mmff_constructor};
pub use multi_harmonic::{
    DihedralMultiHarmonic, dihedral_multi_harmonic_constructor, dihedral_nharmonic_constructor,
};
pub use opls::{DihedralOpls, dihedral_opls_constructor};
pub use periodic::{DihedralPeriodic, dihedral_periodic_constructor};
pub use uff::{DihedralUff, dihedral_uff_constructor};
