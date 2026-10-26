//! Dihedral forms: each built-in dihedral style's energy as a
//! [`ScalarForm`] of the signed dihedral `φ`, priced by [`ScalarBonded`].
//! `dihedral harmonic` is LAMMPS's `K[1 + d cos(nφ)]`, the form of
//! [`improper cvff`](crate::ff::potential::improper::Cvff), and is priced by
//! it.
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`ScalarBonded`]: crate::ff::potential::form_kernel::ScalarBonded

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod mmff;
pub(crate) mod multi_harmonic;
pub(crate) mod opls;
pub(crate) mod periodic;
pub(crate) mod uff;

pub use charmm::Charmm;
pub use class2::Class2;
pub use mmff::Mmff;
pub use multi_harmonic::MultiHarmonic;
pub use opls::Opls;
pub use periodic::Periodic;
pub use uff::Uff;
