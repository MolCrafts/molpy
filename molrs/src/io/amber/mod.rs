//! AMBER's files: the prmtop topology (structure and force field), inpcrd /
//! restrt coordinates, antechamber `.ac`, prep residue templates, and frcmod
//! parameter modifications.
//!
//! The frame doors are functions of [`crate::io`]:
//! [`read_prmtop`](crate::io::read_prmtop) (structure),
//! [`read_inpcrd`](crate::io::read_inpcrd), [`read_ac`](crate::io::read_ac),
//! [`read_prep`](crate::io::read_prep) / [`write_prep`](crate::io::write_prep).
//! The force-field doors are this module's classes:
//! [`AmberPrmtopForcefieldReader`] (`read`, and `read_system` for the force
//! field with its frame) and [`AmberFrcmodWriter`] (`write`). This module
//! also holds the prep records ([`PrepResidue`], [`PrepAtom`]),
//! and [`merge_inpcrd`], which lays an inpcrd's coordinates onto a prmtop's
//! frame.

pub(crate) mod ac;
#[cfg(feature = "ff")]
pub(crate) mod frcmod;
pub(crate) mod inpcrd;
pub(crate) mod prep;
pub(crate) mod prmtop;
#[cfg(all(test, feature = "ff"))]
mod prmtop_check;
#[cfg(feature = "ff")]
pub(crate) mod prmtop_forcefield;
pub(crate) mod prmtop_tables;

#[cfg(feature = "ff")]
pub use frcmod::AmberFrcmodWriter;
pub use inpcrd::merge_inpcrd;
pub use prep::{PrepAtom, PrepResidue};
#[cfg(feature = "ff")]
pub use prmtop_forcefield::AmberPrmtopForcefieldReader;
