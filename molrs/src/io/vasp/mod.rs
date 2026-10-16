//! VASP's structure and volumetric files: POSCAR / CONTCAR and CHGCAR /
//! CHGDIF, which share one header.
//!
//! The doors are functions of [`crate::io`]:
//! [`read_poscar`](crate::io::read_poscar), [`write_poscar`](crate::io::write_poscar)
//! and [`read_chgcar`](crate::io::read_chgcar). This module holds the POSCAR
//! classes, [`VaspPoscarReader`] and [`VaspPoscarWriter`], over any byte stream.

pub(super) mod chgcar;
mod header;
pub(super) mod poscar;

pub use poscar::{VaspPoscarReader, VaspPoscarWriter};
