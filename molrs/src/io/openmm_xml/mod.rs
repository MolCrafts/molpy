//! OpenMM's force-field XML (`<ForceField>` with `<AtomTypes>`,
//! `<HarmonicBondForce>`, …), including the OPLS-AA / CL&P / Foyer packs in
//! the same schema.
//!
//! The doors are functions of [`crate::io`]:
//! [`read_openmm_forcefield`](crate::io::read_openmm_forcefield) /
//! [`read_openmm_forcefield_str`](crate::io::read_openmm_forcefield_str),
//! [`write_openmm_forcefield`](crate::io::write_openmm_forcefield) /
//! [`write_openmm_forcefield_str`](crate::io::write_openmm_forcefield_str)
//! — honest inverses, OpenMM's units (nm, kJ/mol, radians) converted at this
//! boundary. This module holds the format's classes, [`OpenmmXmlReader`] and
//! [`OpenmmXmlWriter`]. The OPLS-AA typing annotations Foyer adds to the same
//! file are read with its force field into a typifier by
//! [`OpenmmXmlReader::read_opls_typifier_str`].

mod opls_typing;
pub(crate) mod reader;
pub(crate) mod writer;

pub use reader::OpenmmXmlReader;
pub use writer::OpenmmXmlWriter;
