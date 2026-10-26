//! OpenMM's force-field XML (`<ForceField>` with `<AtomTypes>`,
//! `<HarmonicBondForce>`, …), including the OPLS-AA / CL&P / Foyer packs in
//! the same schema.
//!
//! The doors are the format's classes: [`OpenmmXmlReader`]'s
//! [`read`](crate::io::reader::ForceFieldReader::read) and
//! [`OpenmmXmlWriter`]'s [`write`](crate::io::writer::ForceFieldWriter::write)
//! — honest inverses, OpenMM's units (nm, kJ/mol, radians) converted at this
//! boundary — and [`OpenmmXmlWriter::write_system`], the force field plus
//! the residue templates of a typed frame, which OpenMM's `createSystem`
//! needs for the PDB [`write_pdb`](crate::io::write_pdb) writes of that frame.
//! The OPLS-AA typing annotations Foyer adds to the same file are read with
//! its force field into a typifier by [`OpenmmXmlReader::read_opls_typifier`].

pub(crate) mod geometric_mixing;
mod opls_typing;
pub(crate) mod reader;
pub(crate) mod writer;

pub use reader::OpenmmXmlReader;
pub use writer::OpenmmXmlWriter;
