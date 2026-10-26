//! MOL2, the Tripos section-delimited molecule format.
//!
//! The doors are functions of [`crate::io`]: [`read_mol2`](crate::io::read_mol2)
//! (the first molecule) and [`write_mol2`](crate::io::write_mol2) (one
//! molecule). This module holds the format's classes: [`Mol2Reader`] (every
//! molecule, random access; [`Mol2Reader::open`] opens a path or bytes) and
//! [`Mol2Writer`] ([`Mol2Writer::create`] writes molecule by molecule).

pub(crate) mod codec;

pub use codec::{Mol2Reader, Mol2Writer};
