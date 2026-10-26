//! CIF, the IUCr Crystallographic Information File (small-molecule CIF and the
//! mmCIF `_atom_site` loop).
//!
//! The doors are functions of [`crate::io`]: [`read_cif`](crate::io::read_cif)
//! (the first `data_` block) and [`write_cif`](crate::io::write_cif) (one
//! block). This module holds the format's classes: [`CifReader`] (every block,
//! random access; [`CifReader::open`] opens a path or bytes) and [`CifWriter`]
//! ([`CifWriter::create`] writes block by block).

pub(crate) mod codec;

pub use codec::{CifReader, CifWriter};
