//! PDB, the Protein Data Bank coordinate format (PDB 3.3,
//! <https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html>).
//!
//! The doors are functions of [`crate::io`]: [`read_pdb`](crate::io::read_pdb)
//! (the first `MODEL`) and [`write_pdb`](crate::io::write_pdb) (one frame, with
//! `END`). This module holds the format's classes: [`PdbReader`] (every `MODEL`,
//! random access; [`PdbReader::open`] opens a path or bytes), [`PdbWriter`]
//! ([`PdbWriter::create`] writes one `MODEL` per frame), and
//! [`PdbIndexBuilder`], the chunked frame indexer.

pub(crate) mod codec;

pub use codec::{PdbIndexBuilder, PdbReader, PdbWriter};
