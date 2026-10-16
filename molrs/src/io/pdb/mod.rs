//! PDB, the Protein Data Bank coordinate format (PDB 3.3,
//! <https://www.wwpdb.org/documentation/file-format-content/format33/sect9.html>).
//!
//! The doors are functions of [`crate::io`]: [`read_pdb`](crate::io::read_pdb),
//! [`read_pdb_trajectory`](crate::io::read_pdb_trajectory) (every `MODEL`),
//! [`read_pdb_bytes`](crate::io::read_pdb_bytes),
//! [`write_pdb`](crate::io::write_pdb) and
//! [`write_pdb_trajectory`](crate::io::write_pdb_trajectory). This module holds
//! the format's classes: [`PdbReader`] / [`PdbWriter`] over any byte stream,
//! and [`PdbIndexBuilder`], the chunked frame indexer.

pub(crate) mod codec;
pub(crate) mod labels;

pub use codec::{PdbIndexBuilder, PdbReader, PdbWriter};
