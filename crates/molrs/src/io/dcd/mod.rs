//! DCD, the CHARMM / NAMD binary trajectory format (also written by OpenMM and
//! LAMMPS).
//!
//! The doors are functions of [`crate::io`]: [`read_dcd`](crate::io::read_dcd)
//! (the first frame) and [`write_dcd`](crate::io::write_dcd) (a one-frame
//! file). This module holds the format's classes: [`DcdReader`] (O(1) random
//! access; [`DcdReader::open`] opens a path or bytes), [`DcdWriter`]
//! ([`DcdWriter::create`] writes frame by frame), [`DcdIndexBuilder`], the
//! chunked frame indexer, and [`DcdFrameDecoder`], which decodes the frame
//! bodies it indexes with the state it captured.

pub(crate) mod codec;

pub use codec::{DcdFrameDecoder, DcdIndexBuilder, DcdReader, DcdWriter};
