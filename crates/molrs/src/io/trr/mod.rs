//! TRR, the GROMACS full-precision trajectory format (XDR-encoded).
//!
//! The doors are functions of [`crate::io`]: [`read_trr`](crate::io::read_trr)
//! (the first frame) and [`write_trr`](crate::io::write_trr) (a one-frame
//! file). This module holds the format's classes: [`TrrReader`]
//! ([`TrrReader::open`] opens a path or bytes), [`TrrWriter`]
//! ([`TrrWriter::create`] writes frame by frame), and [`TrrIndexBuilder`], the
//! chunked frame indexer.

pub(crate) mod codec;

pub use codec::{TrrIndexBuilder, TrrReader, TrrWriter};
