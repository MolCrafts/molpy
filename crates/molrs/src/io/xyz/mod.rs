//! XYZ and extended XYZ (the `Properties=` / `Lattice=` comment-line
//! convention of ASE and QUIP).
//!
//! The doors are functions of [`crate::io`]: [`read_xyz`](crate::io::read_xyz)
//! (the first frame) and [`write_xyz`](crate::io::write_xyz) (one frame). This
//! module holds the format's classes: [`XyzReader`] (every frame, random access;
//! [`XyzReader::open`] opens a path or bytes), [`XyzWriter`]
//! ([`XyzWriter::create`] writes frame by frame), and [`XyzIndexBuilder`], the
//! chunked frame indexer.

pub(crate) mod codec;

pub use codec::{XyzIndexBuilder, XyzReader, XyzWriter};
