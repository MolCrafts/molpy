//! GRO, the GROMACS fixed-column structure and trajectory format.
//!
//! The doors are functions of [`crate::io`]: [`read_gro`](crate::io::read_gro)
//! (the first frame) and [`write_gro`](crate::io::write_gro) (one frame). This
//! module holds the format's classes: [`GroReader`] (every frame, random access;
//! [`GroReader::open`] opens a path or bytes) and [`GroWriter`]
//! ([`GroWriter::create`] writes frame by frame).

pub(crate) mod codec;

pub use codec::{GroReader, GroWriter};
