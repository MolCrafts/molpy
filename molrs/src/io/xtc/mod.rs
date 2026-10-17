//! XTC, the GROMACS compressed trajectory format (XDR-encoded, lossy
//! coordinate compression).
//!
//! The doors are functions of [`crate::io`]: [`read_xtc`](crate::io::read_xtc)
//! (the first frame) and [`write_xtc`](crate::io::write_xtc) (a one-frame
//! file). This module holds the format's classes: [`XtcReader`]
//! ([`XtcReader::open`] opens a path or bytes), [`XtcWriter`]
//! ([`XtcWriter::create`] writes frame by frame), and [`XtcIndexBuilder`], the
//! chunked frame indexer.

pub(crate) mod codec;

pub use codec::{XtcIndexBuilder, XtcReader, XtcWriter};
