//! SDF, the MDL structure-data file (V2000 molfile records).
//!
//! The door is a function of [`crate::io`]: [`read_sdf`](crate::io::read_sdf)
//! (the first record). This module holds the format's classes: [`SdfReader`]
//! (every record, random access; [`SdfReader::open`] opens a path or bytes) and
//! [`SdfIndexBuilder`], the chunked record indexer.

pub(crate) mod codec;

pub use codec::{SdfIndexBuilder, SdfReader};
