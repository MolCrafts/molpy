//! Scientific records (`*.mrec`), the MolRec record format.
//!
//! A **scientific record** is one self-describing package on disk: `meta`
//! plus at least one of a snapshot (`frame`), a topology (`system`), a
//! time-ordered frame sequence (`trajectory`), a force field (`forcefield`),
//! or a run `status`. On disk a record is a **directory** whose name
//! conventionally ends in `.mrec` (for example `water.mrec/`); inside, arrays
//! are stored with **Zarr V3** — an open format that cuts each array into
//! compressed chunks and records their shape in a small JSON document. The
//! Cargo feature that enables this module is still named `zarr` after that
//! encoding; the Zarr storage engine itself is crate-private. A closed
//! directory can be packed into a sibling `*.mrec.zip` archive.
//!
//! # The doors
//!
//! - [`write_mrec`](crate::io::write_mrec) writes a whole [`MolRec`] — every
//!   section it holds — to a `*.mrec` directory, or into a buffer as the bytes
//!   of a packed `*.mrec.zip`.
//! - [`MrecFile`] reads a record back one section at a time:
//!   [`MrecFile::open`] takes a directory, a packed `*.mrec.zip` (read in
//!   place) or a packed record's bytes, [`MrecFile::from_storage`] any open
//!   Zarr store (in-memory, a host's); then [`sections`](MrecFile::sections)
//!   lists them, [`meta`](MrecFile::meta), [`frame`](MrecFile::frame),
//!   [`system`](MrecFile::system), [`forcefield`](MrecFile::forcefield),
//!   [`status`](MrecFile::status), [`method`](MrecFile::method),
//!   [`metrics`](MrecFile::metrics) and [`observables`](MrecFile::observables)
//!   decode one, and [`record`](MrecFile::record) all of them.
//! - A run too large to hold in memory: pin a [`SequenceSchema`], append with
//!   [`MrecWriter`], read one frame at a time with [`MrecReader`]
//!   ([`MrecReader::open`] takes what [`MrecFile::open`] takes,
//!   [`MrecReader::from_storage`] any open storage).
//! - [`pack_mrec_zip`] collapses a closed directory into one `*.mrec.zip`.
//! - [`ForceFieldSection`] — the `forcefield` section as data.
//!   The language-neutral JSON Schema lives in molrec
//!   (`schema/core/record.schema.json`).
//!
//! The path- and byte-taking doors need the `filesystem` feature.
//!
//! Every writer creates the record root and its `meta/` group, written as the
//! producer supplied it; `meta` keys a reader does not recognise are kept as
//! they are. Identity of a store is the
//! `*.mrec/` path suffix plus a Zarr root.
//!
//! # Examples
//!
//! ```
//! # #[cfg(not(feature = "filesystem"))]
//! # fn main() {}
//! # #[cfg(feature = "filesystem")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use molrs::io::mrec::{MolRec, MrecFile};
//! use molrs::io::write_mrec;
//!
//! let dir = tempfile::tempdir().unwrap();
//! let path = dir.path().join("water.mrec");
//!
//! let mut record = MolRec::new();
//! record.frame = Some(molrs::core::Frame::new());
//! write_mrec(&path, &record)?;
//!
//! let file = MrecFile::open(&path)?;
//! assert!(file.sections()?.iter().any(|s| s == "frame"));
//! assert!(file.frame()?.is_some());
//! # Ok(())
//! # }
//! ```

mod forcefield_mapping;
pub(crate) mod forcefield_section;
mod record;
pub(crate) mod zarr_storage;

pub use forcefield_section::{EndpointKey, ForceFieldSection, SectionStyle, style_block_name};
pub use record::{MolRec, Observables};
pub use zarr_storage::{
    Compression, MrecFile, MrecReader, MrecWriter, SequenceSchema, StoredMetrics, StoredObservable,
};

#[cfg(feature = "filesystem")]
pub use zarr_storage::pack_mrec_zip;
