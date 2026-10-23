//! The `trajectory/` frame sequence: one on-disk layout, one encoder, one
//! decoder.

mod block_arrays;
mod block_hints;
mod box_arrays;
mod codec;
mod frame_content;
mod growth;
mod layout;
mod reader;
mod schema;
mod sequence_arrays;
mod series;
mod store_access;
mod writer;

pub use codec::Compression;
pub use reader::MrecReader;
pub use schema::SequenceSchema;
pub use writer::MrecWriter;

/// The whole `trajectory/` sequence contract, pinned before the code exists.
///
/// **Fixtures.** Every test writes a real store under a `tempfile::TempDir`;
/// nothing here is a mock except [`RecordingStorage`], which exists only to
/// observe *write order* and delegates every byte to a real `FilesystemStore`.
/// The stock fixture frame is one `atoms` block carrying one `f64` column
/// [`X`], with non-repeating values, so a mis-sliced CSR row range is a
/// mismatch rather than a coincidence.
///
/// **What is pinned on disk.** The layout of Design § 1 is the contract, so
/// these tests read `trajectory/step`, `trajectory/<block>/offset`,
/// `trajectory/<block>/step_index`, `trajectory/box/step_index` and
/// `trajectory/meta/<key>` directly through `zarrs`, in addition to reading
/// frames back through [`MrecReader`]. A round trip that agreed with itself
/// while writing a private layout would satisfy neither molrec nor molpy.
///
/// **No tolerances anywhere.** Every value assertion is `assert_eq!` on the
/// bits that arrived: the downstream precision study forbids lossy encoding,
/// so a comparison with a tolerance would hide exactly the defect this suite
/// is for.
///
/// **Extents are read from the on-disk array metadata**, because the growth
/// arrays' chunk and shard extents are frozen at the first append and have no
/// other observation point ([`extents`]).
#[cfg(all(test, feature = "filesystem"))]
mod tests;
