mod aligned;
mod contract;
mod footprint;
mod layout;
mod lifecycle;
mod meta;
mod nullable;
mod ordering;
mod precision;
mod reader;
mod references;
mod schema;

use crate::core::Cell;
use std::collections::BTreeMap;

/// The one spelling every binding accepts, with each level checked
/// against its compressor's range rather than wrapped by a cast.
#[test]
fn compression_specs_parse_and_refuse_out_of_range_levels() {
    use super::Compression;
    assert_eq!("none".parse(), Ok(Compression::None));
    assert_eq!("gzip".parse(), Ok(Compression::Gzip(1)));
    assert_eq!("gzip:9".parse(), Ok(Compression::Gzip(9)));
    assert_eq!("zstd:-3".parse(), Ok(Compression::Zstd(-3)));
    for bad in ["gzip:-1", "gzip:10", "zstd:23", "gzip:x", "lz4", "none:1"] {
        assert!(bad.parse::<Compression>().is_err(), "{bad}");
    }
    let err = "gzip:-1".parse::<Compression>().unwrap_err();
    assert!(err.contains("0..=9"), "{err}");
}
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use molrs::core::Frame;
use molrs::core::MetaValue;
use molrs::core::SimBox;
use molrs::core::{Block, Column, DType};
use ndarray::ArrayD;
use tempfile::TempDir;
use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::{Group, GroupBuilder};
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys,
    StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
};

use crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage;
use crate::io::reader::TrajectoryReader;

use super::{MrecReader, MrecWriter, SequenceSchema};
// Named `super::…` by the test bodies of the child modules.
use super::Compression;
use super::codec::dtype_from_tag;

// -- fixtures -----------------------------------------------------------

/// The sequence root group: the on-disk name of the whole layout.
const TRAJ: &str = "/trajectory";
/// The `trajectory/` group attribute the **reference writer** pins its
/// declared schema in.
///
/// Spelled out rather than imported from `super`: this is a name on disk
/// that a foreign writer reads and writes, so a rename of the Rust constant
/// must break these tests rather than travel silently into them — the same
/// reading `every_meta_variant_round_trips_bit_exact_with_its_dtype_tag`
/// takes on `meta_dtype`.
const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
/// The block every stock fixture frame carries.
const ATOMS: &str = "atoms";
/// The one `f64` column in that block.
const X: &str = "x";
/// A second block, used wherever a frame must present a *subset* of the
/// declared union.
const BONDS: &str = "bonds";
/// The `u64` column of [`BONDS`].
const I: &str = "i";
/// A column name outside the canonical schema vocabulary
/// (`core/schema/mod.rs`), so the same name can carry two dtypes.
///
/// [`X`] cannot: `x` is pinned `Float` there, and `Block::insert`
/// rejects a `u64` under that name before any sequence code runs — the
/// dtype-conflict fixtures would then be testing `Block`, not the schema.
/// Same reason `frame_io`'s round-trip tests probe on a name of their own.
const PROBE: &str = "probe";

fn storage_in(dir: &TempDir) -> ReadableWritableListableStorage {
    Arc::new(FilesystemStore::new(dir.path()).unwrap())
}

fn float_column(values: &[f64]) -> Column {
    Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
}

fn uint_column(values: &[u64]) -> Column {
    Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
}

fn block_with(column: &str, col: Column) -> Block {
    let mut block = Block::new();
    block.insert(column, col, None).unwrap();
    block
}

/// The stock fixture frame: one `atoms` block, one `f64` column.
fn atoms_frame(values: &[f64]) -> Frame {
    let mut frame = Frame::new();
    frame.insert(ATOMS, block_with(X, float_column(values)));
    frame
}

/// Column [`X`] of the `atoms` block, as it came back from the store.
fn atoms_x(frame: &Frame) -> Vec<f64> {
    frame
        .get(ATOMS)
        .expect("the frame carries an atoms block")
        .get(X)
        .expect("the atoms block carries column x")
        .get::<f64>()
        .expect("column x arrived as f64")
        .iter()
        .copied()
        .collect()
}

/// The ragged golden run: 3, 5 and 4 rows, every value distinct so a row
/// range taken from the wrong frame cannot match by accident.
fn ragged_frames() -> Vec<Frame> {
    vec![
        atoms_frame(&[1.0, 2.0, 3.0]),
        atoms_frame(&[10.0, 11.0, 12.0, 13.0, 14.0]),
        atoms_frame(&[20.0, 21.0, 22.0, 23.0]),
    ]
}

/// The cell every box fixture carries — default origin and all-periodic
/// boundary, so those optional arrays are omitted.
fn fixed_cell() -> SimBox {
    SimBox::new(
        Cell::Matrix([[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]]),
        [0.0, 0.0, 0.0],
        [true, true, true],
    )
    .unwrap()
}

/// Mint from the union of `frames`, append them all in order, and close.
fn write_all(store: &ReadableWritableListableStorage, frames: &[Frame]) {
    let schema = SequenceSchema::from_frames(frames).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema).unwrap();
    for frame in frames {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();
}

/// [`write_all`] with both extent knobs pinned, for the tests that need a
/// tiny `R` to reach a chunk boundary in a handful of frames.
fn write_all_with(
    store: &ReadableWritableListableStorage,
    frames: &[Frame],
    rows_per_chunk: u64,
    chunks_per_shard: u64,
) {
    let schema = SequenceSchema::from_frames(frames).unwrap();
    let mut writer = MrecWriter::create_in_storage(store.clone(), schema)
        .unwrap()
        .with_rows_per_chunk(rows_per_chunk)
        .unwrap()
        .with_chunks_per_shard(chunks_per_shard)
        .unwrap();
    for frame in frames {
        writer.append(frame).unwrap();
    }
    writer.close().unwrap();
}

/// Drop [`SCHEMA_ATTRIBUTE`] from `trajectory/`'s `zarr.json`, leaving every
/// array, every array attribute and every byte of data untouched.
///
/// That is exactly the store a conforming foreign writer leaves behind:
/// molrec's `docs/spec/trajectory.md` makes the attribute the **reference
/// writer's** own reopen pin, and resolution needs only `step_index` /
/// `offset` plus the array metadata. The removal is asserted rather than
/// assumed, so a renamed attribute cannot turn either caller into a test of
/// an unmodified store.
fn strip_schema_attribute(store: &ReadableWritableListableStorage) {
    let mut group = Group::open(store.clone(), TRAJ).expect("the sequence group exists");
    assert!(
        group.attributes_mut().remove(SCHEMA_ATTRIBUTE).is_some(),
        "the writer must have pinned {SCHEMA_ATTRIBUTE} for this to be a removal"
    );
    group
        .store_metadata()
        .expect("the group metadata rewrites without the attribute");
}

fn open_sequence(store: &ReadableWritableListableStorage) -> MrecReader {
    MrecReader::from_storage(store.clone()).unwrap()
}

/// `frame(i)`, asserting the frame is present.
fn frame_at(seq: &mut MrecReader, index: u64) -> Frame {
    seq.frame(index)
        .expect("reading a frame must not error")
        .expect("frame is present")
}

/// The committed frame count, taken through the `TrajectoryReader` door —
/// the surface `FrameIterator` and every generic consumer already use.
fn committed_len(seq: &mut MrecReader) -> usize {
    TrajectoryReader::len(seq).expect("len must not error")
}

fn u64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<u64> {
    let arr = Array::open(store.clone(), path).expect("array exists");
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    arr.retrieve_array_subset(&subset)
        .expect("array reads back")
}

fn i64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<i64> {
    let arr = Array::open(store.clone(), path).expect("array exists");
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    arr.retrieve_array_subset(&subset)
        .expect("array reads back")
}

/// The frozen extents of a growth array: `(shard extent, inner chunk
/// extent)`, both in rows of the leading axis and whatever the trailing
/// axes are.
///
/// The array's *shape* is deliberately not returned: it grows with every
/// append, while the two extents are frozen by the first append and must
/// never move again (`zarrs::Array::set_shape` rebuilds the grid from the
/// frozen chunk metadata). Under sharding the array's own chunk extent is
/// the shard, and the planned chunk is the sharding codec's subchunk —
/// the same reading `frame_io`'s `a_five_chunk_column_shards_and_round_trips`
/// takes.
fn extents(store: &ReadableWritableListableStorage, path: &str) -> (Vec<u64>, Vec<u64>) {
    let arr = Array::open(store.clone(), path).expect("array exists");
    let metadata = serde_json::to_value(arr.metadata()).unwrap();
    let shard: Vec<u64> =
        serde_json::from_value(metadata["chunk_grid"]["configuration"]["chunk_shape"].clone())
            .expect("a regular chunk grid carries a chunk_shape");
    let sharding = metadata["codecs"]
        .as_array()
        .expect("codecs is an array")
        .iter()
        .find(|codec| codec["name"] == "sharding_indexed")
        .expect("a growth array is sharded, so file count follows bytes")
        .clone();
    let inner: Vec<u64> = serde_json::from_value(sharding["configuration"]["chunk_shape"].clone())
        .expect("the sharding codec carries its subchunk shape");
    (shard, inner)
}

/// Every file under `dir`, keyed by its path relative to `dir`.
fn file_map(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut out = BTreeMap::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else {
                let key = path.strip_prefix(dir).unwrap().to_path_buf();
                out.insert(key, std::fs::read(&path).unwrap());
            }
        }
    }
    out
}

/// The chunk (shard) files under `dir`, without the `zarr.json` metadata —
/// the bytes a shard actually costs.
fn chunk_files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    file_map(dir)
        .into_iter()
        .filter(|(path, _)| path.file_name() != Some(std::ffi::OsStr::new("zarr.json")))
        .collect()
}

fn chunk_bytes(dir: &Path) -> u64 {
    chunk_files(dir).values().map(|v| v.len() as u64).sum()
}

/// Deterministic, poorly compressible `f64` payload in `[1, 2)`: a
/// splitmix hash supplies 52 random mantissa bits, so gzip lands near
/// 0.85x and every value is finite and bit-exactly comparable. Same shape
/// as the `zarrs_pins` payload, for the same reason: a compressible
/// fixture would make a write-amplification bound pass vacuously.
fn noise(index: u64) -> f64 {
    let mut x = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^= x >> 31;
    f64::from_bits((0x3ffu64 << 52) | (x & 0x000f_ffff_ffff_ffff))
}
