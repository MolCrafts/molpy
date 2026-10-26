//! Zarr V3 storage — the reference binding of the MolRec record contract for
//! array-shaped sections (frame / trajectory / system / large observables).
//!
//! **Zarr** is an open format for chunked, compressed N-dimensional arrays: an
//! array is cut into fixed-size *chunks*, each chunk is compressed and stored
//! under its own key, and a small JSON document beside them records the shape,
//! the element type and the codecs. Version 3 is the current specification. The
//! keys can live in any key-value store — here, a directory tree on disk, or a
//! `.zip` archive of one. **MolRec** is the MolCrafts record contract that says
//! which sections a stored record has and what each one means; this module is
//! its reference binding, so the layouts these doors write are the on-disk
//! shapes other MolCrafts tools read.
//!
//! Crate-private: the record's doors are [`crate::io::write_mrec`] and the
//! classes of [`crate::io::mrec`].
//!
//! # The doors
//!
//! Whole records go through `write_mrec` (a directory store on disk, or a
//! packed archive in memory) and `MrecFile` (any of those, or any store the
//! caller already holds), section by section.
//!
//! A long run that cannot be held in memory goes through the streaming trio:
//! [`SequenceSchema`] pins what the run will write, [`MrecWriter`]
//! appends and commits frame by frame, and [`MrecReader`] reads one frame at
//! a time back out. Their shared on-disk layout, and the three-names rule that
//! keeps them apart from the eager in-memory `Trajectory` carrier, are
//! documented on [`MrecReader`] and [`MrecWriter`] themselves.
//!
//! A store nobody is appending to any more can be collapsed into one file with
//! `pack_mrec_zip`; `MrecFile::open` and `MrecReader::open` read the archive
//! in place. Every path- and byte-taking door needs the `filesystem` feature;
//! the storage-taking doors do not.
//!
//! Closed **metrics** are the catalog document plus `series/` (`f64`),
//! `steps/` (`i64`) and `wall_time/` (`string`) arrays. The live JSONL WAL
//! (`metrics/metrics.jsonl`) is stored and read back as text when the record
//! carries it. See [`write_mrec`] and molrec `docs/spec/metrics.md`. This
//! crate does not densify a WAL into those arrays.
//!
//! # Spike verdicts (zarrs 0.23.13)
//!
//! Measured by the `zarrs_pins` test module below, which is the executable
//! record of these three facts.
//!
//! **Q5 — commit semantics.** Partial encoding *can* rewrite a partially
//! filled trailing inner chunk as a tail-only write, so `flush()` may commit
//! every buffered row, ragged tail included, and commit granularity is any
//! step rather than chunk-aligned. (Since 2026-09-02 the writer lands whole
//! frame-aligned chunks on its own cadence and rewrites a partial tail only
//! on an explicit `flush`; the shard index sits at the *start* of the shard
//! so a whole-chunk append leaves no dead bytes, and no shard is ever
//! re-encoded whole — see `sequence/mod.rs`'s module doc.) Evidence: extending the trailing
//! inner chunk of shard 1 from 500 to 800 rows left the completed shard 0 file
//! **byte identical**, cost a **single** store write of 18 180 B against a
//! 146 399 B shard (the codec read back only the 11 486 B straddling chunk),
//! and all 14 800 rows came back bit exact. One caveat for the writer: the
//! overwritten copy of the rewritten chunk stays in the file as dead bytes
//! (shard 1 grew 146 399 → 164 447 B), so repeatedly extending one tail chunk
//! trades file size for latency. A pure append leaves no dead bytes — Q6
//! measures growth exactly equal to the new chunk.
//!
//! **Q7 — zip read path: `zarrs_zip` works.** Its read-only
//! `ZipStorageAdapter` opened a zip whose entries are STORED (method 0) copies
//! of a molrs directory store and returned the array bit exact, taking the
//! adapter's documented `Method::Store` fast path (byte ranges translated to
//! absolute offsets, no inflate). Evidence: `q7_*` reads `/frame/atoms/x` out
//! of a zipped `write_mrec` store and gets `[1.0, 2.5, -3.75, 1e-300]`
//! back unchanged. `zarrs_zip` 0.5.2 resolves against the locked
//! `zarrs_storage` 0.4.5, so no in-tree minimal zip reader is needed.
//! Cross-read verified **out of tree** (never in the default gate, which admits
//! no third-party scientific software): zarr-python 3.3.0 from molrec's dev
//! extra opened the same layout through `zarr.storage.ZipStore` and returned
//! `frame/atoms/x` bit exact, both for a zip written by the `zip` crate and for
//! one written by `python3 -m zipfile` with `ZIP_STORED` (both 1650 B).
//!
//! **Store-level write amplification — a `zarrs_filesystem` 0.3.12 caveat.**
//! Q4 and Q6 pin the *codec → store* contract: one append hands the store one
//! compressed inner chunk plus one 16 B × k shard index. `FilesystemStore`
//! then turns that into a whole-file rewrite —
//! `WritableStorageTraits::set_partial_many` delegates to
//! `zarrs_storage::store_set_partial_many`, which reads the entire value,
//! patches it in memory and `set`s it back, even though
//! `supports_set_partial()` reports `true`. Measured directly against the
//! store API: a 16 B write at the tail of a sparse shard file allocated the
//! whole file and scaled linearly — 1 MiB / 1.6 ms, 8 MiB / 5.0 ms,
//! 64 MiB / 52 ms, 256 MiB / 247 ms. Shard sizing must therefore be chosen
//! against per-flush latency, not only against file count.

mod chunking;
mod error;
mod forcefield_io;
mod frame_io;
#[cfg(feature = "filesystem")]
mod pack;
#[cfg(feature = "filesystem")]
mod positional_write;
pub(crate) mod record_io;
mod sequence;
// Built where it is registered (no C `zstd`), and for the tests that hold it
// to the C encoder's frames.
#[cfg(any(test, not(feature = "zarr-codecs")))]
mod zstd_decode;

#[cfg(feature = "filesystem")]
pub use pack::pack_mrec_zip;
#[cfg(feature = "filesystem")]
pub use record_io::write_mrec;
// The store-taking record doors need no filesystem: an in-memory or host
// store (wasm) writes a whole record through `write_mrec_storage` and reads it
// back through `MrecFile::from_storage`.
pub use record_io::{MrecFile, StoredMetrics, StoredObservable, write_mrec_storage};
pub use sequence::{Compression, MrecReader, MrecWriter, SequenceSchema};

/// Mechanics pins for `zarrs` 0.23.13 — the append fast path the
/// `trajectory/` frame sequence is built on.
///
/// Every test writes a real `FilesystemStore` under a `tempfile::tempdir`;
/// nothing here is a mock. Q1–Q4 and Q6 are **pins**: they must stay green,
/// because a `zarrs` upgrade that breaks any of them silently turns each
/// `flush()` into an O(shard) rewrite. Q5 and Q7 assert the invariants
/// recorded in the module doc above, so a regression is loud.
///
/// One shared shape: an `[nrows, 3]` f64 array whose chunk (= shard) is
/// `rows_per_chunk * chunks_per_shard` rows, with a `sharding_indexed` codec
/// carrying `gzip` on the inner chunks and its index at the **end** of the
/// shard. Row width is 24 B, so an inner chunk of 1000 rows is 24 000 B
/// uncompressed and the shard index is `16 * chunks_per_shard + 4` B
/// (2 × u64 per inner chunk, plus the `crc32c` checksum).
#[cfg(all(test, feature = "filesystem"))]
mod zarrs_pins {
    use std::io::Write as _;
    use std::num::NonZeroU64;
    use std::path::Path;
    use std::sync::Arc;

    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::MolRsError;
    use molrs::io::mrec::MolRec;
    use ndarray::ArrayD;
    use tempfile::tempdir;
    use zarrs::array::codec::{GzipCodec, ShardingCodecBuilder, ShardingIndexLocation};
    use zarrs::array::{Array, ArrayBuilder, ArraySubset, CodecOptions, data_type};
    use zarrs::config::global_config;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
    use zarrs::storage::{
        ReadableWritableListableStorage, ReadableWritableListableStorageTraits, StoreKey,
    };

    use super::{MrecFile, write_mrec};

    /// Trailing extent of the pinned array: an `[nrows, 3]` f64 table.
    const NCOLS: u64 = 3;
    /// Rows in one inner (sub)chunk. 1000 × 3 × 8 B = 24 000 B uncompressed.
    const ROWS_PER_CHUNK: u64 = 1000;
    /// Uncompressed bytes of one inner chunk.
    const CHUNK_BYTES: u64 = ROWS_PER_CHUNK * NCOLS * 8;

    /// Encoded size of a shard index holding `k` inner chunks: two `u64` per
    /// chunk plus the `crc32c` checksum the `zarr` feature tier enables.
    const fn index_bytes(chunks_per_shard: u64) -> u64 {
        chunks_per_shard * 16 + 4
    }

    /// The options every append must carry. Built explicitly and threaded into
    /// the `_opt` methods — `global_config_mut()` is a process-wide `RwLock`
    /// and is never touched (see Q4).
    fn partial_encoding_options() -> CodecOptions {
        global_config()
            .codec_options()
            .with_experimental_partial_encoding(true)
    }

    /// Deterministic, poorly-compressible f64 payload in `[1, 2)`: a splitmix
    /// hash supplies 52 random mantissa bits, so gzip lands near 0.85× and the
    /// values are finite and bit-exactly comparable.
    fn rows(start_row: u64, count: u64) -> Vec<f64> {
        let mut out = Vec::with_capacity((count * NCOLS) as usize);
        for row in start_row..start_row + count {
            for col in 0..NCOLS {
                let mut x = (row * NCOLS + col).wrapping_add(0x9e37_79b9_7f4a_7c15);
                x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                x ^= x >> 31;
                out.push(f64::from_bits(
                    (0x3ffu64 << 52) | (x & 0x000f_ffff_ffff_ffff),
                ));
            }
        }
        out
    }

    /// Build the `[0, NCOLS]` growth array at `/traj`, sharded and gzipped, and
    /// write its metadata.
    fn build_sequence_array(
        store: ReadableWritableListableStorage,
        chunks_per_shard: u64,
    ) -> Array<dyn ReadableWritableListableStorageTraits> {
        let subchunk = vec![
            NonZeroU64::new(ROWS_PER_CHUNK).unwrap(),
            NonZeroU64::new(NCOLS).unwrap(),
        ];
        let mut sharding = ShardingCodecBuilder::new(subchunk, &data_type::float64());
        sharding
            .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5).unwrap())])
            .index_location(ShardingIndexLocation::End);
        let array = ArrayBuilder::new(
            vec![0, NCOLS],
            vec![ROWS_PER_CHUNK * chunks_per_shard, NCOLS],
            data_type::float64(),
            0.0f64,
        )
        .array_to_bytes_codec(Arc::new(sharding.build()))
        .build(store, "/traj")
        .unwrap();
        array.store_metadata().unwrap();
        array
    }

    /// Grow the leading axis and write `count` rows at `start_row`.
    fn append(
        array: &mut Array<dyn ReadableWritableListableStorageTraits>,
        start_row: u64,
        count: u64,
        options: &CodecOptions,
    ) {
        array.set_shape(vec![start_row + count, NCOLS]).unwrap();
        let subset =
            ArraySubset::new_with_start_shape(vec![start_row, 0], vec![count, NCOLS]).unwrap();
        array
            .store_array_subset_opt(&subset, rows(start_row, count).as_slice(), options)
            .unwrap();
    }

    /// Size of the shard file backing shard `shard_index` of `/traj`.
    fn shard_size(root: &Path, shard_index: u64) -> u64 {
        std::fs::metadata(root.join(format!("traj/c/{shard_index}/0")))
            .unwrap()
            .len()
    }

    /// Raw bytes of the shard file backing shard `shard_index` of `/traj`.
    fn shard_bytes(root: &Path, shard_index: u64) -> Vec<u8> {
        std::fs::read(root.join(format!("traj/c/{shard_index}/0"))).unwrap()
    }

    // -- Q1 ---------------------------------------------------------------

    /// Appending one inner chunk grows the shard **file** by about one
    /// compressed chunk, not by a whole shard extent.
    #[test]
    fn q1_append_grows_the_shard_file_by_one_chunk_not_one_shard() {
        const CHUNKS_PER_SHARD: u64 = 16;
        let dir = tempdir().unwrap();
        let root = dir.path();
        let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(root).unwrap());
        let mut array = build_sequence_array(store, CHUNKS_PER_SHARD);
        let options = partial_encoding_options();

        append(&mut array, 0, ROWS_PER_CHUNK, &options);
        let after_one = shard_size(root, 0);
        append(&mut array, ROWS_PER_CHUNK, ROWS_PER_CHUNK, &options);
        let after_two = shard_size(root, 0);

        // Measured on zarrs 0.23.13: 22 756 B -> 45 222 B, i.e. +22 466 B.
        let growth = after_two - after_one;
        let shard_extent_bytes = CHUNK_BYTES * CHUNKS_PER_SHARD; // 384 000 B
        assert!(
            (CHUNK_BYTES / 2..=CHUNK_BYTES + index_bytes(CHUNKS_PER_SHARD)).contains(&growth),
            "expected about one compressed {CHUNK_BYTES} B chunk, grew by {growth} B"
        );
        assert!(
            after_two < shard_extent_bytes / 4,
            "two chunks of a {shard_extent_bytes} B shard extent occupy {after_two} B; \
             the file must not be sized by the shard"
        );
    }

    // -- Q2 ---------------------------------------------------------------

    /// `set_shape` grows the leading axis only; the chunk (shard) extent and
    /// its grid metadata stay frozen at what creation pinned.
    #[test]
    fn q2_set_shape_grows_the_leading_axis_with_frozen_chunk_extents() {
        const CHUNKS_PER_SHARD: u64 = 16;
        const SHARD_ROWS: u64 = ROWS_PER_CHUNK * CHUNKS_PER_SHARD;
        let dir = tempdir().unwrap();
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(dir.path()).unwrap());
        let mut array = build_sequence_array(store, CHUNKS_PER_SHARD);

        let grid_metadata = array.chunk_grid().metadata();
        let full_chunk = vec![
            NonZeroU64::new(SHARD_ROWS).unwrap(),
            NonZeroU64::new(NCOLS).unwrap(),
        ];
        assert_eq!(array.shape(), &[0, NCOLS]);

        array.set_shape(vec![1, NCOLS]).unwrap();
        assert_eq!(array.chunk_grid().metadata(), grid_metadata);
        assert_eq!(array.chunk_shape(&[0, 0]).unwrap(), full_chunk);
        assert_eq!(array.chunk_grid_shape(), &[1, 1]);

        array.set_shape(vec![SHARD_ROWS + 1, NCOLS]).unwrap();
        assert_eq!(array.chunk_grid().metadata(), grid_metadata);
        // The boundary shard keeps the full extent; it is fill-padded, not clipped.
        assert_eq!(array.chunk_shape(&[1, 0]).unwrap(), full_chunk);
        assert_eq!(array.chunk_grid_shape(), &[2, 1]);
        assert_eq!(array.shape(), &[SHARD_ROWS + 1, NCOLS]);
    }

    // -- Q3 ---------------------------------------------------------------

    /// A `[0, N]` array with non-zero chunk extents is legal metadata, is
    /// writable, and reads back as empty.
    #[test]
    fn q3_zero_length_array_with_nonzero_chunks_is_legal_and_reads_empty() {
        let dir = tempdir().unwrap();
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(dir.path()).unwrap());
        build_sequence_array(store.clone(), 16);

        let reopened = Array::open(store, "/traj").unwrap();
        assert_eq!(reopened.shape(), &[0, NCOLS]);
        assert_eq!(reopened.chunk_grid_shape(), &[0, 1]);
        let empty: Vec<f64> = reopened
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![0, NCOLS]))
            .unwrap();
        assert!(empty.is_empty());
    }

    // -- Q4 ---------------------------------------------------------------

    /// The explicitly built `CodecOptions` is load-bearing: threaded through
    /// `store_array_subset_opt` it writes one chunk, while the non-`_opt`
    /// method (which builds its own default options) rewrites the whole shard.
    #[test]
    fn q4_partial_encoding_only_applies_when_threaded_through_the_opt_method() {
        const CHUNKS_PER_SHARD: u64 = 8;
        const WARM_ROWS: u64 = 4 * ROWS_PER_CHUNK;

        fn warm_storage(
            root: &Path,
        ) -> (
            Array<dyn ReadableWritableListableStorageTraits>,
            Arc<PerformanceMetricsStorageAdapter<FilesystemStore>>,
        ) {
            let metrics = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
                FilesystemStore::new(root).unwrap(),
            )));
            let store: ReadableWritableListableStorage = metrics.clone();
            let mut array = build_sequence_array(store, CHUNKS_PER_SHARD);
            append(&mut array, 0, WARM_ROWS, &partial_encoding_options());
            metrics.reset();
            (array, metrics)
        }

        let opt_dir = tempdir().unwrap();
        let (mut opt_array, opt_metrics) = warm_storage(opt_dir.path());
        append(
            &mut opt_array,
            WARM_ROWS,
            ROWS_PER_CHUNK,
            &partial_encoding_options(),
        );
        let opt_bytes = opt_metrics.bytes_written() as u64;

        let default_dir = tempdir().unwrap();
        let (mut default_array, default_metrics) = warm_storage(default_dir.path());
        default_array
            .set_shape(vec![WARM_ROWS + ROWS_PER_CHUNK, NCOLS])
            .unwrap();
        let subset =
            ArraySubset::new_with_start_shape(vec![WARM_ROWS, 0], vec![ROWS_PER_CHUNK, NCOLS])
                .unwrap();
        default_array
            .store_array_subset(&subset, rows(WARM_ROWS, ROWS_PER_CHUNK).as_slice())
            .unwrap();
        // Measured on zarrs 0.23.13: 22 630 B with the option, 112 546 B without
        // it — and that is with only 5 of 8 inner chunks populated.
        let default_bytes = default_metrics.bytes_written() as u64;

        assert!(
            opt_bytes < CHUNK_BYTES + index_bytes(CHUNKS_PER_SHARD),
            "partial encoding wrote {opt_bytes} B for one {CHUNK_BYTES} B chunk"
        );
        assert!(
            default_bytes > 4 * opt_bytes,
            "the default options must rewrite the whole shard: {default_bytes} B \
             vs {opt_bytes} B with partial encoding"
        );
    }

    // -- Q5 ---------------------------------------------------------------

    /// EXPERIMENT (verdict in the module doc): extending a partially filled
    /// trailing inner chunk is a tail-only write — earlier shards are byte
    /// identical, one store write carries far less than a shard, and every row
    /// reads back bit exact.
    #[test]
    fn q5_extending_the_trailing_inner_chunk_is_a_tail_only_write() {
        const CHUNKS_PER_SHARD: u64 = 8;
        const SHARD_ROWS: u64 = ROWS_PER_CHUNK * CHUNKS_PER_SHARD; // 8000
        const PARTIAL_END: u64 = SHARD_ROWS + 6 * ROWS_PER_CHUNK + 500; // 14 500
        const FINAL_END: u64 = PARTIAL_END + 300; // 14 800

        let dir = tempdir().unwrap();
        let root = dir.path();
        let metrics = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            FilesystemStore::new(root).unwrap(),
        )));
        let store: ReadableWritableListableStorage = metrics.clone();
        let mut array = build_sequence_array(store, CHUNKS_PER_SHARD);
        let options = partial_encoding_options();

        append(&mut array, 0, SHARD_ROWS, &options);
        append(&mut array, SHARD_ROWS, PARTIAL_END - SHARD_ROWS, &options);
        let shard0_before = shard_bytes(root, 0);
        metrics.reset();

        // Measured on zarrs 0.23.13: shard 1 grew 146 399 B -> 164 447 B, one
        // store write of 18 180 B, and the codec read back only the straddling
        // trailing chunk (11 486 B), not the 146 399 B shard.
        append(&mut array, PARTIAL_END, FINAL_END - PARTIAL_END, &options);

        assert_eq!(
            shard_bytes(root, 0),
            shard0_before,
            "the completed shard must not be touched by a tail append"
        );
        assert_eq!(metrics.writes(), 1, "a tail append is one store write");
        let written = metrics.bytes_written() as u64;
        assert!(
            written < 2 * CHUNK_BYTES,
            "tail append handed {written} B to the store; a whole-shard rewrite of \
             the 7 populated chunks would be about {} B",
            7 * CHUNK_BYTES
        );

        let all: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(vec![FINAL_END, NCOLS]))
            .unwrap();
        assert_eq!(all, rows(0, FINAL_END), "rows must survive bit exact");
    }

    // -- Q6 ---------------------------------------------------------------

    /// Write amplification: one flush hands the store exactly one compressed
    /// inner chunk plus one shard index — never O(shard).
    #[test]
    fn q6_one_flush_writes_one_compressed_chunk_plus_one_shard_index() {
        const CHUNKS_PER_SHARD: u64 = 8;
        const WARM_ROWS: u64 = 6 * ROWS_PER_CHUNK;

        let dir = tempdir().unwrap();
        let root = dir.path();
        let metrics = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            FilesystemStore::new(root).unwrap(),
        )));
        let store: ReadableWritableListableStorage = metrics.clone();
        let mut array = build_sequence_array(store, CHUNKS_PER_SHARD);
        let options = partial_encoding_options();

        append(&mut array, 0, WARM_ROWS, &options);
        let size_before = shard_size(root, 0);
        metrics.reset();

        append(&mut array, WARM_ROWS, ROWS_PER_CHUNK, &options);

        let written = metrics.bytes_written() as u64;
        // Measured on zarrs 0.23.13: 22 609 B written = 22 477 B of file growth
        // + the 132 B shard index, against a 192 000 B shard extent.
        let file_growth = shard_size(root, 0) - size_before;
        assert_eq!(metrics.writes(), 1, "one flush is one store write");
        // The index sits at the end of the shard, so the new chunk lands exactly
        // on the old index and the file grows by the chunk alone.
        assert_eq!(
            written,
            file_growth + index_bytes(CHUNKS_PER_SHARD),
            "a flush is one compressed chunk plus one shard index"
        );
        assert!(
            written < CHUNK_BYTES * CHUNKS_PER_SHARD / 4,
            "flush wrote {written} B against a {} B shard extent",
            CHUNK_BYTES * CHUNKS_PER_SHARD
        );
    }

    // -- Q7 ---------------------------------------------------------------

    /// Pack `dir` into `zip_path` with every entry STORED (method 0), the way
    /// `pack()` will: the chunks are already gzipped, so packing is
    /// concatenation plus a central directory.
    fn zip_stored(dir: &Path, zip_path: &Path) {
        let file = std::fs::File::create(zip_path).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored);
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            let mut entries: Vec<_> = std::fs::read_dir(&current)
                .unwrap()
                .flatten()
                .map(|e| e.path())
                .collect();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                let name = path
                    .strip_prefix(dir)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .to_string();
                writer.start_file(name, options).unwrap();
                writer.write_all(&std::fs::read(&path).unwrap()).unwrap();
            }
        }
        writer.finish().unwrap();

        let mut archive = zip::ZipArchive::new(std::fs::File::open(zip_path).unwrap()).unwrap();
        for i in 0..archive.len() {
            let entry = archive.by_index(i).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{} is not a stored entry",
                entry.name().unwrap()
            );
        }
    }

    fn record_with_x(values: &[f64]) -> Result<MolRec, MolRsError> {
        let mut block = Block::new();
        block.insert(
            "x",
            ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap(),
            None,
        )?;
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        let mut record = MolRec::new();
        record.frame = Some(frame);
        Ok(record)
    }

    /// EXPERIMENT (verdict in the module doc): `zarrs_zip`'s read-only
    /// `ZipStorageAdapter` opens a STORED-entry zip of a molrs directory store
    /// and returns the array bit exact.
    #[test]
    fn q7_zarrs_zip_reads_a_stored_entry_zip_of_a_molrs_store() {
        const VALUES: [f64; 4] = [1.0, 2.5, -3.75, 1.0e-300];

        let dir = tempdir().unwrap();
        let storage_dir = dir.path().join("rec.mrec");
        write_mrec(&storage_dir, &record_with_x(&VALUES).unwrap()).unwrap();
        // The directory store is the reference the zip must reproduce.
        assert!(
            MrecFile::open(&storage_dir)
                .unwrap()
                .frame()
                .unwrap()
                .is_some()
        );

        let zip_path = dir.path().join("rec.mrec.zip");
        zip_stored(&storage_dir, &zip_path);

        let outer = Arc::new(FilesystemStore::new(dir.path()).unwrap());
        let zip_store = Arc::new(
            zarrs_zip::ZipStorageAdapter::new(outer, StoreKey::new("rec.mrec.zip").unwrap())
                .unwrap(),
        );
        let array = Array::open(zip_store, "/frame/atoms/x").unwrap();
        let read: Vec<f64> = array
            .retrieve_array_subset(&ArraySubset::new_with_shape(array.shape().to_vec()))
            .unwrap();
        assert_eq!(read, VALUES.to_vec());
    }
}
