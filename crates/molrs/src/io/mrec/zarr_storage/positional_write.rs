//! Positional-write filesystem store adapter.
//!
//! A Zarr *store* is the key-value layer under an array: the codec hands it
//! `(key, bytes)` and, for an append, `(key, offset, bytes)`. [`FilesystemStore`]
//! maps keys to files under a root directory. This module wraps it with one
//! change — the offset form is written straight at that offset instead of being
//! emulated by reading the whole file back, patching it in memory and rewriting
//! it. [`PositionalWriteStorage`] carries the measurement that makes the
//! difference matter.
//!
//! Both platforms can do that, under different names: `write_all_at` on Unix
//! and `seek_write` on Windows. [`write_positional`] is the one-line shim over
//! the pair.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use molrs::core::MolRsError;
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

/// Write `buf` at `offset`, moving nothing else and reading nothing back.
///
/// `std` offers this on both platforms under different names and with
/// different contracts: Unix `write_all_at` already loops until the buffer is
/// drained, while Windows `seek_write` is one `WriteFile` call that may write
/// less, so the loop lives here. Writing past the end extends the file on both,
/// and the gap reads back as zeros — the same bytes the read-modify-write path
/// produced when it resized its buffer.
#[cfg(unix)]
fn write_positional(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    file.write_all_at(buf, offset)
}

/// Windows arm of [`write_positional`]: `seek_write` until the buffer is drained.
#[cfg(windows)]
fn write_positional(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    let mut buf = buf;
    let mut offset = offset;
    while !buf.is_empty() {
        match file.seek_write(buf, offset) {
            Ok(0) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::WriteZero,
                    "positional write made no progress",
                ));
            }
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// A [`FilesystemStore`] whose partial writes land where they are aimed.
///
/// **"Positional", not "Partial", because two layers say "partial" and mean
/// different things**: the *codec* layer's partial encoding re-encodes one
/// inner chunk of a shard and hands the result to the *store* layer's
/// [`WritableStorageTraits::set_partial_many`] — and only here does that become
/// a true positional write at the disk (bytes at an offset; nothing read,
/// nothing truncated). The name records which of the two partials this type
/// is; do not "fix" it back to `Partial*`.
///
/// It exists because `zarrs_filesystem` 0.3.12 does not do that: its
/// `set_partial_many` delegates to `zarrs_storage::store_set_partial_many`,
/// which reads the whole value, patches it in memory and `set`s it back
/// (truncate + full rewrite), while `supports_set_partial()` still reports
/// `true` — so the codec layer takes the append fast path and the disk pays
/// O(file) anyway. Measured through the store API: one 16 B tail write cost
/// 247 ms on a 256 MiB shard (see the [`crate::io::mrec::zarr_storage`] module doc). Through
/// this adapter the same call writes 16 B.
///
/// `bytes_written` (test-only) is the only observation that tells
/// the two apart — `get` and the file length are identical either way.
///
/// Unix only: the positional write is
/// [`write_positional`].
///
/// Two more disk-level guarantees live here, because the sequence writer's
/// commit protocol needs them and no store trait spells them:
///
/// - **Metadata is replaced atomically.** A `set` of any `zarr.json` key is
///   written to a sibling temporary file and renamed over the old one, so a
///   crash mid-write leaves the previous metadata rather than a truncated
///   file that makes the whole array unopenable.
/// - **Durability on request.** Every path this store writes is remembered
///   until [`sync_dirty`](Self::sync_dirty) fsyncs it (and its directory).
///   The writer calls that around the commit marker when a flush is durable.
#[derive(Debug)]
pub(in crate::io::mrec::zarr_storage) struct PositionalWriteStorage {
    inner: FilesystemStore,
    bytes_written: AtomicU64,
    /// Paths written since the last `sync_dirty`.
    dirty: Mutex<BTreeSet<PathBuf>>,
}

impl PositionalWriteStorage {
    /// Open a positional-write store rooted at `path`.
    ///
    /// The wrapped [`FilesystemStore`] is built here rather than handed in, so
    /// its root cannot disagree with the paths this adapter writes at, and a
    /// path-taking write door reaches it in one call. Default options are
    /// deliberate: they leave the wrapped store's file-handle cache disabled,
    /// and a cached handle would not see a positional write, because the
    /// invalidation hook is private to that store.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] carrying the wrapped store's own message when
    /// `path` cannot be made a store root — the directory cannot be created, or
    /// a non-directory already stands there.
    pub(in crate::io::mrec::zarr_storage) fn new(
        path: impl AsRef<Path>,
    ) -> Result<Self, MolRsError> {
        Ok(Self {
            inner: FilesystemStore::new(path.as_ref())
                .map_err(|e| MolRsError::zarr(e.to_string()))?,
            bytes_written: AtomicU64::new(0),
            dirty: Mutex::new(BTreeSet::new()),
        })
    }

    /// Remember `path` as touched since the last sync.
    fn mark_dirty(&self, path: PathBuf) {
        if let Ok(mut dirty) = self.dirty.lock() {
            dirty.insert(path);
        }
    }

    /// `fsync` every file written since the last call, then every directory
    /// that holds one of them, so both the bytes and the directory entries
    /// (the renamed metadata files) survive a power loss.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] carrying the OS error when a file or directory
    /// cannot be opened or synced. A path that has since been erased is
    /// skipped.
    pub(in crate::io::mrec::zarr_storage) fn sync_dirty(&self) -> Result<(), MolRsError> {
        let paths: Vec<PathBuf> = match self.dirty.lock() {
            Ok(mut dirty) => std::mem::take(&mut *dirty).into_iter().collect(),
            Err(_) => return Err(MolRsError::zarr("store dirty set is poisoned")),
        };
        let mut directories = BTreeSet::new();
        for path in &paths {
            // Write access: Windows' FlushFileBuffers refuses a read-only handle.
            match OpenOptions::new().write(true).open(path) {
                Ok(file) => file
                    .sync_data()
                    .map_err(|e| MolRsError::zarr(format!("fsync {}: {e}", path.display())))?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(e) => {
                    return Err(MolRsError::zarr(format!(
                        "open for fsync {}: {e}",
                        path.display()
                    )));
                }
            }
            if let Some(parent) = path.parent() {
                directories.insert(parent.to_path_buf());
            }
        }
        for directory in directories {
            if let Ok(dir) = File::open(&directory) {
                dir.sync_all().map_err(|e| {
                    MolRsError::zarr(format!("fsync directory {}: {e}", directory.display()))
                })?;
            }
        }
        Ok(())
    }

    /// Bytes this store has handed to the disk since it was opened.
    ///
    /// Observation only: nothing reads it to decide anything, and there is no
    /// reset. It is how a caller proves a tail write cost the size of the tail
    /// and not the size of the file.
    ///
    /// Test-only: the write-amplification assertions in `sequence.rs` are its
    /// only readers, by design.
    #[cfg(test)]
    pub(in crate::io::mrec::zarr_storage) fn bytes_written(&self) -> u64 {
        self.bytes_written.load(Ordering::Relaxed)
    }
}

impl ReadableStorageTraits for PositionalWriteStorage {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.inner.get(key)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.inner.get_partial(key, byte_range)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.inner.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.inner.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.inner.supports_get_partial()
    }
}

impl ListableStorageTraits for PositionalWriteStorage {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.inner.list()
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        self.inner.list_prefix(prefix)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        self.inner.list_dir(prefix)
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.inner.size()
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        self.inner.size_prefix(prefix)
    }
}

impl WritableStorageTraits for PositionalWriteStorage {
    /// A whole-value write.
    ///
    /// A `zarr.json` key — array or group metadata — is written to a sibling
    /// temporary file and renamed into place, so the metadata a reader finds
    /// is always either the old document or the new one, never a truncated
    /// mix. Every other key is delegated verbatim: the wrapped store truncates
    /// to the new length, and that behaviour is the contract.
    fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
        let len = value.len() as u64;
        let path = self.inner.key_to_fspath(key);
        if key.as_str().ends_with("zarr.json") {
            if let Some(parent) = path.parent()
                && !parent.exists()
            {
                std::fs::create_dir_all(parent)?;
            }
            let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
            std::fs::write(&temporary, &value)?;
            std::fs::rename(&temporary, &path)?;
        } else {
            self.inner.set(key, value)?;
        }
        self.mark_dirty(path);
        self.bytes_written.fetch_add(len, Ordering::Relaxed);
        Ok(())
    }

    /// The reason this type exists: each `(offset, bytes)` is written where it
    /// points, and the existing value is never read.
    ///
    /// [`set_partial`](WritableStorageTraits::set_partial) is provided sugar
    /// over this method in `zarrs_storage` 0.4.5, so overriding it here is what
    /// redirects every partial write off the read-modify-write path.
    fn set_partial_many(
        &self,
        key: &StoreKey,
        offset_values: OffsetBytesIterator,
    ) -> Result<(), StorageError> {
        let path = self.inner.key_to_fspath(key);
        if let Some(parent) = path.parent()
            && !parent.exists()
        {
            std::fs::create_dir_all(parent)?;
        }

        // `truncate(false)` is the whole point: the bytes already on disk are
        // the context of a positional write. A write past the end extends the
        // file and leaves a hole, which reads as zeros — the same bytes the
        // read-modify-write path produced by resizing its buffer with zeros.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;

        let mut written = 0u64;
        for (offset, value) in offset_values {
            write_positional(&file, &value, offset)?;
            written += value.len() as u64;
        }
        self.mark_dirty(path);
        self.bytes_written.fetch_add(written, Ordering::Relaxed);
        Ok(())
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.inner.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.inner.erase_prefix(prefix)
    }

    /// True, and — unlike the wrapped store's `true` — true at the disk.
    fn supports_set_partial(&self) -> bool {
        true
    }
}

/// The disk-level contract of [`PositionalWriteStorage`].
///
/// Every test writes a real file under a `tempfile::tempdir` and then reads it
/// back with `std::fs`, **outside** the store: the defect this type exists for
/// is invisible from the store API, because `zarrs_filesystem` 0.3.12 reports
/// `supports_set_partial() == true` while `set_partial_many` delegates to
/// `zarrs_storage::store_set_partial_many`, which reads the whole value,
/// patches it in memory and `set`s it back (see the `io::mrec::zarr_storage` module doc).
/// Both stores answer `get` identically afterwards; only the bytes that reached
/// the disk differ.
///
/// Constructor pinned by these tests: `PositionalWriteStorage::new(path)` builds
/// its own wrapped `FilesystemStore` over the same root, so a caller cannot
/// hand it a store whose root disagrees with the paths it writes at, and the
/// two path-taking write doors reach it in one call. `Arc<PositionalWriteStorage>`
/// must coerce to `ReadableWritableListableStorage`, which is the shape
/// `MrecWriter` binds an array to.
#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;
    use zarrs::filesystem::FilesystemStore;
    use zarrs::storage::{
        Bytes, ListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorage,
        StoreKey, WritableStorageTraits,
    };

    use super::PositionalWriteStorage;

    /// The one key every test writes: a shard file, which is the only value the
    /// sequence writer ever hands a partial write to.
    const KEY: &str = "traj/c/0";
    /// Bytes of the tail write under test — 16 B, the size of one shard index
    /// entry, and the write the module doc measured at 247 ms on 256 MiB.
    const PATCH_LEN: usize = 16;
    /// The value size the offset cases are written into.
    const VALUE_LEN: usize = 4096;

    fn key() -> StoreKey {
        StoreKey::new(KEY).unwrap()
    }

    fn storage_in(dir: &TempDir) -> Arc<PositionalWriteStorage> {
        Arc::new(PositionalWriteStorage::new(dir.path()).unwrap())
    }

    /// The bytes actually on disk for [`KEY`], read outside the store.
    fn on_disk(dir: &TempDir) -> Vec<u8> {
        std::fs::read(dir.path().join(KEY)).unwrap()
    }

    /// A value whose every byte is a function of its index, so a zero fill, a
    /// shift or a truncation each show up as a mismatch. The modulus keeps
    /// every byte below 199, disjoint from [`patch`]'s range.
    fn patterned(len: usize) -> Vec<u8> {
        (0..len).map(|i| ((i * 7 + 11) % 199) as u8).collect()
    }

    /// The 16 bytes written at an offset: non-constant (so a reordering is
    /// caught) and entirely above [`patterned`]'s range (so landing on top of
    /// the pattern is never a coincidence).
    fn patch() -> Vec<u8> {
        (0..PATCH_LEN as u8).map(|i| 0xF0 | i).collect()
    }

    /// A partial write changes exactly the bytes it was handed: the file keeps
    /// its length and every byte outside the written range is the byte that was
    /// there before.
    #[test]
    fn a_partial_write_lands_at_the_offset_and_touches_nothing_else() {
        const OFFSET: usize = 1000;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let original = patterned(VALUE_LEN);
        store.set(&key(), Bytes::from(original.clone())).unwrap();

        store
            .set_partial(&key(), OFFSET as u64, Bytes::from(patch()))
            .unwrap();

        let after = on_disk(&dir);
        assert_eq!(
            after.len(),
            VALUE_LEN,
            "a positional write must not resize the file"
        );
        assert_ne!(
            &original[OFFSET..OFFSET + PATCH_LEN],
            &patch()[..],
            "the patch must differ from what it replaced, or this proves nothing"
        );
        assert_eq!(
            &after[OFFSET..OFFSET + PATCH_LEN],
            &patch()[..],
            "the patch must land at the offset"
        );
        let mut expected = original;
        expected[OFFSET..OFFSET + PATCH_LEN].copy_from_slice(&patch());
        assert_eq!(
            after, expected,
            "every byte outside the written range must be untouched"
        );
    }

    /// A write past the end extends the file, and the gap it leaves reads as
    /// zeros.
    ///
    /// Zero fill is the chosen semantic because it is what `write_at` past EOF
    /// gives (the file is sparse, and a hole reads as zeros) and what the
    /// read-modify-write path this adapter replaces already produced
    /// (`store_set_partial_many` resizes its buffer with zeros). A reader
    /// therefore sees the same bytes either way.
    #[test]
    fn a_write_past_eof_extends_the_file() {
        const GAP: usize = 100;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let original = patterned(VALUE_LEN);
        store.set(&key(), Bytes::from(original.clone())).unwrap();

        store
            .set_partial(&key(), (VALUE_LEN + GAP) as u64, Bytes::from(patch()))
            .unwrap();

        let after = on_disk(&dir);
        assert_eq!(
            after.len(),
            VALUE_LEN + GAP + PATCH_LEN,
            "the file must extend to cover the written range"
        );
        assert_eq!(&after[..VALUE_LEN], &original[..], "the old value stands");
        assert_eq!(
            &after[VALUE_LEN..VALUE_LEN + GAP],
            &[0u8; GAP][..],
            "the gap between the old end and the write reads as zeros"
        );
        assert_eq!(
            &after[VALUE_LEN + GAP..],
            &patch()[..],
            "the patch must land at the offset it was given"
        );
    }

    /// A partial write is never a truncation: writing eight bytes at zero on a
    /// 4096-byte value leaves 4096 bytes.
    #[test]
    fn a_partial_write_never_shortens() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let original = patterned(VALUE_LEN);
        store.set(&key(), Bytes::from(original.clone())).unwrap();

        let head = patch()[..8].to_vec();
        store
            .set_partial(&key(), 0, Bytes::from(head.clone()))
            .unwrap();

        let after = on_disk(&dir);
        assert_eq!(
            after.len(),
            VALUE_LEN,
            "opening for a positional write must not truncate"
        );
        assert_eq!(&after[..8], &head[..], "the head was rewritten");
        assert_eq!(&after[8..], &original[8..], "the tail survives untouched");
    }

    /// `set` is still a whole-value write: it replaces the value and the file
    /// ends at the new length, however much longer the old value was.
    #[test]
    fn set_truncates_to_the_new_length() {
        const SHORT_LEN: usize = 100;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        store
            .set(&key(), Bytes::from(patterned(VALUE_LEN)))
            .unwrap();
        store
            .set_partial(&key(), 1000, Bytes::from(patch()))
            .unwrap();

        let short = patterned(SHORT_LEN);
        store.set(&key(), Bytes::from(short.clone())).unwrap();

        let after = on_disk(&dir);
        assert_eq!(
            after.len(),
            SHORT_LEN,
            "set replaces the value; the file must not keep the old tail"
        );
        assert_eq!(after, short, "and the new value is exactly what was set");
    }

    /// `bytes_written()` counts the bytes handed to the store, not the size of
    /// the file they landed in.
    #[test]
    fn bytes_written_counts_handed_bytes_not_file_size() {
        const LARGE_LEN: usize = 1 << 20;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        store
            .set(&key(), Bytes::from(patterned(LARGE_LEN)))
            .unwrap();

        let before = store.bytes_written();
        store
            .set_partial(&key(), 1000, Bytes::from(patch()))
            .unwrap();

        assert_eq!(
            store.bytes_written() - before,
            PATCH_LEN as u64,
            "a {PATCH_LEN} B partial write into a {LARGE_LEN} B value costs {PATCH_LEN} B"
        );
    }

    /// Reads and listings are the wrapped store's answers verbatim — the
    /// adapter only changes how writes reach the disk.
    #[test]
    fn reads_and_lists_delegate_to_the_wrapped_store() {
        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        let sibling = StoreKey::new("traj/zarr.json").unwrap();
        let absent = StoreKey::new("traj/c/1").unwrap();
        store
            .set(&key(), Bytes::from(patterned(VALUE_LEN)))
            .unwrap();
        store.set(&sibling, Bytes::from(b"{}".to_vec())).unwrap();
        store
            .set_partial(&key(), 1000, Bytes::from(patch()))
            .unwrap();

        let plain = FilesystemStore::new(dir.path()).unwrap();
        assert_eq!(
            store.get(&key()).unwrap(),
            plain.get(&key()).unwrap(),
            "a patched value reads the same through both stores"
        );
        assert_eq!(store.get(&sibling).unwrap(), plain.get(&sibling).unwrap());
        assert_eq!(
            store.get(&absent).unwrap(),
            plain.get(&absent).unwrap(),
            "an absent key is absent through both"
        );

        let mut listed = store.list().unwrap();
        let mut plain_listed = plain.list().unwrap();
        listed.sort();
        plain_listed.sort();
        assert_eq!(listed, plain_listed, "the listing is the wrapped store's");
    }

    /// The claim the codec layer acts on: this store supports partial writes,
    /// and unlike the stock one it means it at the disk.
    #[test]
    fn supports_set_partial_is_true_at_construction() {
        let dir = TempDir::new().unwrap();
        assert!(storage_in(&dir).supports_set_partial());
    }

    /// The disk-level guard: a 16 B write at the tail of an 8 MiB value costs
    /// 16 B, not 8 MiB.
    ///
    /// This is the whole reason the adapter exists. Handed to the stock
    /// `FilesystemStore` the same call is a whole-value read-modify-write
    /// (`zarrs_filesystem` 0.3.12; measured at 247 ms for one 16 B write on
    /// 256 MiB, see the `io::mrec::zarr_storage` module doc), and no store-API observation
    /// distinguishes the two — only this accounting does. The stock store's
    /// half of the contrast is ac-011's store-half assertion, which runs the
    /// same measurement through a real flush; there is no portable way to
    /// observe the stock rewrite from here (file length and `get` are
    /// identical either way, and allocated-block counts are Unix-only).
    #[test]
    fn a_tail_write_on_a_large_file_is_not_a_rewrite() {
        const HUGE_LEN: usize = 8 << 20;

        let dir = TempDir::new().unwrap();
        let store = storage_in(&dir);
        store.set(&key(), Bytes::from(patterned(HUGE_LEN))).unwrap();

        let before = store.bytes_written();
        store
            .set_partial(&key(), (HUGE_LEN - PATCH_LEN) as u64, Bytes::from(patch()))
            .unwrap();

        assert_eq!(
            store.bytes_written() - before,
            PATCH_LEN as u64,
            "a tail write must cost {PATCH_LEN} B at the disk, not the {HUGE_LEN} B file"
        );
    }

    /// The shape the sequence writer binds an array to: an `Arc` of this store
    /// is a `ReadableWritableListableStorage`, and it works through the
    /// erased trait object.
    #[test]
    fn an_arc_of_the_store_is_readable_writable_listable_storage() {
        let dir = TempDir::new().unwrap();
        let storage: ReadableWritableListableStorage = storage_in(&dir);

        let value = patterned(64);
        storage.set(&key(), Bytes::from(value.clone())).unwrap();

        assert_eq!(storage.get(&key()).unwrap().as_deref(), Some(&value[..]));
        assert_eq!(storage.list().unwrap(), vec![key()]);
    }
}
