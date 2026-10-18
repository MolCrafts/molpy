//! Pack a closed directory store into a single stored-entry `.mrec.zip`, and
//! open one.

use std::ffi::OsStr;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zarrs::filesystem::FilesystemStore;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableStorageTraits,
    ReadableWritableListableStorage, StoreKey, WritableStorageTraits,
};

use molrs::core::MolRsError;

use super::record_io::zerr;

/// The suffix a packed store carries.
const ZIP_SUFFIX: &str = ".zip";
/// The suffix a directory store is expected to carry already.
const MREC_SUFFIX: &str = "mrec";

/// Pack the closed directory store at `storage_path` into a sibling
/// `.mrec.zip` and remove the directory, returning the path of the archive.
///
/// The archive's name is derived from the directory's: `traj.mrec` becomes
/// `traj.mrec.zip`, and a directory without the `.mrec` suffix gains the whole
/// `.mrec.zip`. Entries
/// are walked in sorted order and written STORED (method 0) — the chunks
/// arrived gzipped, so a second compression pass would buy nothing and cost a
/// full re-encode.
///
/// The parameter is a path, never a live
/// [`MrecWriter`](crate::io::mrec::MrecWriter): packing a
/// store still being appended to is the live single-file write this backend
/// rejects. `writer.close()` followed by `pack_mrec_zip(path)` is the caller's
/// composition.
///
/// # Errors
///
/// Returns a [`MolRsError::Zarr`] naming `storage_path`
/// when no directory is there — which is also what a second `pack` of the same
/// store meets, since the first one removed the directory. Nothing is created
/// in that case, so a refused pack leaves no half-written archive behind.
///
/// Every other failure is also a [`MolRsError::Zarr`], and it names the path it
/// happened on, since neither `std::fs` nor the zip writer puts the path in its
/// own message. `storage_path` naming no directory component at all fails before
/// anything is created. After that the archive is open, and a failure while
/// walking the directory, reading an entry, encoding an entry's name as a store
/// key (its path must be valid UTF-8) or writing into the archive **leaves a
/// partial `.mrec.zip` beside the still-intact directory store**: the directory
/// is removed only once the archive is closed, so the run's data always exists
/// in at least one of the two places, but the caller deletes the partial
/// archive before retrying. The removal itself can fail last of all, and then
/// both the finished archive and the directory survive, the directory being the
/// redundant copy.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use molrs::core::Trajectory;
/// use molrs::io::mrec::{MolRec, MrecReader, pack_mrec_zip};
/// use molrs::io::write_mrec;
///
/// let dir = tempfile::tempdir().unwrap();
/// let store = dir.path().join("traj.mrec");
/// let mut record = MolRec::new();
/// record.trajectory = Some(Trajectory::from_frames(vec![molrs::core::Frame::new()]));
/// write_mrec(&store, &record)?;
///
/// let zip = pack_mrec_zip(&store)?;
/// assert!(zip.ends_with("traj.mrec.zip"));
/// assert!(!store.exists(), "pack removes the directory it consumed");
///
/// let seq = MrecReader::open(&zip)?;
/// assert_eq!(seq.to_trajectory()?.len(), 1);
/// # Ok(())
/// # }
/// ```
pub fn pack_mrec_zip(storage_path: impl AsRef<Path>) -> Result<PathBuf, MolRsError> {
    let storage_path = storage_path.as_ref();
    if !storage_path.is_dir() {
        return Err(at(storage_path, "pack", "no directory store is there"));
    }
    let zip_path = packed_path(storage_path)?;

    let file = std::fs::File::create(&zip_path).map_err(|e| at(storage_path, "pack", e))?;
    let mut writer = zip::ZipWriter::new(file);
    for entry in sorted_files(storage_path).map_err(|e| at(storage_path, "pack", e))? {
        let name = entry
            .strip_prefix(storage_path)
            .map_err(zerr)?
            .to_str()
            .ok_or_else(|| {
                at(
                    storage_path,
                    "pack",
                    format!("{} is not a valid store key", entry.display()),
                )
            })?
            .to_string();
        let bytes = std::fs::read(&entry).map_err(|e| at(&entry, "pack", e))?;
        // `large_file` writes the zip64 extra field, which is required — and
        // only legal to omit — below 4 GiB. Sized per entry rather than set
        // once, so an ordinary store keeps the plainest possible archive.
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(bytes.len() as u64 >= u64::from(u32::MAX));
        // ZIP keys always use '/', including on Windows directory stores.
        writer
            .start_file_from_path(name, options)
            .map_err(|e| at(&entry, "pack", e))?;
        writer
            .write_all(&bytes)
            .map_err(|e| at(&entry, "pack", e))?;
    }
    writer.finish().map_err(|e| at(storage_path, "pack", e))?;

    // The directory goes last: until the archive is closed, the store the
    // caller handed over is still the only copy of its data.
    std::fs::remove_dir_all(storage_path).map_err(|e| at(storage_path, "pack", e))?;
    Ok(zip_path)
}

/// Open a packed `.mrec.zip` read-only, through `zarrs_zip`'s
/// `ZipStorageAdapter`.
///
/// The doors are [`MrecFile::open`](crate::io::mrec::MrecFile::open) and
/// [`MrecReader::open`](crate::io::mrec::MrecReader::open), which open a
/// packed record in place through this.
///
/// The returned store is readable and listable and nothing more, which is exactly
/// what a read function such as [`crate::io::mrec::MrecReader::open`] asks
/// for. Stored entries are read through the adapter's byte-range fast path, so
/// a frame costs the bytes of that frame rather than the bytes of the archive.
///
/// # Errors
///
/// Returns a [`MolRsError::Zarr`] if `path` has no file name, if its directory cannot be opened, or if the file
/// is not a readable zip archive.
pub(crate) fn open_mrec_zip(path: impl AsRef<Path>) -> Result<ReadableListableStorage, MolRsError> {
    let path = path.as_ref();
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| at(path, "open", "it names no file"))?;
    // The adapter reads the archive as one key of the directory holding it.
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let directory = Arc::new(FilesystemStore::new(parent).map_err(|e| at(path, "open", e))?);
    let key = StoreKey::new(name).map_err(|e| at(path, "open", e))?;
    Ok(Arc::new(
        zarrs_zip::ZipStorageAdapter::new(directory, key).map_err(|e| at(path, "open", e))?,
    ))
}

/// The bytes of a packed `.mrec.zip` unpacked into an in-memory store.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] when the bytes are not a zip archive of stored
/// (uncompressed) entries whose names are store keys.
pub(in crate::io::mrec::zarr_storage) fn unpack_zip_bytes(
    bytes: &[u8],
) -> Result<ReadableWritableListableStorage, MolRsError> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| MolRsError::zarr(format!("packed record bytes: {e}")))?;
    let store = Arc::new(MemoryStore::new());
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|e| {
            MolRsError::zarr(format!(
                "packed record entry {index}: {e} (a packed record stores its entries uncompressed)"
            ))
        })?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().trim_start_matches('/').to_string();
        let mut content = Vec::with_capacity(entry.size() as usize);
        std::io::Read::read_to_end(&mut entry, &mut content)
            .map_err(|e| MolRsError::zarr(format!("packed record entry {name}: {e}")))?;
        let key = StoreKey::new(&name).map_err(zerr)?;
        store.set(&key, content.into()).map_err(zerr)?;
    }
    Ok(store)
}

/// Every key of `store`, in sorted order, packed STORED into a zip archive
/// written to `out` — the archive [`pack_mrec_zip`] makes of a directory.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] when the store cannot be listed or read, or the
/// archive cannot be written.
pub(in crate::io::mrec::zarr_storage) fn pack_storage_zip(
    store: &ReadableWritableListableStorage,
    out: &mut Vec<u8>,
) -> Result<(), MolRsError> {
    let mut keys = store.list().map_err(zerr)?;
    keys.sort();
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(out));
    for key in keys {
        let Some(bytes) = store.get(&key).map_err(zerr)? else {
            continue;
        };
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(bytes.len() as u64 >= u64::from(u32::MAX));
        writer.start_file(key.as_str(), options).map_err(zerr)?;
        writer.write_all(&bytes).map_err(zerr)?;
    }
    writer.finish().map_err(zerr)?;
    Ok(())
}

/// An error naming the path it happened on.
///
/// Neither `std::fs` nor `zarrs_zip` puts the path in its message, and a door
/// whose whole argument is a path must not report a bare "permission denied".
fn at(path: &Path, verb: &str, cause: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("cannot {verb} {}: {cause}", path.display()))
}

/// The archive name derived from a directory store's: `.mrec` gains only
/// `.zip`, anything else gains the whole `.mrec.zip`.
fn packed_path(storage_path: &Path) -> Result<PathBuf, MolRsError> {
    let name = storage_path
        .file_name()
        .ok_or_else(|| at(storage_path, "pack", "it names no directory"))?;
    let mut packed = name.to_os_string();
    if Path::new(name).extension() == Some(OsStr::new(MREC_SUFFIX)) {
        packed.push(ZIP_SUFFIX);
    } else {
        packed.push(format!(".{MREC_SUFFIX}{ZIP_SUFFIX}"));
    }
    Ok(storage_path.with_file_name(packed))
}

/// Every file under `root`, sorted by path.
///
/// The walk order is a stack's and therefore arbitrary; the sort at the end is
/// what makes it deterministic, so two packs of the same store lay their
/// entries down in the same order.
fn sorted_files(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        for entry in std::fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.is_dir() {
                directories.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use std::path::{Path, PathBuf};

    use molrs::core::Frame;
    use molrs::core::Trajectory;
    use molrs::core::{Block, Column};
    use ndarray::ArrayD;
    use tempfile::tempdir;

    use super::pack_mrec_zip;
    use crate::io::mrec::MolRec;
    use crate::io::mrec::zarr_storage::{MrecReader, write_mrec};

    /// The one block every fixture frame carries.
    const ATOMS: &str = "atoms";
    /// Its one `f64` column.
    const X: &str = "x";

    fn float_column(values: &[f64]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn atoms_frame(values: &[f64]) -> Frame {
        let mut block = Block::new();
        block.insert(X, float_column(values), None).unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, block);
        frame
    }

    /// Three frames of 3, 5 and 4 rows: ragged, so the CSR row pointer has to
    /// survive the pack, with every value distinct across frames so a row range
    /// taken from the wrong frame cannot match by accident. Step numbers and
    /// times of its own, so both index arrays are exercised too.
    fn trajectory() -> Trajectory {
        Trajectory {
            frames: vec![
                atoms_frame(&[1.0, -2.5, 1.0e-300]),
                atoms_frame(&[0.1, 0.2, 0.3, 4.5, -6.25]),
                atoms_frame(&[f64::MIN_POSITIVE, 2.0e300, 3.5, -0.125]),
            ],
            step: Some(vec![0, 5, 10]),
            time: Some(vec![0.0, 0.5, 1.25]),
        }
    }

    /// Column `x` of the `atoms` block as raw bits.
    ///
    /// Bits, not values: the trajectory dtype contract admits no tolerance, and
    /// comparing `f64` by value would let a sign-of-zero flip through.
    fn atoms_x_bits(frame: &Frame) -> Vec<u64> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .get(X)
            .expect("the atoms block carries column x")
            .get::<f64>()
            .expect("column x arrived as f64")
            .iter()
            .map(|value| value.to_bits())
            .collect()
    }

    /// Write [`trajectory`] into a fresh directory store `name` under `parent`.
    fn make_storage(parent: &Path, name: &str) -> PathBuf {
        let path = parent.join(name);
        let mut record = MolRec::new();
        record.trajectory = Some(trajectory());
        write_mrec(&path, &record).expect("the fixture store writes");
        path
    }

    /// The pinned name: `<dir>.mrec` packs to `<dir>.mrec.zip`, one file, and
    /// the directory store is gone.
    #[test]
    fn pack_produces_one_zip_and_removes_the_directory() {
        let dir = tempdir().unwrap();
        let storage_path = make_storage(dir.path(), "traj.mrec");

        let zip_path = pack_mrec_zip(&storage_path).expect("packing a closed store succeeds");

        assert_eq!(
            zip_path,
            dir.path().join("traj.mrec.zip"),
            "a path already ending in .mrec gains only .zip"
        );
        assert!(zip_path.is_file(), "the packed store is a single file");
        assert!(
            !storage_path.exists(),
            "pack removes the directory store it consumed"
        );
        let left: Vec<PathBuf> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert_eq!(
            left,
            vec![zip_path],
            "the zip is the only thing left at rest"
        );
    }

    /// A directory with no `.mrec` suffix gains the whole suffix.
    #[test]
    fn packing_a_directory_without_the_mrec_suffix_appends_the_whole_suffix() {
        let dir = tempdir().unwrap();
        let storage_path = make_storage(dir.path(), "traj");

        let zip_path = pack_mrec_zip(&storage_path).expect("packing a closed store succeeds");

        assert_eq!(zip_path, dir.path().join("traj.mrec.zip"));
    }

    /// Every entry is STORED (method 0) — the chunks are already gzipped, so
    /// packing must not spend a second compression pass on them.
    #[test]
    fn every_zip_entry_is_stored_method_zero() {
        let dir = tempdir().unwrap();
        let storage_path = make_storage(dir.path(), "traj.mrec");
        let zip_path = pack_mrec_zip(&storage_path).expect("packing a closed store succeeds");

        let mut archive =
            zip::ZipArchive::new(std::fs::File::open(&zip_path).unwrap()).expect("a readable zip");

        assert!(!archive.is_empty(), "a packed store carries entries");
        for index in 0..archive.len() {
            let entry = archive.by_index(index).unwrap();
            assert_eq!(
                entry.compression(),
                zip::CompressionMethod::Stored,
                "{} is not a stored entry",
                entry.name()
            );
        }
    }

    /// Every frame, step and time read out of the packed zip is bit identical
    /// to what the directory store gave before packing.
    #[test]
    fn frames_read_back_from_the_zip_bit_exact() {
        let dir = tempdir().unwrap();
        let storage_path = make_storage(dir.path(), "traj.mrec");
        // The reference is taken from the directory form, before it is gone.
        let before = MrecReader::open(&storage_path)
            .expect("the directory store opens")
            .to_trajectory()
            .expect("the directory store reads");

        let zip_path = pack_mrec_zip(&storage_path).expect("packing a closed store succeeds");
        let sequence = MrecReader::open(&zip_path).expect("the packed sequence opens");
        let after = sequence.to_trajectory().expect("the packed sequence reads");

        assert_eq!(after.step, before.step, "step numbers survive the pack");
        assert_eq!(after.time, before.time, "times survive the pack");
        assert_eq!(
            after.frames.len(),
            before.frames.len(),
            "frame count survives the pack"
        );
        for (index, (packed, directory)) in after.frames.iter().zip(&before.frames).enumerate() {
            assert_eq!(
                atoms_x_bits(packed),
                atoms_x_bits(directory),
                "frame {index} column x came back with different bits"
            );
        }
    }

    /// Packing a path that is not there is an error that names the path.
    #[test]
    fn packing_a_missing_directory_errs_naming_it() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("absent.mrec");

        let error = pack_mrec_zip(&missing).expect_err("a missing store cannot be packed");

        let message = error.to_string();
        assert!(
            message.contains(&missing.display().to_string()),
            "the error must name the path it could not pack: {message}"
        );
        assert!(
            !dir.path().join("absent.mrec.zip").exists(),
            "a failed pack leaves no half-written zip behind"
        );
    }

    /// Packing twice is not idempotent and does not pretend to be: the second
    /// call meets the directory `pack` itself removed and errs naming it, the
    /// same way any other missing path does.
    #[test]
    fn packing_twice_errs_naming_the_removed_directory() {
        let dir = tempdir().unwrap();
        let storage_path = make_storage(dir.path(), "traj.mrec");
        let zip_path = pack_mrec_zip(&storage_path).expect("the first pack succeeds");

        let error =
            pack_mrec_zip(&storage_path).expect_err("the directory is gone after the first pack");

        let message = error.to_string();
        assert!(
            message.contains(&storage_path.display().to_string()),
            "the error must name the path it could not pack: {message}"
        );
        assert!(
            zip_path.is_file(),
            "the failed second pack must not disturb the first pack's zip"
        );
    }
}
