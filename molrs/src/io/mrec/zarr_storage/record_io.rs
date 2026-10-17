//! Zarr V3 binding for [`MolRec`] — the reference L4 binding of the MolRec
//! contract (<https://github.com/MolCrafts/molrec>) for **array sections**.
//!
//! # The `trajectory/` section has one owner
//!
//! This module owns the **record-level** sections — `meta`, `frame`, `system`,
//! `observables`, the JSON groups — and the two path-taking doors. It does
//! **not** own the `trajectory/` layout: that layout has exactly one encoder
//! and one decoder, both in `sequence`, and the record
//! writer and reader drive them rather than restating them. Nothing here
//! encodes or decodes a row pointer, a step index, or a frame group.

use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "zarr")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
#[cfg(feature = "zarr")]
use zarrs::storage::WritableStorageTraits;
use zarrs::storage::{
    ReadableListableStorage, ReadableStorageTraits, ReadableWritableListableStorage,
};

use crate::io::mrec::MrecReader;
#[cfg(feature = "zarr")]
use crate::io::mrec::zarr_storage::forcefield_io::write_forcefield_group;
use crate::io::mrec::zarr_storage::forcefield_io::{FORCEFIELD_GROUP, read_forcefield_group};
use crate::io::mrec::zarr_storage::frame_io::{
    check_declared_references, join_path, read_column, read_frame_group,
};
#[cfg(feature = "zarr")]
use crate::io::mrec::zarr_storage::frame_io::{node_prefix, write_column, write_frame_group};
#[cfg(feature = "filesystem")]
use crate::io::mrec::zarr_storage::pack::{open_mrec_zip, pack_storage_zip, unpack_zip_bytes};
#[cfg(feature = "filesystem")]
use crate::io::mrec::zarr_storage::positional_write::PositionalWriteStorage;
#[cfg(feature = "zarr")]
use crate::io::mrec::{MrecWriter, SequenceSchema};
#[cfg(feature = "filesystem")]
use crate::io::{Sink, Source};
use molrs::core::Column;
use molrs::core::MolRsError;
#[cfg(feature = "filesystem")]
use zarrs::storage::store::MemoryStore;
// Not `filesystem`-gated: the store-taking section door below names it in
// every configuration, wasm included.
use molrs::core::Frame;
use molrs::core::{ObservableKind, ObservableRecord, ObservableValues};
use molrs::io::mrec::MolRec;

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a [`crate::io::mrec::MolRec`]: to a path as a `*.mrec` directory
/// store, or into a buffer as the bytes of a packed `*.mrec.zip` (what
/// [`pack_mrec_zip`](crate::io::mrec::pack_mrec_zip) makes of the directory).
///
/// The conventional suffix is `.mrec` (for example `water.mrec/`); other
/// names are accepted. A second write to the same path replaces the previous record
/// entirely — leftover sections from a wider record do not survive.
///
/// `meta` is written as the producer supplied it. A trajectory section is encoded by
/// [`crate::io::mrec::MrecWriter`]. [`MrecFile`] reads the sections back.
///
/// # Layout
///
/// One record is one openable root:
///
/// ```text
/// <root>/
/// ├── meta/          the producer's keys
/// ├── system/        frame-shaped group (topology / types)
/// ├── frame/         frame-shaped group (snapshot)
/// ├── trajectory/    the frame sequence — see [`MrecWriter`](crate::io::mrec::MrecWriter)
/// ├── forcefield/    document attrs + one block group per style table
/// ├── observables/   meta/<name> (semantics) + <name> (data)
/// ├── method/        JSON attributes
/// ├── status/        JSON attributes
/// └── metrics/       dense series arrays + catalog attrs; optional JSONL WAL
/// ```
///
/// # Metrics: dense Zarr SoT, JSONL WAL for live append
///
/// Closed training / monitor curves densify to **Zarr series arrays** under
/// `metrics/` (molrec `docs/spec/metrics.md`). Live append uses
/// `metrics/metrics.jsonl` — do **not** use per-step Zarr chunk append.
/// Higher layers (molexp / molnex) own the WAL → densify path.
///
/// When `write_record_*` materialises a `metrics` map into a Zarr group, that
/// is a **closed catalog / summary**. Readers that need the full curve MUST
/// open dense series arrays when present, else fall back to the JSONL WAL.
///
/// Root sections the reader does not interpret are ignored: they are never
/// reinterpreted as frame groups, and they never fail a read. The section
/// reads of [`MrecFile`] decode only the section they name, so a section they
/// were not asked for cannot break them either.
///
/// # Errors
///
/// The frame-IO error: the OS error of a path that cannot be written, and
/// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying the
/// [`MolRsError`] otherwise — a [`MolRsError::Zarr`] when a section fails to
/// encode, a [`MolRsError::Validation`] when [`crate::io::mrec::MolRec::validate`]
/// rejects the record (no state section, or a `step`/`time` length that does
/// not match the frame count).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use molrs::io::mrec::{MolRec, MrecFile};
/// use molrs::io::write_mrec;
///
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("water.mrec");
///
/// let mut record = MolRec::new();
/// record.frame = Some(molrs::core::Frame::new());
/// write_mrec(&path, &record)?;
/// assert!(MrecFile::open(&path)?.frame()?.is_some());
///
/// let mut packed = Vec::new();
/// write_mrec(&mut packed, &record)?;
/// assert!(MrecFile::open(&packed)?.record()?.frame.is_some());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "filesystem")]
pub fn write_mrec<'a>(sink: impl Into<Sink<'a>>, record: &MolRec) -> std::io::Result<()> {
    let written = match sink.into() {
        Sink::Path(path) => PositionalWriteStorage::new(path)
            .and_then(|store| write_mrec_storage(Arc::new(store), record)),
        Sink::Buffer(out) => {
            let store: ReadableWritableListableStorage = Arc::new(MemoryStore::new());
            write_mrec_storage(store.clone(), record).and_then(|()| {
                out.clear();
                pack_storage_zip(&store, out)
            })
        }
    };
    written.map_err(io_error)
}

/// The frame-IO error a record door reports: an OS error stays itself, and
/// every store, schema or validation refusal is
/// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying the
/// [`MolRsError`] — its message is the error's, and the typed error is
/// [`get_ref`](std::io::Error::get_ref)`()` downcast to `MolRsError`.
pub(crate) fn io_error(e: MolRsError) -> std::io::Error {
    match e {
        MolRsError::Io(e) => e,
        other => std::io::Error::new(std::io::ErrorKind::InvalidData, other),
    }
}

/// A MolRec record opened for reading, one section at a time.
///
/// Sections are independent: a record may carry a `frame`, a `system`, a
/// `trajectory`, a `forcefield`, or several at once, and a caller that wants
/// one of them does not pay for the rest — each read decodes only the section
/// it names, and asking which sections exist ([`sections`](Self::sections))
/// decodes none. [`record`](Self::record) reads them all. The `trajectory`
/// section is [`MrecReader`]'s, which reads it one frame at a time.
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use molrs::io::mrec::{MolRec, MrecFile};
/// use molrs::io::write_mrec;
///
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("water.mrec");
/// let mut record = MolRec::new();
/// record.frame = Some(molrs::core::Frame::new());
/// write_mrec(&path, &record)?;
///
/// let file = MrecFile::open(&path)?;
/// assert!(file.sections()?.iter().any(|s| s == "frame"));
/// assert!(file.frame()?.is_some());
/// assert!(file.system()?.is_none());
/// # Ok(())
/// # }
/// ```
pub struct MrecFile {
    store: ReadableListableStorage,
}

impl MrecFile {
    /// Open a record: a `*.mrec` directory store, a packed `*.mrec.zip`
    /// (read in place, a section costing the bytes of that section), or the
    /// bytes of a packed one.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when the path holds neither a directory store
    /// nor a packed record, or the bytes are not a packed record.
    #[cfg(feature = "filesystem")]
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<Self> {
        Ok(Self {
            store: open_record(&src.into()).map_err(io_error)?,
        })
    }

    /// A record in a store the caller already holds (in-memory, a host's).
    pub fn from_storage(store: ReadableListableStorage) -> Self {
        Self { store }
    }

    /// The record's top-level section names (`meta`, `frame`, `system`, …),
    /// sorted, without decoding any of them.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when the store root cannot be listed.
    pub fn sections(&self) -> std::io::Result<Vec<String>> {
        section_names_of(&self.store).map_err(io_error)
    }

    /// The record's `meta` document; empty when the store has none.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when the `meta` group cannot be decoded.
    pub fn meta(&self) -> std::io::Result<JsonMap<String, JsonValue>> {
        read_meta(&self.store).map_err(io_error)
    }

    /// The `frame` section — the snapshot — or `None` when the record has
    /// none.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when the section fails to decode, or a
    /// reference into another section is out of its range.
    pub fn frame(&self) -> std::io::Result<Option<Frame>> {
        read_frame_section(&self.store, "frame").map_err(io_error)
    }

    /// The `system` section — the topology — or `None` when the record has
    /// none.
    ///
    /// # Errors
    ///
    /// As [`frame`](Self::frame).
    pub fn system(&self) -> std::io::Result<Option<Frame>> {
        read_frame_section(&self.store, "system").map_err(io_error)
    }

    /// The `forcefield` section, validated
    /// ([`ForceFieldSection::validate`](crate::io::mrec::ForceFieldSection::validate)),
    /// or `None` when the record has none. A table no style names, and a
    /// document key this build does not know, are kept.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when a table fails to decode, or what the
    /// validation refuses.
    pub fn forcefield(&self) -> std::io::Result<Option<crate::io::mrec::ForceFieldSection>> {
        if !self.sections()?.iter().any(|name| name == FORCEFIELD_GROUP) {
            return Ok(None);
        }
        read_forcefield_group(&self.store, &join_path("/", FORCEFIELD_GROUP))
            .map(Some)
            .map_err(io_error)
    }

    /// Every section of the record. Root sections this build does not
    /// interpret are ignored, never misread, and `meta` keys it does not
    /// recognise are kept.
    ///
    /// # Errors
    ///
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) carrying a
    /// [`MolRsError::Zarr`] when a section fails to decode.
    pub fn record(&self) -> std::io::Result<MolRec> {
        read_record(&self.store).map_err(io_error)
    }
}

/// The store a [`Source`] names: a directory store, a packed record read in
/// place, or packed bytes unpacked into memory.
#[cfg(feature = "filesystem")]
fn open_record(src: &Source<'_>) -> Result<ReadableListableStorage, MolRsError> {
    match src.path() {
        Some(path) if path.is_dir() => Ok(Arc::new(FilesystemStore::new(path).map_err(zerr)?)),
        Some(path) if path.is_file() => open_mrec_zip(path),
        Some(path) => Err(MolRsError::zarr(format!(
            "cannot open {}: neither a record directory nor a packed record is there",
            path.display()
        ))),
        None => {
            let bytes = src.bytes().map_err(MolRsError::Io)?;
            Ok(unpack_zip_bytes(&bytes)?.readable_listable())
        }
    }
}

/// Write a record into an open store, rooted at `/`.
#[cfg(feature = "zarr")]
pub fn write_mrec_storage(
    store: ReadableWritableListableStorage,
    record: &MolRec,
) -> Result<(), MolRsError> {
    record.validate()?;
    check_absolute_references(record)?;
    let prefix = "/";

    // Erase before writing: this record is the whole content of the store root,
    // so a rewrite must not inherit the previous record's sections, blocks,
    // columns or frames. Every writer clears its own target node this way.
    store.erase_prefix(&node_prefix(prefix)?)?;

    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    write_json_group(&store, &join_path(prefix, "meta"), &record.meta)?;

    if let Some(system) = &record.system {
        write_frame_group(&store, &join_path(prefix, "system"), system)?;
    }
    if let Some(frame) = &record.frame {
        write_frame_group(&store, &join_path(prefix, "frame"), frame)?;
    }
    if let Some(trajectory) = &record.trajectory {
        // The `trajectory/` layout has exactly one encoder and it is not here:
        // this door mints the schema from the frames themselves and drives
        // `MrecWriter`. The root erase above has already emptied the
        // node, which is what `create` insists on before it will mint.
        let mut writer = MrecWriter::create_in_storage(
            store.clone(),
            SequenceSchema::from_frames(&trajectory.frames)?,
        )?;
        for (index, frame) in trajectory.frames.iter().enumerate() {
            // `record.validate()` above pinned `step` and `time` to
            // `frames.len()`, so both indexings are in range.
            let time = trajectory.time.as_ref().map(|times| times[index]);
            match (&trajectory.step, time) {
                (Some(steps), _) => writer.append_at(frame, steps[index], time)?,
                // No step numbers of its own and no time to carry: the auto
                // door owns the numbering, so nothing restates it.
                (None, None) => writer.append(frame)?,
                // A time but no step number. The auto door takes no time, so
                // its numbering (0, 1, 2, …) has to be spelled out — dropping
                // the time instead would be silent data loss.
                (None, Some(_)) => writer.append_at(frame, index as i64, time)?,
            }
        }
        writer.close()?;
    }
    if let Some(forcefield) = &record.forcefield {
        write_forcefield_group(&store, &join_path(prefix, FORCEFIELD_GROUP), forcefield)?;
    }
    if !record.observables.is_empty() {
        write_observables(&store, &join_path(prefix, "observables"), record)?;
    }
    for (name, section) in [("method", &record.method), ("status", &record.status)] {
        if !section.is_empty() {
            write_json_group(&store, &join_path(prefix, name), section)?;
        }
    }
    if !record.metrics.is_empty() || !record.metrics_series.is_empty() {
        write_metrics(&store, &join_path(prefix, "metrics"), record)?;
    }

    Ok(())
}

#[cfg(feature = "zarr")]
fn write_json_group(
    store: &ReadableWritableListableStorage,
    path: &str,
    attrs: &JsonMap<String, JsonValue>,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .attributes(attrs.clone())
        .build(store.clone(), path)?
        .store_metadata()?;
    Ok(())
}

/// Percent-encode a metrics series name into a legal array name
/// (`train/loss` → `train%2Floss`): every byte outside `[A-Za-z0-9._-]`
/// becomes `%XX` with uppercase hex. A name Zarr forbids as a node — `.`,
/// `..`, or one starting with `__` — has its first byte escaped too (`.` →
/// `%2E`, `..` → `%2E.`, `__x` → `%5F_x`). Mirrors molrec's `safe_name`
/// (`metrics.md`) — the two implementations must mangle identically or
/// produce stores neither can read back.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] for the empty name, which is not a series key.
fn safe_series_name(name: &str) -> Result<String, MolRsError> {
    use std::fmt::Write as _;
    if name.is_empty() {
        return Err(MolRsError::zarr(
            "a metrics series needs a name: the empty string is not a series key",
        ));
    }
    let mut out = String::with_capacity(name.len());
    for byte in name.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-' => out.push(byte as char),
            _ => {
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    if out == "." || out == ".." || out.starts_with("__") {
        let first = out.as_bytes()[0];
        let mut escaped = String::with_capacity(out.len() + 2);
        let _ = write!(escaped, "%{first:02X}");
        escaped.push_str(&out[1..]);
        out = escaped;
    }
    Ok(out)
}

/// The inverse of [`safe_series_name`].
fn original_series_name(encoded: &str) -> Result<String, MolRsError> {
    let bytes = encoded.as_bytes();
    let mut raw = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = encoded.get(index + 1..index + 3).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "metrics series name '{encoded}': truncated %-escape"
                ))
            })?;
            raw.push(u8::from_str_radix(hex, 16).map_err(|_| {
                MolRsError::zarr(format!(
                    "metrics series name '{encoded}': bad %-escape '%{hex}'"
                ))
            })?);
            index += 3;
        } else {
            raw.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(raw)
        .map_err(|_| MolRsError::zarr(format!("metrics series name '{encoded}' is not UTF-8")))
}

/// Write the `metrics/` section: the catalog document as group attributes,
/// and each closed series as one float64 array at
/// `metrics/series/<safe_name>`.
///
/// The live JSONL WAL (`metrics/metrics.jsonl`) is host-owned and never
/// written here.
#[cfg(feature = "zarr")]
fn write_metrics(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    record: &MolRec,
) -> Result<(), MolRsError> {
    write_json_group(store, prefix, &record.metrics)?;
    if record.metrics_series.is_empty() {
        return Ok(());
    }
    let series_path = join_path(prefix, "series");
    GroupBuilder::new()
        .build(store.clone(), &series_path)?
        .store_metadata()?;
    for (name, values) in &record.metrics_series {
        let column = Column::from(
            ndarray::ArrayD::from_shape_vec(vec![values.len()], values.clone())
                .map_err(|e| MolRsError::zarr(format!("metrics series '{name}': {e}")))?,
        );
        write_column(
            store,
            &join_path(&series_path, &safe_series_name(name)?),
            &column,
            None,
        )?;
    }
    Ok(())
}

/// Read `metrics/series/<name>` float64 arrays back into
/// [`MolRec::metrics_series`]. A `metrics/` group without a `series/` child
/// is a document-only section; a non-float64 series is an error rather than
/// a silent narrowing.
fn read_metrics_series(
    store: &ReadableListableStorage,
    metrics_path: &str,
    record: &mut MolRec,
) -> Result<(), MolRsError> {
    let node = Node::open(store, metrics_path)?;
    let has_series = node.children().iter().any(|child| {
        matches!(child.metadata(), NodeMetadata::Group(_))
            && child.path().as_str().ends_with("/series")
    });
    if !has_series {
        return Ok(());
    }
    let series_path = join_path(metrics_path, "series");
    let series_node = Node::open(store, &series_path)?;
    for child in series_node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let column = read_column(
            store,
            path,
            &ArraySubset::new_with_shape(Array::open(store.clone(), path)?.shape().to_vec()),
        )?;
        let values = column.get::<f64>().ok_or_else(|| {
            MolRsError::zarr(format!(
                "metrics series '{name}' must be a float64 array, got {}",
                column.dtype().name()
            ))
        })?;
        record.metrics_series.insert(
            original_series_name(name)?,
            values.iter().copied().collect(),
        );
    }
    Ok(())
}

#[cfg(feature = "zarr")]
fn write_observables(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    record: &MolRec,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;
    let meta_path = join_path(prefix, "meta");
    GroupBuilder::new()
        .build(store.clone(), &meta_path)?
        .store_metadata()?;

    for (name, obs) in record.observables.iter() {
        let mut attrs = obs.extra.clone();
        attrs.insert("kind".into(), obs.kind.as_str().into());
        attrs.insert("description".into(), obs.description.clone().into());
        attrs.insert("time_dependent".into(), obs.time_dependent.into());
        if let Some(unit) = &obs.unit {
            attrs.insert("unit".into(), unit.clone().into());
        }
        if !obs.axes.is_empty() {
            attrs.insert("axes".into(), obs.axes.clone().into());
        }
        if let Some(sampling) = &obs.sampling {
            attrs.insert("sampling".into(), sampling.clone().into());
        }
        if let Some(domain) = &obs.domain {
            attrs.insert("domain".into(), domain.clone().into());
        }
        if let Some(target) = &obs.target {
            attrs.insert("target".into(), target.clone().into());
        }
        write_json_group(store, &join_path(&meta_path, name), &attrs)?;

        let ObservableValues::Column(column) = &obs.values;
        write_column(store, &join_path(prefix, name), column, None)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// One frame-shaped section of the record in `store`, or `None` when it has
/// none; see [`MrecFile::frame`].
fn read_frame_section(
    store: &ReadableListableStorage,
    section: &str,
) -> Result<Option<Frame>, MolRsError> {
    let sections = section_names_of(store)?;
    if !sections.iter().any(|name| name == section) {
        return Ok(None);
    }
    let frame = read_frame_group(store, &join_path("/", section))?;
    // An absolute reference into another section is checked against that
    // block's `count` attribute, without decoding the section.
    let rows_of = |target: &str| -> Option<Option<usize>> {
        let (target_section, block) = target.strip_prefix('/')?.split_once('/')?;
        if !matches!(target_section, "frame" | "system") {
            return None;
        }
        if target_section == section {
            return Some(frame.get(block).map(|b| b.n_rows().unwrap_or(0)));
        }
        if !sections.iter().any(|name| name == target_section) {
            return None;
        }
        let path = join_path(&join_path("/", target_section), block);
        Some(
            zarrs::group::Group::open(store.clone(), &path)
                .ok()
                .and_then(|group| group.attributes().get("count").and_then(|n| n.as_u64()))
                .map(|n| n as usize),
        )
    };
    check_declared_references(&frame, section, &rows_of)?;
    Ok(Some(frame))
}

/// The record's top-level section names; see [`MrecFile::sections`].
fn section_names_of(store: &ReadableListableStorage) -> Result<Vec<String>, MolRsError> {
    let root = Node::open(store, "/")?;
    let mut names = Vec::new();
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = child
            .path()
            .as_str()
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        if !name.is_empty() {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Every section of the record in `store`, rooted at `/`.
fn read_record(store: &ReadableListableStorage) -> Result<MolRec, MolRsError> {
    let prefix = "/";
    let mut record = MolRec::new();

    record.meta = read_meta(store)?;

    let root = Node::open(store, prefix)?;
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let path = child.path().as_str().to_string();
        let name = path.rsplit('/').next().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        match name.as_str() {
            "meta" => {}
            "system" => record.system = Some(read_frame_group(store, &path)?),
            "frame" => record.frame = Some(read_frame_group(store, &path)?),
            "trajectory" => {
                // One decoder, and it is not here. `MrecReader::open`
                // resolves the same `/trajectory` node this child is.
                // Demoted on the way in: the decoder is a read door, so it is
                // handed the store's read-only view rather than this one.
                let sequence = MrecReader::from_storage(store.clone())?;
                record.trajectory = Some(sequence.to_trajectory()?);
            }
            FORCEFIELD_GROUP => {
                record.forcefield = Some(read_forcefield_group(store, &path)?);
            }
            "observables" => read_observables(store, &path, &mut record)?,
            "method" => record.method = read_json_group(store, &path)?,
            "status" => record.status = read_json_group(store, &path)?,
            "metrics" => {
                record.metrics = read_json_group(store, &path)?;
                read_metrics_series(store, &path, &mut record)?;
            }
            // A root section this build does not interpret is ignored. Its
            // layout is unknown — it may hold arrays directly, or be shaped
            // like a sequence — so reading it as a frame group would misread
            // it or fail the whole record over data nobody asked for.
            _ => {}
        }
    }

    check_absolute_references(&record)?;
    Ok(record)
}

/// Check every absolute row reference (`/frame/atoms`, `/system/atoms`) of
/// the record's `frame`, `system` and trajectory frames against the section
/// it names, when that section is present: the block must exist and every
/// non-null value must be below its row count. A reference into a section
/// the record lacks cannot be checked and is left alone.
fn check_absolute_references(record: &MolRec) -> Result<(), MolRsError> {
    let rows_of = |target: &str| -> Option<Option<usize>> {
        let (section, block) = target.strip_prefix('/')?.split_once('/')?;
        let frame = match section {
            "frame" => record.frame.as_ref(),
            "system" => record.system.as_ref(),
            _ => None,
        }?;
        Some(frame.get(block).map(|b| b.n_rows().unwrap_or(0)))
    };
    for (what, frame) in [("frame", &record.frame), ("system", &record.system)] {
        if let Some(frame) = frame {
            check_declared_references(frame, what, &rows_of)?;
        }
    }
    if let Some(trajectory) = &record.trajectory {
        for (index, frame) in trajectory.frames.iter().enumerate() {
            check_declared_references(frame, &format!("trajectory frame {index}"), &rows_of)?;
        }
    }
    Ok(())
}

/// Read the `meta` section of the record rooted at `/`.
///
/// Every writer creates the group, but a reader tolerates its absence — an
/// empty document — so a store a foreign tool assembled without one still
/// opens.
pub(in crate::io::mrec::zarr_storage) fn read_meta<S>(
    store: &Arc<S>,
) -> Result<JsonMap<String, JsonValue>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let attrs = match zarrs::group::Group::open(store.clone(), "/meta") {
        Ok(group) => group.attributes().clone(),
        Err(zarrs::group::GroupCreateError::MissingMetadata) => JsonMap::new(),
        Err(e) => return Err(e.into()),
    };
    Ok(attrs)
}

fn read_json_group(
    store: &ReadableListableStorage,
    path: &str,
) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    Ok(zarrs::group::Group::open(store.clone(), path)?
        .attributes()
        .clone())
}

fn read_observables(
    store: &ReadableListableStorage,
    prefix: &str,
    record: &mut MolRec,
) -> Result<(), MolRsError> {
    let meta_path = join_path(prefix, "meta");
    let node = Node::open(store, prefix)?;

    for child in node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() {
            continue;
        }

        let attrs = zarrs::group::Group::open(store.clone(), &join_path(&meta_path, name))
            .map_err(|_| {
                MolRsError::zarr(format!(
                    "observable '{name}' has data but no 'observables/meta/{name}' entry"
                ))
            })?
            .attributes()
            .clone();

        // A kind this build does not define is carried through as
        // `ObservableKind::Other` and written back unchanged.
        let kind = attrs
            .get("kind")
            .and_then(JsonValue::as_str)
            .map(ObservableKind::from)
            .ok_or_else(|| MolRsError::zarr(format!("observable '{name}' is missing 'kind'")))?;

        let mut extra = attrs.clone();
        for key in [
            "kind",
            "description",
            "time_dependent",
            "unit",
            "axes",
            "sampling",
            "domain",
            "target",
        ] {
            extra.remove(key);
        }

        let obs = ObservableRecord {
            name: name.to_string(),
            kind,
            description: attrs
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string(),
            time_dependent: attrs
                .get("time_dependent")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            unit: attrs
                .get("unit")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            axes: attrs
                .get("axes")
                .and_then(JsonValue::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            sampling: attrs
                .get("sampling")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            domain: attrs
                .get("domain")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            target: attrs
                .get("target")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            extra,
            // An observable's array is its whole column.
            values: ObservableValues::Column(read_column(
                store,
                path,
                &ArraySubset::new_with_shape(Array::open(store.clone(), path)?.shape().to_vec()),
            )?),
        };
        record.observables.insert(obs)?;
    }
    Ok(())
}

#[cfg(feature = "filesystem")]
impl MrecReader {
    /// Open the `trajectory` section of a record for reading one frame at a
    /// time: a `*.mrec` directory store, a packed `*.mrec.zip`, or the bytes
    /// of a packed one — the sources [`MrecFile::open`] takes. The cursor is
    /// index-only at open; each [`MrecReader::frame`] call decodes one
    /// committed frame. A store the caller already holds goes through
    /// [`MrecReader::from_storage`].
    ///
    /// # Errors
    ///
    /// [`MrecFile::open`]'s; [`InvalidData`](std::io::ErrorKind::InvalidData)
    /// carrying a [`MolRsError::Zarr`] for a record with no `trajectory`
    /// section, or carrying [`MrecReader::from_storage`]'s store errors
    /// (schema mismatch, missing index).
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use molrs::core::{Frame, Trajectory};
    /// use molrs::io::mrec::{MolRec, MrecReader};
    /// use molrs::io::write_mrec;
    ///
    /// let dir = tempfile::tempdir().unwrap();
    /// let path = dir.path().join("run.mrec");
    ///
    /// let mut record = MolRec::new();
    /// record.trajectory = Some(Trajectory::from_frames(vec![Frame::new()]));
    /// write_mrec(&path, &record)?;
    ///
    /// let mut seq = MrecReader::open(&path)?;
    /// assert!(seq.frame(0)?.is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn open<'a>(src: impl Into<Source<'a>>) -> std::io::Result<MrecReader> {
        let src = src.into();
        let store = open_record(&src).map_err(io_error)?;
        let sections = section_names_of(&store).map_err(io_error)?;
        if !sections.iter().any(|name| name == "trajectory") {
            return Err(io_error(MolRsError::zarr(format!(
                "{}: the record has no 'trajectory' section (its sections: {}); \
                 read the others with MrecFile",
                src.name(),
                sections.join(", ")
            ))));
        }
        MrecReader::from_storage(store).map_err(io_error)
    }
}

pub(in crate::io::mrec::zarr_storage) fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use crate::core::Cell;
    use molrs::core::Frame;
    use molrs::core::Trajectory;
    use molrs::core::{Block, Column};
    use molrs::io::mrec::ForceFieldSection;
    use molrs::op::F;
    use ndarray::ArrayD;
    use std::path::Path;
    use tempfile::tempdir;

    /// The record at `path`, opened.
    fn opened(path: &Path) -> MrecFile {
        MrecFile::open(path).unwrap()
    }

    /// The `frame` section of the record at `path`, which must carry one.
    fn frame_section(path: &Path) -> Frame {
        opened(path).frame().unwrap().expect("a frame section")
    }

    /// The `system` section of the record at `path`, which must carry one.
    fn system_section(path: &Path) -> Frame {
        opened(path).system().unwrap().expect("a system section")
    }

    /// Every frame of the record's `trajectory` section.
    fn trajectory_of(path: &Path) -> Result<Trajectory, MolRsError> {
        MrecReader::open(path)?.to_trajectory()
    }

    /// A record whose one state section is `trajectory`.
    fn trajectory_record(trajectory: &Trajectory) -> MolRec {
        let mut record = MolRec::new();
        record.trajectory = Some(trajectory.clone());
        record
    }

    /// A record whose state sections are `frame`, and `system` when given.
    fn frame_record(frame: &Frame, system: Option<&Frame>) -> MolRec {
        let mut record = MolRec::new();
        record.frame = Some(frame.clone());
        record.system = system.cloned();
        record
    }

    /// A record whose one state section is `forcefield`.
    fn forcefield_record(forcefield: &ForceFieldSection) -> MolRec {
        let mut record = MolRec::new();
        record.forcefield = Some(forcefield.clone());
        record
    }

    fn float_column(values: &[F]) -> Column {
        Column::from(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn frame_with_atoms(n: usize) -> Frame {
        let mut block = Block::new();
        block
            .insert(
                "x",
                ArrayD::from_shape_vec(vec![n], vec![1.0; n]).unwrap(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    /// One `atoms` block with one `f64` column, at the caller's values.
    ///
    /// [`frame_with_atoms`] fills with 1.0, which cannot tell a rewritten
    /// store from a stale one; these values can.
    fn frame_with_x(values: &[F]) -> Frame {
        let mut block = Block::new();
        block.insert("x", float_column(values), None).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    /// Column `x` of the `atoms` block, as it came back from the store.
    fn atoms_x(frame: &Frame) -> Vec<F> {
        frame
            .get("atoms")
            .expect("the frame carries an atoms block")
            .get("x")
            .and_then(|c| c.get::<f64>())
            .expect("column x arrived as f64")
            .iter()
            .copied()
            .collect()
    }

    fn write_then_read(record: &MolRec) -> MolRec {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        write_mrec(&path, record).unwrap();
        opened(&path).record().unwrap()
    }

    /// A producer that supplied no metadata gets an empty `meta` back.
    #[test]
    fn a_record_written_without_meta_reads_back_an_empty_meta() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let loaded = write_then_read(&rec);
        assert!(loaded.meta.is_empty(), "{:?}", loaded.meta);
    }

    fn walk_json(root: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "json") {
                    out.push(path);
                }
            }
        }
        out
    }

    #[test]
    fn producer_meta_and_method_round_trip() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(2));
        rec.meta
            .insert("creator".into(), serde_json::json!({"name": "unit-test"}));
        rec.method.insert("type".into(), "static_structure".into());

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.meta["creator"]["name"], "unit-test");
        assert_eq!(loaded.method["type"], "static_structure");
    }

    #[test]
    fn frame_blocks_round_trip() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(5));
        let loaded = write_then_read(&rec);
        assert_eq!(
            loaded.frame.unwrap().get("atoms").unwrap().n_rows(),
            Some(5)
        );
    }

    #[test]
    fn trajectory_round_trips_with_index_arrays() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(4));
        rec.add_frame(frame_with_atoms(4));
        rec.add_frame(frame_with_atoms(4));
        let traj = rec.trajectory.as_mut().unwrap();
        traj.step = Some(vec![0, 1]);
        traj.time = Some(vec![0.0, 0.5]);

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.n_frames(), 2);
        let traj = loaded.trajectory.as_ref().unwrap();
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.step, Some(vec![0, 1]));
        assert_eq!(traj.time, Some(vec![0.0, 0.5]));
    }

    #[test]
    fn observables_round_trip_with_semantics() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let mut obs = ObservableRecord::new(
            "total_energy",
            ObservableKind::Scalar,
            float_column(&[1.0, 1.5, 2.0]),
        );
        obs.description = "Total energy by step".into();
        obs.unit = Some("eV".into());
        obs.axes = vec!["timestep".into()];
        obs.time_dependent = true;
        obs.domain = Some("trajectory".into());
        rec.observables.insert(obs).unwrap();

        let loaded = write_then_read(&rec);
        let got = loaded.observables.get("total_energy").unwrap();
        assert_eq!(got.kind, ObservableKind::Scalar);
        assert_eq!(got.unit.as_deref(), Some("eV"));
        assert_eq!(got.axes, vec!["timestep".to_string()]);
        assert!(got.time_dependent);
        assert_eq!(got.domain.as_deref(), Some("trajectory"));
    }

    /// A kind this build does not define is carried through and written back
    /// unchanged, rather than failing the whole record read.
    #[test]
    fn an_unknown_observable_kind_round_trips() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let mut obs = ObservableRecord::new(
            "spectrum",
            ObservableKind::Scalar,
            float_column(&[0.5, 0.25]),
        );
        obs.kind = ObservableKind::from("spectrum_density");
        obs.description = "From a producer module".into();
        rec.observables.insert(obs).unwrap();

        let loaded = write_then_read(&rec);
        let got = loaded.observables.get("spectrum").unwrap();
        assert_eq!(got.kind, ObservableKind::Other("spectrum_density".into()));
        assert_eq!(got.description, "From a producer module");

        let again = write_then_read(&loaded);
        assert_eq!(
            again.observables.get("spectrum").unwrap().kind.as_str(),
            "spectrum_density"
        );
    }

    /// Observable data without its `observables/meta/<name>` entry is still
    /// refused: the pairing is mandatory.
    #[test]
    fn observable_data_without_meta_is_refused() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.observables
            .insert(ObservableRecord::new(
                "energy",
                ObservableKind::Scalar,
                float_column(&[1.0]),
            ))
            .unwrap();
        write_mrec(&path, &rec).unwrap();
        std::fs::remove_dir_all(path.join("observables/meta/energy")).unwrap();
        let err = opened(&path).record().unwrap_err().to_string();
        assert!(err.contains("energy") && err.contains("meta"), "{err}");
    }

    /// A record carrying a frame, plus a root section this build does not
    /// know whose arrays sit directly under it and which is shaped like a
    /// sequence (`step` array, CSR-ish children).
    fn record_with_foreign_section() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(3));
        write_mrec(&path, &rec).unwrap();
        let store: ReadableWritableListableStorage =
            Arc::new(PositionalWriteStorage::new(&path).unwrap());
        for group in ["/future", "/future/atoms"] {
            GroupBuilder::new()
                .build(store.clone(), group)
                .unwrap()
                .store_metadata()
                .unwrap();
        }
        let steps = Column::from(ArrayD::from_shape_vec(vec![2], vec![0_i64, 1]).unwrap());
        write_column(&store, "/future/step", &steps, None).unwrap();
        write_column(&store, "/future/atoms/x", &float_column(&[1.0, 2.0]), None).unwrap();
        write_column(
            &store,
            "/future/atoms/offset",
            &Column::from(ArrayD::from_shape_vec(vec![3], vec![0_u64, 1, 5]).unwrap()),
            None,
        )
        .unwrap();
        (dir, path)
    }

    /// Unknown root sections are ignored by every typed reader: never
    /// reinterpreted as a frame group, never a reason to fail.
    #[test]
    fn unknown_root_sections_are_ignored() {
        let (_dir, path) = record_with_foreign_section();
        assert!(
            opened(&path)
                .sections()
                .unwrap()
                .contains(&"future".to_string())
        );

        let record = opened(&path).record().unwrap();
        assert_eq!(
            record.frame.unwrap().get("atoms").unwrap().n_rows(),
            Some(3)
        );
        assert_eq!(frame_section(&path).get("atoms").unwrap().n_rows(), Some(3));
        let err = trajectory_of(&path).unwrap_err().to_string();
        assert!(err.contains("no 'trajectory' section"), "{err}");
    }

    /// The frame and system doors decode only their own section, so a broken
    /// trajectory or observables section does not stop them; the whole-record
    /// reader still reports it.
    #[test]
    fn a_broken_sibling_section_does_not_fail_the_frame_or_system_door() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(3));
        rec.system = Some(frame_with_atoms(2));
        rec.add_frame(frame_with_atoms(3));
        rec.observables
            .insert(ObservableRecord::new(
                "energy",
                ObservableKind::Scalar,
                float_column(&[1.0]),
            ))
            .unwrap();
        write_mrec(&path, &rec).unwrap();
        // Break both: observable data without its meta, and a trajectory
        // whose schema pin is not a schema.
        std::fs::remove_dir_all(path.join("observables/meta/energy")).unwrap();
        let trajectory_path = path.join("trajectory/zarr.json");
        let mut trajectory: JsonValue =
            serde_json::from_slice(&std::fs::read(&trajectory_path).unwrap()).unwrap();
        trajectory["attributes"]["sequence_schema"] = "not a schema".into();
        std::fs::write(&trajectory_path, serde_json::to_vec(&trajectory).unwrap()).unwrap();

        assert!(opened(&path).record().is_err());
        assert!(trajectory_of(&path).is_err());
        assert_eq!(frame_section(&path).get("atoms").unwrap().n_rows(), Some(3));
        assert_eq!(
            system_section(&path).get("atoms").unwrap().n_rows(),
            Some(2)
        );
    }

    /// `molrec_version` is an ordinary producer key: written as given, read
    /// back unchanged, and checked by no read door, whatever its value.
    #[test]
    fn a_molrec_version_key_is_an_ordinary_meta_key() {
        for value in [
            JsonValue::from(99_u64),
            JsonValue::Null,
            JsonValue::from("1"),
        ] {
            let dir = tempdir().unwrap();
            let path = dir.path().join("record.mrec");
            let mut rec = MolRec::new();
            rec.frame = Some(frame_with_atoms(3));
            rec.system = Some(frame_with_atoms(3));
            rec.add_frame(frame_with_atoms(3));
            rec.meta.insert("molrec_version".into(), value.clone());
            write_mrec(&path, &rec).unwrap();

            frame_section(&path);
            system_section(&path);
            trajectory_of(&path).unwrap();
            MrecReader::open(&path).unwrap();
            assert_eq!(opened(&path).meta().unwrap()["molrec_version"], value);
            assert_eq!(
                opened(&path).record().unwrap().meta["molrec_version"],
                value
            );
        }
    }

    /// A deleted `meta/` group still reads as an empty document: a foreign
    /// store may write no metadata group at all.
    #[test]
    fn a_deleted_meta_group_reads_as_an_empty_document() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        write_mrec(&path, &rec).unwrap();
        std::fs::remove_dir_all(path.join("meta")).unwrap();
        let loaded = opened(&path).record().unwrap();
        assert!(loaded.meta.is_empty());
    }

    #[test]
    fn a_record_with_no_state_section_is_refused_at_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        assert!(write_mrec(&path, &MolRec::new()).is_err());
    }

    #[test]
    fn trajectory_door_writes_the_producer_meta() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut meta = JsonMap::new();
        meta.insert("creator".into(), serde_json::json!({"name": "unit-test"}));
        let mut record = trajectory_record(&Trajectory::from_frames(vec![frame_with_atoms(2)]));
        record.meta = meta.clone();
        write_mrec(&path, &record).unwrap();
        let back = opened(&path).meta().unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn trajectory_door_round_trips_through_the_record_layout() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut frame = frame_with_atoms(3);
        frame.meta.insert("key", "value");
        let traj = Trajectory::from_frames(vec![frame]);

        write_mrec(&path, &trajectory_record(&traj)).unwrap();
        let loaded = trajectory_of(&path).unwrap();
        assert_eq!(loaded.frames.len(), 1);
        assert_eq!(
            loaded.frames[0].meta.get("key").unwrap().as_str(),
            Some("value")
        );

        // The narrow door writes a conforming record, not a private layout:
        // a root, a `meta/` group, and the trajectory section.
        assert!(
            opened(&path)
                .sections()
                .unwrap()
                .contains(&"meta".to_string())
        );
        let record = opened(&path).record().unwrap();
        assert!(record.trajectory.is_some());
    }

    #[test]
    fn frame_door_writes_the_frame_section() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snapshot.mrec");
        let frame = frame_with_atoms(3);
        write_mrec(&path, &frame_record(&frame, None)).unwrap();

        let loaded = frame_section(&path);
        assert_eq!(loaded.get("atoms").unwrap().n_rows(), Some(3));
        assert!(opened(&path).system().unwrap().is_none());
        assert!(
            opened(&path)
                .sections()
                .unwrap()
                .contains(&"meta".to_string())
        );
        let record = opened(&path).record().unwrap();
        assert!(record.trajectory.is_none());
    }

    #[test]
    fn system_door_writes_the_system_section() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("system.mrec");
        write_mrec(
            &path,
            &MolRec {
                system: Some(frame_with_atoms(2)),
                ..MolRec::new()
            },
        )
        .unwrap();

        let loaded = system_section(&path);
        assert_eq!(loaded.get("atoms").unwrap().n_rows(), Some(2));
        assert!(opened(&path).frame().unwrap().is_none());
    }

    #[test]
    fn frame_door_can_carry_a_system_section_beside_the_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("both.mrec");
        let frame = frame_with_atoms(3);
        let system = frame_with_atoms(3);
        write_mrec(&path, &frame_record(&frame, Some(&system))).unwrap();

        assert_eq!(
            opened(&path).sections().unwrap(),
            vec!["frame", "meta", "system"]
        );
        // The identity document is present and, with nothing handed in, empty.
        assert!(opened(&path).meta().unwrap().is_empty());

        assert_eq!(frame_section(&path).get("atoms").unwrap().n_rows(), Some(3));
        assert_eq!(
            system_section(&path).get("atoms").unwrap().n_rows(),
            Some(3)
        );
    }

    #[test]
    fn metrics_series_round_trip_as_float64_arrays() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("catalog".into(), "summary".into());
        rec.metrics_series
            .insert("train/loss".into(), vec![1.0, 0.5, 0.25]);
        rec.metrics_series.insert("lr".into(), vec![1e-3, 1e-4]);

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.metrics["catalog"], "summary");
        assert_eq!(loaded.metrics_series.len(), 2);
        assert_eq!(loaded.metrics_series["train/loss"], vec![1.0, 0.5, 0.25]);
        assert_eq!(loaded.metrics_series["lr"], vec![1e-3, 1e-4]);

        // A second write of what was read must not shed the series — this is
        // the preserve-the-unknown guarantee for densified curves.
        let again = write_then_read(&loaded);
        assert_eq!(again.metrics_series, rec.metrics_series);
    }

    /// Names Zarr forbids as nodes have their first byte escaped too, and
    /// every name reads back; the empty name is not a series key.
    #[test]
    fn series_names_zarr_forbids_are_escaped_and_read_back() {
        for (name, safe) in [
            (".", "%2E"),
            ("..", "%2E."),
            ("__x", "%5F_x"),
            ("__", "%5F_"),
            ("_x", "_x"),
            ("a.b", "a.b"),
            ("train/loss", "train%2Floss"),
        ] {
            assert_eq!(safe_series_name(name).unwrap(), safe, "{name}");
            assert_eq!(original_series_name(safe).unwrap(), name);
        }
        assert!(safe_series_name("").is_err());

        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        for name in [".", "..", "__dunder"] {
            rec.metrics_series.insert(name.into(), vec![1.0]);
        }
        assert_eq!(write_then_read(&rec).metrics_series, rec.metrics_series);

        rec.metrics_series.insert(String::new(), vec![1.0]);
        let dir = tempdir().unwrap();
        assert!(write_mrec(dir.path().join("r.mrec").as_path(), &rec).is_err());
    }

    /// A live host WAL (`metrics/metrics.jsonl`) is a stray text file inside
    /// the store; reading the record must tolerate it rather than error.
    #[test]
    fn a_stray_metrics_wal_file_is_tolerated_on_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("live".into(), true.into());
        write_mrec(&path, &rec).unwrap();

        std::fs::write(
            path.join("metrics/metrics.jsonl"),
            "{\"t\":\"scalar\",\"k\":\"loss\",\"v\":0.5}\n",
        )
        .unwrap();

        let loaded = opened(&path).record().unwrap();
        assert_eq!(loaded.metrics["live"], true);
        assert!(loaded.metrics_series.is_empty());
    }

    #[test]
    fn a_non_float64_metrics_series_is_rejected_loud() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("catalog".into(), "summary".into());
        write_mrec(&path, &rec).unwrap();

        // Plant a u64 array where a float64 series belongs.
        let store: ReadableWritableListableStorage =
            std::sync::Arc::new(super::PositionalWriteStorage::new(&path).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/metrics/series")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_column(
            &store,
            "/metrics/series/steps",
            &Column::from(ndarray::ArrayD::from_shape_vec(vec![2], vec![1u64, 2]).unwrap()),
            None,
        )
        .unwrap();

        let err = opened(&path).record().unwrap_err().to_string();
        assert!(err.contains("steps"), "{err}");
        assert!(err.contains("float64"), "{err}");
    }

    /// Every node in the store, as a path relative to the record root (the
    /// root group itself is the empty string). One `zarr.json` is one node.
    fn node_paths(root: &Path) -> Vec<String> {
        let mut paths: Vec<String> = walk_json(root)
            .iter()
            .map(|file| {
                file.parent()
                    .unwrap()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn rewrite_leaves_no_stale_node() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");

        // A wide record first: a second block, a second column, and a
        // trajectory/frames/ subtree.
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                ArrayD::from_shape_vec(vec![5], vec![1.0; 5]).unwrap(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                ArrayD::from_shape_vec(vec![5], vec![2.0; 5]).unwrap(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                ArrayD::from_shape_vec(vec![2], vec![0u64, 1u64]).unwrap(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        let mut wide = MolRec::new();
        wide.frame = Some(frame);
        wide.add_frame(frame_with_atoms(5));
        write_mrec(&path, &wide).unwrap();

        // Then a strictly smaller one into the same path.
        let mut narrow = MolRec::new();
        narrow.frame = Some(frame_with_atoms(3));
        write_mrec(&path, &narrow).unwrap();

        assert_eq!(
            node_paths(&path),
            vec!["", "frame", "frame/atoms", "frame/atoms/x", "meta"]
        );
    }

    /// The trajectory door writes the frames **once**.
    ///
    /// Today it copies `frames[0]` into the record's `frame` section to get
    /// past `MolRec::validate`, so frame 0 lands in the store twice — once as
    /// `frame/` and once as row 0 of the sequence. That duplicate is a second
    /// copy of real data (bytes, and a wrong `frame` section a reader will
    /// believe), so the pin is on the store's own node list: a trajectory-only
    /// record has a `trajectory` section and no `frame` one at all.
    #[test]
    fn a_written_trajectory_store_holds_no_duplicate_frame_zero() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");

        let traj = Trajectory {
            frames: vec![
                frame_with_x(&[1.0, 2.0, 3.0]),
                frame_with_x(&[4.0, 5.0, 6.0]),
            ],
            step: Some(vec![0, 1]),
            time: None,
        };
        write_mrec(&path, &trajectory_record(&traj)).unwrap();

        let nodes = node_paths(&path);
        let frame_nodes: Vec<&String> = nodes
            .iter()
            .filter(|node| node.as_str() == "frame" || node.starts_with("frame/"))
            .collect();
        assert!(
            frame_nodes.is_empty(),
            "a trajectory-only record must not duplicate frame 0 into a \
             'frame' section; store holds {frame_nodes:?} among {nodes:?}"
        );
        assert!(
            nodes.iter().any(|node| node == "trajectory"),
            "the frames belong to the trajectory section: {nodes:?}"
        );
    }

    #[test]
    fn simbox_geometry_roundtrips_as_f64() {
        use molrs::core::SimBox;

        let dir = tempdir().unwrap();
        let path = dir.path().join("simbox_f64.mrec");
        let h_val = 123.456789012345;
        let h = [[h_val, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 40.0]];
        let origin = [0.1, 0.2, 0.3];
        let mut frame = Frame::new();
        frame.simbox = Some(SimBox::new(Cell::Matrix(h), origin, [true, true, true]).unwrap());
        let mut rec = MolRec::new();
        rec.frame = Some(frame);
        write_mrec(&path, &rec).unwrap();
        let back = opened(&path).record().unwrap();
        let sb = back.frame.as_ref().unwrap().simbox.as_ref().unwrap();
        let h_back = sb.matrix()[0][0];
        assert!(
            (h_back - h_val).abs() < 1e-15,
            "f64 simbox lost precision: {h_back} vs {h_val}"
        );
        assert!((sb.origin()[0] - 0.1).abs() < 1e-15);
    }

    /// A second trajectory record written over the same path replaces the
    /// first, and the read-back is the second one.
    ///
    /// This is the seam between two rules that only look compatible:
    /// [`write_mrec_storage`] erases the root before writing, and
    /// `MrecWriter::create` **refuses** a path that still holds a
    /// `trajectory/step` array rather than silently overwriting somebody's
    /// run. Get the order or the prefix wrong and the second write is a hard
    /// `Err`, not a wrong answer — so the pin is that it succeeds *and* that
    /// no frame, step or time of the first record survives it.
    #[test]
    fn rewriting_a_trajectory_record_over_the_same_path_succeeds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");

        let first = Trajectory {
            frames: vec![
                frame_with_x(&[1.0, 2.0, 3.0]),
                frame_with_x(&[4.0, 5.0, 6.0]),
            ],
            step: Some(vec![0, 1]),
            time: None,
        };
        write_mrec(&path, &trajectory_record(&first)).unwrap();

        // Different in every axis the layout carries: fewer rows per frame,
        // more frames, other step numbers, and times where there were none.
        let second = Trajectory {
            frames: vec![
                frame_with_x(&[-1.5, -2.5]),
                frame_with_x(&[-3.5, -4.5]),
                frame_with_x(&[-5.5, -6.5]),
            ],
            step: Some(vec![7, 8, 9]),
            time: Some(vec![0.25, 0.5, 0.75]),
        };
        write_mrec(&path, &trajectory_record(&second)).unwrap();

        let loaded = trajectory_of(&path).unwrap();
        assert_eq!(
            loaded.frames.iter().map(atoms_x).collect::<Vec<Vec<F>>>(),
            second.frames.iter().map(atoms_x).collect::<Vec<Vec<F>>>(),
            "the read-back must be the second record's frames, bit for bit"
        );
        assert_eq!(loaded.step, second.step, "and its step numbers");
        assert_eq!(loaded.time, second.time, "and its times");
    }

    // -- the wasm32 precision fixture ---------------------------------------

    /// The packed record molrs-wasm reads to prove a wasm32 reader decodes a
    /// precision column (`numcodecs.shuffle` + `zstd`, written here by the C
    /// encoder) through the pure-Rust `zstd` plugin.
    const WASM_PRECISION_FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../molrs-wasm/tests/fixtures/precision.mrec.zip"
    );

    /// The values the fixture's `x` columns were presented with; the wasm
    /// test rounds them with the same rule and compares bit for bit.
    const FIXTURE_X: [[F; 3]; 2] = [[0.123_456_789, -1.000_488, 7.3], [0.2, 1.75, -3.062_57]];

    fn precision_fixture_record() -> MolRec {
        let precise = |values: &[F]| {
            let mut frame = frame_with_x(values);
            frame
                .get_mut("atoms")
                .unwrap()
                .set_precision("x", 1e-3)
                .unwrap();
            frame
        };
        let mut record = MolRec::new();
        record.frame = Some(precise(&FIXTURE_X[0]));
        record.trajectory = Some(Trajectory::from_frames(
            FIXTURE_X.iter().map(|values| precise(values)).collect(),
        ));
        record
    }

    /// `cargo mrs-test -- --ignored regenerate_the_wasm_precision_fixture`
    /// rewrites the checked-in fixture.
    #[test]
    #[ignore = "rewrites a checked-in fixture"]
    fn regenerate_the_wasm_precision_fixture() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("precision.mrec");
        write_mrec(&path, &precision_fixture_record()).unwrap();
        let packed = crate::io::mrec::zarr_storage::pack_mrec_zip(&path).unwrap();
        std::fs::copy(packed, WASM_PRECISION_FIXTURE).unwrap();
    }

    #[test]
    fn the_wasm_precision_fixture_is_a_shuffled_zstd_precision_record() {
        let store =
            crate::io::mrec::zarr_storage::pack::open_mrec_zip(WASM_PRECISION_FIXTURE).unwrap();
        let q = molrs::core::quantum(1e-3).unwrap();
        let rounded = |values: &[F]| -> Vec<F> {
            values
                .iter()
                .map(|&x| molrs::core::quantize(x, q))
                .collect()
        };
        let array = Array::open(store.clone(), "/frame/atoms/x").unwrap();
        let metadata = serde_json::to_string(array.metadata()).unwrap();
        assert!(metadata.contains("numcodecs.shuffle") && metadata.contains("zstd"));
        let sequence = MrecReader::from_storage(store).unwrap();
        for (index, values) in FIXTURE_X.iter().enumerate() {
            let frame = sequence.frame(index as u64).unwrap().unwrap();
            assert_eq!(atoms_x(&frame), rounded(values));
            assert_eq!(frame.get("atoms").unwrap().precision("x"), Some(1e-3));
        }
    }

    // -- absolute row references (molrec F4) -------------------------------

    fn members_into_frame(atom: u64) -> Frame {
        let mut members = Block::new();
        members
            .insert(
                "ibead",
                Column::from(ArrayD::from_shape_vec(vec![1], vec![0_u64]).unwrap()),
                None,
            )
            .unwrap();
        members
            .insert(
                "atom",
                Column::from(ArrayD::from_shape_vec(vec![1], vec![atom]).unwrap()),
                None,
            )
            .unwrap();
        members.set_target("atom", "/frame/atoms").unwrap();
        let mut system = frame_with_atoms(1);
        system.insert("members", members);
        system
    }

    #[test]
    fn an_absolute_target_is_range_checked_against_its_section() {
        let mut record = MolRec::new();
        record.frame = Some(frame_with_atoms(3));
        record.system = Some(members_into_frame(2));
        let back = write_then_read(&record);
        assert_eq!(
            back.system.unwrap()["members"].target("atom"),
            Some("/frame/atoms")
        );

        record.system = Some(members_into_frame(3));
        let dir = tempdir().unwrap();
        let err = write_mrec(dir.path().join("r.mrec").as_path(), &record)
            .unwrap_err()
            .to_string();
        assert!(err.contains("/frame/atoms"), "{err}");

        // Without a `frame` section the reference cannot be checked: kept.
        record.frame = None;
        let back = write_then_read(&record);
        assert!(back.system.is_some());
    }

    /// The typed door reads only its own section, but still range-checks an
    /// absolute reference against the target block's `count`.
    #[test]
    fn the_system_door_range_checks_an_absolute_target() {
        let mut record = MolRec::new();
        record.frame = Some(frame_with_atoms(3));
        record.system = Some(members_into_frame(2));
        let dir = tempdir().unwrap();
        let path = dir.path().join("r.mrec");
        write_mrec(&path, &record).unwrap();
        assert!(opened(&path).system().is_ok());

        let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
        let mut atoms = zarrs::group::Group::open(store, "/frame/atoms").unwrap();
        atoms
            .attributes_mut()
            .insert("count".into(), serde_json::json!(2));
        atoms.store_metadata().unwrap();
        let err = opened(&path).system().unwrap_err().to_string();
        assert!(err.contains("/frame/atoms"), "{err}");
    }

    fn string_column(values: &[&str]) -> Column {
        Column::from(
            ArrayD::from_shape_vec(
                vec![values.len()],
                values.iter().map(|v| v.to_string()).collect(),
            )
            .unwrap(),
        )
    }

    /// A force field that exercises every corner of the layout: an
    /// unconverted unit system, a percent-encoded style, a wildcard
    /// endpoint, an absent parameter, string parameters, a zero-row table,
    /// an unknown document key and a table no style names.
    fn awkward_forcefield() -> ForceFieldSection {
        let document = serde_json::json!({
            "name": "awkward",
            "units": {"length": "nm", "energy": "kJ/mol", "angle": "radian"},
            "source": {"format": "gromacs-top", "uri": "ff.itp"},
            "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
            "styles": [
                {"category": "atom", "style": "full"},
                {"category": "dihedral", "style": "periodic"},
                {"category": "pair", "style": "lj/cut/coul/long",
                 "params": {"cutoff": 1.2, "mixing": "arithmetic"}},
                {"category": "bond", "style": "mmff_bond"},
            ],
            "aromaticity_model": "OEAroModel_MDL",
        });
        let mut atoms = Block::new();
        atoms
            .insert("name", string_column(&["CT", "HC"]), None)
            .unwrap();
        atoms
            .insert("class", string_column(&["CT", "HC"]), None)
            .unwrap();
        atoms
            .insert("smarts", string_column(&["[C;X4]", ""]), None)
            .unwrap();
        atoms.set_validity("smarts", vec![true, false]).unwrap();
        atoms
            .insert("mass", float_column(&[12.011, 1.008]), None)
            .unwrap();
        atoms
            .insert(
                "atomic_number",
                Column::from(ArrayD::from_shape_vec(vec![2], vec![6u64, 1]).unwrap()),
                None,
            )
            .unwrap();
        let mut dihedrals = Block::new();
        for (column, values) in [
            ("name", ["X-CT-CT-X", "HC-CT-CT-HC"]),
            ("itom", ["", "HC"]),
            ("jtom", ["CT", "CT"]),
            ("ktom", ["CT", "CT"]),
            ("ltom", ["", "HC"]),
        ] {
            dihedrals
                .insert(column, string_column(&values), None)
                .unwrap();
        }
        dihedrals
            .insert("k1", float_column(&[0.6276, 0.8]), None)
            .unwrap();
        dihedrals
            .insert("k2", float_column(&[0.25, 0.0]), None)
            .unwrap();
        dihedrals.set_validity("k2", vec![true, false]).unwrap();
        let mut pairs = Block::new();
        for (column, values) in [("name", ["CT"]), ("itom", ["CT"]), ("jtom", ["CT"])] {
            pairs.insert(column, string_column(&values), None).unwrap();
        }
        pairs.insert("sigma", float_column(&[0.35]), None).unwrap();
        pairs
            .insert("epsilon", float_column(&[0.276144]), None)
            .unwrap();
        let mut per_instance = Block::new();
        for column in ["name", "itom", "jtom"] {
            per_instance
                .insert(column, string_column(&[]), None)
                .unwrap();
        }
        let mut notes = Block::new();
        notes
            .insert("text", string_column(&["kept as unknown content"]), None)
            .unwrap();

        let mut tables = indexmap::IndexMap::new();
        tables.insert("atom.full".to_owned(), atoms);
        tables.insert("dihedral.periodic".to_owned(), dihedrals);
        tables.insert("pair.lj%2Fcut%2Fcoul%2Flong".to_owned(), pairs);
        tables.insert("bond.mmff_bond".to_owned(), per_instance);
        tables.insert("notes.free%20text".to_owned(), notes);
        ForceFieldSection {
            document: document.as_object().unwrap().clone(),
            tables,
        }
    }

    fn same_block(a: &Block, b: &Block, what: &str) {
        assert_eq!(a.n_rows(), b.n_rows(), "{what}: rows");
        let mut keys_a: Vec<&str> = a.keys().collect();
        let mut keys_b: Vec<&str> = b.keys().collect();
        keys_a.sort_unstable();
        keys_b.sort_unstable();
        assert_eq!(keys_a, keys_b, "{what}: columns");
        for key in keys_a {
            assert_eq!(a.dtype(key), b.dtype(key), "{what}.{key}: dtype");
            assert_eq!(a.validity(key), b.validity(key), "{what}.{key}: validity");
            let (ca, cb) = (a.get(key).unwrap(), b.get(key).unwrap());
            if let (Some(x), Some(y)) = (ca.get::<f64>(), cb.get::<f64>()) {
                let bits = |v: &ArrayD<F>| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(x), bits(y), "{what}.{key}: values, bit for bit");
            }
            if let (Some(x), Some(y)) = (ca.get::<String>(), cb.get::<String>()) {
                assert_eq!(x, y, "{what}.{key}: strings");
            }
            if let (Some(x), Some(y)) = (ca.get::<u64>(), cb.get::<u64>()) {
                assert_eq!(x, y, "{what}.{key}: u64");
            }
        }
    }

    /// `pair14` is no force-field category (1-4 parameters are `lj/charmm`'s
    /// `epsilon14` / `sigma14` and the frame's per-pair override columns): a
    /// `pair14` table is unknown content, kept as written — a restatement with
    /// other parameters included, which a `pair` table may not hold.
    #[test]
    fn a_pair14_table_round_trips_as_an_unknown_category() {
        let mut ff = awkward_forcefield();
        ff.document["styles"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"category": "pair14", "style": "lj/cut"}));
        let mut pair14 = Block::new();
        for (column, values) in [
            ("name", ["CT-HC", "HC-CT"]),
            ("itom", ["CT", "HC"]),
            ("jtom", ["HC", "CT"]),
        ] {
            pair14.insert(column, string_column(&values), None).unwrap();
        }
        pair14
            .insert("epsilon", float_column(&[0.1, 0.2]), None)
            .unwrap();
        ff.tables.insert("pair14.lj%2Fcut".to_owned(), pair14);
        ff.validate().unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join("ff.mrec");
        write_mrec(&path, &forcefield_record(&ff)).unwrap();
        let back = opened(&path).forcefield().unwrap().unwrap();
        assert_eq!(back.document, ff.document);
        same_block(
            &ff.tables["pair14.lj%2Fcut"],
            &back.tables["pair14.lj%2Fcut"],
            "pair14",
        );
    }

    #[test]
    fn a_forcefield_section_round_trips_through_a_record() {
        let ff = awkward_forcefield();
        let dir = tempdir().unwrap();
        let path = dir.path().join("ff.mrec");
        write_mrec(&path, &forcefield_record(&ff)).unwrap();

        assert!(
            opened(&path)
                .sections()
                .unwrap()
                .contains(&"forcefield".to_owned())
        );
        let back = opened(&path).forcefield().unwrap().unwrap();
        assert_eq!(back.document, ff.document, "the document, key for key");
        assert_eq!(
            serde_json::to_string(&back.document).unwrap(),
            serde_json::to_string(&ff.document).unwrap(),
            "and in the stored key order"
        );
        let mut names: Vec<&String> = back.tables.keys().collect();
        names.sort();
        let mut want: Vec<&String> = ff.tables.keys().collect();
        want.sort();
        assert_eq!(names, want);
        for (name, table) in &ff.tables {
            same_block(table, &back.tables[name], name);
        }

        // The whole-record door carries it too, beside a system.
        let mut record = MolRec::new();
        record.system = Some(frame_with_atoms(2));
        record.forcefield = Some(ff.clone());
        let back = write_then_read(&record);
        assert_eq!(back.forcefield.unwrap().document, ff.document);
    }

    #[test]
    fn a_record_without_a_forcefield_reads_none() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("frame.mrec");
        write_mrec(&path, &frame_record(&frame_with_atoms(1), None)).unwrap();
        assert!(opened(&path).forcefield().unwrap().is_none());
    }

    #[test]
    fn a_malformed_forcefield_is_refused_on_write_and_on_read() {
        let mut ff = awkward_forcefield();
        ff.tables.shift_remove("dihedral.periodic");
        let dir = tempdir().unwrap();
        let path = dir.path().join("ff.mrec");
        let err = write_mrec(&path, &forcefield_record(&ff)).unwrap_err();
        assert!(err.to_string().contains("dihedral.periodic"), "{err}");

        // Laid down behind the writer's back, the reader refuses it too.
        write_mrec(&path, &forcefield_record(&awkward_forcefield())).unwrap();
        std::fs::remove_dir_all(path.join("forcefield").join("dihedral.periodic")).unwrap();
        let err = opened(&path).forcefield().unwrap_err();
        assert!(err.to_string().contains("dihedral.periodic"), "{err}");
    }
}
