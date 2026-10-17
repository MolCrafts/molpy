//! WASM bindings for MolRec scientific records (`*.mrec`, Zarr V3) — the
//! faces of `molrs::io::mrec::MrecReader` and `MrecFile`.
//!
//! Both classes open one [`MrecSource`](JsMrecSource), the record however JS
//! holds it, in increasing laziness:
//!
//! - a **`Uint8Array`** — the bytes of a packed `*.mrec.zip`, whose stored
//!   entries are unpacked into an in-memory store;
//! - a **`Map<path, Uint8Array>`** of every file of a `*.mrec` directory,
//!   copied once into an in-memory store;
//! - a **host object** that serves keys on demand: `get(key)`,
//!   `getRange(key, offset, length)`, `size(key)` and `list(prefix)`. Only the
//!   chunks a frame touches ever cross into wasm, so a multi-gigabyte run
//!   opens in a worker that reads its files (or an HTTP range server)
//!   synchronously.
//!
//! [`MrecReader`] is the `trajectory` section, one frame at a time;
//! [`MrecFile`] reads any record one section at a time (`sections()`,
//! `frame()`, `system()`, `meta()`) — a molpack configuration is `meta` +
//! `frame`, which `new MrecFile(bytes).frame()` reads.

use crate::core::frame::Frame;
use crate::core::simbox::Box as JsBox;
use molrs::io::mrec::{MrecFile as RsMrecFile, MrecReader as RsMrecReader};
use molrs::io::reader::TrajectoryReader;
use std::io::Read;
use std::sync::Arc;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, ReadableListableStorage,
    ReadableStorageTraits, StorageError, StoreKey, StoreKeys, StoreKeysPrefixes, StorePrefix,
    WritableStorageTraits,
};

fn js_string_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen(typescript_custom_section)]
const MREC_SOURCE_TYPE: &'static str = r#"
/**
 * A store host: serves a record's keys on demand, every method synchronous.
 * Keys are store-relative paths without a leading slash
 * (`trajectory/step/zarr.json`).
 */
export interface MrecStorageHost {
    /** The whole value, or `null` when absent. */
    get(key: string): Uint8Array | null;
    /**
     * Optional ranged read; `length` may be `-1` for "to the end". Without it
     * the reader falls back to `get` and slices, which reads whole shards.
     */
    getRange?(key: string, offset: number, length: number): Uint8Array | null;
    /** The value's byte length, or `null` when absent. */
    size(key: string): number | null;
    /** Every key under `prefix` (`""` for all). */
    list(prefix: string): string[];
}

/**
 * A record as `MrecReader` and `MrecFile` open it: a packed `*.mrec.zip`'s
 * bytes, a `Map` of a `*.mrec` directory's files, or a host serving its keys.
 */
export type MrecSource = Uint8Array | Map<string, Uint8Array> | MrecStorageHost;
"#;

#[wasm_bindgen]
extern "C" {
    /// A record: see the `MrecSource` TypeScript type.
    #[wasm_bindgen(typescript_type = "MrecSource")]
    pub type JsMrecSource;
}

/// The store `source` names: a packed zip's bytes, a `Map` of files, or a
/// host object. `op` names the door in the error.
fn open_store(op: &str, source: &JsMrecSource) -> Result<ReadableListableStorage, JsValue> {
    if let Some(bytes) = source.dyn_ref::<js_sys::Uint8Array>() {
        return memory_storage_from_zip(&bytes.to_vec());
    }
    if let Some(files) = source.dyn_ref::<js_sys::Map>() {
        return memory_storage_from_files(files);
    }
    if source.is_object() {
        return Ok(Arc::new(HostStorage::new(JsValue::clone(source))?));
    }
    Err(JsValue::from_str(&format!(
        "{op}: the record must be a packed *.mrec.zip's bytes (Uint8Array), a Map of its \
         files, or a store host object"
    )))
}

/// Reader for the `trajectory` section of a record, one frame at a time.
///
/// The sequence is opened **once**, in the constructor. `MrecReader` is the
/// lazy store cursor, so `readFrame` decodes exactly the frame it was asked
/// for (keeping the last decoded chunk of every column, so consecutive frames
/// are slices), and `nFrames` answers off the index the open already
/// cached. Every method takes `&self`: the cursor holds caches, not a
/// position.
#[wasm_bindgen(js_name = MrecReader)]
pub struct MrecReader {
    sequence: RsMrecReader,
}

#[wasm_bindgen(js_class = MrecReader)]
impl MrecReader {
    /// Open the `trajectory` section of `source` — a packed `*.mrec.zip`'s
    /// bytes, a `Map` of the record's files, or a store host (see
    /// `MrecSource`). Index-only: the schema plus each section's step index
    /// and offset; no frame is decoded here.
    ///
    /// # Errors
    ///
    /// Throws if `source` is none of the three, a zip holds compressed
    /// entries (packed stores are stored-only), a host lacks `get` / `size` /
    /// `list`, or the store holds no readable frame sequence.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsMrecSource) -> Result<MrecReader, JsValue> {
        let store = open_store("new MrecReader", source)?;
        let sequence = RsMrecReader::from_storage(store).map_err(js_string_err)?;
        Ok(MrecReader { sequence })
    }

    /// Frame `t` (0-based), or `undefined` past the end.
    #[wasm_bindgen(js_name = readFrame)]
    pub fn read_frame(&self, t: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self.sequence.frame(t as u64).map_err(js_string_err)?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    /// Read frame `t` carrying only the named `"block/column"` pairs.
    ///
    /// A viewer that needs coordinates asks for `["atoms/x", "atoms/y",
    /// "atoms/z"]` and decodes nothing else. The cell and per-step metadata
    /// always come along.
    #[wasm_bindgen(js_name = readColumns)]
    pub fn read_columns(&self, t: usize, columns: Vec<String>) -> Result<Option<Frame>, JsValue> {
        let pairs: Vec<(&str, &str)> = columns
            .iter()
            .map(|spec| {
                spec.split_once('/').ok_or_else(|| {
                    JsValue::from_str(&format!("column spec {spec:?} is not \"block/column\""))
                })
            })
            .collect::<Result<_, JsValue>>()?;
        let rs_frame = self
            .sequence
            .frame_columns(t as u64, &pairs)
            .map_err(js_string_err)?;
        match rs_frame {
            Some(frame) => Ok(Some(Frame::from_rs(frame)?)),
            None => Ok(None),
        }
    }

    /// The update of block `name` that frame `t` resolves to, or `undefined`
    /// when the block is absent there.
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows — a stage can keep its bond buffers when `blockUpdateAt("bonds",
    /// t)` did not change, without comparing a single value.
    #[wasm_bindgen(js_name = blockUpdateAt)]
    pub fn block_update_at(&self, name: &str, t: usize) -> Result<Option<f64>, JsValue> {
        Ok(self
            .sequence
            .block_update_at(name, t as u64)
            .map_err(js_string_err)?
            .map(|update| update as f64))
    }

    /// The cell at frame `t`, or `undefined` before any cell was written.
    #[wasm_bindgen(js_name = boxAt)]
    pub fn box_at(&self, t: usize) -> Result<Option<JsBox>, JsValue> {
        Ok(self
            .sequence
            .box_at(t as u64)
            .map_err(js_string_err)?
            .map(|inner| JsBox { inner }))
    }

    #[wasm_bindgen(js_name = nFrames)]
    pub fn n_frames(&self) -> Result<usize, JsValue> {
        Ok(self.sequence.steps().len())
    }

    /// Atom count of frame 0, decoded on demand.
    ///
    /// Named for what it is: a ragged store grows its atom count per frame, so
    /// this is the *first* frame's count, not the maximum or the current one.
    #[wasm_bindgen(js_name = nAtomsAtFirstFrame)]
    pub fn n_atoms_at_first_frame(&self) -> Result<usize, JsValue> {
        Ok(self
            .sequence
            .frame(0)
            .map_err(js_string_err)?
            .and_then(|frame| frame.get("atoms").and_then(|block| block.n_rows()))
            .unwrap_or(0))
    }

    /// Step numbers of the committed frames — frame labels for a replay UI.
    #[wasm_bindgen(js_name = steps)]
    pub fn steps(&self) -> Vec<i64> {
        self.sequence.steps().to_vec()
    }

    /// Physical times (fs) of the committed frames, when the run wrote any.
    #[wasm_bindgen(js_name = times)]
    pub fn times(&self) -> Option<Vec<f64>> {
        self.sequence.times().map(<[f64]>::to_vec)
    }

    /// Whether the store carries a block section of this name with at least
    /// one committed update.
    #[wasm_bindgen(js_name = hasBlock)]
    pub fn has_block(&self, name: &str) -> bool {
        self.sequence.has_block(name)
    }

    /// Names of every block section present in the store.
    #[wasm_bindgen(js_name = blockNames)]
    pub fn block_names(&self) -> Vec<String> {
        self.sequence.block_names().map(str::to_string).collect()
    }
}

impl TrajectoryReader for MrecReader {
    fn build_index(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> std::io::Result<Option<molrs::core::Frame>> {
        self.sequence
            .frame(index as u64)
            .map_err(std::io::Error::other)
    }

    fn len(&mut self) -> std::io::Result<usize> {
        Ok(self.sequence.steps().len())
    }
}

// ---------------------------------------------------------------------------
// Host-served store
// ---------------------------------------------------------------------------

/// A read-only Zarr store whose keys are served by synchronous JS callbacks.
///
/// wasm32 has one thread, so the `Send`/`Sync` the store traits ask for are
/// vacuous here; the `unsafe impl`s below say exactly that and nothing more.
struct HostStorage {
    host: JsValue,
    get: js_sys::Function,
    get_range: Option<js_sys::Function>,
    size: js_sys::Function,
    list: js_sys::Function,
}

// SAFETY: wasm32-unknown-unknown is single-threaded; no JsValue ever crosses
// a thread boundary because there is none.
unsafe impl Send for HostStorage {}
unsafe impl Sync for HostStorage {}

fn storage_err(context: &str, e: JsValue) -> StorageError {
    StorageError::Other(format!(
        "{context}: {}",
        e.as_string().unwrap_or_else(|| format!("{e:?}"))
    ))
}

impl HostStorage {
    fn new(host: JsValue) -> Result<Self, JsValue> {
        let method = |name: &str| -> Result<Option<js_sys::Function>, JsValue> {
            let value = js_sys::Reflect::get(&host, &JsValue::from_str(name))?;
            if value.is_undefined() || value.is_null() {
                return Ok(None);
            }
            value.dyn_into::<js_sys::Function>().map(Some).map_err(|_| {
                JsValue::from_str(&format!("store host property {name:?} is not a function"))
            })
        };
        let required = |name: &str| -> Result<js_sys::Function, JsValue> {
            method(name)?.ok_or_else(|| JsValue::from_str(&format!("store host lacks {name}(…)")))
        };
        Ok(Self {
            get: required("get")?,
            get_range: method("getRange")?,
            size: required("size")?,
            list: required("list")?,
            host,
        })
    }

    fn bytes_of(value: JsValue) -> Option<Bytes> {
        if value.is_undefined() || value.is_null() {
            return None;
        }
        Some(js_sys::Uint8Array::new(&value).to_vec().into())
    }

    fn call_get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        let value = self
            .get
            .call1(&self.host, &JsValue::from_str(key.as_str()))
            .map_err(|e| storage_err("get", e))?;
        Ok(Self::bytes_of(value))
    }

    fn call_size(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        let value = self
            .size
            .call1(&self.host, &JsValue::from_str(key.as_str()))
            .map_err(|e| storage_err("size", e))?;
        Ok(value.as_f64().map(|size| size as u64))
    }

    fn call_list(&self, prefix: &str) -> Result<Vec<String>, StorageError> {
        let value = self
            .list
            .call1(&self.host, &JsValue::from_str(prefix))
            .map_err(|e| storage_err("list", e))?;
        let array = js_sys::Array::from(&value);
        Ok(array.iter().filter_map(|item| item.as_string()).collect())
    }

    fn range(&self, key: &StoreKey, range: &ByteRange) -> Result<MaybeBytes, StorageError> {
        match &self.get_range {
            Some(get_range) => {
                let (offset, length) = match range {
                    ByteRange::FromStart(offset, Some(length)) => (*offset as f64, *length as f64),
                    ByteRange::FromStart(offset, None) => (*offset as f64, -1.0),
                    ByteRange::Suffix(length) => {
                        let Some(size) = self.call_size(key)? else {
                            return Ok(None);
                        };
                        (size.saturating_sub(*length) as f64, *length as f64)
                    }
                };
                let value = get_range
                    .call3(
                        &self.host,
                        &JsValue::from_str(key.as_str()),
                        &JsValue::from_f64(offset),
                        &JsValue::from_f64(length),
                    )
                    .map_err(|e| storage_err("getRange", e))?;
                Ok(Self::bytes_of(value))
            }
            None => {
                let Some(whole) = self.call_get(key)? else {
                    return Ok(None);
                };
                let slice = range.to_range_usize(whole.len() as u64);
                Ok(Some(whole.slice(slice)))
            }
        }
    }
}

impl ReadableStorageTraits for HostStorage {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.call_get(key)
    }

    fn get_partial(
        &self,
        key: &StoreKey,
        byte_range: ByteRange,
    ) -> Result<MaybeBytes, StorageError> {
        self.range(key, &byte_range)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        let mut out: Vec<Result<Bytes, StorageError>> = Vec::new();
        for range in byte_ranges {
            match self.range(key, &range)? {
                Some(bytes) => out.push(Ok(bytes)),
                None => return Ok(None),
            }
        }
        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.call_size(key)
    }

    fn supports_get_partial(&self) -> bool {
        self.get_range.is_some()
    }
}

impl ListableStorageTraits for HostStorage {
    fn list(&self) -> Result<StoreKeys, StorageError> {
        self.list_prefix(&StorePrefix::root())
    }

    fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
        let mut keys: Vec<StoreKey> = self
            .call_list(prefix.as_str())?
            .into_iter()
            .filter_map(|name| StoreKey::new(name).ok())
            .collect();
        keys.sort();
        Ok(keys)
    }

    fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
        let mut keys = Vec::new();
        let mut prefixes = std::collections::BTreeSet::new();
        for key in self.list_prefix(prefix)? {
            let rest = key
                .as_str()
                .strip_prefix(prefix.as_str())
                .unwrap_or(key.as_str());
            match rest.find('/') {
                Some(slash) => {
                    let child = format!("{}{}", prefix.as_str(), &rest[..=slash]);
                    if let Ok(child) = StorePrefix::new(child) {
                        prefixes.insert(child);
                    }
                }
                None => keys.push(key),
            }
        }
        Ok(StoreKeysPrefixes::new(keys, prefixes.into_iter().collect()))
    }

    fn size(&self) -> Result<u64, StorageError> {
        self.size_prefix(&StorePrefix::root())
    }

    fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
        let mut total = 0u64;
        for key in self.list_prefix(prefix)? {
            total += self.call_size(&key)?.unwrap_or(0);
        }
        Ok(total)
    }
}

/// Load a `Map<path, Uint8Array>` of a record's files into an in-memory store.
fn memory_storage_from_files(files: &js_sys::Map) -> Result<ReadableListableStorage, JsValue> {
    let store = Arc::new(MemoryStore::new());
    for key_res in files.keys() {
        let key = key_res.map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;
        let path = key
            .as_string()
            .ok_or_else(|| JsValue::from_str("Invalid path key"))?;
        let content_value = files.get(&key);
        let content = js_sys::Uint8Array::new(&content_value).to_vec();
        let storage_path = path.strip_prefix('/').unwrap_or(&path);
        let skey = StoreKey::new(storage_path).map_err(js_string_err)?;
        store.set(&skey, content.into()).map_err(js_string_err)?;
    }
    Ok(store as ReadableListableStorage)
}

/// Unpack a packed `*.mrec.zip` into an in-memory store.
///
/// Every entry of a packed store is *stored* (never deflated), so this is a
/// central-directory walk plus one copy per entry. A zip with compressed
/// entries is refused by name.
fn memory_storage_from_zip(bytes: &[u8]) -> Result<ReadableListableStorage, JsValue> {
    let cursor = std::io::Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor).map_err(js_string_err)?;
    let store = Arc::new(MemoryStore::new());
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(|e| {
            JsValue::from_str(&format!(
                "zip entry {index}: {e} (packed stores use stored entries only)"
            ))
        })?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_string();
        let mut content = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut content).map_err(js_string_err)?;
        let skey = StoreKey::new(name.trim_start_matches('/')).map_err(js_string_err)?;
        store.set(&skey, content.into()).map_err(js_string_err)?;
    }
    Ok(store as ReadableListableStorage)
}

/// A record read one section at a time — the WASM face of
/// `molrs::io::mrec::MrecFile`.
///
/// Opening lists nothing and decodes nothing; each method reads the one
/// section it names. A record's sections are independent, so asking which
/// exist never costs a read of any of them.
///
/// # Example (JavaScript)
///
/// ```js
/// const file = new MrecFile(new Uint8Array(await blob.arrayBuffer()));
/// if (file.sections().includes("frame")) {
///   const frame = file.frame();
/// }
/// ```
#[wasm_bindgen(js_name = MrecFile)]
pub struct MrecFile {
    inner: RsMrecFile,
}

#[wasm_bindgen(js_class = MrecFile)]
impl MrecFile {
    /// Open the record `source` — a packed `*.mrec.zip`'s bytes, a `Map` of
    /// the record's files, or a store host (see `MrecSource`).
    ///
    /// # Errors
    ///
    /// Throws if `source` is none of the three, a zip holds compressed
    /// entries, or a host lacks `get` / `size` / `list`.
    #[wasm_bindgen(constructor)]
    pub fn new(source: &JsMrecSource) -> Result<MrecFile, JsValue> {
        Ok(MrecFile {
            inner: RsMrecFile::from_storage(open_store("new MrecFile", source)?),
        })
    }

    /// The record's top-level section names (`"meta"`, `"frame"`,
    /// `"trajectory"`, …), sorted, without decoding any of them.
    ///
    /// # Errors
    ///
    /// Throws when the store root cannot be listed.
    pub fn sections(&self) -> Result<Vec<String>, JsValue> {
        self.inner.sections().map_err(js_string_err)
    }

    /// The `frame` section — the snapshot — or `undefined` when the record
    /// has none.
    ///
    /// # Errors
    ///
    /// Throws when the section fails to decode.
    pub fn frame(&self) -> Result<Option<Frame>, JsValue> {
        self.inner
            .frame()
            .map_err(js_string_err)?
            .map(Frame::from_rs)
            .transpose()
    }

    /// The `system` section — the topology — or `undefined` when the record
    /// has none.
    ///
    /// # Errors
    ///
    /// Throws when the section fails to decode.
    pub fn system(&self) -> Result<Option<Frame>, JsValue> {
        self.inner
            .system()
            .map_err(js_string_err)?
            .map(Frame::from_rs)
            .transpose()
    }

    /// The record's `meta` document as a JSON object; `{}` when the store
    /// has none.
    ///
    /// # Errors
    ///
    /// Throws when the `meta` group cannot be decoded.
    pub fn meta(&self) -> Result<JsValue, JsValue> {
        use serde::Serialize;
        self.inner
            .meta()
            .map_err(js_string_err)?
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(js_string_err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{quantize, quantum};
    use wasm_bindgen_test::*;

    /// A record whose `frame/atoms/x` and `trajectory/atoms/x` declare
    /// precision 1e-3 and were written natively with `numcodecs.shuffle` +
    /// `zstd` (C encoder). Regenerated by molrs's ignored test
    /// `regenerate_the_wasm_precision_fixture`.
    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/precision.mrec.zip");

    /// The values the fixture's `x` columns were presented with.
    const X: [[f64; 3]; 2] = [[0.123_456_789, -1.000_488, 7.3], [0.2, 1.75, -3.062_57]];

    fn rounded(values: &[f64]) -> Vec<f64> {
        let q = quantum(1e-3).unwrap();
        values.iter().map(|&x| quantize(x, q)).collect()
    }

    fn x_of(frame: &molrs::core::Frame) -> Vec<f64> {
        frame
            .get("atoms")
            .and_then(|atoms| atoms.get("x"))
            .and_then(|x| x.get::<f64>())
            .expect("atoms.x is f64")
            .iter()
            .copied()
            .collect()
    }

    fn zip_source() -> JsMrecSource {
        JsValue::from(js_sys::Uint8Array::from(FIXTURE)).unchecked_into()
    }

    /// zstd decodes on wasm32 through the pure-Rust plugin, with no C.
    #[wasm_bindgen_test]
    fn a_precision_trajectory_written_with_zstd_reads_back() {
        let reader = MrecReader::new(&zip_source()).unwrap();
        assert_eq!(reader.n_frames().unwrap(), X.len());
        for (index, values) in X.iter().enumerate() {
            let frame = reader.sequence.frame(index as u64).unwrap().unwrap();
            assert_eq!(x_of(&frame), rounded(values));
        }
        let frame = reader.read_frame(0).unwrap().unwrap();
        assert_eq!(
            frame.get("atoms").unwrap().precision("x").unwrap(),
            Some(1e-3)
        );
    }

    #[wasm_bindgen_test]
    fn a_precision_frame_written_with_zstd_reads_back() {
        let file = MrecFile::new(&zip_source()).unwrap();
        let frame = file.inner.frame().unwrap().unwrap();
        assert_eq!(x_of(&frame), rounded(&X[0]));
        assert_eq!(frame.get("atoms").unwrap().precision("x"), Some(1e-3));
        assert!(file.frame().unwrap().is_some());
        let sections = file.sections().unwrap();
        assert!(sections.iter().any(|name| name == "frame"), "{sections:?}");
    }

    /// A `Map` of the record's files opens the same record as its zip.
    #[wasm_bindgen_test]
    fn a_map_of_files_opens_the_same_record() {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(FIXTURE)).unwrap();
        let files = js_sys::Map::new();
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index).unwrap();
            if entry.is_dir() {
                continue;
            }
            let mut content = Vec::new();
            entry.read_to_end(&mut content).unwrap();
            files.set(
                &JsValue::from_str(entry.name()),
                &js_sys::Uint8Array::from(content.as_slice()).into(),
            );
        }
        let source: JsMrecSource = JsValue::from(files).unchecked_into();
        assert_eq!(
            MrecReader::new(&source).unwrap().n_frames().unwrap(),
            X.len()
        );
        assert!(MrecFile::new(&source).unwrap().frame().unwrap().is_some());
        let not_a_record: JsMrecSource = JsValue::from_f64(1.0).unchecked_into();
        assert!(MrecFile::new(&not_a_record).is_err());
    }
}
