//! WASM bindings for [`Frame`] -- the top-level hierarchical data container.
//!
//! A `Frame` holds a collection of named [`Block`]s (e.g., `"atoms"`,
//! `"bonds"`, `"angles"`) and an optional [`SimBox`](super::simbox::Box)
//! defining periodic boundary conditions.
//!
//! # Typical block layout
//!
//! | Block key   | Expected columns | Column types |
//! |-------------|------------------|--------------|
//! | `"atoms"`   | `atomic_number` (u64, the species every molrs consumer reads), `x`, `y`, `z` (F), optionally `element` (string, raw symbols), `mass`, `charge` (F) | u64, F, string |
//! | `"bonds"`   | `atomi`, `atomj` (u64), `bond_type` (u64), `bond_number` (u64) | u64 |
//! | `"angles"`  | `atomi`, `atomj`, `atomk` (u64) | u64 |
//!
//! # Example (JavaScript)
//!
//! ```js
//! const frame = new Frame();
//! const atoms = frame.createBlock("atoms");
//! atoms.set("atomic_number", new BigUint64Array([6n, 6n, 8n]));
//! atoms.set("x", xCoords); // Float64Array
//! atoms.set("y", yCoords);
//! atoms.set("z", zCoords);
//!
//! const bonds = frame.createBlock("bonds");
//! bonds.set("atomi", new BigUint64Array([0n, 1n]));
//! bonds.set("atomj", new BigUint64Array([1n, 2n]));
//! bonds.set("bond_type", bondTypes);     // BigUint64Array; 4 = aromatic
//! bonds.set("bond_number", bondNumbers); // BigUint64Array; localized 1/2/3
//!
//! frame.get("atoms").view("x"); // zero-copy Float64Array
//! ```

use serde::Serialize;
use wasm_bindgen::prelude::*;

use js_sys::{
    Array as JsArray, BigInt, BigInt64Array, BigUint64Array, Float64Array, Int32Array, Uint32Array,
};
use molrs::core::Block as RsBlock;
use molrs::core::MetaValue;
use molrs_ffi::{BlockRef, FrameRef};
use wasm_bindgen::JsCast;

use super::block::Block;
use super::js_err;

/// Hierarchical data container mapping string keys to typed [`Block`]s.
///
/// A `Frame` owns a set of named blocks (column stores) and an optional
/// simulation box ([`Box`](super::simbox::Box)). This is the
/// primary interchange type for molecular data in the WASM API.
///
/// # Conventions
///
/// - The `"atoms"` block should contain per-atom properties:
///   `atomic_number` (u64, 0..=118, 0 = wildcard — the one species field
///   molrs reads), `x`/`y`/`z` (F, coordinates in angstrom), and optionally
///   `element` (string, raw symbols nothing reads; convert with
///   `fillAtomicNumbers` / `fillSymbols`), `mass` (F, atomic mass units) and
///   `charge` (F, elementary charges).
/// - The `"bonds"` block should contain bond topology: `atomi`/`atomj` (u64,
///   zero-based atom indices), `bond_type` (u64: 1 single, 2 double, 3 triple,
///   4 aromatic) and `bond_number` (u64: the localized Lewis/Kekulé integer).
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = new Frame();
/// const atoms = frame.createBlock("atoms");
/// atoms.set("x", xCoords);
/// ```
#[wasm_bindgen]
pub struct Frame {
    /// Paired frame id + shared store. All lifetime management lives in
    /// the shared `molrs_ffi::FrameRef` type so each binding layer (wasm,
    /// python, capi) has only the attribute plumbing to write.
    pub(crate) inner: FrameRef,
}

#[wasm_bindgen]
impl Frame {
    /// Create a new, empty `Frame` with no blocks and no simulation box.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = new Frame();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Frame {
            inner: FrameRef::default(),
        }
    }

    /// Create a new empty [`Block`] and register it under `key`.
    ///
    /// If a block with the same key already exists it is replaced.
    ///
    /// # Arguments
    ///
    /// * `key` - Block name (e.g., `"atoms"`, `"bonds"`)
    ///
    /// # Returns
    ///
    /// A mutable [`Block`] handle that can be used to add columns.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the underlying store operation fails
    /// (e.g., the frame has been dropped).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const atoms = frame.createBlock("atoms");
    /// atoms.set("x", xCoords);
    /// ```
    #[wasm_bindgen(js_name = createBlock)]
    pub fn create_block(&self, key: &str) -> Result<Block, JsValue> {
        let rs_block = RsBlock::new();
        self.inner
            .arena
            .borrow_mut()
            .set_block(self.inner.id, key, rs_block)
            .map_err(js_err)?;
        let handle = self
            .inner
            .arena
            .borrow()
            .get_block(self.inner.id, key)
            .map_err(js_err)?;
        Ok(Block {
            inner: BlockRef::new(self.inner.arena.clone(), handle),
        })
    }

    /// The [`Block`] named `key`: a live handle, so writes through it land
    /// in this frame. `frame.get("atoms")` is the block. `view("x")` is the
    /// zero-copy numeric column; `copy("x")` is the owned column.
    ///
    /// # Errors
    ///
    /// Throws if no block is named `key` (test with [`has`](Self::has)), or
    /// if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const x = frame.get("atoms").get("x");
    /// const nBonds = frame.has("bonds") ? frame.get("bonds").nRows : 0;
    /// ```
    #[wasm_bindgen(js_name = get)]
    pub fn get(&self, key: &str) -> Result<Block, JsValue> {
        if !self.has(key)? {
            return Err(JsValue::from_str(&format!("block '{key}' not found")));
        }
        let handle = self
            .inner
            .arena
            .borrow()
            .get_block(self.inner.id, key)
            .map_err(js_err)?;
        Ok(Block {
            inner: BlockRef::new(self.inner.arena.clone(), handle),
        })
    }

    /// Whether a block named `key` exists.
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped. A live frame that lacks `key`
    /// returns `false`.
    #[wasm_bindgen(js_name = has)]
    pub fn has(&self, key: &str) -> Result<bool, JsValue> {
        self.inner.has_block(key).map_err(js_err)
    }

    /// Store a deep copy of `block` under `key`, replacing any block there.
    ///
    /// The source is copied: later writes to `block` do not reach this frame,
    /// and `block` stays usable.
    ///
    /// # Errors
    ///
    /// Throws if either the source block or this frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.set("atoms", other.get("atoms"));
    /// ```
    #[wasm_bindgen(js_name = set)]
    pub fn set(&self, key: &str, block: &Block) -> Result<(), JsValue> {
        let rs_block = block.inner.clone_block().map_err(js_err)?;
        self.inner
            .arena
            .borrow_mut()
            .set_block(self.inner.id, key, rs_block)
            .map_err(js_err)
    }

    /// Delete the block named `key`.
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped or `key` does not exist.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.delBlock("bonds");
    /// ```
    #[wasm_bindgen(js_name = delBlock)]
    pub fn del_block(&self, key: &str) -> Result<(), JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .del_block(self.inner.id, key)
            .map_err(js_err)
    }

    /// Remove all blocks from this frame (but keep the frame alive).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has already been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.clear();
    /// ```
    #[wasm_bindgen(js_name = clear)]
    pub fn clear(&self) -> Result<(), JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .clear(self.inner.id)
            .map_err(js_err)
    }

    /// Rename a block from `old_key` to `new_key`.
    ///
    /// # Arguments
    ///
    /// * `old_key` - Current block name
    /// * `new_key` - New block name
    ///
    /// # Returns
    ///
    /// `true` if the block was found and renamed, `false` if `old_key`
    /// did not exist.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.renameBlock("atoms", "particles");
    /// ```
    #[wasm_bindgen(js_name = renameBlock)]
    pub fn rename_block(&self, old_key: &str, new_key: &str) -> Result<bool, JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .with_frame_mut(self.inner.id, |f| f.rename_block(old_key, new_key))
            .map_err(js_err)
    }

    /// The metadata value `name`, as the typed value it is stored as, or
    /// `undefined` when the frame has no such key.
    ///
    /// Frame meta is typed (`MetaValue`); each dtype reaches JS as its own
    /// type, never through a string:
    ///
    /// | dtype | JS |
    /// |-------|----|
    /// | `bool` | `boolean` |
    /// | `i32` / `u32` / `f64` | `number` |
    /// | `i64` / `u64` | `bigint` (exact) |
    /// | `string` | `string` |
    /// | `bool3` | `boolean[]` |
    /// | `i32x3` / `u32x3` | `Int32Array` / `Uint32Array` |
    /// | `i64x3` / `u64x3` | `BigInt64Array` / `BigUint64Array` |
    /// | `f64x3` / `f64x6` / `f64x9` | `Float64Array` of 3 / 6 / 9 |
    /// | `json` | the JSON value (object, array, `null`, …) |
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const energy = frame.getMeta("energy");        // number, or undefined
    /// const isBondTable = frame.getMeta("dump_local_label") === "BONDS";
    /// ```
    #[wasm_bindgen(js_name = getMeta)]
    pub fn get_meta(&self, name: &str) -> Result<JsMetaValueOrUndefined, JsValue> {
        self.with_frame(|frame| match frame.meta.get(name) {
            Some(value) => meta_to_js(value),
            None => Ok(JsValue::UNDEFINED),
        })
        .map(JsCast::unchecked_into)
    }

    /// The metadata keys on this frame, in insertion order.
    ///
    /// Includes every key whatever its dtype; read each value with
    /// [`getMeta`](Self::get_meta).
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped. A live frame with no metadata
    /// returns an empty list.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const keys = frame.metaKeys(); // e.g. ["energy", "config", "temp"]
    /// ```
    #[wasm_bindgen(js_name = metaKeys)]
    pub fn meta_keys(&self) -> Result<Vec<String>, JsValue> {
        self.inner
            .with(|frame| frame.meta.keys().cloned().collect::<Vec<String>>())
            .map_err(js_err)
    }

    /// Names of the blocks in this frame, in insertion order (the order the
    /// file or the caller added them).
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped. A live frame with no blocks
    /// returns an empty list.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const names = frame.keys(); // e.g. ["atoms", "bonds"]
    /// ```
    #[wasm_bindgen(js_name = keys)]
    pub fn keys(&self) -> Result<Vec<String>, JsValue> {
        self.inner
            .with(|frame| frame.keys().map(|k| k.to_string()).collect())
            .map_err(js_err)
    }

    /// Store `value` as metadata `name`, replacing any value there. The
    /// dtype is the JS value's own — the inverse of
    /// [`getMeta`](Self::get_meta):
    ///
    /// | JS | dtype |
    /// |----|-------|
    /// | `string` | `string` |
    /// | `boolean` | `bool` |
    /// | `number` | `f64` (a JS number is a double) |
    /// | `bigint` | `i64`, or `u64` above `i64`'s range |
    /// | `Float64Array` of 3 / 6 / 9 | `f64x3` / `f64x6` / `f64x9` |
    /// | `Int32Array` / `Uint32Array` of 3 | `i32x3` / `u32x3` |
    /// | `BigInt64Array` / `BigUint64Array` of 3 | `i64x3` / `u64x3` |
    /// | `boolean[]` of 3 | `bool3` |
    /// | any other array, a plain object, `null` | `json` |
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped, or `value` has no metadata
    /// dtype (`undefined`, a function, a typed array of another length or
    /// element type, a `bigint` beyond 64 bits, a value JSON cannot hold).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.setMeta("note", "run-42");
    /// frame.setMeta("energy", -3.14);
    /// frame.setMeta("step", 1000n);
    /// ```
    #[wasm_bindgen(js_name = setMeta)]
    pub fn set_meta(&self, name: &str, value: JsMetaValue) -> Result<(), JsValue> {
        let value = meta_from_js(name, &value)?;
        self.inner
            .arena
            .borrow_mut()
            .with_frame_meta_mut(self.inner.id, |meta| {
                meta.insert(name.to_string(), value);
            })
            .map_err(js_err)
    }

    /// Get the simulation box attached to this frame (if any).
    ///
    /// # Returns
    ///
    /// The [`Box`](super::simbox::Box) if one has been set, or `undefined`
    /// when this frame is live and has no box.
    ///
    /// # Errors
    ///
    /// Throws if the frame has been dropped. That is not the absent-box
    /// signal: a live frame with no box is `undefined`, and a dropped frame
    /// is an exception.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const box = frame.box;
    /// if (box) {
    ///   console.log("Volume:", box.volume());
    /// }
    /// ```
    #[wasm_bindgen(getter, js_name = box)]
    pub fn get_box(&self) -> Result<Option<super::simbox::Box>, JsValue> {
        self.inner
            .arena
            .borrow()
            .with_frame(self.inner.id, |f| {
                f.simbox
                    .as_ref()
                    .map(|s| super::simbox::Box { inner: s.clone() })
            })
            .map_err(js_err)
    }

    /// Attach or detach a simulation box.
    ///
    /// Pass a [`Box`](super::simbox::Box) to attach, or
    /// `undefined`/`null` to detach.
    ///
    /// # Arguments
    ///
    /// * `box` - The simulation box, or `undefined`/`null` to remove it
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.box = new Box(Cell.lengths([10, 10, 10]), [0, 0, 0], [true, true, true]);
    /// ```
    #[wasm_bindgen(setter, js_name = box)]
    pub fn set_box(&self, simbox: Option<super::simbox::Box>) -> Result<(), JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .set_frame_box(self.inner.id, simbox.map(|b| b.inner))
            .map_err(js_err)
    }

    /// Judge this frame against the canonical Frame schema.
    ///
    /// Throws a string describing every violation when the frame does not
    /// conform. The rules live in molrs (`Validator::canonical`); JavaScript
    /// must not re-check endpoint ranges or column dtypes by hand.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string with the full schema report when validation
    /// fails, or when the frame handle has been dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.validate();  // throws if bonds.atomi is out of range, …
    /// ```
    #[wasm_bindgen]
    pub fn validate(&self) -> Result<(), JsValue> {
        self.with_frame(|f| f.validate().map_err(|e| JsValue::from_str(&e.to_string())))
    }

    /// Explicitly release this frame and all its blocks from the store.
    ///
    /// After calling `drop()`, any subsequent operations on this frame
    /// or its blocks will throw. This is optional -- the frame will also
    /// be released when garbage-collected by the JS engine.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the frame was already dropped.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// frame.drop();
    /// // frame.clear() would now throw
    /// ```
    #[wasm_bindgen(js_name = drop)]
    pub fn drop_frame(&self) -> Result<(), JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .drop_frame(self.inner.id)
            .map_err(js_err)
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(typescript_custom_section)]
const META_VALUE_TYPE: &'static str = r#"
/**
 * A frame metadata value, as `Frame.getMeta` returns it and `Frame.setMeta`
 * takes it: each metadata dtype as its own JS type (see `Frame.getMeta`).
 */
export type MetaValue =
    | boolean | number | bigint | string
    | boolean[] | Int32Array | Uint32Array | BigInt64Array | BigUint64Array | Float64Array
    | null | object;
"#;

#[wasm_bindgen]
extern "C" {
    /// A metadata value: see the `MetaValue` TypeScript type.
    #[wasm_bindgen(typescript_type = "MetaValue")]
    pub type JsMetaValue;

    /// A metadata value, or `undefined` for a missing key.
    #[wasm_bindgen(typescript_type = "MetaValue | undefined")]
    pub type JsMetaValueOrUndefined;
}

/// `value` as the JS type of its dtype (the table on `Frame.getMeta`).
fn meta_to_js(value: &MetaValue) -> Result<JsValue, JsValue> {
    Ok(match value {
        MetaValue::Bool(v) => JsValue::from_bool(*v),
        MetaValue::I32(v) => JsValue::from(*v),
        MetaValue::U32(v) => JsValue::from(*v),
        MetaValue::F64(v) => JsValue::from_f64(*v),
        MetaValue::I64(v) => BigInt::from(*v).into(),
        MetaValue::U64(v) => BigInt::from(*v).into(),
        MetaValue::String(v) => JsValue::from_str(v),
        MetaValue::Bool3(v) => v
            .iter()
            .map(|&b| JsValue::from_bool(b))
            .collect::<JsArray>()
            .into(),
        MetaValue::I32x3(v) => Int32Array::from(&v[..]).into(),
        MetaValue::U32x3(v) => Uint32Array::from(&v[..]).into(),
        MetaValue::I64x3(v) => BigInt64Array::from(&v[..]).into(),
        MetaValue::U64x3(v) => BigUint64Array::from(&v[..]).into(),
        MetaValue::F64x3(v) => Float64Array::from(&v[..]).into(),
        MetaValue::F64x6(v) => Float64Array::from(&v[..]).into(),
        MetaValue::F64x9(v) => Float64Array::from(&v[..]).into(),
        MetaValue::Json(v) => v
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&format!("getMeta: {e}")))?,
    })
}

/// The metadata value a JS value is (the table on `Frame.setMeta`).
fn meta_from_js(name: &str, value: &JsValue) -> Result<MetaValue, JsValue> {
    let refuse = |what: &str| {
        JsValue::from_str(&format!(
            "setMeta('{name}'): {what} has no metadata dtype; pass a string, boolean, \
             number, bigint, a typed array of 3 (Float64Array also 6 or 9), \
             boolean[3], or a JSON object/array"
        ))
    };
    fn fixed<T: Copy, const N: usize>(values: Vec<T>) -> Option<[T; N]> {
        values.try_into().ok()
    }
    if let Some(v) = value.as_string() {
        return Ok(MetaValue::String(v));
    }
    if let Some(v) = value.as_bool() {
        return Ok(MetaValue::Bool(v));
    }
    if let Some(v) = value.as_f64() {
        return Ok(MetaValue::F64(v));
    }
    if value.is_bigint() {
        let big: &BigInt = value.unchecked_ref();
        if let Ok(v) = i64::try_from(big.clone()) {
            return Ok(MetaValue::I64(v));
        }
        return u64::try_from(big.clone())
            .map(MetaValue::U64)
            .map_err(|_| refuse("a bigint beyond 64 bits"));
    }
    if let Some(a) = value.dyn_ref::<Float64Array>() {
        let v = a.to_vec();
        return match v.len() {
            3 => Ok(MetaValue::F64x3(fixed(v).expect("len 3"))),
            6 => Ok(MetaValue::F64x6(fixed(v).expect("len 6"))),
            9 => Ok(MetaValue::F64x9(fixed(v).expect("len 9"))),
            n => Err(refuse(&format!("a Float64Array of {n}"))),
        };
    }
    macro_rules! triple {
        ($js:ty, $variant:ident) => {
            if let Some(a) = value.dyn_ref::<$js>() {
                return fixed(a.to_vec())
                    .map(MetaValue::$variant)
                    .ok_or_else(|| refuse(&format!("a {} of {}", stringify!($js), a.length())));
            }
        };
    }
    triple!(Int32Array, I32x3);
    triple!(Uint32Array, U32x3);
    triple!(BigInt64Array, I64x3);
    triple!(BigUint64Array, U64x3);
    if value.is_undefined() || value.is_function() || value.is_symbol() {
        return Err(refuse(&value.js_typeof().as_string().unwrap_or_default()));
    }
    if js_sys::ArrayBuffer::is_view(value) {
        return Err(refuse("this typed array"));
    }
    if let Some(a) = value.dyn_ref::<JsArray>()
        && a.length() == 3
        && a.iter().all(|v| v.as_bool().is_some())
    {
        let flags: Vec<bool> = a.iter().filter_map(|v| v.as_bool()).collect();
        return Ok(MetaValue::Bool3(fixed(flags).expect("len 3")));
    }
    serde_wasm_bindgen::from_value(value.clone())
        .map(MetaValue::Json)
        .map_err(|e| refuse(&format!("this value ({e})")))
}

/// Internal helpers (not exposed to JS).
impl Frame {
    pub(crate) fn from_rs(rs_frame: molrs::core::Frame) -> Result<Self, JsValue> {
        let arena = molrs_ffi::FrameArenaCell::default();
        let id = arena.borrow_mut().frame_new();
        arena.borrow_mut().set_frame(id, rs_frame).map_err(js_err)?;
        Ok(Frame {
            inner: FrameRef::new(arena, id),
        })
    }

    /// Borrow the inner core frame for the duration of a closure.
    ///
    /// Zero-copy: no deep clone. The closure runs while the FFI store is
    /// immutably borrowed, so it must not attempt to mutate the store.
    pub(crate) fn with_frame<R>(
        &self,
        f: impl FnOnce(&molrs::core::Frame) -> Result<R, JsValue>,
    ) -> Result<R, JsValue> {
        self.inner
            .arena
            .borrow()
            .with_frame(self.inner.id, f)
            .map_err(js_err)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_frame_lifecycle() {
        let frame = Frame::new();
        assert!(frame.clear().is_ok());
        frame.drop_frame().unwrap();
        assert!(frame.clear().is_err());
    }

    /// The JS error text. `unwrap_err` needs the success type to be `Debug`,
    /// and [`Block`] is not.
    fn js_error<T>(result: Result<T, JsValue>) -> String {
        match result {
            Ok(_) => panic!("expected an error"),
            Err(err) => err.as_string().expect("error string"),
        }
    }

    /// Helper: build a wrapped `Frame` with two typed meta entries.
    fn frame_with_meta() -> Frame {
        use molrs::core::MetaValue;
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame
            .meta
            .insert("energy".to_string(), MetaValue::F64(-1.23));
        rs_frame
            .meta
            .insert("config".to_string(), MetaValue::String("trans".into()));
        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn get_meta_returns_the_stored_dtype() {
        let frame = frame_with_meta();
        assert_eq!(frame.get_meta("energy").unwrap().as_f64(), Some(-1.23));
        let config: JsValue = frame.get_meta("config").unwrap().into();
        assert_eq!(config.as_string().as_deref(), Some("trans"));
        assert!(JsValue::from(frame.get_meta("missing").unwrap()).is_undefined());
    }

    #[wasm_bindgen_test]
    fn set_meta_round_trips_every_js_type() {
        use wasm_bindgen::JsCast;
        let frame = Frame::new();
        let cases: Vec<(&str, JsValue, &str)> = vec![
            ("s", JsValue::from_str("run-42"), "string"),
            ("b", JsValue::TRUE, "bool"),
            ("f", JsValue::from_f64(2.5), "f64"),
            ("i", js_sys::BigInt::from(-7_i64).into(), "i64"),
            ("u", js_sys::BigInt::from(u64::MAX).into(), "u64"),
            (
                "v3",
                Float64Array::from(&[1.0, 2.0, 3.0][..]).into(),
                "f64x3",
            ),
            ("v9", Float64Array::from(&[0.0; 9][..]).into(), "f64x9"),
            ("i3", Int32Array::from(&[1, -2, 3][..]).into(), "i32x3"),
            ("j", js_sys::Object::new().into(), "json"),
        ];
        for (key, value, dtype) in &cases {
            frame.set_meta(key, value.clone().unchecked_into()).unwrap();
            let stored = frame
                .with_frame(|f| Ok(f.meta.get(key).unwrap().dtype()))
                .unwrap();
            assert_eq!(stored, *dtype, "{key}");
        }
        let i: JsValue = frame.get_meta("i").unwrap().into();
        assert_eq!(
            i64::try_from(i.unchecked_into::<js_sys::BigInt>()).unwrap(),
            -7
        );
        let v3: Float64Array = JsValue::from(frame.get_meta("v3").unwrap()).unchecked_into();
        assert_eq!(v3.to_vec(), vec![1.0, 2.0, 3.0]);
        for bad in [
            JsValue::UNDEFINED,
            Float64Array::from(&[1.0, 2.0][..]).into(),
            Int32Array::from(&[1, 2][..]).into(),
        ] {
            assert!(frame.set_meta("bad", bad.unchecked_into()).is_err());
        }
    }

    #[wasm_bindgen_test]
    fn get_returns_a_live_block_and_throws_when_missing() {
        use wasm_bindgen::JsCast;
        let frame = Frame::new();
        assert!(!frame.has("atoms").unwrap());
        let e = frame.get("atoms").err().expect("missing block throws");
        assert!(e.as_string().unwrap().contains("'atoms' not found"));

        frame.create_block("atoms").unwrap();
        assert!(frame.has("atoms").unwrap());
        let mut atoms = frame.get("atoms").unwrap();
        let x = js_sys::Float64Array::from(&[1.0, 2.0][..]);
        atoms
            .set("x", JsValue::from(x).unchecked_into(), None)
            .unwrap();
        // A second handle sees the write: `get` is not a copy.
        assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 2);
    }

    #[wasm_bindgen_test]
    fn set_copies_and_leaves_the_source_usable() {
        use wasm_bindgen::JsCast;
        let src = Frame::new();
        let mut atoms = src.create_block("atoms").unwrap();
        let x = js_sys::Float64Array::from(&[1.0][..]);
        atoms
            .set("x", JsValue::from(x).unchecked_into(), None)
            .unwrap();

        let dst = Frame::new();
        dst.set("atoms", &atoms).unwrap();
        let y = js_sys::Float64Array::from(&[2.0][..]);
        atoms
            .set("y", JsValue::from(y).unchecked_into(), None)
            .unwrap();
        assert!(!dst.get("atoms").unwrap().has("y").unwrap());
        assert_eq!(dst.keys().unwrap(), vec!["atoms".to_string()]);

        dst.del_block("atoms").unwrap();
        assert!(!dst.has("atoms").unwrap());
        assert!(dst.del_block("atoms").is_err());
    }

    #[wasm_bindgen_test]
    fn meta_keys_follow_insertion_order() {
        let frame = frame_with_meta();
        assert_eq!(
            frame.meta_keys().unwrap(),
            vec!["energy".to_string(), "config".to_string()]
        );
    }

    #[wasm_bindgen_test]
    fn a_dropped_frame_is_an_error_on_every_probe() {
        let frame = Frame::new();
        frame.create_block("atoms").unwrap();
        let block = frame.get("atoms").unwrap();
        frame.drop_frame().unwrap();
        for text in [
            js_error(frame.has("atoms")),
            js_error(frame.get("atoms")),
            js_error(frame.keys()),
            js_error(frame.meta_keys()),
            js_error(block.has("x")),
            js_error(frame.get_box()),
        ] {
            assert!(
                text.contains("Invalid frame") || text.contains("Invalid block"),
                "{text}"
            );
            assert!(!text.contains("not found"), "{text}");
        }
    }
}
