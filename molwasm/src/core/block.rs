//! WASM bindings for [`Block`] -- typed columnar data container.
//!
//! A `Block` stores named columns, each backed by a homogeneously typed
//! array. All columns within a block share the same row count (`nRows`).
//!
//! # Column access API
//!
//! The column's dtype picks the JS array type; no method names a dtype.
//!
//! | Method | JS signature | Semantics |
//! |--------|--------------|-----------|
//! | `view` | `(key: string) -> NumericColumn` | Zero-copy typed-array view of a numeric column. The primary immediate read |
//! | `copy` | `(key: string, fallback?: Column) -> Column` | Owned copy of every dtype, including bool, string, and complex; an absent key returns `fallback` or throws |
//! | `set` | `(key: string, data: Column, shape?: number[])` | Insert or replace; dtype inferred from `data` |
//! | `has` | `(key: string) -> boolean` | Presence; throws if the handle is dead |
//! | `dtype` | `(key: string) -> DType` | Dtype name; missing throws |
//! | `shape` | `(key: string) -> number[]` | Column shape (`[nRows]`, `[nRows, 3]`, ...); missing throws |
//! | `keys` | `() -> string[]` | Column names in insertion order |
//! | `nRows` | getter `number` | Shared row count (`0` when empty) |
//!
//! # dtype <-> JS type
//!
//! The names are core `DType::name()`'s, the vocabulary every binding and the
//! Frame schema share (`float` is `f64`, `int` is `i32`, `uint` is `u64`).
//!
//! | `dtype(key)` | `copy` / `view` returns | `set` infers it from |
//! |--------------|------------------------|----------------------|
//! | `"f64"` | `Float64Array` | `Float64Array` |
//! | `"i8"` / `"i16"` / `"i32"` / `"i64"` | `Int8Array` / `Int16Array` / `Int32Array` / `BigInt64Array` | same |
//! | `"u8"` / `"u16"` / `"u32"` / `"u64"` | `Uint8Array` / `Uint16Array` / `Uint32Array` / `BigUint64Array` | same |
//! | `"bool"` | `boolean[]` (`copy`; `view` throws) | `boolean[]` |
//! | `"string"` | `string[]` (`copy`; `view` throws) | `string[]` (also the empty `[]`) |
//! | `"c64"` / `"c128"` | `{ real, imag, shape, dtype }` (`copy`; `view` throws) | never |
//!
//! Numeric results are the typed array itself, with `shape` and `dtype`
//! properties attached, so `arr[i]` still works. Complex values are not
//! interleaved: `real` and `imag` each have one entry per element.
//! `bool` is `boolean[]` rather than `Uint8Array` so a `copy` -> `set`
//! round trip keeps the dtype (a `Uint8Array` would come back as `u8`).
//! `Float32Array` is refused on `set`: the store keeps every real float as `float` (`f64`).
//!
//! # Memory safety note
//!
//! `view` returns a typed array backed by WASM linear memory. It becomes
//! **invalid** (detached, length 0) as soon as WASM memory grows, which any
//! allocation may cause, and it dangles once the column is replaced or the
//! block is freed. Use it immediately and do not keep it; use `copy` for data
//! you hold on to. Writes through a view land in the column in place.

use js_sys::{
    Array as JsArray, BigInt64Array, BigUint64Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Uint8Array, Uint8ClampedArray, Uint16Array, Uint32Array,
};
use ndarray::{ArrayD, IxDyn};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use molrs::core::{Block as RsBlock, BlockDtype, Column};
use molrs_ffi::BlockRef;

use super::js_err;

#[wasm_bindgen(typescript_custom_section)]
const COLUMN_TYPES: &'static str = r#"
/**
 * Column dtype as `Block.dtype` reports it: molrs core's dtype name, the one
 * the Frame schema, a `.mrec` store and every binding use — molrec's
 * concrete-width names (`f64` is a `Float64Array`, `u64` a `BigUint64Array`).
 */
export type DType =
    | "f64" | "i8" | "i16" | "i32" | "i64"
    | "u8" | "u16" | "u32" | "u64"
    | "bool" | "string" | "c64" | "c128";

/** A numeric column as a typed array, in the column's own dtype. */
export type NumericColumn =
    | Float64Array | Int8Array | Int16Array | Int32Array | BigInt64Array
    | Uint8Array | Uint16Array | Uint32Array | BigUint64Array;

/** `copy` of a complex column. `real` and `imag` are not interleaved. */
export type ComplexColumn = {
    real: Float32Array | Float64Array;
    imag: Float32Array | Float64Array;
    shape: number[];
    dtype: "c64" | "c128";
};

/**
 * Any column value `Block.copy` returns.
 * `Block.set` accepts the numeric, boolean, and string forms.
 */
export type Column = NumericColumn | boolean[] | string[] | ComplexColumn;
"#;

#[wasm_bindgen]
extern "C" {
    /// A column value: see the `Column` TypeScript type.
    #[wasm_bindgen(typescript_type = "Column")]
    pub type JsColumn;

    /// A zero-copy numeric column view: see `NumericColumn`.
    #[wasm_bindgen(typescript_type = "NumericColumn")]
    pub type JsNumericColumn;

    /// A dtype name: see `DType`.
    #[wasm_bindgen(typescript_type = "DType")]
    pub type JsDType;

    /// An array shape: a plain `number[]`.
    #[wasm_bindgen(typescript_type = "number[]")]
    pub type JsShape;
}

// ---------------------------------------------------------------------------
// Block
// ---------------------------------------------------------------------------

/// Column-oriented data store with typed arrays.
///
/// Each column is identified by a string key and has a fixed dtype. All
/// columns in a block have the same number of rows. See the module docs for
/// the dtype <-> JS type table.
///
/// # Example (JavaScript)
///
/// ```js
/// const atoms = frame.createBlock("atoms");
/// atoms.set("x", new Float64Array([0, 1, 2]));
/// atoms.set("element", ["C", "C", "O"]);
/// atoms.set("id", new BigUint64Array([0n, 1n, 2n]));
/// atoms.nRows;            // 3
/// atoms.dtype("id");      // "u64"
/// const x = atoms.view("x"); // zero-copy Float64Array; copy("x") to keep it
/// ```
#[wasm_bindgen]
pub struct Block {
    /// Paired handle + shared store. All lifetime management lives in the
    /// shared `molrs_ffi::BlockRef` type (single definition across every
    /// binding — wasm, python, capi).
    pub(crate) inner: BlockRef,
}

#[wasm_bindgen]
impl Block {
    /// Create a new, standalone empty `Block`.
    ///
    /// The block is backed by its own temporary store. Prefer
    /// [`Frame.createBlock()`](crate::Frame::create_block) to create
    /// blocks that are immediately attached to a frame.
    ///
    /// # Errors
    ///
    /// Throws if the internal store allocation fails.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<Block, JsValue> {
        Block::from_rs(RsBlock::new())
    }

    // ---- block metadata ----

    /// Number of columns in this block.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    #[wasm_bindgen(js_name = len)]
    pub fn len(&self) -> Result<usize, JsValue> {
        self.with(|b| b.len())
    }

    /// Whether this block has zero columns.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }

    /// Number of rows, shared across all columns; `0` for a block with no
    /// columns.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const n = atoms.nRows; // e.g. 100
    /// ```
    #[wasm_bindgen(getter, js_name = nRows)]
    pub fn n_rows(&self) -> Result<usize, JsValue> {
        self.with(|b| b.n_rows().unwrap_or(0))
    }

    /// The declared N-D structural shape (`[Nx, Ny, Nz]` for a volumetric
    /// grid), or `undefined` for a plain row table.
    ///
    /// Set with [`setShape`](Self::set_shape). Its product always equals
    /// `nRows`. Distinct from [`shape`](Self::shape), which is the shape of
    /// one column.
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// grid.structuralShape;  // [32, 32, 32]
    /// atoms.structuralShape; // undefined
    /// ```
    #[wasm_bindgen(getter, js_name = structuralShape)]
    pub fn structural_shape(&self) -> Result<Option<JsShape>, JsValue> {
        self.with(|b| b.structural_shape().map(shape_to_js))
    }

    /// Declare this block as N-dimensional with the given `shape`.
    ///
    /// `shape`'s product must equal the block's current `nRows` when the
    /// block has columns. Pass `[]` to clear it and revert to plain
    /// row-table semantics.
    ///
    /// This does **not** reshape column storage: columns remain row-major
    /// buffers of `product(shape)` rows. The shape is structural metadata
    /// consumers (e.g. a volumetric renderer) use to unflatten the row index.
    ///
    /// # Errors
    ///
    /// Throws if `shape` holds anything but non-negative integers, if its
    /// product does not match `nRows`, or if the handle has been invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const grid = frame.createBlock("grid");
    /// grid.set("electron_density", values); // values.length === 32*32*32
    /// grid.setShape([32, 32, 32]);
    /// ```
    #[wasm_bindgen(js_name = setShape)]
    pub fn set_shape(&mut self, shape: JsShape) -> Result<(), JsValue> {
        let dims = shape_from_js(&shape)?;
        self.with_mut(|b| {
            b.set_shape(&dims)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    /// Column names in insertion order (the order the file or the caller
    /// wrote the columns).
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated.
    #[wasm_bindgen(js_name = keys)]
    pub fn keys(&self) -> Result<Vec<String>, JsValue> {
        self.with(|b| b.keys().map(str::to_owned).collect())
    }

    // ---- per-column metadata ----

    /// Whether column `key` exists (of any dtype).
    ///
    /// # Errors
    ///
    /// Throws if the block handle has been invalidated. A live block that
    /// lacks `key` returns `false`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// if (bonds.has("bond_type") && bonds.dtype("bond_type") === "u64") { … }
    /// ```
    #[wasm_bindgen(js_name = has)]
    pub fn has(&self, key: &str) -> Result<bool, JsValue> {
        self.with(|b| b.contains_key(key))
    }

    /// Dtype of column `key`; see the `DType` type for the names.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// atoms.dtype("x");       // "f64"
    /// atoms.dtype("element"); // "string"
    /// ```
    #[wasm_bindgen(js_name = dtype)]
    pub fn dtype(&self, key: &str) -> Result<JsDType, JsValue> {
        self.with_col(key, |col| {
            Ok(JsValue::from_str(col.dtype().name()).unchecked_into())
        })
    }

    /// Shape of column `key`: `[nRows]` for a per-row scalar, `[nRows, 3]`
    /// for a per-row vector, and so on.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// block.set("pos", flat, [n, 3]);
    /// block.shape("pos"); // [n, 3]
    /// ```
    #[wasm_bindgen(js_name = shape)]
    pub fn shape(&self, key: &str) -> Result<JsShape, JsValue> {
        self.with_col(key, |col| Ok(shape_to_js(col.shape())))
    }

    /// Validity mask of column `key`: one byte per row, `1` where the row
    /// holds a value and `0` where it is null.
    ///
    /// A null cell still has a filled value in the column itself (whatever
    /// the producer wrote there, typically `0` or `""`), so a consumer that
    /// must tell "no value" from "zero" reads this mask beside the column.
    /// Masks travel with the block through
    /// [`readWireFrame`](crate::io::frame_encoding::read_wire_frame).
    ///
    /// # Returns
    ///
    /// A `Uint8Array` of length `nRows` when at least one row of the column
    /// is null; `undefined` when the column is fully valid.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const fragId = atoms.copy("frag_id");
    /// const valid = atoms.validity("frag_id"); // undefined → no nulls
    /// const isNull = (i) => valid !== undefined && valid[i] === 0;
    /// ```
    #[wasm_bindgen(js_name = validity)]
    pub fn validity(&self, key: &str) -> Result<Option<Uint8Array>, JsValue> {
        self.with(|b| {
            if !b.contains_key(key) {
                return Err(missing_column(key));
            }
            Ok(b.validity(key).map(|mask| {
                let bytes: Vec<u8> = mask.iter().map(|&valid| u8::from(valid)).collect();
                Uint8Array::from(bytes.as_slice())
            }))
        })?
    }

    /// The declared row-reference target of column `key`: the block its
    /// values index (`"atoms"`, or `"/frame/atoms"` in another section of the
    /// record), as the store declared it (molrec `targets`).
    ///
    /// # Returns
    ///
    /// The target, or `undefined` when the column declares none.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, or if the handle has been
    /// invalidated.
    #[wasm_bindgen(js_name = target)]
    pub fn target(&self, key: &str) -> Result<Option<String>, JsValue> {
        self.with(|b| {
            if !b.contains_key(key) {
                return Err(missing_column(key));
            }
            Ok(b.target(key).map(str::to_string))
        })?
    }

    /// The declared precision of column `key`: the absolute tolerance its
    /// stored values were rounded to (molrec "declared precision").
    ///
    /// # Returns
    ///
    /// The precision, or `undefined` when the column declares none.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const p = atoms.precision("x"); // e.g. 0.001, or undefined
    /// ```
    #[wasm_bindgen(js_name = precision)]
    pub fn precision(&self, key: &str) -> Result<Option<f64>, JsValue> {
        self.with(|b| {
            if !b.contains_key(key) {
                return Err(missing_column(key));
            }
            Ok(b.precision(key))
        })?
    }

    /// Rename column `old_key` to `new_key`.
    ///
    /// # Errors
    ///
    /// Throws if `old_key` does not exist, if the moved column violates the
    /// Frame schema's spec for `new_key`, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// block.renameColumn("symbol", "element");
    /// ```
    #[wasm_bindgen(js_name = renameColumn)]
    pub fn rename_column(&mut self, old_key: &str, new_key: &str) -> Result<(), JsValue> {
        // A rename is a write into `new_key`, so the Frame schema checks the
        // moved column against that key's spec — the error surfaces here
        // rather than a wrong-typed column landing silently.
        self.with_mut(|b| {
            b.rename_column(old_key, new_key)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }

    // ---- column data ----

    /// Owned copy of column `key`. Numeric columns are the typed array
    /// itself, with `shape` and `dtype` properties. Bool is `boolean[]`,
    /// string is a flat row-major `string[]`, and complex is
    /// `{ real, imag, shape, dtype }` (`Float32Array` for `c64`,
    /// `Float64Array` for `c128`).
    ///
    /// This is the read to keep; [`view`](Self::view) is the zero-copy
    /// numeric read. An absent `key` returns `fallback` when one was passed
    /// (as given, not copied or checked), and throws otherwise.
    ///
    /// # Errors
    ///
    /// Throws if `key` is absent and no `fallback` was passed, or if the
    /// handle has been invalidated (a dead handle throws even with a
    /// `fallback`).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const x = atoms.copy("x"); // owned Float64Array
    /// const charge = atoms.copy("charge", new Float64Array(atoms.nRows));
    /// ```
    #[wasm_bindgen(js_name = copy)]
    pub fn copy(&self, key: &str, fallback: Option<JsColumn>) -> Result<JsColumn, JsValue> {
        // The fallback answers an absent column only: presence is read off
        // the block (a dead handle throws here), never off an error's text.
        let present = self.with(|b| b.get(key).is_some())?;
        match (present, fallback) {
            (false, Some(fallback)) => Ok(fallback),
            _ => Ok(self
                .with_col(key, |col| column_to_js(key, col))?
                .unchecked_into()),
        }
    }

    /// Zero-copy typed-array view of numeric column `key`, in the column's
    /// own dtype. Flat, row-major, with `shape` and `dtype` properties.
    /// This is the primary immediate read for numeric columns.
    ///
    /// **Warning**: the view is backed by WASM linear memory. It is
    /// invalidated (detached) whenever WASM memory grows — any allocation may
    /// do that — and dangles once the column is replaced or the block freed.
    /// Use it immediately; call [`copy`](Self::copy) for data you keep. Writes
    /// through the view modify the column in place.
    ///
    /// # Errors
    ///
    /// Throws if the column does not exist, if it is bool, string, or complex
    /// (those have no typed-array view — the error says to use `copy`), if
    /// its storage is not contiguous row-major, or if the handle has been
    /// invalidated.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const x = atoms.view("x"); // Float64Array over WASM memory
    /// for (let i = 0; i < x.length; i++) x[i] += 1.0; // in-place write
    /// ```
    #[wasm_bindgen(js_name = view)]
    pub fn view(&self, key: &str) -> Result<JsNumericColumn, JsValue> {
        self.with_col(key, |col| column_view(key, col).map(JsCast::unchecked_into))
    }

    /// Insert or replace column `key`, with the dtype inferred from `data`:
    /// each typed array stores as its own dtype (`Float64Array` -> `f64`,
    /// `Int32Array` -> `i32`, `BigUint64Array` -> `u64`, …), `string[]` ->
    /// `string`, `boolean[]` -> `bool`. An empty `[]` stores as `string`.
    ///
    /// `shape` (row-major) makes a multi-dimensional column, e.g. `[n, 3]`
    /// for per-row vectors; without it the column is 1-D.
    ///
    /// # Errors
    ///
    /// Throws if `data` is a `Float32Array` (the store keeps floats as
    /// `f64`), a `Uint8ClampedArray`, a plain `number[]` or any other value
    /// with no dtype of its own, or an `Array` mixing element types; if
    /// `shape`'s product differs from `data.length`; if the row count
    /// conflicts with existing columns; or if the Frame schema declares a
    /// different dtype for `key`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// atoms.set("x", new Float64Array([0, 1, 2]));
    /// atoms.set("pos", new Float64Array(9), [3, 3]);
    /// atoms.set("element", ["C", "C", "O"]);
    /// bonds.set("atomi", new BigUint64Array([0n, 1n]));
    /// ```
    #[wasm_bindgen(js_name = set)]
    pub fn set(
        &mut self,
        key: &str,
        data: JsColumn,
        shape: Option<JsShape>,
    ) -> Result<(), JsValue> {
        let dims = shape.as_ref().map(shape_from_js).transpose()?;
        let col = column_from_js(key, &data, dims.as_deref())?;
        self.with_mut(|b| {
            b.insert(key, col, None)
                .map_err(|e| JsValue::from_str(&e.to_string()))
        })
    }
}

impl Default for Block {
    fn default() -> Self {
        Self::new().expect("Block::new on fresh arena")
    }
}

// ---------------------------------------------------------------------------
// Column <-> JS conversion
// ---------------------------------------------------------------------------

fn missing_column(key: &str) -> JsValue {
    JsValue::from_str(&format!("column '{key}' not found"))
}

/// Attach the column's `shape` and `dtype` without wrapping the value,
/// so a typed array stays a typed array.
fn stamp(value: JsValue, shape: &[usize], dtype: &str) -> JsValue {
    let shape_js = shape_to_js(shape);
    let _ = js_sys::Reflect::set(&value, &JsValue::from_str("shape"), shape_js.as_ref());
    let _ = js_sys::Reflect::set(
        &value,
        &JsValue::from_str("dtype"),
        &JsValue::from_str(dtype),
    );
    value
}

/// The values of `arr` in logical row-major order: borrowed when the storage
/// already is standard-layout, collected otherwise.
fn row_major<T: Clone>(arr: &ArrayD<T>) -> std::borrow::Cow<'_, [T]> {
    match arr.as_slice() {
        Some(s) => std::borrow::Cow::Borrowed(s),
        None => std::borrow::Cow::Owned(arr.iter().cloned().collect()),
    }
}

/// Owned JS copy of `col` in its natural JS array type.
fn column_to_js(key: &str, col: &Column) -> Result<JsValue, JsValue> {
    let shape = col.shape().to_vec();
    let dtype = col.dtype().name();
    macro_rules! typed_copy {
        ($arr:expr, $js:ty) => {
            if let Some(arr) = $arr {
                let value: JsValue = <$js>::from(row_major(arr).as_ref()).into();
                return Ok(stamp(value, &shape, dtype));
            }
        };
    }
    typed_copy!(col.get::<f64>(), Float64Array);
    typed_copy!(col.get::<i8>(), Int8Array);
    typed_copy!(col.get::<i16>(), Int16Array);
    typed_copy!(col.get::<i32>(), Int32Array);
    typed_copy!(col.get::<i64>(), BigInt64Array);
    typed_copy!(col.get::<u8>(), Uint8Array);
    typed_copy!(col.get::<u16>(), Uint16Array);
    typed_copy!(col.get::<u32>(), Uint32Array);
    typed_copy!(col.get::<u64>(), BigUint64Array);
    if let Some(arr) = col.get::<bool>() {
        let value: JsValue = arr
            .iter()
            .map(|&v| JsValue::from_bool(v))
            .collect::<JsArray>()
            .into();
        return Ok(stamp(value, &shape, dtype));
    }
    if let Some(arr) = col.get::<String>() {
        let value: JsValue = arr
            .iter()
            .map(|s| JsValue::from_str(s))
            .collect::<JsArray>()
            .into();
        return Ok(stamp(value, &shape, dtype));
    }
    if let Some(arr) = col.get::<num_complex::Complex<f32>>() {
        return Ok(complex_to_js(
            arr,
            dtype,
            |z| z.re,
            |z| z.im,
            |re| Float32Array::from(re).into(),
        ));
    }
    if let Some(arr) = col.get::<num_complex::Complex<f64>>() {
        return Ok(complex_to_js(
            arr,
            dtype,
            |z| z.re,
            |z| z.im,
            |re| Float64Array::from(re).into(),
        ));
    }
    Err(JsValue::from_str(&format!(
        "column '{key}' is {dtype}: no JS copy for this dtype"
    )))
}

fn complex_to_js<T: Copy, U: Copy>(
    arr: &ArrayD<T>,
    dtype: &str,
    re: impl Fn(&T) -> U,
    im: impl Fn(&T) -> U,
    component: impl Fn(&[U]) -> JsValue,
) -> JsValue {
    let flat = row_major(arr);
    let mut real = Vec::with_capacity(flat.len());
    let mut imag = Vec::with_capacity(flat.len());
    for value in flat.iter() {
        real.push(re(value));
        imag.push(im(value));
    }
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("real"), &component(&real));
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("imag"), &component(&imag));
    let shape = shape_to_js(arr.shape());
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("shape"), shape.as_ref());
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("dtype"), &JsValue::from_str(dtype));
    obj.into()
}

/// Zero-copy typed-array view over `col`'s storage.
fn column_view(key: &str, col: &Column) -> Result<JsValue, JsValue> {
    let shape = col.shape().to_vec();
    let dtype = col.dtype().name();
    macro_rules! typed_view {
        ($arr:expr, $js:ty) => {
            if let Some(arr) = $arr {
                let slice = arr.as_slice().ok_or_else(|| {
                    JsValue::from_str(&format!(
                        "column '{key}' is not contiguous row-major; use copy()"
                    ))
                })?;
                // SAFETY: `slice` lives in WASM linear memory, owned by the
                // store behind this block. The view is only valid until the
                // next memory growth or until the column is replaced/freed;
                // that contract is documented on `Block.view` and is the JS
                // caller's to keep.
                let value: JsValue = unsafe { <$js>::view(slice) }.into();
                return Ok(stamp(value, &shape, dtype));
            }
        };
    }
    typed_view!(col.get::<f64>(), Float64Array);
    typed_view!(col.get::<i8>(), Int8Array);
    typed_view!(col.get::<i16>(), Int16Array);
    typed_view!(col.get::<i32>(), Int32Array);
    typed_view!(col.get::<i64>(), BigInt64Array);
    typed_view!(col.get::<u8>(), Uint8Array);
    typed_view!(col.get::<u16>(), Uint16Array);
    typed_view!(col.get::<u32>(), Uint32Array);
    typed_view!(col.get::<u64>(), BigUint64Array);
    Err(JsValue::from_str(&format!(
        "column '{key}' is {dtype}: only numeric columns have a typed-array view; use copy()"
    )))
}

/// Build a `Column` of `T` from `data`, shaped by `shape` (1-D when `None`).
fn shaped<T: BlockDtype>(
    key: &str,
    data: Vec<T>,
    shape: Option<&[usize]>,
) -> Result<Column, JsValue> {
    let n = data.len();
    let dims = shape.map_or_else(|| vec![n], <[usize]>::to_vec);
    let arr = ArrayD::from_shape_vec(IxDyn(&dims), data).map_err(|_| {
        JsValue::from_str(&format!(
            "column '{key}': shape {dims:?} does not hold {n} values"
        ))
    })?;
    Ok(Column::from(arr))
}

/// Infer the dtype of `data` from its constructor and build the column.
fn column_from_js(key: &str, data: &JsValue, shape: Option<&[usize]>) -> Result<Column, JsValue> {
    macro_rules! typed_set {
        ($js:ty) => {
            if let Some(arr) = data.dyn_ref::<$js>() {
                return shaped(key, arr.to_vec(), shape);
            }
        };
    }
    typed_set!(Float64Array);
    typed_set!(Int8Array);
    typed_set!(Int16Array);
    typed_set!(Int32Array);
    typed_set!(BigInt64Array);
    typed_set!(Uint8Array);
    typed_set!(Uint16Array);
    typed_set!(Uint32Array);
    typed_set!(BigUint64Array);
    if data.is_instance_of::<Float32Array>() {
        return Err(JsValue::from_str(&format!(
            "column '{key}': Float32Array is not a stored dtype; floats are stored as f64, pass a Float64Array"
        )));
    }
    if data.is_instance_of::<Uint8ClampedArray>() {
        return Err(JsValue::from_str(&format!(
            "column '{key}': Uint8ClampedArray is not a stored dtype; pass a Uint8Array"
        )));
    }
    if let Some(arr) = data.dyn_ref::<JsArray>() {
        return array_column(key, arr, shape);
    }
    Err(JsValue::from_str(&format!(
        "column '{key}': cannot infer a dtype from {}; pass a typed array, string[] or boolean[]",
        js_type_name(data)
    )))
}

/// A `string[]` or `boolean[]` column; the first element decides which.
fn array_column(key: &str, arr: &JsArray, shape: Option<&[usize]>) -> Result<Column, JsValue> {
    let first = arr.get(0);
    if arr.length() == 0 || first.is_string() {
        let values = arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_string()
                    .ok_or_else(|| mixed_array(key, i, "string", &v))
            })
            .collect::<Result<Vec<String>, JsValue>>()?;
        return shaped(key, values, shape);
    }
    if first.as_bool().is_some() {
        let values = arr
            .iter()
            .enumerate()
            .map(|(i, v)| {
                v.as_bool()
                    .ok_or_else(|| mixed_array(key, i, "boolean", &v))
            })
            .collect::<Result<Vec<bool>, JsValue>>()?;
        return shaped(key, values, shape);
    }
    Err(JsValue::from_str(&format!(
        "column '{key}': cannot infer a dtype from an Array of {}; numeric data must be a typed array (e.g. Float64Array)",
        js_type_name(&first)
    )))
}

fn mixed_array(key: &str, index: usize, expected: &str, got: &JsValue) -> JsValue {
    JsValue::from_str(&format!(
        "column '{key}': element {index} is {}, expected every element to be a {expected}",
        js_type_name(got)
    ))
}

/// `typeof value`, or the constructor name for objects.
fn js_type_name(value: &JsValue) -> String {
    if value.is_object()
        && let Some(obj) = value.dyn_ref::<js_sys::Object>()
    {
        return String::from(obj.constructor().name());
    }
    value.js_typeof().as_string().unwrap_or_default()
}

/// A `number[]` from a Rust shape.
fn shape_to_js(dims: &[usize]) -> JsShape {
    dims.iter()
        .map(|&d| JsValue::from_f64(d as f64))
        .collect::<JsArray>()
        .unchecked_into()
}

/// A Rust shape from a JS `number[]` (any array-like of non-negative
/// integers is accepted).
fn shape_from_js(shape: &JsShape) -> Result<Vec<usize>, JsValue> {
    JsArray::from(shape.as_ref())
        .iter()
        .map(|v| match v.as_f64() {
            Some(d) if d >= 0.0 && d.fract() == 0.0 && d <= u32::MAX as f64 => Ok(d as usize),
            _ => Err(JsValue::from_str(&format!(
                "shape entries must be non-negative integers, got {v:?}"
            ))),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl Block {
    fn with<R>(&self, f: impl FnOnce(&RsBlock) -> R) -> Result<R, JsValue> {
        self.inner
            .arena
            .borrow()
            .with_block(&self.inner.handle, f)
            .map_err(js_err)
    }

    fn with_mut<R>(
        &mut self,
        f: impl FnOnce(&mut RsBlock) -> Result<R, JsValue>,
    ) -> Result<R, JsValue> {
        self.inner
            .arena
            .borrow_mut()
            .with_block_mut(&mut self.inner.handle, f)
            .map_err(js_err)?
    }

    /// Run `f` on column `key`; a missing column throws.
    fn with_col<R>(
        &self,
        key: &str,
        f: impl FnOnce(&Column) -> Result<R, JsValue>,
    ) -> Result<R, JsValue> {
        self.with(|b| b.get(key).map_or_else(|| Err(missing_column(key)), f))?
    }

    /// A standalone block holding `block` (what a Rust-side reader such as
    /// `readCsvBlock` returns).
    pub(crate) fn from_rs(block: RsBlock) -> Result<Block, JsValue> {
        let arena = molrs_ffi::FrameArenaCell::default();
        let fid = arena.borrow_mut().frame_new();
        arena
            .borrow_mut()
            .set_block(fid, "temp", block)
            .map_err(js_err)?;
        let handle = arena.borrow().get_block(fid, "temp").map_err(js_err)?;
        Ok(Block {
            inner: BlockRef::new(arena, handle),
        })
    }

    /// Run `f` on the Rust block (what a Rust-side writer such as
    /// `writeCsvBlock` reads).
    pub(crate) fn with_rs<R>(&self, f: impl FnOnce(&RsBlock) -> R) -> Result<R, JsValue> {
        self.with(f)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::frame::Frame;
    use wasm_bindgen_test::*;

    fn block() -> Block {
        Frame::new().create_block("scratch").unwrap()
    }

    fn col(value: impl Into<JsValue>) -> JsColumn {
        value.into().unchecked_into()
    }

    fn shape_of(v: &JsShape) -> Vec<usize> {
        shape_from_js(v).unwrap()
    }

    fn err_text(e: JsValue) -> String {
        e.as_string().unwrap_or_default()
    }

    /// `set` a typed array, then check `dtype`, `copy` and `view` hand back the
    /// same values in the same typed-array type.
    macro_rules! round_trip {
        ($name:ident, $js:ty, $t:ty, $dtype:literal, [$($v:expr),*]) => {
            #[wasm_bindgen_test]
            fn $name() {
                let mut b = block();
                let values: Vec<$t> = vec![$($v),*];
                b.set("c", col(<$js>::from(values.as_slice())), None).unwrap();
                assert_eq!(b.dtype("c").unwrap().as_string().unwrap(), $dtype);
                assert_eq!(shape_of(&b.shape("c").unwrap()), vec![values.len()]);
                let got: JsValue = b.copy("c", None).unwrap().into();
                assert!(got.is_instance_of::<$js>(), "get returned the wrong type");
                assert_eq!(got.unchecked_into::<$js>().to_vec(), values);
                let view: JsValue = b.view("c").unwrap().into();
                assert!(view.is_instance_of::<$js>(), "view returned the wrong type");
                assert_eq!(view.unchecked_into::<$js>().to_vec(), values);
            }
        };
    }

    round_trip!(round_trip_f64, Float64Array, f64, "f64", [1.5, -2.0, 3.25]);
    round_trip!(round_trip_i8, Int8Array, i8, "i8", [-1, 0, 7]);
    round_trip!(round_trip_i16, Int16Array, i16, "i16", [-300, 0, 300]);
    round_trip!(round_trip_i32, Int32Array, i32, "i32", [1, -2, 3]);
    round_trip!(round_trip_i64, BigInt64Array, i64, "i64", [-5, 0, 1 << 40]);
    round_trip!(round_trip_u8, Uint8Array, u8, "u8", [0, 128, 255]);
    round_trip!(round_trip_u16, Uint16Array, u16, "u16", [0, 1, 65535]);
    round_trip!(
        round_trip_u32,
        Uint32Array,
        u32,
        "u32",
        [0, 1, 4_000_000_000]
    );
    round_trip!(round_trip_u64, BigUint64Array, u64, "u64", [0, 7, 1 << 50]);

    #[wasm_bindgen_test]
    fn round_trip_string() {
        let mut b = block();
        let values: JsArray = ["C", "C", "O"]
            .iter()
            .map(|s| JsValue::from_str(s))
            .collect();
        b.set("element", col(values), None).unwrap();
        assert_eq!(b.dtype("element").unwrap().as_string().unwrap(), "string");
        let got: JsArray = JsValue::from(b.copy("element", None).unwrap()).unchecked_into();
        let got: Vec<String> = got.iter().map(|v| v.as_string().unwrap()).collect();
        assert_eq!(got, vec!["C", "C", "O"]);
    }

    #[wasm_bindgen_test]
    fn round_trip_bool_keeps_dtype() {
        let mut b = block();
        let values: JsArray = [true, false, true]
            .iter()
            .map(|&v| JsValue::from_bool(v))
            .collect();
        b.set("flag", col(values), None).unwrap();
        assert_eq!(b.dtype("flag").unwrap().as_string().unwrap(), "bool");
        let got = b.copy("flag", None).unwrap();
        // boolean[] back, so feeding it to `set` again keeps `bool`.
        b.set("flag2", got, None).unwrap();
        assert_eq!(b.dtype("flag2").unwrap().as_string().unwrap(), "bool");
        let got: JsArray = JsValue::from(b.copy("flag2", None).unwrap()).unchecked_into();
        let got: Vec<bool> = got.iter().map(|v| v.as_bool().unwrap()).collect();
        assert_eq!(got, vec![true, false, true]);
    }

    #[wasm_bindgen_test]
    fn empty_array_stores_as_string() {
        let mut b = block();
        b.set("names", col(JsArray::new()), None).unwrap();
        assert_eq!(b.dtype("names").unwrap().as_string().unwrap(), "string");
        assert_eq!(b.n_rows().unwrap(), 0);
    }

    #[wasm_bindgen_test]
    fn set_with_shape_is_multidimensional() {
        let mut b = block();
        let shape: JsShape = col(JsArray::of2(&2.into(), &3.into())).unchecked_into();
        let data = Float64Array::from(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0][..]);
        b.set("pos", col(data), Some(shape)).unwrap();
        assert_eq!(b.n_rows().unwrap(), 2);
        assert_eq!(shape_of(&b.shape("pos").unwrap()), vec![2, 3]);
        let got: Float64Array = JsValue::from(b.copy("pos", None).unwrap()).unchecked_into();
        assert_eq!(got.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[wasm_bindgen_test]
    fn set_replaces_with_new_dtype() {
        let mut b = block();
        b.set("c", col(Float64Array::from(&[1.0][..])), None)
            .unwrap();
        b.set("c", col(Int32Array::from(&[1][..])), None).unwrap();
        assert_eq!(b.dtype("c").unwrap().as_string().unwrap(), "i32");
        assert_eq!(b.keys().unwrap(), vec!["c".to_string()]);
    }

    #[wasm_bindgen_test]
    fn view_writes_through_in_place() {
        let mut b = block();
        b.set("x", col(Float64Array::new_with_length(3)), None)
            .unwrap();
        let view: Float64Array = JsValue::from(b.view("x").unwrap()).unchecked_into();
        view.set_index(0, 1.0);
        view.set_index(2, 3.0);
        let got: Float64Array = JsValue::from(b.copy("x", None).unwrap()).unchecked_into();
        assert_eq!(got.to_vec(), vec![1.0, 0.0, 3.0]);
    }

    #[wasm_bindgen_test]
    fn has_keys_n_rows() {
        let mut b = block();
        assert!(!b.has("x").unwrap());
        assert_eq!(b.n_rows().unwrap(), 0);
        b.set("x", col(Float64Array::from(&[1.0, 2.0][..])), None)
            .unwrap();
        b.set("id", col(BigUint64Array::from(&[1_u64, 2][..])), None)
            .unwrap();
        assert!(b.has("x").unwrap());
        assert!(!b.has("y").unwrap());
        assert_eq!(b.keys().unwrap(), vec!["x".to_string(), "id".to_string()]);
        assert_eq!(b.n_rows().unwrap(), 2);
    }

    // ---- errors ----

    #[wasm_bindgen_test]
    fn copy_missing_throws_unless_fallback() {
        let b = block();
        assert!(err_text(b.copy("nope", None).err().unwrap()).contains("'nope' not found"));
        let fallback = BigUint64Array::from(&[9_u64][..]);
        let got: BigUint64Array =
            JsValue::from(b.copy("nope", Some(col(fallback))).unwrap()).unchecked_into();
        assert_eq!(got.to_vec(), vec![9]);
    }

    #[wasm_bindgen_test]
    fn copy_ignores_fallback_when_present() {
        let mut b = block();
        b.set("x", col(Float64Array::from(&[1.0][..])), None)
            .unwrap();
        let got: JsValue = b
            .copy("x", Some(col(Int32Array::from(&[5][..]))))
            .unwrap()
            .into();
        assert!(got.is_instance_of::<Float64Array>());
    }

    #[wasm_bindgen_test]
    fn metadata_on_missing_column_throws() {
        let b = block();
        assert!(b.dtype("nope").is_err());
        assert!(b.shape("nope").is_err());
        assert!(b.view("nope").is_err());
        assert!(b.validity("nope").is_err());
    }

    #[wasm_bindgen_test]
    fn view_refuses_non_numeric() {
        let mut b = block();
        let names: JsArray = ["a"].iter().map(|s| JsValue::from_str(s)).collect();
        b.set("s", col(names), None).unwrap();
        let flags: JsArray = [true].iter().map(|&v| JsValue::from_bool(v)).collect();
        b.set("f", col(flags), None).unwrap();
        assert!(err_text(b.view("s").err().unwrap()).contains("use copy()"));
        assert!(err_text(b.view("f").err().unwrap()).contains("use copy()"));
        let copied = b.copy("s", None).unwrap();
        assert!(JsValue::from(copied).is_instance_of::<JsArray>());
    }

    #[wasm_bindgen_test]
    fn set_refuses_float32_and_clamped() {
        let mut b = block();
        let e = b
            .set("x", col(Float32Array::new_with_length(2)), None)
            .unwrap_err();
        assert!(err_text(e).contains("Float64Array"));
        let e = b
            .set("x", col(Uint8ClampedArray::new_with_length(2)), None)
            .unwrap_err();
        assert!(err_text(e).contains("Uint8ClampedArray"));
        assert!(!b.has("x").unwrap());
    }

    #[wasm_bindgen_test]
    fn set_refuses_untyped_numbers_and_mixed_arrays() {
        let mut b = block();
        let numbers: JsArray = [1.0, 2.0].iter().map(|&v| JsValue::from_f64(v)).collect();
        assert!(err_text(b.set("n", col(numbers), None).unwrap_err()).contains("typed array"));
        let mixed = JsArray::of2(&JsValue::from_str("a"), &JsValue::from_f64(1.0));
        assert!(err_text(b.set("m", col(mixed), None).unwrap_err()).contains("element 1"));
        assert!(b.set("o", col(js_sys::Object::new()), None).is_err());
    }

    #[wasm_bindgen_test]
    fn set_refuses_shape_mismatch_and_ragged_rows() {
        let mut b = block();
        let shape: JsShape = col(JsArray::of2(&2.into(), &2.into())).unchecked_into();
        let e = b
            .set("pos", col(Float64Array::new_with_length(3)), Some(shape))
            .unwrap_err();
        assert!(err_text(e).contains("does not hold 3 values"));
        b.set("x", col(Float64Array::new_with_length(2)), None)
            .unwrap();
        assert!(
            b.set("y", col(Float64Array::new_with_length(3)), None)
                .is_err()
        );
    }

    #[wasm_bindgen_test]
    fn set_honours_the_frame_schema() {
        // `id` is declared u64 by the Frame schema; an i32 column is refused.
        let mut b = block();
        assert!(b.set("id", col(Int32Array::from(&[1][..])), None).is_err());
    }

    #[wasm_bindgen_test]
    fn structural_shape_round_trip() {
        let mut b = block();
        assert!(b.structural_shape().unwrap().is_none());
        b.set("rho", col(Float64Array::new_with_length(8)), None)
            .unwrap();
        let dims: JsShape = col([2, 2, 2]
            .iter()
            .map(|&d| JsValue::from(d))
            .collect::<JsArray>())
        .unchecked_into();
        b.set_shape(dims).unwrap();
        assert_eq!(
            shape_of(&b.structural_shape().unwrap().unwrap()),
            vec![2, 2, 2]
        );
        let bad: JsShape = col(JsArray::of1(&3.into())).unchecked_into();
        assert!(b.set_shape(bad).is_err());
    }

    #[wasm_bindgen_test]
    fn validity_reports_null_rows() {
        let mut rs_atoms = RsBlock::new();
        let frag = ndarray::Array1::from_vec(vec![7_i32, 0, 0]).into_dyn();
        rs_atoms
            .insert("frag_id", frag, Some(vec![true, false, true]))
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", rs_atoms);
        let frame = Frame::from_rs(rs_frame).unwrap();
        let block = frame.get("atoms").unwrap();

        let mask = block.validity("frag_id").unwrap().expect("mask");
        assert_eq!(mask.to_vec(), vec![1_u8, 0, 1]);
    }

    #[wasm_bindgen_test]
    fn validity_is_undefined_for_fully_valid_column() {
        let mut b = block();
        b.set("x", col(Float64Array::from(&[0.0, 1.0][..])), None)
            .unwrap();
        assert!(b.validity("x").unwrap().is_none());
    }
}
