//! WASM bindings for inspecting the Frame schema.
//!
//! Projected from the compiled-in Rust tables, so the JS view of the contract
//! cannot describe something the Rust enforcement does not.
//!
//! `schemaDocument()` hands back a real JS object via `serde-wasm-bindgen`
//! rather than a string the caller has to `JSON.parse` — a browser consumer
//! wants the object (`JSON.stringify` it to persist or diff).

use wasm_bindgen::prelude::*;

use molrs::core::schema;

use crate::core::block::JsDType;

/// The whole Frame vocabulary as a JS object.
///
/// ```js
/// const doc = molrs.schemaDocument();
/// doc.columns.find(c => c.key === "atomi").dtype;  // "uint"
/// ```
#[wasm_bindgen(js_name = schemaDocument)]
pub fn schema_document() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&schema::document())
        .map_err(|e| JsValue::from_str(&format!("schema: {e}")))
}

/// Declared dtype of a canonical column, named as `Block.dtype` and
/// `schemaDocument()` name it (`"float"`, `"uint"`, `"string"`, …: core
/// `DType::name()`), so the three compare directly.
///
/// Returns `undefined` when the key carries no declared dtype. That means the
/// key is **unconstrained**, not invalid: the column vocabulary is closed but
/// unspecified keys are the documented extension point.
#[wasm_bindgen(js_name = schemaColumnDtype)]
pub fn schema_column_dtype(key: &str) -> Option<JsDType> {
    schema::column(key).map(|spec| JsValue::from_str(spec.dtype.name()).unchecked_into())
}

/// Whether a block name is part of the canonical vocabulary.
///
/// `false` does not mean the block is illegal — the block set is open, and a
/// frame may carry blocks the vocabulary does not name.
#[wasm_bindgen(js_name = schemaHasBlock)]
pub fn schema_has_block(name: &str) -> bool {
    schema::block(name).is_some()
}

/// Every canonical column key, in vocabulary order.
#[wasm_bindgen(js_name = schemaColumnKeys)]
pub fn schema_column_keys() -> Vec<String> {
    schema::SCHEMA_COLUMNS
        .iter()
        .map(|c| c.key.to_string())
        .collect()
}

/// Every canonical block name, in vocabulary order.
#[wasm_bindgen(js_name = schemaBlockNames)]
pub fn schema_block_names() -> Vec<String> {
    schema::SCHEMA_BLOCKS
        .iter()
        .map(|b| b.name.to_string())
        .collect()
}

/// Column keys, groups, block names, and frame-meta keys.
///
/// The same tables as [`schema_document`], projected as constant names rather
/// than dtypes. JavaScript reads this instead of transcribing the strings.
///
/// ```js
/// const keys = molrs.keysDocument();
/// keys.columns.find(c => c.constName === "ATOMI").value; // "atomi"
/// keys.blocks.find(b => b.constName === "BONDS").value;  // "bonds"
/// ```
#[wasm_bindgen(js_name = keysDocument)]
pub fn keys_document() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&molrs::core::keys::keys_document())
        .map_err(|e| JsValue::from_str(&format!("keys: {e}")))
}
