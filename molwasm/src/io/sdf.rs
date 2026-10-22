//! MDL SDF / MOL for the WASM API — the face of `molrs::io::sdf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `SdfStream` | `SdfIndexBuilder` + `read_sdf` (the chunk-fed SDF reader) |
//! | `readSdf` | `read_sdf` |
//!
//! molrs has no SDF writer, so SDF is read-only.

use molrs::io::sdf::SdfIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = SdfStream,
    indexer = SdfIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_sdf(bytes),
}

read_door!(
    /// Read the first record of SDF / MDL molfile: the file's text or bytes.
    readSdf => read_sdf, "SDF"
);
