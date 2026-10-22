//! Core data model exported to JavaScript — the WASM face of `molrs::core`.
//!
//! - [`Frame`] -- hierarchical container of named [`Block`]s, plus an
//!   optional [`Box`] (simulation box).
//! - [`Block`] -- column-oriented data store with typed arrays.
//! - [`NDArray`] -- owned float array with shape metadata for passing
//!   multi-dimensional numeric data across the WASM boundary.
//! - `schema` -- the Frame schema vocabulary (`schemaDocument`, …).
//! - [`Topology`] -- the bond graph of a frame (`molrs::core::Topology`).
//! - `covalentRadius` -- per-element data, and `fillAtomicNumbers` /
//!   `fillSymbols` -- the explicit species conversions
//!   (`molrs::core::Element`).
//! - [`Box`] -- the periodic cell (Rust's `SimBox`), built from a [`Cell`].
//! - the regions -- the geometric solids and their composition.
//! - [`TriMesh`] -- triangle surfaces.
//! - [`NeighborList`] (self) / [`NeighborQuery`] (cross) -- neighbor search,
//!   and the [`Neighbors`] pair table they produce.
//!
//! # Internal details
//!
//! All mutable state is managed through a [`FrameArenaCell`] (an
//! `Rc<RefCell<FrameArena>>`) that is **not** `Send + Sync`. This is
//! intentional: WebAssembly is single-threaded, so no locking overhead
//! is required. Native multi-threaded consumers should use
//! `Arc<Mutex<FrameArena>>` instead.
//!
//! The [`FrameArenaCell`] alias and its paired [`FrameRef`](molrs_ffi::FrameRef) /
//! [`BlockRef`](molrs_ffi::BlockRef) wrappers live in `molrs-ffi` so every
//! binding layer (wasm, python, capi, cxx) consumes the same canonical
//! lifetime-management plumbing.
//!
//! [`FrameArenaCell`]: molrs_ffi::FrameArenaCell

use wasm_bindgen::JsValue;

use molrs_ffi::FfiError;

pub(crate) mod block;
pub(crate) mod element;
pub(crate) mod frame;
pub(crate) mod mesh;
pub(crate) mod nd_array;
pub(crate) mod neighbors;
pub(crate) mod region;
pub(crate) mod schema;
pub(crate) mod simbox;
pub(crate) mod topology;

pub use block::Block;
pub use element::{covalent_radius, fill_atomic_numbers, fill_symbols};
pub use frame::Frame;
pub use mesh::TriMesh;
pub use nd_array::NDArray;
pub use neighbors::{NeighborList, NeighborQuery, Neighbors};
pub use region::*;
pub use schema::*;
pub use simbox::{Box, Cell};
pub use topology::Topology;

/// Convert an [`FfiError`] into a [`JsValue`] string for propagation
/// to JavaScript as a thrown exception.
pub(crate) fn js_err(err: FfiError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

/// The N×3 positions of a core frame (`molrs::core::Frame::coords`), as a JS
/// error when the frame has no `atoms` block with float `x` / `y` / `z`.
pub(crate) fn frame_coords(frame: &molrs::core::Frame) -> Result<molrs::op::Fnx3, JsValue> {
    frame
        .coords()
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
