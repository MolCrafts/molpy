//! FFI layer for molrs with handle-based abstractions.
//!
//! This crate provides a stable, handle-based API for Python and WASM bindings.
//! It separates the Rust-idiomatic core API from cross-language FFI concerns.
//!
//! # Usage
//!
//! A consumer holds a [`FrameRef`] (a frame id paired with the
//! [`FrameArenaCell`] that owns the frame) and
//! borrows columns zero-copy through [`BlockRef`]. See `docs/interop.md` for the
//! full recipe and the data contract.
//!
//! ```no_run
//! use molrs::op::Idx;
//! use molrs_ffi::FrameRef;
//!
//! let frame = FrameRef::default();      // a frame inside a fresh FrameArenaCell
//! // ... populate it via frame.with_mut(|f| ...) ...
//! if let Ok(atoms) = frame.block("atoms") {
//!     // zero-copy borrow of the uint atom-id column (the uint-index contract)
//!     let n_ids = atoms.borrow::<Idx, _>("id", |ids, _shape| ids.len()).ok().flatten();
//!     let _ = n_ids;
//! }
//! ```

pub mod abi;
mod error;
#[cfg(feature = "ff")]
mod forcefield;
mod frame_arena;
mod frame_ref;
mod handle;
mod region;

pub use error::FfiError;
#[cfg(feature = "ff")]
pub use forcefield::ForceFieldRef;
pub use frame_arena::FrameArena;
pub use frame_ref::{BlockRef, FrameArenaCell, FrameRef, OwnedColumn};
pub use handle::{BlockHandle, FrameId};
pub use region::RegionRef;
