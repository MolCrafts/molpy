//! WebAssembly bindings for the molrs molecular simulation toolkit.
//!
//! Provides a JavaScript/TypeScript-friendly API for molecular data
//! manipulation, file I/O, 3D coordinate generation, and trajectory
//! analysis. Built on top of [`molrs_ffi`] handle-based architecture
//! for safe, single-threaded WASM usage.
//!
//! # Architecture
//!
//! The WASM API mirrors the core Rust data model:
//!
//! - **[`Frame`]** -- hierarchical container mapping string keys
//!   (e.g., `"atoms"`, `"bonds"`) to typed [`Block`]s.
//! - **[`Block`]** -- column-oriented data store with typed columns
//!   (`F`, `i32`, `u64`, `string`). Float columns are the compute scalar
//!   `F = f64` and map to `Float64Array`.
//! - **[`Box`]** (exported as `Box` in JS) -- simulation box defining
//!   periodic boundary conditions and coordinate transformations.
//! - **[`NDArray`]** -- owned float array with ndarray-compatible shape
//!   metadata for passing multi-dimensional data across the WASM boundary.
//!
//! # Modules
//!
//! Each module binds one molrs owner; the JS namespace itself is flat.
//!
//! | Module      | molrs owner | Exports |
//! |-------------|-------------|---------|
//! | `core`      | `core` | Frame, Block, Box, Cell, NDArray, schema, Topology, `covalentRadius`, `fillAtomicNumbers` / `fillSymbols`, regions, TriMesh, NeighborList / NeighborQuery / Neighbors |
//! | `io`        | `io` | File readers/writers (XYZ, PDB, LAMMPS, `*.mrec` records, …), `SmilesIr` / `CgSmilesIr` |
//! | `perceive`  | `perceive` | Chemical perception: `assign*` mutate the frame in place, `addHydrogens` / `removeHydrogens` return a new one |
//! | `compute`   | `compute` | Analysis: one class per `ComputeSpec` entry (`Rdf`, `Msd`, `Cluster`, …), their result handles, and the `ComputeSpec` registry |
//! | `conformer` | `conformer` | 3D conformer generation (`Conformer`) |
//! | `ff`        | `ff` | Typifiers (UFF, MMFF94, MMFF94s), the `ForceField` they output, `PotentialCompiler` and the `Potentials` it compiles |
//! | `optimize`  | `optimize` | `Lbfgs` / `OptimizationReport` |
//! | `builder`   | `builder` | `CarbonTubeBuilder`, `TubeExtent` |
//!
//! # Quick start (JavaScript)
//!
//! The npm package is a `bundler` build: importing it loads the module.
//!
//! ```js
//! import { SmilesIr, Conformer, fillSymbols, writeXyz } from "@molcrafts/molrs";
//!
//! const frame = SmilesIr.parse("CCO").toFrame();
//! const mol3d = new Conformer("fast").generate(frame);
//! fillSymbols(mol3d);                 // XYZ spells symbols
//! const xyz   = writeXyz(mol3d);
//! console.log(xyz);
//! ```

use js_sys::WebAssembly::Memory;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// WASM module entry point. Installs the panic hook so that Rust panics
/// are forwarded to the browser console as readable stack traces.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Return a handle to the WASM linear memory.
///
/// Useful for advanced interop where JS code needs direct access to the
/// WASM memory buffer (e.g., for zero-copy typed-array views).
///
/// # Example (JavaScript)
///
/// ```js
/// const mem = wasmMemory();
/// const buf = new Float64Array(mem.buffer, ptr, len);
/// ```
#[wasm_bindgen(js_name = wasmMemory)]
pub fn wasm_memory() -> Memory {
    wasm_bindgen::memory().unchecked_into()
}

// Module declarations — one per molrs owner, mirroring the Rust crate.
#[cfg(feature = "builder")]
mod builder;
#[cfg(feature = "compute")]
mod compute;
#[cfg(feature = "conformer")]
mod conformer;
mod core;
/// Force-field composition (typify / forcefield / PotentialCompiler / Potentials) — requires `conformer` (→ `ff`).
#[cfg(feature = "conformer")]
mod ff;
#[cfg(feature = "io")]
mod io;
/// Geometry optimization (`Lbfgs`) over force-field potentials.
#[cfg(feature = "conformer")]
mod optimize;
/// Chemical perception (rings, aromaticity, hydrogens, …) — WASM face of
/// `molrs::perceive`.
mod perceive;

// The JS namespace is flat; so is the crate root.
#[cfg(feature = "builder")]
pub use builder::{CarbonTubeBuilder, TubeExtent};
#[cfg(feature = "compute")]
pub use compute::*;
#[cfg(feature = "conformer")]
pub use conformer::*;
pub use core::*;
#[cfg(feature = "conformer")]
pub use ff::*;
#[cfg(feature = "io")]
pub use io::*;
#[cfg(feature = "conformer")]
pub use optimize::*;
pub use perceive::{
    add_hydrogens, assign_aromaticity, assign_kekule_bond_orders, assign_rings, remove_hydrogens,
};
