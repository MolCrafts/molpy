//! Gaussian Cube for the WASM API — the face of `molrs::io::cube`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readCube` | `read_cube` |
//! | `writeCube` | `write_cube` (needs a `"grid"` block) |
//!
//! molrs reads cubes back to back with `CubeReader`; JS reads one cube.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read a Gaussian cube (the file's text or bytes) into a frame with:
    /// - `"atoms"` block: `atomic_number` (u64, the Z the file states — no
    ///   `element` symbols are synthesized; `fillSymbols(frame)` writes them),
    ///   `charge` (F), `x`/`y`/`z` (F, **always Å** — Bohr files are
    ///   converted on read).
    /// - `"grid"` block: structural shape `[nx, ny, nz]` and one f64 column
    ///   per scalar field — `density` for single-density files,
    ///   `mo_<idx>` for negative-natoms multi-orbital files.
    /// - `box`: voxel cell × dims in Å.
    ///
    /// ```js
    /// const frame   = readCube(await file.text());
    /// const density = frame.get("grid").copy("density"); // owned Float64Array
    /// ```
    readCube => read_cube, "Cube"
);
write_door!(
    /// Write `frame` as Gaussian cube text (needs a `"grid"` block).
    writeCube => write_cube, "Cube"
);
