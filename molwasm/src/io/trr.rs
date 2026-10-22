//! GROMACS TRR trajectories for the WASM API — the face of `molrs::io::trr`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `TrrStream` | `TrrIndexBuilder` + `read_trr` (the chunk-fed TRR reader) |
//! | `readTrr`, `writeTrr` | `read_trr`, `write_trr` (nm ↔ Å) |

use molrs::io::trr::TrrIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = TrrStream,
    indexer = TrrIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_trr(bytes),
}

read_bytes_door!(
    /// Read the first GROMACS TRR frame of the file's bytes (nm → Å).
    readTrr => read_trr, "TRR"
);
write_bytes_door!(
    /// Write `frame` as a one-frame GROMACS TRR trajectory (Å → nm).
    writeTrr => write_trr, "TRR"
);
