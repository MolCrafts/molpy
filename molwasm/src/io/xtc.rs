//! GROMACS XTC trajectories for the WASM API — the face of `molrs::io::xtc`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XtcStream` | `XtcIndexBuilder` + `read_xtc` (the chunk-fed XTC reader) |
//! | `readXtc`, `writeXtc` | `read_xtc`, `write_xtc` (nm ↔ Å) |

use molrs::io::xtc::XtcIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = XtcStream,
    indexer = XtcIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_xtc(bytes),
}

read_bytes_door!(
    /// Read the first GROMACS XTC frame of the file's bytes (nm → Å).
    readXtc => read_xtc, "XTC"
);
write_bytes_door!(
    /// Write `frame` as a one-frame GROMACS XTC trajectory (Å → nm).
    writeXtc => write_xtc, "XTC"
);
