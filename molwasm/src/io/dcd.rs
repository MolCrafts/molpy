//! DCD trajectories for the WASM API — the face of `molrs::io::dcd`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `DcdStream` | `DcdIndexBuilder` + `DcdFrameDecoder` (the chunk-fed DCD reader; the only stream that uses `decoderState`) |
//! | `readDcd`, `writeDcd` | `read_dcd`, `write_dcd` |

use molrs::io::dcd::{DcdFrameDecoder, DcdIndexBuilder};
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = DcdStream,
    indexer = DcdIndexBuilder::new(),
    parse   = |bytes, state| match state {
        // A frame body the indexer delimited, decoded with the header it
        // captured (fixed-atom later frames need the frame-0 seed).
        Some(state) => DcdFrameDecoder::new(state)?.decode(bytes),
        None => molrs::io::read_dcd(bytes),
    },
}

read_bytes_door!(
    /// Read the first DCD frame of the file's bytes — a whole file (what
    /// `writeDcd` returns), or one frame body as `DcdStream` delimits it
    /// (self-describing when the file has no fixed atoms).
    readDcd => read_dcd, "DCD"
);
write_bytes_door!(
    /// Write `frame` as a one-frame DCD trajectory.
    writeDcd => write_dcd, "DCD"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{float_col, two_atom_frame};
    use wasm_bindgen_test::*;

    /// `readDcd` reads back the one-frame file `writeDcd` wrote.
    #[wasm_bindgen_test]
    fn read_dcd_reads_back_write_dcd() {
        let bytes = write_dcd(&two_atom_frame()).expect("dcd bytes");
        let frame = read_dcd(&bytes).expect("read back");
        let z = float_col(&frame.get("atoms").expect("atoms"), "z");
        assert_eq!(z.to_vec(), vec![0.0, 0.5]);
    }
}
