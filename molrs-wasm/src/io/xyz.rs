//! XYZ / Extended XYZ for the WASM API — the face of `molrs::io::xyz`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XyzStream` | `XyzIndexBuilder` + `read_xyz` (the chunk-fed XYZ reader) |
//! | `readXyz`, `writeXyz` | `read_xyz`, `write_xyz` |

use molrs::io::xyz::XyzIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = XyzStream,
    indexer = XyzIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_xyz(bytes),
}

read_door!(
    /// Read the first frame of (extended) XYZ: the file's text or bytes.
    readXyz => read_xyz, "XYZ"
);
write_door!(
    /// Write `frame` as (extended) XYZ text.
    writeXyz => write_xyz, "XYZ"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{bytes, float_col, text, two_atom_frame};
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_xyz_starts_with_the_atom_count() {
        let out = write_xyz(&two_atom_frame()).expect("xyz output");
        assert!(out.lines().next().unwrap_or("").starts_with('2'));
    }

    /// `readXyz` reads what `writeXyz` wrote, as text and as bytes alike.
    #[wasm_bindgen_test]
    fn read_xyz_reads_text_and_bytes_alike() {
        let out = write_xyz(&two_atom_frame()).expect("xyz output");
        for frame in [
            read_xyz(&text(&out)).expect("text"),
            read_xyz(&bytes(out.as_bytes())).expect("bytes"),
        ] {
            let x = float_col(&frame.get("atoms").expect("atoms"), "x");
            assert_eq!(x.to_vec(), vec![0.0, 1.0]);
        }
    }

    /// A source that is neither text nor bytes is refused by name.
    #[wasm_bindgen_test]
    fn read_xyz_refuses_a_non_source() {
        let not_a_source = wasm_bindgen::JsCast::unchecked_into(JsValue::from_f64(1.0));
        let err = read_xyz(&not_a_source).err().expect("refused");
        assert!(err.as_string().unwrap().contains("readXyz"));
    }
}
