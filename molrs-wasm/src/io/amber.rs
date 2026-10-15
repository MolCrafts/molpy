//! AMBER for the WASM API — the face of `molrs::io::amber`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readInpcrdStr` | `read_inpcrd_str` (ASCII inpcrd / restrt) |
//! | `readAcStr` | `read_ac_str` (Antechamber `.ac`) |
//! | `readPrmtopStr` | `read_prmtop_str` (prmtop structure) |
//!
//! molrs reads these formats with functions only, so JS has no reader class.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read AMBER ASCII inpcrd / restrt text: atoms with `x`/`y`/`z`
    /// (and `vel` when present), the box when present.
    readInpcrdStr => read_inpcrd_str(text: &str), "AMBER inpcrd"
);
read_door!(
    /// Read Antechamber `.ac` text.
    readAcStr => read_ac_str(text: &str), "AC"
);
read_door!(
    /// Read AMBER prmtop text's structure (atoms, topology; no force field).
    readPrmtopStr => read_prmtop_str(text: &str), "AMBER prmtop"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::float_col;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn read_inpcrd_str_reads_the_coordinates() {
        let inpcrd = "title\n     2\n   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000\n";
        let frame = read_inpcrd_str(inpcrd).expect("inpcrd");
        let x = float_col(&frame.get("atoms").expect("atoms"), "x");
        assert_eq!(x.to_vec(), vec![1.0, 4.0]);
    }
}
