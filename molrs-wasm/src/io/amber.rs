//! AMBER for the WASM API — the face of `molrs::io::amber`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readInpcrd` | `read_inpcrd` (ASCII inpcrd / restrt) |
//! | `readAc` | `read_ac` (Antechamber `.ac`) |
//! | `readPrmtop` | `read_prmtop` (prmtop structure) |
//!
//! molrs reads these formats with functions only, so JS has no reader class.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read AMBER ASCII inpcrd / restrt (text or bytes): atoms with `x`/`y`/`z`
    /// (and `vel` when present), the box when present.
    readInpcrd => read_inpcrd, "AMBER inpcrd"
);
read_door!(
    /// Read Antechamber `.ac` (text or bytes).
    readAc => read_ac, "AC"
);
read_door!(
    /// Read an AMBER prmtop's structure (atoms, topology; no force field).
    readPrmtop => read_prmtop, "AMBER prmtop"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{float_col, text};
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn read_inpcrd_reads_the_coordinates() {
        let inpcrd = "title\n     2\n   1.0000000   2.0000000   3.0000000   4.0000000   5.0000000   6.0000000\n";
        let frame = read_inpcrd(&text(inpcrd)).expect("inpcrd");
        let x = float_col(&frame.get("atoms").expect("atoms"), "x");
        assert_eq!(x.to_vec(), vec![1.0, 4.0]);
    }
}
