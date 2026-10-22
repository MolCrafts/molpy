//! Per-element data and the explicit species conversions — the WASM face of
//! `molrs::core::Element`.
//!
//! molrs reads species only from `atoms.atomic_number` (`uint`, 0..=118, 0 =
//! the wildcard `*`); `atoms.element` is raw symbol data that nothing reads.
//! Nothing converts between them implicitly: `fillAtomicNumbers` and
//! `fillSymbols` are the two conversions, and a frame read from a symbol
//! format (XYZ, PDB, SDF, …) goes through `fillAtomicNumbers` before any
//! chemistry (`Conformer.generate`, the typifiers, perception,
//! `SmilesIr.fromFrame`) accepts it.

use wasm_bindgen::prelude::*;

use super::frame::Frame;
use super::js_err;

/// Covalent radius (in angstrom) of the element with atomic number
/// `atomicNumber`.
///
/// Lookup against the built-in periodic table — the single source of truth
/// for covalent radii; downstream code (e.g. bond perception) calls this
/// rather than carrying its own table. Keyed by atomic number, the field
/// every molrs consumer reads: a symbol is converted first
/// (`fillAtomicNumbers` on a frame).
///
/// # Errors
///
/// Throws when `atomicNumber` is not an element, 1..=118 — the wildcard 0
/// included, which has no radius.
///
/// # Example (JavaScript)
///
/// ```js
/// covalentRadius(6);   // 0.76
/// covalentRadius(1);   // 0.31
/// covalentRadius(0);   // throws: the wildcard has no radius
/// ```
#[wasm_bindgen(js_name = covalentRadius)]
pub fn covalent_radius(atomic_number: u32) -> Result<f64, JsValue> {
    u8::try_from(atomic_number)
        .ok()
        .and_then(molrs::core::Element::by_number)
        .map(|el| f64::from(el.covalent_radius()))
        .ok_or_else(|| {
            JsValue::from_str(&format!(
                "covalentRadius: {atomic_number} is not the atomic number of an element (1..=118)"
            ))
        })
}

/// Write `atoms.atomic_number` from the `atoms.element` symbols, in place —
/// `Element::fill_atomic_numbers`.
///
/// Symbols match case-insensitively; `"*"` is the wildcard, Z = 0. An
/// existing `atomic_number` column is replaced; `element` is kept. Block
/// handles taken from `frame` before the call are invalidated — take them
/// again (`frame.get("atoms")`).
///
/// # Errors
///
/// Throws when the frame has no `"atoms"` block, the block has no string
/// `element` column, or a row holds a symbol that is not an element (the
/// message names the row).
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = readXyz(text);        // atoms.element: ["O", "H", "H"]
/// fillAtomicNumbers(frame);           // atoms.atomic_number: [8n, 1n, 1n]
/// const mol3d = new Conformer("fast").generate(frame);
/// ```
#[wasm_bindgen(js_name = fillAtomicNumbers)]
pub fn fill_atomic_numbers(frame: &Frame) -> Result<(), JsValue> {
    frame
        .inner
        .with_mut(molrs::core::Element::fill_atomic_numbers)
        .map_err(js_err)?
        .map_err(|e| JsValue::from_str(&format!("fillAtomicNumbers: {e}")))
}

/// Write `atoms.element` symbols from `atoms.atomic_number`, in place —
/// `Element::fill_symbols`.
///
/// Z = 0 becomes the wildcard `"*"`. An existing `element` column is
/// replaced; `atomic_number` is kept. This is the step before a writer that
/// spells symbols (`writeXyz`, `writePdb`, …) when the frame came from
/// chemistry (`SmilesIr.toFrame`, `Conformer.generate`) or a Z format (cube,
/// XSF). Block handles taken from `frame` before the call are invalidated.
///
/// # Errors
///
/// Throws when the frame has no `"atoms"` block, the block has no `uint`
/// `atomic_number` column, or a row holds a number outside 0..=118 (the
/// message names the row).
///
/// # Example (JavaScript)
///
/// ```js
/// const frame = SmilesIr.parse("CCO").toFrame(); // atoms.atomic_number
/// fillSymbols(frame);                  // atoms.element: ["C", "C", "O"]
/// const text = writeXyz(frame);
/// ```
#[wasm_bindgen(js_name = fillSymbols)]
pub fn fill_symbols(frame: &Frame) -> Result<(), JsValue> {
    frame
        .inner
        .with_mut(molrs::core::Element::fill_symbols)
        .map_err(js_err)?
        .map_err(|e| JsValue::from_str(&format!("fillSymbols: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, keys};
    use molrs::op::Idx;
    use ndarray::{Array1, ArrayD};
    use wasm_bindgen_test::*;

    fn frame_with_atoms(atoms: Block) -> Frame {
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn covalent_radius_is_keyed_by_atomic_number() {
        assert_eq!(
            covalent_radius(6).unwrap(),
            f64::from(molrs::core::Element::C.covalent_radius())
        );
        assert!(covalent_radius(0).is_err());
        assert!(covalent_radius(119).is_err());
        assert!(covalent_radius(256 + 6).is_err());
    }

    #[wasm_bindgen_test]
    fn fill_atomic_numbers_converts_symbols_in_place() {
        let mut atoms = Block::new();
        let symbols: ArrayD<String> =
            Array1::from_vec(vec!["O".to_string(), "h".into(), "*".into()]).into_dyn();
        atoms.insert(keys::ELEMENT, symbols, None).unwrap();
        let frame = frame_with_atoms(atoms);
        fill_atomic_numbers(&frame).unwrap();
        let z: Vec<Idx> = frame
            .with_frame(|f| {
                Ok(f["atoms"]
                    .get(keys::ATOMIC_NUMBER)
                    .and_then(|c| c.get::<Idx>())
                    .expect("atomic_number written")
                    .iter()
                    .copied()
                    .collect())
            })
            .unwrap();
        assert_eq!(z, vec![8, 1, 0]);
    }

    #[wasm_bindgen_test]
    fn fill_symbols_converts_atomic_numbers_in_place() {
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::ATOMIC_NUMBER,
                Array1::<Idx>::from_vec(vec![6, 0]).into_dyn(),
                None,
            )
            .unwrap();
        let frame = frame_with_atoms(atoms);
        fill_symbols(&frame).unwrap();
        let symbols: Vec<String> = frame
            .with_frame(|f| {
                Ok(f["atoms"]
                    .get(keys::ELEMENT)
                    .and_then(|c| c.get::<String>())
                    .expect("element written")
                    .iter()
                    .cloned()
                    .collect())
            })
            .unwrap();
        assert_eq!(symbols, ["C", "*"]);
    }

    #[wasm_bindgen_test]
    fn conversions_refuse_a_missing_source_column() {
        let frame = frame_with_atoms(Block::new());
        assert!(fill_atomic_numbers(&frame).is_err());
        assert!(fill_symbols(&frame).is_err());
    }
}
