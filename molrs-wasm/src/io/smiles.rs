//! The SMILES and CGsmiles line notations for the WASM API — the faces of
//! `molrs::io::smiles::SmilesIr` and `molrs::io::cgsmiles::CgSmilesIr`.
//!
//! A notation is its class, as in molrs: [`SmilesIr.parse`](SmilesIr::parse)
//! reads SMILES text, `toFrame()` builds the molecule, and
//! [`SmilesIr.fromFrame`](SmilesIr::from_frame) + `toString()` write one;
//! [`CgSmilesIr.parse`](CgSmilesIr::parse) + `toFrame()` read the molecule a
//! CGsmiles string states.
//!
//! | JS | molrs |
//! |----|-------|
//! | `SmilesIr.parse(text)`, `ir.toFrame()` | `SmilesIr::parse`, `to_atomistic` + `to_frame` |
//! | `SmilesIr.fromFrame(frame)`, `ir.toString()` | `SmilesIr::from_atomistic` (default emit options), `Display` |
//! | `CgSmilesIr.parse(text)`, `ir.toFrame()` | `CgSmilesIr::parse`, `to_atomistic` + `to_frame` |
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { SmilesIr, Conformer } from "@molcrafts/molrs";
//!
//! const frame = SmilesIr.parse("c1ccccc1").toFrame(); // benzene, 2D graph (no coords)
//! const mol3d = new Conformer("fast").generate(frame); // embed 3D coords
//! SmilesIr.fromFrame(frame).toString();                // "c1ccccc1"
//! ```
//!
//! # References
//!
//! - Weininger, D. (1988). SMILES, a chemical language and information
//!   system. *J. Chem. Inf. Comput. Sci.*, 28(1), 31-36.

use crate::core::frame::Frame;
use wasm_bindgen::prelude::*;

/// Intermediate representation of a parsed SMILES string.
///
/// Holds the molecular graph(s) parsed from a SMILES string. A single
/// SMILES string can encode multiple disconnected molecules separated
/// by `.` (e.g., `"[Na+].[Cl-]"`).
///
/// Call [`toFrame()`](SmilesIr::to_frame) to convert to a [`Frame`]
/// with `"atoms"` and `"bonds"` blocks.
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = SmilesIr.parse("CCO");
/// console.log(ir.nComponents); // 1
///
/// const frame = ir.toFrame();
/// const atoms = frame.get("atoms");
/// console.log(atoms.copy("atomic_number")); // BigUint64Array [6n, 6n, 8n, 1n, ...], an owned copy
/// ```
#[wasm_bindgen(js_name = SmilesIr)]
pub struct SmilesIr {
    inner: molrs::io::smiles::SmilesIr,
}

#[wasm_bindgen(js_class = SmilesIr)]
impl SmilesIr {
    /// Parse a SMILES string — a `.`-separated set of molecules included —
    /// into its intermediate representation.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the SMILES string is malformed
    /// (e.g., unmatched ring closure digits, invalid atom symbols).
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(smiles: &str) -> Result<SmilesIr, JsValue> {
        let inner = molrs::io::smiles::SmilesIr::parse(smiles)
            .map_err(|e| JsValue::from_str(&format!("SMILES parse error: {e}")))?;
        Ok(SmilesIr { inner })
    }

    /// Return the number of disconnected components in the SMILES.
    ///
    /// Components are separated by `.` in the SMILES string. For
    /// example, `"[Na+].[Cl-]"` has 2 components.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const ir = SmilesIr.parse("[Na+].[Cl-]");
    /// console.log(ir.nComponents); // 2
    /// ```
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.components.len()
    }

    /// Convert the intermediate representation to a [`Frame`].
    ///
    /// The resulting frame contains:
    ///
    /// - `"atoms"` block: `atomic_number` (u64; no `element` symbols — call
    ///   `fillSymbols(frame)` before a writer that spells them). Connectivity
    ///   only: implicit hydrogens are not added (`addHydrogens(frame)` adds
    ///   them) and no coordinates are generated — use
    ///   [`Conformer`](crate::conformer::Conformer) to embed coordinates.
    /// - `"bonds"` block: `atomi`, `atomj` (u64, zero-based atom indices),
    ///   `bond_type` (u64: 1 single, 2 double, 3 triple, 4 aromatic) and
    ///   `bond_number` (u64: the localized Lewis/Kekulé integer, 0 when the
    ///   notation declared aromaticity without a phase — call
    ///   `assignAromaticity(frame)` to fill it in).
    ///
    /// # Returns
    ///
    /// A new [`Frame`] with atoms and bonds.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the conversion fails (e.g., invalid
    /// valence), or if a property of the result contradicts the Frame schema
    /// on the way out.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = ir.toFrame();
    /// const bonds = frame.get("bonds");
    /// const types = bonds.copy("bond_type");
    /// const numbers = bonds.copy("bond_number");
    /// ```
    #[wasm_bindgen(js_name = toFrame)]
    pub fn to_frame(&self) -> Result<Frame, JsValue> {
        let mol = self
            .inner
            .to_atomistic()
            .map_err(|e| JsValue::from_str(&format!("IR -> Atomistic: {e}")))?;
        Frame::from_rs(
            mol.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }
}

#[wasm_bindgen(js_class = SmilesIr)]
impl SmilesIr {
    /// The SMILES IR of the molecule in `frame` (its `"atoms"` and
    /// `"bonds"`) — `SmilesIr::from_atomistic` with molrs's default emit
    /// options (canonical, aromatic as marked, organic-subset hydrogens, no
    /// stereo, one component). `toString()` is its SMILES text.
    ///
    /// Species is read from `atoms.atomic_number`; a frame that spells
    /// symbols (`atoms.element`, e.g. from `readXyz`) needs
    /// `fillAtomicNumbers(frame)` first.
    ///
    /// # Errors
    ///
    /// Throws when an atom has no valid `atomic_number`, or when molrs cannot
    /// write the molecule.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const text = SmilesIr.fromFrame(SmilesIr.parse("OCC").toFrame()).toString();
    /// ```
    #[wasm_bindgen(js_name = fromFrame)]
    pub fn from_frame(frame: &Frame) -> Result<SmilesIr, JsValue> {
        let mol = frame.with_frame(|rs| {
            molrs::core::Atomistic::from_frame(rs)
                .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
        })?;
        molrs::io::smiles::SmilesIr::from_atomistic(
            &mol,
            &molrs::io::smiles::SmilesEmitOptions::default(),
        )
        .map(|inner| SmilesIr { inner })
        .map_err(|e| JsValue::from_str(&format!("SMILES writing error: {e}")))
    }

    /// The IR written as SMILES text — `Display`.
    ///
    /// # Errors
    ///
    /// Throws for an IR no parser or `fromFrame` produces (none here).
    #[wasm_bindgen(js_name = toString)]
    pub fn to_text(&self) -> Result<String, JsValue> {
        use std::fmt::Write;
        let mut text = String::new();
        write!(text, "{}", self.inner)
            .map_err(|_| JsValue::from_str("SMILES writing error: the IR has no SMILES text"))?;
        Ok(text)
    }
}

/// A parsed CGsmiles string — the WASM face of
/// `molrs::io::cgsmiles::CgSmilesIr`: the coarse-grained levels and their
/// fragment tables, no atoms until [`toFrame`](Self::to_frame) expands the
/// lowest level.
#[wasm_bindgen(js_name = CgSmilesIr)]
pub struct CgSmilesIr {
    inner: molrs::io::cgsmiles::CgSmilesIr,
}

#[wasm_bindgen(js_class = CgSmilesIr)]
impl CgSmilesIr {
    /// Parse a CGsmiles string.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the string is not CGsmiles.
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(text: &str) -> Result<CgSmilesIr, JsValue> {
        molrs::io::cgsmiles::CgSmilesIr::parse(text)
            .map(|inner| CgSmilesIr { inner })
            .map_err(|e| JsValue::from_str(&format!("CGsmiles parse error: {e}")))
    }

    /// The molecule the string states, its lowest level expanded into atoms
    /// — `CgSmilesIr::to_atomistic` — as a [`Frame`] laid out as
    /// [`SmilesIr.toFrame`](SmilesIr::to_frame)'s.
    ///
    /// # Errors
    ///
    /// Throws if the levels do not expand into a molecule.
    #[wasm_bindgen(js_name = toFrame)]
    pub fn to_frame(&self) -> Result<Frame, JsValue> {
        let mol = self
            .inner
            .to_atomistic()
            .map_err(|e| JsValue::from_str(&format!("CGsmiles -> Atomistic: {e}")))?;
        Frame::from_rs(
            mol.to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    /// `fromFrame` + `toString` is the core's canonical writer, and its text
    /// reads back as the same molecule.
    #[wasm_bindgen_test]
    fn from_frame_writes_the_core_canonical_string() {
        let frame = SmilesIr::parse("OCC").unwrap().to_frame().expect("smiles");
        let text = SmilesIr::from_frame(&frame)
            .expect("from frame")
            .to_text()
            .expect("write");
        let mol = molrs::io::smiles::SmilesIr::parse("OCC")
            .unwrap()
            .to_atomistic()
            .unwrap();
        let core = molrs::io::smiles::SmilesIr::from_atomistic(
            &mol,
            &molrs::io::smiles::SmilesEmitOptions::default(),
        )
        .unwrap()
        .to_string();
        assert_eq!(text, core);
        let back = SmilesIr::parse(&text).unwrap().to_frame().unwrap();
        assert_eq!(back.get("atoms").unwrap().n_rows().unwrap(), 3);
    }

    #[wasm_bindgen_test]
    fn cgsmiles_ir_reads_a_molecule() {
        let frame = CgSmilesIr::parse("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}")
            .expect("cgsmiles")
            .to_frame()
            .expect("frame");
        assert!(frame.get("atoms").is_ok());
    }
}
