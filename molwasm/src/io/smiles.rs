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
//! | `SmilesIr.fromFragment(body)`, `SmilesIr.fromPsmiles(unit)`, `fragment.toTemplate()` | `SmilesIr::from_fragment`, `from_psmiles`, `FragmentIr::to_template` + `to_frame` |
//! | `BigSmilesIr.parse(text)`, `ir.toMolecules()`, `ir.toSystem()` | `BigSmilesIr::parse`, `to_molecules`, `to_system` |
//!
//! # Typical workflow (JavaScript)
//!
//! ```js
//! import { SmilesIr, Conformer } from "@molcrafts/molwasm";
//!
//! const frame = SmilesIr.parse("c1ccccc1").toFrame(); // benzene, 2D graph (no coords)
//! const mol3d = new Conformer().generate(frame); // embed 3D coords
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

#[wasm_bindgen(js_class = SmilesIr)]
impl SmilesIr {
    /// Parse a fragment body — SMILES plus bonding descriptors
    /// (`[<]OCC[>]`, `[$]COC[$]`) — `SmilesIr::from_fragment`.
    ///
    /// # Errors
    ///
    /// Throws for text that is not fragment notation.
    #[wasm_bindgen(js_name = fromFragment)]
    pub fn from_fragment(body: &str) -> Result<FragmentIr, JsValue> {
        molrs::io::smiles::SmilesIr::from_fragment(body)
            .map(|inner| FragmentIr { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Parse a PSMILES repeat unit (`[*]CC[*]`) — `SmilesIr::from_psmiles`.
    /// Each `*` is a joining site with no head or tail, read as the `$`
    /// descriptor: the result is the fragment `fromFragment("[$]CC[$]")`
    /// gives.
    ///
    /// # Errors
    ///
    /// Throws for a `*` bonded to anything but one atom, a unit without
    /// exactly two `*`, or text that is not SMILES.
    #[wasm_bindgen(js_name = fromPsmiles)]
    pub fn from_psmiles(unit: &str) -> Result<FragmentIr, JsValue> {
        molrs::io::smiles::SmilesIr::from_psmiles(unit)
            .map(|inner| FragmentIr { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

/// A fragment body — SMILES plus bonding descriptors — the WASM face of
/// `molrs::io::smiles::FragmentIr`.
#[wasm_bindgen(js_name = FragmentIr)]
pub struct FragmentIr {
    inner: molrs::io::smiles::FragmentIr,
}

#[wasm_bindgen(js_class = FragmentIr)]
impl FragmentIr {
    /// The ported unit of this body as a [`Frame`] — `FragmentIr::to_template`:
    /// heavy atoms plus one hydrogen handle per descriptor (`atoms`, `bonds`)
    /// and one row of `ports` per descriptor. No coordinates.
    ///
    /// # Errors
    ///
    /// Throws if the body does not convert, or the template cannot be
    /// written as a frame.
    #[wasm_bindgen(js_name = toTemplate)]
    pub fn to_template(&self) -> Result<Frame, JsValue> {
        let template = self
            .inner
            .to_template()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Frame::from_rs(
            template
                .to_frame()
                .map_err(|e| JsValue::from_str(&format!("toTemplate: {e}")))?,
        )
    }
}

/// A parsed BigSMILES / generative-BigSMILES string — the WASM face of
/// `molrs::io::bigsmiles::BigSmilesIr`.
#[cfg(feature = "builder")]
#[wasm_bindgen(js_name = BigSmilesIr)]
pub struct BigSmilesIr {
    inner: molrs::io::bigsmiles::BigSmilesIr,
}

#[cfg(feature = "builder")]
#[wasm_bindgen(js_class = BigSmilesIr)]
impl BigSmilesIr {
    /// Parse a BigSMILES or generative-BigSMILES string: stochastic objects
    /// `{[] …; … []}`, port weights `[<|0.5|]`, size distributions
    /// `|schulz_zimm(1500, 1400)|` and system amounts `.|5e5|`.
    ///
    /// # Errors
    ///
    /// Throws for text that is not (generative) BigSMILES, or uses notation
    /// molrs does not model.
    #[wasm_bindgen(js_name = parse)]
    pub fn parse(text: &str) -> Result<BigSmilesIr, JsValue> {
        molrs::io::bigsmiles::BigSmilesIr::parse(text)
            .map(|inner| BigSmilesIr { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// The number of `.`-separated molecules the string writes.
    #[wasm_bindgen(getter, js_name = nMolecules)]
    pub fn n_molecules(&self) -> usize {
        self.inner.molecules.len()
    }

    /// Each molecule as a [`Molecule`](crate::polymer::Molecule); fragments
    /// are named by position (`"m.p"`, `"m.p.rk"`, `"m.p.ek"`, and
    /// `"m.p.hk"` for the hydrogen that ends an object written without end
    /// groups). Sizes count heavy-atom mass.
    ///
    /// # Errors
    ///
    /// Throws for a string the polymer model refuses.
    #[wasm_bindgen(js_name = toMolecules)]
    pub fn to_molecules(&self) -> Result<Vec<crate::polymer::Molecule>, JsValue> {
        let molecules = self
            .inner
            .to_molecules()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(molecules
            .into_iter()
            .map(|inner| crate::polymer::Molecule { inner })
            .collect())
    }

    /// The string as a [`System`](crate::polymer::System): every molecule in
    /// its written amount, counting heavy-atom mass.
    ///
    /// # Errors
    ///
    /// Throws for a molecule without an amount, or amounts the system
    /// refuses.
    #[wasm_bindgen(js_name = toSystem)]
    pub fn to_system(&self) -> Result<crate::polymer::System, JsValue> {
        self.inner
            .to_system()
            .map(|inner| crate::polymer::System { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
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

    #[wasm_bindgen_test]
    fn a_psmiles_unit_is_the_dollar_fragment() {
        let star = SmilesIr::from_psmiles("[*]CC[*]")
            .unwrap()
            .to_template()
            .unwrap();
        let dollar = SmilesIr::from_fragment("[$]CC[$]")
            .unwrap()
            .to_template()
            .unwrap();
        for frame in [&star, &dollar] {
            // Two carbons and one hydrogen handle per port.
            assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 4);
            assert_eq!(frame.get("ports").unwrap().n_rows().unwrap(), 2);
        }
        assert!(SmilesIr::from_psmiles("*CC").is_err());
    }

    #[cfg(feature = "builder")]
    #[wasm_bindgen_test]
    fn a_bigsmiles_system_samples_to_its_amount() {
        let ir = BigSmilesIr::parse("{[][$]CC[$][]}|poisson(300)|.|3000|").unwrap();
        assert_eq!(ir.n_molecules(), 1);
        assert_eq!(ir.to_molecules().unwrap().len(), 1);
        let system = ir.to_system().unwrap();
        assert_eq!(system.measure(), "heavy_atom_mass");
        let sample = system.sample(7).unwrap();
        assert!(
            sample
                .sites()
                .unwrap()
                .get("atoms")
                .unwrap()
                .n_rows()
                .unwrap()
                > 0
        );
        // The implicit hydrogen end group is in the library.
        assert!(system.library().unwrap().has(&JsValue::from_str("0.0.h0")));
        assert!(BigSmilesIr::parse("{[][<]CC[>][]}|zipf(2)|").is_err());
    }

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
