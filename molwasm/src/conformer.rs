//! 3D coordinate generation (distance geometry + force-field refinement).
//!
//! Generates realistic 3D molecular geometries from a molecular graph
//! (2D connectivity). The pipeline uses distance geometry embedding
//! followed by MMFF94-based energy minimization.
//!
//! # Pipeline stages
//!
//! 1. **ETKDGv3 constraints** -- build and smooth the distance-bounds matrix.
//! 2. **Distance-geometry embedding** -- sample distances and embed in 4D.
//! 3. **ETKDG refinement** -- apply distance, chirality, and torsion knowledge.
//! 4. **MMFF94 cleanup** -- relax the generated 3D structure.
//! 5. **Stereo guards** -- verify stereochemistry is preserved.
//!
//! # References
//!
//! - Halgren, T.A. (1996). Merck Molecular Force Field (MMFF94).
//!   *J. Comput. Chem.*, 17(5-6), 490-519.

use wasm_bindgen::prelude::*;

use molrs::conformer::{Conformer as RsConformer, ConformerOptions};
use molrs::core::Atomistic;

use crate::core::frame::Frame;

/// 3D conformer generator — molrs `conformer::Conformer`, as Python's
/// `molrs.conformer.Conformer`.
///
/// Construct with the generation parameters, then call
/// [`generate`](Self::generate) on a molecular [`Frame`].
///
/// # Example (JavaScript)
///
/// ```js
/// const ir = SmilesIr.parse("c1ccccc1"); // benzene
/// const frame3d = new Conformer(true, 42).generate(ir.toFrame());
///
/// const atoms = frame3d.get("atoms");
/// const x = atoms.view("x"); // zero-copy Float64Array of the 3D x-coords
/// ```
#[wasm_bindgen]
pub struct Conformer {
    inner: RsConformer,
}

#[wasm_bindgen]
impl Conformer {
    /// The pipeline is fixed: ETKDG embedding, then an MMFF94 cleanup
    /// minimization.
    ///
    /// * `addHydrogens` - Add implicit hydrogens before embedding (default
    ///   `true`).
    /// * `seed` - Optional RNG seed (`u32`) for reproducibility. If
    ///   omitted, a random seed is used.
    #[wasm_bindgen(constructor)]
    pub fn new(add_hydrogens: Option<bool>, seed: Option<u32>) -> Conformer {
        let defaults = ConformerOptions::default();
        Conformer {
            inner: RsConformer::new(ConformerOptions {
                add_hydrogens: add_hydrogens.unwrap_or(defaults.add_hydrogens),
                rng_seed: seed.map(u64::from),
                ..defaults
            }),
        }
    }

    /// Generate 3D coordinates for a molecular [`Frame`].
    ///
    /// The input frame must have an `"atoms"` block with an
    /// `"atomic_number"` (`uint`) column on every atom — what `SmilesIr.toFrame`
    /// writes. Species is never read from `"element"`: a frame that spells
    /// symbols (e.g. from `readXyz`) needs `fillAtomicNumbers(frame)`
    /// first, and without it this throws. A `"bonds"` block with `atomi`,
    /// `atomj` and the bond order (`bond_type` / `bond_number`, as
    /// `SmilesIr.toFrame` writes them) is required for correct geometry.
    ///
    /// Returns a **new** [`Frame`] with 3D coordinates as `x`, `y`, `z`
    /// (angstrom) in the `"atoms"` block; the input is not modified.
    ///
    /// # Errors
    ///
    /// Throws if:
    /// - The frame has no `"atoms"` block, or an atom has no valid
    ///   `atomic_number` (the message names it; convert with
    ///   `fillAtomicNumbers`)
    /// - The molecular graph has invalid valences or topology
    /// - The 3D embedding fails to converge
    /// - A property of the result contradicts the Frame schema on the way out
    pub fn generate(&self, frame: &Frame) -> Result<Frame, JsValue> {
        let atomistic = frame.with_frame(|rs_frame| {
            Atomistic::from_frame(rs_frame)
                .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
        })?;

        let (result, _report) = self
            .inner
            .generate(&atomistic)
            .map_err(|e| JsValue::from_str(&format!("conformer: {e}")))?;

        Frame::from_rs(
            result
                .to_frame()
                .map_err(|e| JsValue::from_str(&format!("toFrame: {e}")))?,
        )
    }
}
