//! 3D coordinate generation: ETKDGv3 distance-geometry embedding.
//!
//! Embeds a molecular graph (2D connectivity) in 3D — RDKit's `EmbedMolecule`.
//! No force field runs; a force-field relaxation is a separate step.
//!
//! # Pipeline stages
//!
//! 1. **ETKDGv3 constraints** -- build and smooth the distance-bounds matrix.
//! 2. **Distance-geometry embedding** -- sample distances and embed in 4D.
//! 3. **ETKDG refinement** -- apply distance, chirality, and torsion knowledge.
//! 4. **Stereo guards** -- verify stereochemistry is preserved.
//!
//! # References
//!
//! - Riniker, S.; Landrum, G.A. (2015). *J. Chem. Inf. Model.* 55, 2562-2574.

use wasm_bindgen::prelude::*;

use molrs::conformer::Conformer as RsConformer;
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
/// const frame3d = new Conformer(true, 0, 42, false, true, true).generate(ir.toFrame());
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
    /// RDKit `EmbedParameters` names; an omitted argument takes RDKit's
    /// `ETKDGv3()` default.
    ///
    /// * `addHydrogens` - Add implicit hydrogens before embedding (`true`).
    /// * `maxIterations` - Embedding attempts; `0` is `10 × nAtoms` (`0`).
    /// * `seed` - RNG seed (`u32`) for reproducibility; omitted, one is drawn.
    /// * `useRandomCoords` - Start from random coordinates (`false`).
    /// * `useExpTorsionAnglePrefs` - The experimental-torsion terms (`true`).
    /// * `useBasicKnowledge` - Flat rings and planar sp2 centres (`true`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        add_hydrogens: Option<bool>,
        max_iterations: Option<u32>,
        seed: Option<u32>,
        use_random_coords: Option<bool>,
        use_exp_torsion_angle_prefs: Option<bool>,
        use_basic_knowledge: Option<bool>,
    ) -> Conformer {
        Conformer {
            inner: RsConformer::new(
                add_hydrogens.unwrap_or(true),
                max_iterations.map_or(0, |n| n as usize),
                seed.map(u64::from),
                use_random_coords.unwrap_or(false),
                use_exp_torsion_angle_prefs.unwrap_or(true),
                use_basic_knowledge.unwrap_or(true),
            ),
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
