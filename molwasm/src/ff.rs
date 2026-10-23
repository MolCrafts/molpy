//! Force-field WASM face — typifiers, the force field they output and the
//! compiler that turns it into potentials; the native molrs composition
//! (`molrs::ff::typifier`, `molrs::ff::potential`), as in Python. Minimizing
//! with those potentials is `optimize`'s job (`Lbfgs`).
//!
//! ```js
//! const typifier = new UffTypifier();
//! const typed    = typifier.typify(frame);
//! const pots     = new PotentialCompiler(typifier.forcefield()).compile(typed);
//! ```
//!
//! A typifier class is named after the native typifier and, as Python's
//! typifier classes do, folds in the `Typing<…>` driver: `typify` labels a
//! frame and is the only writer of the output force field `forcefield()`.
//! There is no shortcut from a typifier to potentials.

use std::sync::Arc;

use wasm_bindgen::prelude::*;

use molrs::core::Atomistic;
use molrs::ff::compile::PotentialCompiler as RsPotentialCompiler;
use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{Accumulator, Potential, Potentials as RsPotentials};
use molrs::ff::typifier::Typing;
use molrs::ff::typifier::UffTypifier as RsUff;
use molrs::ff::typifier::mmff::{Mmff94Typifier as RsMmff94, Mmff94sTypifier as RsMmff94s};

use crate::core::frame::Frame;

// ── Typifiers ───────────────────────────────────────────────────────────────

macro_rules! wasm_typifier {
    (
        $(#[$meta:meta])*
        $JsName:ident, $RsType:ty, $ctor:expr
    ) => {
        $(#[$meta])*
        #[wasm_bindgen(js_name = $JsName)]
        pub struct $JsName {
            inner: Typing<$RsType>,
        }

        #[wasm_bindgen(js_class = $JsName)]
        impl $JsName {
            #[wasm_bindgen(constructor)]
            pub fn new() -> $JsName {
                $JsName {
                    inner: Typing::new($ctor),
                }
            }

            /// Typify a molecular [`Frame`]. Returns a **new** labeled frame;
            /// `frame` is untouched. Species is read from
            /// `atoms.atomic_number` only: a frame that spells symbols
            /// (`atoms.element`) needs `fillAtomicNumbers(frame)` first, and
            /// an atom without a valid atomic number throws. Every call
            /// accumulates its definitions
            /// into this typifier's private output force field; a conflicting
            /// definition is an error and leaves the output unchanged.
            ///
            /// Native: `Typing::new(typifier).typify(&mol)?.to_frame()?`.
            pub fn typify(&mut self, frame: &Frame) -> Result<Frame, JsValue> {
                let mol = frame.with_frame(|rs| {
                    Atomistic::from_frame(rs).map_err(|e| {
                        JsValue::from_str(&format!("Frame → Atomistic: {e}"))
                    })
                })?;
                let typed = self
                    .inner
                    .typify(&mol)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                Frame::from_rs(
                    typed.to_frame().map_err(|e| {
                        JsValue::from_str(&format!("toFrame: {e}"))
                    })?,
                )
            }

            /// A copy of the output force field: exactly the definitions
            /// [`typify`](Self::typify) has assigned so far.
            ///
            /// Native: `Typing::forcefield()`.
            pub fn forcefield(&self) -> ForceField {
                ForceField {
                    inner: self.inner.forcefield().clone(),
                }
            }
        }

        impl Default for $JsName {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

wasm_typifier!(
    /// Universal Force Field typifier (full RDKit default table).
    UffTypifier,
    RsUff,
    RsUff::new()
);

wasm_typifier!(
    /// MMFF94 typifier.
    Mmff94Typifier,
    RsMmff94,
    RsMmff94::new()
);

wasm_typifier!(
    /// MMFF94s typifier (static / planar amide N).
    Mmff94sTypifier,
    RsMmff94s,
    RsMmff94s::new()
);

// ── ForceField ──────────────────────────────────────────────────────────────

/// A force field (molrs `ff::forcefield::ForceField`): what a typifier's
/// `forcefield()` returns and a [`PotentialCompiler`] compiles.
#[wasm_bindgen(js_name = ForceField)]
pub struct ForceField {
    pub(crate) inner: RsForceField,
}

// ── PotentialCompiler ───────────────────────────────────────────────────────

/// Compiles a [`ForceField`] into evaluable [`Potentials`] — molrs
/// `ff::compile::PotentialCompiler`. Holds a copy of the force field taken
/// at construction.
#[wasm_bindgen(js_name = PotentialCompiler)]
pub struct PotentialCompiler {
    ff: RsForceField,
}

#[wasm_bindgen(js_class = PotentialCompiler)]
impl PotentialCompiler {
    #[wasm_bindgen(constructor)]
    pub fn new(forcefield: &ForceField) -> PotentialCompiler {
        PotentialCompiler {
            ff: forcefield.inner.clone(),
        }
    }

    /// Compile the potentials of a **typed** frame. Non-bonded terms need a
    /// `pairs` block; `Lbfgs.minimize` installs that list (from the
    /// [`Neighbors`](crate::core::Neighbors) table it was constructed with)
    /// and recompiles before minimizing, so compiling a frame with no `pairs`
    /// yields its bonded kernels only.
    pub fn compile(&self, frame: &Frame) -> Result<Potentials, JsValue> {
        let pots = frame.with_frame(|rs| {
            RsPotentialCompiler::new(&self.ff)
                .compile(rs)
                .map_err(|e| JsValue::from_str(&format!("compile: {e}")))
        })?;
        Ok(Potentials {
            ff: self.ff.clone(),
            inner: Arc::new(pots),
        })
    }
}

// ── Potentials ──────────────────────────────────────────────────────────────

/// Compiled kernels (molrs `ff::potential::Potentials`). Holds the force field so [`Lbfgs`](crate::optimize::Lbfgs) can recompile
/// after installing a neighbour list.
#[wasm_bindgen(js_name = Potentials)]
pub struct Potentials {
    pub(crate) ff: RsForceField,
    pub(crate) inner: Arc<RsPotentials>,
}

#[wasm_bindgen(js_class = Potentials)]
impl Potentials {
    /// `{ energy: number, forces: Float64Array }` for flat 3N coordinates —
    /// molrs `Potential::evaluate` into a fresh `Accumulator`.
    ///
    /// # Errors
    ///
    /// Throws if `coords` is not a flat `[x0, y0, z0, x1, …]` array (its
    /// length a multiple of 3).
    pub fn evaluate(&self, coords: &[f64]) -> Result<JsValue, JsValue> {
        if !coords.len().is_multiple_of(3) {
            return Err(JsValue::from_str(&format!(
                "Potentials.evaluate: coords must be flat 3N; got {} values",
                coords.len()
            )));
        }
        let mut acc = Accumulator::new(coords.len());
        self.inner.evaluate(coords, &mut acc);
        let e = acc.energy();
        let f = acc.into_forces();
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"energy".into(), &JsValue::from_f64(e))?;
        let fa = js_sys::Float64Array::new_with_length(f.len() as u32);
        fa.copy_from(&f);
        js_sys::Reflect::set(&obj, &"forces".into(), &fa)?;
        Ok(obj.into())
    }
}
