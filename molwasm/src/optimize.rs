//! Geometry optimization — the WASM face of `molrs::optimize` (`Lbfgs`).
//!
//! ```js
//! const typifier = new UffTypifier();
//! const typed    = typifier.typify(frame);
//! const pots     = new PotentialCompiler(typifier.forcefield()).compile(typed);
//! const nl       = new NeighborList(12.5);         // or NeighborList.bruteForce
//! nl.build(typed);
//! const report   = new Lbfgs(pots, nl.neighbors()).minimize(typed);
//! ```
//!
//! The non-bonded pairs come from a [`NeighborList`](crate::core::NeighborList)
//! when the frame has no `pairs` block. A frame that already has one keeps
//! that list, cut to the neighbor table. The optimizer builds no pair list
//! of its own.

use std::sync::Arc;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use molrs::core::Frame as RsFrame;
use molrs::core::Neighbors as RsNeighbors;
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{Potential, Potentials as RsPotentials, install_pairs};
use molrs::optimize::{Lbfgs as RsLbfgs, LbfgsSettings, Optimizer, set_free_mask};

use crate::core::Neighbors;
use crate::core::frame::Frame;
use crate::ff::Potentials;

// ── Lbfgs ───────────────────────────────────────────────────────────────────

/// Limited-memory BFGS.
///
/// Construct with potentials and a neighbor table, then `minimize(frame)`.
///
/// The table is a [`NeighborList`](crate::core::NeighborList)'s
/// [`Neighbors`], built at the force field's non-bonded cutoff (or
/// `NeighborList.bruteForce` for a small molecule). Its pairs are installed
/// as the frame's `pairs` with the force field's `special_bonds` applied, by
/// `molrs::ff::potential::install_pairs`.
#[wasm_bindgen(js_name = Lbfgs)]
pub struct Lbfgs {
    ff: RsForceField,
    /// Latest compiled kernels (rebuilt at each `minimize` after pair install).
    pots: Arc<RsPotentials>,
    /// The constructor's neighbour table.
    neighbors: RsNeighbors,
    settings: LbfgsSettings,
}

#[wasm_bindgen(js_class = Lbfgs)]
impl Lbfgs {
    /// Bind `pots` and the [`Neighbors`] table from
    /// [`NeighborList::neighbors`](crate::core::NeighborList::neighbors).
    ///
    /// Knobs default to molrs `LbfgsSettings::DEFAULT` when omitted:
    /// `fmax` (0.05), `maxStep` (0.2), `memory` (8). The step budget is
    /// `minimize`'s.
    ///
    /// # Errors
    ///
    /// Throws for a given `fmax` or `maxStep` that is not a positive finite
    /// number, or a `memory` of 0 — a value is refused, never adjusted.
    #[wasm_bindgen(constructor)]
    pub fn new(
        pots: &Potentials,
        neighbors: &Neighbors,
        fmax: Option<f64>,
        max_step: Option<f64>,
        memory: Option<usize>,
    ) -> Result<Lbfgs, JsValue> {
        let defaults = LbfgsSettings::DEFAULT;
        let positive = |name: &str, value: Option<f64>, default: f64| match value {
            None => Ok(default),
            Some(v) if v.is_finite() && v > 0.0 => Ok(v),
            Some(v) => Err(JsValue::from_str(&format!(
                "new Lbfgs: {name} must be a positive finite number, got {v}"
            ))),
        };
        let memory = match memory {
            None => defaults.memory,
            Some(0) => return Err(JsValue::from_str("new Lbfgs: memory must be at least 1")),
            Some(m) => m,
        };
        Ok(Lbfgs {
            ff: pots.ff.clone(),
            pots: Arc::clone(&pots.inner),
            neighbors: RsNeighbors::clone(&neighbors.inner),
            settings: LbfgsSettings {
                fmax: positive("fmax", fmax, defaults.fmax)?,
                max_step: positive("maxStep", max_step, defaults.max_step)?,
                memory,
                max_steps: defaults.max_steps,
            },
        })
    }

    /// Minimize `frame` coordinates **in place** for up to `maxSteps`
    /// iterations (molrs `LbfgsSettings::DEFAULT.max_steps`, 500, when
    /// omitted).
    ///
    /// Installs the constructor's neighbour pairs, recompiles potentials,
    /// then runs L-BFGS. Optional `fixed`: dense atom indices held fixed.
    ///
    /// # Errors
    ///
    /// Throws for a `maxSteps` of 0, a `fixed` index outside the frame's
    /// atoms, or what pair installation, compilation or the minimizer
    /// refuses.
    pub fn minimize(
        &mut self,
        frame: &Frame,
        max_steps: Option<usize>,
        fixed: Option<Uint32Array>,
    ) -> Result<OptimizationReport, JsValue> {
        let max_steps = match max_steps {
            None => self.settings.max_steps,
            Some(0) => {
                return Err(JsValue::from_str(
                    "Lbfgs.minimize: maxSteps must be at least 1",
                ));
            }
            Some(n) => n,
        };
        let settings = LbfgsSettings {
            max_steps,
            ..self.settings
        };

        // Install pairs + recompile, then minimize — all on the same borrow.
        let report = frame
            .inner
            .with_mut(
                |rs| -> Result<molrs::optimize::OptimizationReport, String> {
                    let pairs = install_pairs(
                        rs,
                        self.ff.required_special_bonds()?,
                        Some(&self.neighbors),
                    )?;
                    rs.insert("pairs", pairs);
                    let compiled = PotentialCompiler::new(&self.ff)
                        .compile(rs)
                        .map_err(|e| format!("compile: {e}"))?;
                    self.pots = Arc::new(compiled);

                    if let Some(ref fixed) = fixed {
                        apply_fixed_mask(rs, fixed)?;
                    }

                    let pot: Arc<dyn Potential> = Arc::clone(&self.pots) as Arc<dyn Potential>;
                    let report = RsLbfgs::new(pot, settings)
                        .minimize(rs)
                        .map_err(|e| e.to_string())?;

                    if fixed.is_some()
                        && let Some(atoms) = rs.get_mut("atoms")
                    {
                        let _ = atoms.remove("free");
                    }
                    Ok(report)
                },
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?
            .map_err(|e| JsValue::from_str(&e))?;

        Ok(OptimizationReport { inner: report })
    }
}

// ── OptimizationReport ──────────────────────────────────────────────────────

/// Outcome of one minimization (molrs `optimize::OptimizationReport`).
#[wasm_bindgen(js_name = OptimizationReport)]
pub struct OptimizationReport {
    inner: molrs::optimize::OptimizationReport,
}

#[wasm_bindgen(js_class = OptimizationReport)]
impl OptimizationReport {
    /// Whether `fmax` convergence was reached within the step budget.
    #[wasm_bindgen(getter)]
    pub fn converged(&self) -> bool {
        self.inner.converged
    }
    /// Outer iterations performed.
    #[wasm_bindgen(getter, js_name = nSteps)]
    pub fn n_steps(&self) -> usize {
        self.inner.n_steps
    }
    /// Potential energy at the returned geometry.
    #[wasm_bindgen(getter, js_name = finalEnergy)]
    pub fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }
    /// Largest per-atom force magnitude at the returned geometry.
    #[wasm_bindgen(getter, js_name = finalFmax)]
    pub fn final_fmax(&self) -> f64 {
        self.inner.final_fmax
    }
    /// RMS gradient component at the returned geometry.
    #[wasm_bindgen(getter, js_name = finalGradRms)]
    pub fn final_grad_rms(&self) -> f64 {
        self.inner.final_grad_rms
    }
}

fn apply_fixed_mask(frame: &mut RsFrame, fixed: &Uint32Array) -> Result<(), String> {
    if fixed.length() == 0 {
        return Ok(());
    }
    let n = frame.get("atoms").and_then(|b| b.n_rows()).unwrap_or(0);
    let mut free = vec![true; n];
    for idx in fixed.to_vec() {
        let slot = free.get_mut(idx as usize).ok_or_else(|| {
            format!("Lbfgs.minimize: fixed atom {idx} is outside the frame's 0..{n} atoms")
        })?;
        *slot = false;
    }
    set_free_mask(frame, &free).map_err(|e| e.to_string())
}
