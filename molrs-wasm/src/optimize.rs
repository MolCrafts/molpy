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
//! and nowhere else: the optimizer builds no pair list of its own.

use std::sync::Arc;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use molrs::core::Frame as RsFrame;
use molrs::core::Neighbors as RsNeighbors;
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::forcefield::ForceField as RsForceField;
use molrs::ff::potential::{
    Potential, Potentials as RsPotentials, intramolecular_pairs_from_neighbors,
};
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
/// `molrs::ff::potential::intramolecular_pairs_from_neighbors`.
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
    /// Knobs default to molrs `LbfgsSettings::DEFAULT`: `fmax` (0.05),
    /// `maxStep` (0.2), `memory` (8). The step budget is `minimize`'s.
    #[wasm_bindgen(constructor)]
    pub fn new(
        pots: &Potentials,
        neighbors: &Neighbors,
        fmax: Option<f64>,
        max_step: Option<f64>,
        memory: Option<usize>,
    ) -> Lbfgs {
        let defaults = LbfgsSettings::DEFAULT;
        Lbfgs {
            ff: pots.ff.clone(),
            pots: Arc::clone(&pots.inner),
            neighbors: RsNeighbors::clone(&neighbors.inner),
            settings: LbfgsSettings {
                fmax: fmax.unwrap_or(defaults.fmax).max(1e-8),
                max_step: max_step.unwrap_or(defaults.max_step).max(1e-6),
                memory: memory.unwrap_or(defaults.memory).max(1),
                max_steps: defaults.max_steps,
            },
        }
    }

    /// Minimize `frame` coordinates **in place** for up to `maxSteps`
    /// iterations (molrs `LbfgsSettings::DEFAULT.max_steps`, 500, when
    /// omitted).
    ///
    /// Installs the constructor's neighbour pairs, recompiles potentials,
    /// then runs L-BFGS. Optional `fixed`: dense atom indices held fixed.
    pub fn minimize(
        &mut self,
        frame: &Frame,
        max_steps: Option<usize>,
        fixed: Option<Uint32Array>,
    ) -> Result<OptimizationReport, JsValue> {
        let settings = LbfgsSettings {
            max_steps: max_steps.unwrap_or(self.settings.max_steps).max(1),
            ..self.settings
        };

        // Install pairs + recompile, then minimize — all on the same borrow.
        let report = frame
            .inner
            .with_mut(
                |rs| -> Result<molrs::optimize::OptimizationReport, String> {
                    let pairs = intramolecular_pairs_from_neighbors(
                        rs,
                        self.ff.special_bonds(),
                        &self.neighbors,
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
                    let report = RsLbfgs::new(pot, settings).minimize(rs)?;

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
    let n = frame.get("atoms").and_then(|b| b.n_rows()).unwrap_or(0);
    if n == 0 || fixed.length() == 0 {
        return Ok(());
    }
    let mut free = vec![true; n];
    for i in 0..fixed.length() {
        let idx = fixed.get_index(i) as usize;
        if idx < n {
            free[idx] = false;
        }
    }
    set_free_mask(frame, &free)
}
