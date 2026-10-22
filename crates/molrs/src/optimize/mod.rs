//! Geometry optimization of a [`Frame`].
//!
//! The optimizer contract lives here and is always compiled: the
//! [`Optimizer`] trait, its [`OptimizationReport`], the [`LbfgsSettings`]
//! defaults and the `atoms.free` mask
//! ([`set_free_mask`]). A purely geometric optimizer (e.g. a packer's
//! torsion Monte Carlo) implements [`Optimizer`] without enabling `ff`. The
//! force-field-agnostic L-BFGS engine behind `Lbfgs` is crate-internal; the
//! ETKDG conformer stages drive it on their own distance-geometry energies.
//!
//! The optimizer that minimizes a force-field potential
//! (`ff::potential::Potential`), `Lbfgs` — over an owned or a borrowed
//! potential; a `Frame`, one flat coordinate buffer or a batch of them — is
//! gated on `ff`. It is the one front door for every potential — a potential
//! that rebuilds its pairs as the atoms move
//! (`ff::potential::soft::SoftPotential`) does so itself.
//! The dependency points one way: `optimize` consumes `ff`, never the
//! reverse.

#[cfg(feature = "ff")]
mod lbfgs;
#[cfg(feature = "ff")]
mod lbfgs_driver;

#[cfg(feature = "ff")]
pub use lbfgs_driver::Lbfgs;

use crate::core::Frame;
use crate::core::keys::FREE;
use crate::core::schema::block_names::ATOMS;
use crate::op::F;
use ndarray::Array1;

#[cfg(feature = "conformer")]
pub(crate) use lbfgs::minimize_lbfgs_rms;

/// Outcome of one minimization — the one report every minimizer in molrs
/// returns, the geometry optimizer and the conformer stages alike.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationReport {
    /// Whether the convergence criterion was met within the step budget.
    pub converged: bool,
    /// Number of outer iterations performed.
    pub n_steps: usize,
    /// Potential energy at the returned point, in the potential's energy unit
    /// (kcal/mol for the force fields molrs ships).
    pub final_energy: F,
    /// Maximum per-atom force magnitude at the returned point (energy / Å).
    pub final_fmax: F,
    /// Root-mean-square gradient component at the returned point (energy / Å).
    pub final_grad_rms: F,
}

impl OptimizationReport {
    /// The report of a minimization that stopped at `energy` with gradient
    /// `grad` (= −forces, flat `3·n_atoms`).
    #[cfg(feature = "ff")]
    pub(crate) fn from_gradient(converged: bool, n_steps: usize, energy: F, grad: &[F]) -> Self {
        Self {
            converged,
            n_steps,
            final_energy: energy,
            final_fmax: lbfgs::fmax_from_grad(grad),
            final_grad_rms: lbfgs::grad_rms(grad),
        }
    }

    /// The report of an empty system: nothing to move, converged at zero.
    #[cfg(any(feature = "ff", test))]
    pub(crate) const EMPTY: Self = Self {
        converged: true,
        n_steps: 0,
        final_energy: 0.0,
        final_fmax: 0.0,
        final_grad_rms: 0.0,
    };
}

/// The knobs of an L-BFGS geometry minimization, and their one set of
/// defaults — every binding reads [`LbfgsSettings::DEFAULT`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbfgsSettings {
    /// Converged when the largest per-atom force falls below this (energy / Å).
    pub fmax: F,
    /// Most outer iterations before giving up.
    pub max_steps: usize,
    /// Largest displacement of any coordinate in one step (Å): the trust
    /// region.
    pub max_step: F,
    /// Correction pairs kept in the L-BFGS history.
    pub memory: usize,
}

impl LbfgsSettings {
    /// `fmax = 0.05`, `max_steps = 500`, `max_step = 0.2`, `memory = 8`.
    pub const DEFAULT: Self = Self {
        fmax: 0.05,
        max_steps: 500,
        max_step: 0.2,
        memory: 8,
    };
}

impl Default for LbfgsSettings {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Geometry optimizer: minimize a [`Frame`] in place.
///
/// Optional free mask: bool column `atoms.free` (missing ⇒ every atom free).
/// Fixed atoms stay in the potential evaluation but are not optimizable DOFs.
pub trait Optimizer: Send + Sync {
    /// Relax `frame` in place. Coordinates are `atoms.{x,y,z}`.
    fn minimize(&mut self, frame: &mut Frame) -> Result<OptimizationReport, String>;
}

/// Ensure `atoms.free` exists with the given mask (helper for callers assembling Frames).
pub fn set_free_mask(frame: &mut Frame, free: &[bool]) -> Result<(), String> {
    let atoms = frame
        .get_mut(ATOMS)
        .ok_or_else(|| "Frame has no atoms block".to_string())?;
    let n = atoms.n_rows().unwrap_or(0);
    if free.len() != n {
        return Err(format!(
            "free mask length {} != atoms nrows {n}",
            free.len()
        ));
    }
    atoms
        .insert(FREE, Array1::from_vec(free.to_vec()).into_dyn(), None)
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A purely geometric optimizer: translates the atoms so their centroid
    /// sits at the origin. Needs no force field, so it builds without `ff`.
    struct Center;

    impl Optimizer for Center {
        fn minimize(&mut self, frame: &mut Frame) -> Result<OptimizationReport, String> {
            let mut xyz = frame.coords().map_err(|e| e.to_string())?;
            let centroid = xyz.mean_axis(ndarray::Axis(0)).unwrap_or_default();
            xyz -= &centroid;
            frame.set_coords(xyz.view()).map_err(|e| e.to_string())?;
            Ok(OptimizationReport {
                n_steps: 1,
                ..OptimizationReport::EMPTY
            })
        }
    }

    #[test]
    fn geometric_optimizer_runs_on_a_frame_without_ff() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[1.0 as F, 0.0, 0.0], [3.0, 2.0, 0.0]].view())
            .unwrap();
        let report = Center.minimize(&mut frame).unwrap();
        assert!(report.converged);
        assert_eq!(
            frame.coords().unwrap(),
            array![[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]
        );
    }

    #[test]
    fn set_free_mask_rejects_a_length_mismatch() {
        let mut frame = Frame::new();
        frame
            .set_coords(array![[0.0 as F, 0.0, 0.0], [1.0, 0.0, 0.0]].view())
            .unwrap();
        assert!(set_free_mask(&mut frame, &[true]).is_err());
        set_free_mask(&mut frame, &[false, true]).unwrap();
        let free = frame
            .get(ATOMS)
            .unwrap()
            .get(FREE)
            .and_then(|c| c.get::<bool>())
            .unwrap();
        assert_eq!(free.iter().copied().collect::<Vec<_>>(), vec![false, true]);
    }
}
