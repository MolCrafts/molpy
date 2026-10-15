//! The L-BFGS geometry optimizer over a force-field [`Potential`]: [`Lbfgs`],
//! the one object for a [`Frame`] ([`Optimizer::minimize`]), a flat
//! coordinate buffer ([`Lbfgs::minimize_coords`]) and a homogeneous batch of
//! them ([`Lbfgs::minimize_batch`]).
//!
//! Gated on `ff` (the potential trait lives there); the [`Optimizer`] trait
//! it implements is always compiled.

use std::ops::Deref;

use super::lbfgs::{Converge, minimize_core};
use super::{LbfgsSettings, OptimizationReport, Optimizer};
use crate::core::Frame;
use crate::core::keys::FREE;
use crate::core::schema::block_names::ATOMS;
use crate::ff::potential::{Accumulator, Potential};
use crate::op::F;

/// Check a flat coordinate buffer: `Ok(false)` when there is nothing to
/// move, `Err` when its length is not `3·n_atoms`.
fn has_atoms(coords: &[F]) -> Result<bool, String> {
    if coords.is_empty() {
        return Ok(false);
    }
    if !coords.len().is_multiple_of(3) {
        return Err(format!(
            "coords length {} is not a multiple of 3 (expected 3·n_atoms)",
            coords.len()
        ));
    }
    Ok(true)
}

/// One evaluation of `potential` at `coords` into fresh storage: the
/// `(energy, forces)` the L-BFGS core consumes.
fn energy_forces<P: Potential + ?Sized>(potential: &P, coords: &[F]) -> (F, Vec<F>) {
    let mut acc = Accumulator::new(coords.len());
    potential.evaluate(coords, &mut acc);
    (acc.energy(), acc.into_forces())
}

/// Limited-memory BFGS over a force-field [`Potential`].
///
/// One front door for every potential, including one that rebuilds its own
/// pairs as the atoms move (`ff::potential::soft::SoftPotential`).
///
/// `P` is whatever holds the potential: an owned handle
/// (`Arc<dyn Potential>`, `Box<Potentials>`, …) for an optimizer that is
/// stored, or a borrow (`&dyn Potential`) for a one-shot run under a
/// potential someone else owns.
///
/// Construct with [`Lbfgs::new`] (potential + [`LbfgsSettings`]). Primary call
/// is [`Optimizer::minimize`] on a [`Frame`], which honours the frame's
/// `atoms.free` mask; [`minimize_coords`](Lbfgs::minimize_coords) relaxes one
/// flat coordinate buffer and [`minimize_batch`](Lbfgs::minimize_batch) a
/// stack of them when a Frame is not available.
pub struct Lbfgs<P> {
    potential: P,
    settings: LbfgsSettings,
}

impl<P> Lbfgs<P>
where
    P: Deref,
    P::Target: Potential,
{
    /// Bind `potential` with `settings` ([`LbfgsSettings::DEFAULT`] for the
    /// defaults).
    pub fn new(potential: P, settings: LbfgsSettings) -> Self {
        Self {
            potential,
            settings,
        }
    }

    /// The settings this optimizer runs with.
    pub fn settings(&self) -> &LbfgsSettings {
        &self.settings
    }

    /// Relax flat `3·n_atoms` coordinates in place, every atom free.
    ///
    /// # Errors
    /// Returns `Err` if `coords.len()` is not a multiple of three.
    pub fn minimize_coords(&self, coords: &mut [F]) -> Result<OptimizationReport, String> {
        if !has_atoms(coords)? {
            return Ok(OptimizationReport::EMPTY);
        }
        let potential = &*self.potential;
        let (energy, grad, n_steps, converged) = minimize_core(
            coords,
            self.settings.max_steps,
            Converge::Fmax(self.settings.fmax),
            self.settings.max_step,
            self.settings.memory,
            |c| energy_forces(potential, c),
        );
        Ok(OptimizationReport::from_gradient(
            converged, n_steps, energy, &grad,
        ))
    }

    /// Relax a homogeneous batch — `n_structs` structures of `n_atoms` atoms,
    /// stacked in one flat buffer — each independently, one report per
    /// structure.
    ///
    /// # Errors
    ///
    /// When `coords.len() != n_structs · n_atoms · 3`, or `n_atoms` is zero
    /// for a non-empty batch.
    pub fn minimize_batch(
        &self,
        coords: &mut [F],
        n_atoms: usize,
        n_structs: usize,
    ) -> Result<Vec<OptimizationReport>, String> {
        let stride = n_atoms * 3;
        let expected = n_structs * stride;
        if coords.len() != expected {
            return Err(format!(
                "coords length {} != n_structs ({}) · n_atoms ({}) · 3 = {}",
                coords.len(),
                n_structs,
                n_atoms,
                expected
            ));
        }
        if n_structs == 0 {
            return Ok(Vec::new());
        }
        if stride == 0 {
            return Err(format!(
                "n_atoms must be > 0 for a batch of {n_structs} structures"
            ));
        }
        coords
            .chunks_mut(stride)
            .map(|block| self.minimize_coords(block))
            .collect()
    }

    /// Minimize free DOFs only, evaluating the potential on the full system.
    fn minimize_masked(&self, full: &mut [F], free: &[bool]) -> Result<OptimizationReport, String> {
        let n = free.len();
        if full.len() != n * 3 {
            return Err(format!(
                "free mask length {n} does not match coords atom count {}",
                full.len() / 3
            ));
        }
        let free_idx: Vec<usize> = free
            .iter()
            .enumerate()
            .filter_map(|(i, &f)| if f { Some(i) } else { None })
            .collect();
        if free_idx.is_empty() {
            let (e, forces) = energy_forces(&*self.potential, full);
            let grad: Vec<F> = forces.iter().map(|f| -f).collect();
            return Ok(OptimizationReport::from_gradient(true, 0, e, &grad));
        }
        if free_idx.len() == n {
            return self.minimize_coords(full);
        }

        let mut x_free = Vec::with_capacity(free_idx.len() * 3);
        for &i in &free_idx {
            x_free.extend_from_slice(&full[3 * i..3 * i + 3]);
        }

        let pot = &*self.potential;
        let free_idx_c = free_idx.clone();
        let mut full_buf = full.to_vec();

        let (final_energy, grad_free, n_steps, converged) = minimize_core(
            &mut x_free,
            self.settings.max_steps,
            Converge::Fmax(self.settings.fmax),
            self.settings.max_step,
            self.settings.memory,
            |xf| {
                for (k, &i) in free_idx_c.iter().enumerate() {
                    full_buf[3 * i] = xf[3 * k];
                    full_buf[3 * i + 1] = xf[3 * k + 1];
                    full_buf[3 * i + 2] = xf[3 * k + 2];
                }
                let (e, forces) = energy_forces(pot, &full_buf);
                let mut f_free = vec![0.0; xf.len()];
                for (k, &i) in free_idx_c.iter().enumerate() {
                    f_free[3 * k] = forces[3 * i];
                    f_free[3 * k + 1] = forces[3 * i + 1];
                    f_free[3 * k + 2] = forces[3 * i + 2];
                }
                (e, f_free)
            },
        );

        for (k, &i) in free_idx.iter().enumerate() {
            full[3 * i] = x_free[3 * k];
            full[3 * i + 1] = x_free[3 * k + 1];
            full[3 * i + 2] = x_free[3 * k + 2];
        }

        Ok(OptimizationReport::from_gradient(
            converged,
            n_steps,
            final_energy,
            &grad_free,
        ))
    }
}

impl<P> Optimizer for Lbfgs<P>
where
    P: Deref + Send + Sync,
    P::Target: Potential,
{
    fn minimize(&mut self, frame: &mut Frame) -> Result<OptimizationReport, String> {
        let mut xyz = frame.coords().map_err(|e| e.to_string())?;
        let free = frame_free_mask(frame, xyz.nrows())?;
        // `Frame::coords` builds a fresh row-major N×3 array, so its buffer is
        // the flat `[x0, y0, z0, x1, …]` the minimizer works on.
        let coords = xyz
            .as_slice_mut()
            .expect("Frame::coords returns a standard-layout array");
        let report = match free {
            None => self.minimize_coords(coords)?,
            Some(mask) => self.minimize_masked(coords, &mask)?,
        };
        frame.set_coords(xyz.view()).map_err(|e| e.to_string())?;
        Ok(report)
    }
}

// ── Frame helpers ────────────────────────────────────────────────────────────

/// `None` ⇒ all free. `Some(mask)` length = n_atoms.
fn frame_free_mask(frame: &Frame, n_atoms: usize) -> Result<Option<Vec<bool>>, String> {
    let Some(atoms) = frame.get(ATOMS) else {
        return Ok(None);
    };
    let Some(col) = atoms.get(FREE).and_then(|c| c.get::<bool>()) else {
        return Ok(None);
    };
    if col.len() != n_atoms {
        return Err(format!(
            "atoms.free length {} != n_atoms {n_atoms}",
            col.len()
        ));
    }
    Ok(Some(col.iter().copied().collect()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Block;
    use crate::optimize::set_free_mask;
    use ndarray::Array1;
    use std::sync::Arc;

    struct HarmonicBond {
        k: F,
        r0: F,
    }

    impl Potential for HarmonicBond {
        fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
            let d = [
                coords[3] - coords[0],
                coords[4] - coords[1],
                coords[5] - coords[2],
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            acc.add_energy(0.5 * self.k * (r - self.r0) * (r - self.r0));
            if r > 1e-12 {
                let coeff = self.k * (r - self.r0) / r;
                let f = acc.forces_without_virial();
                for i in 0..3 {
                    let fi = coeff * d[i];
                    f[i] += fi;
                    f[3 + i] -= fi;
                }
            }
        }
    }

    fn opt<T: Potential>(pot: T) -> Lbfgs<Arc<T>> {
        Lbfgs::new(Arc::new(pot), LbfgsSettings::DEFAULT)
    }

    fn frame_from_coords(coords: &[F]) -> Frame {
        let n = coords.len() / 3;
        let mut atoms = Block::new();
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            x.push(coords[3 * i]);
            y.push(coords[3 * i + 1]);
            z.push(coords[3 * i + 2]);
        }
        atoms
            .insert("x", Array1::from_vec(x).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(y).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(z).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    #[test]
    fn relaxes_harmonic_bond_to_equilibrium() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let report = opt(pot).minimize_coords(&mut coords).unwrap();
        assert!(report.converged, "should converge: {report:?}");
        let r = coords[3] - coords[0];
        assert!((r.abs() - 1.0).abs() < 1e-6, "bond length got {r}");
        assert!(report.final_energy < 1e-9);
        assert!(report.final_fmax <= 0.05);
    }

    #[test]
    fn minimize_frame_updates_xyz() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]);
        let report = opt(pot).minimize(&mut frame).unwrap();
        assert!(report.converged);
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!((x[[1]] - x[[0]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn free_mask_freezes_fixed_atom() {
        // Atom 0 fixed at origin; atom 1 free. Bond wants r0=1.
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]);
        set_free_mask(&mut frame, &[false, true]).unwrap();
        opt(pot).minimize(&mut frame).unwrap();
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!(x[[0]].abs() < 1e-9, "fixed atom moved: {}", x[[0]]);
        assert!((x[[1]] - 1.0).abs() < 1e-5, "free atom should sit at r0");
    }

    #[test]
    fn fmax_convergence_semantics() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let r = Lbfgs::new(
            Arc::new(pot),
            LbfgsSettings {
                fmax: 0.05,
                max_steps: 1,
                max_step: 0.2,
                memory: 8,
            },
        )
        .minimize_coords(&mut coords)
        .unwrap();
        assert!(!r.converged);
        assert_eq!(r.n_steps, 1);
    }

    #[test]
    fn idempotent_at_minimum() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let r = opt(pot).minimize_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert!(r.n_steps <= 1);
    }

    #[test]
    fn single_atom_converges_immediately() {
        struct Free;
        impl Potential for Free {
            fn evaluate(&self, _coords: &[F], _acc: &mut Accumulator<'_>) {}
        }
        let mut coords = vec![0.3, -0.2, 0.1];
        let r = opt(Free).minimize_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert!(r.n_steps <= 1);
    }

    #[test]
    fn rejects_non_multiple_of_three() {
        struct Free;
        impl Potential for Free {
            fn evaluate(&self, _coords: &[F], _acc: &mut Accumulator<'_>) {}
        }
        let mut coords = vec![0.0, 0.0, 0.0, 1.0];
        assert!(opt(Free).minimize_coords(&mut coords).is_err());
    }

    #[test]
    fn empty_coords_is_converged_noop() {
        struct Free;
        impl Potential for Free {
            fn evaluate(&self, _coords: &[F], _acc: &mut Accumulator<'_>) {}
        }
        let mut coords: Vec<F> = vec![];
        let r = opt(Free).minimize_coords(&mut coords).unwrap();
        assert!(r.converged);
        assert_eq!(r.n_steps, 0);
    }

    #[test]
    fn trust_region_caps_step() {
        let pot = HarmonicBond { k: 500.0, r0: 1.0 };
        let mut coords = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let before = coords.clone();
        Lbfgs::new(
            Arc::new(pot),
            LbfgsSettings {
                fmax: 0.05,
                max_steps: 1,
                max_step: 0.01,
                memory: 8,
            },
        )
        .minimize_coords(&mut coords)
        .unwrap();
        for (a, b) in coords.iter().zip(&before) {
            assert!((a - b).abs() <= 0.01 + 1e-12);
        }
    }

    #[test]
    fn batch_equals_serial() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let single_start = vec![0.0, 0.0, 0.0, 1.4, 0.0, 0.0];
        let mut single = single_start.clone();
        let single_report = opt(HarmonicBond { k: 100.0, r0: 1.0 })
            .minimize_coords(&mut single)
            .unwrap();

        let b = 4;
        let mut batch: Vec<F> = Vec::new();
        for _ in 0..b {
            batch.extend_from_slice(&single_start);
        }
        let reports = Lbfgs::new(&pot, LbfgsSettings::DEFAULT)
            .minimize_batch(&mut batch, 2, b)
            .unwrap();
        assert_eq!(reports.len(), b);
        for (i, rep) in reports.iter().enumerate() {
            assert!((rep.final_energy - single_report.final_energy).abs() < 1e-10);
            let block = &batch[i * 6..i * 6 + 6];
            for (a, s) in block.iter().zip(&single) {
                assert!((a - s).abs() < 1e-9);
            }
        }
    }

    /// A borrowed trait object is a potential holder like an owned one: the
    /// one-shot run under a potential the caller keeps.
    #[test]
    fn borrowed_dyn_potential_relaxes_like_an_owned_one() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let borrowed: &dyn Potential = &pot;
        let mut coords = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0];
        let report = Lbfgs::new(borrowed, LbfgsSettings::DEFAULT)
            .minimize_coords(&mut coords)
            .unwrap();
        assert!(report.converged);
        assert!(((coords[3] - coords[0]).abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_rejects_size_mismatch() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords = vec![0.0; 6 * 3 + 1];
        assert!(
            Lbfgs::new(&pot, LbfgsSettings::DEFAULT)
                .minimize_batch(&mut coords, 2, 3)
                .is_err()
        );
    }

    #[test]
    fn batch_zero_structs_is_empty() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        let reports = Lbfgs::new(&pot, LbfgsSettings::DEFAULT)
            .minimize_batch(&mut coords, 2, 0)
            .unwrap();
        assert!(reports.is_empty());
    }

    #[test]
    fn batch_zero_atoms_errors_not_panics() {
        let pot = HarmonicBond { k: 100.0, r0: 1.0 };
        let mut coords: Vec<F> = vec![];
        assert!(
            Lbfgs::new(&pot, LbfgsSettings::DEFAULT)
                .minimize_batch(&mut coords, 0, 3)
                .is_err()
        );
    }

    /// The soft packing potential minimizes through the one front door: two
    /// overlapping atoms are pushed to the soft core's edge, and a pair that
    /// was out of range when the run started still counts once it comes in.
    #[test]
    fn lbfgs_relaxes_the_soft_potential_that_rebuilds_its_own_pairs() {
        use crate::ff::potential::soft::SoftSpec;
        let mut frame = frame_from_coords(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 30.0, 0.0, 0.0]);
        let pot = SoftSpec::from_frame(&frame).potential(None);
        let report = Lbfgs::new(
            Arc::new(pot),
            LbfgsSettings {
                fmax: 1e-4,
                max_steps: 500,
                max_step: 0.2,
                memory: 8,
            },
        )
        .minimize(&mut frame)
        .unwrap();
        assert!(report.converged, "{report:?}");
        assert!(report.final_energy < 1e-6, "{report:?}");
        let x = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!(x[[1]] - x[[0]] > 2.59, "pushed apart to sigma: {x:?}");
    }
}
