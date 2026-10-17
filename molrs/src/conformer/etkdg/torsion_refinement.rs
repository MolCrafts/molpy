//! The second ETKDG stage: 3D experimental-torsion refinement.
//!
//! Port of RDKit's `construct3DForceField` / `minimizeWithExpTorsions`
//! (`$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`, BSD-3, Copyright
//! (C) 2004-2025 Greg Landrum / Sereina Riniker and other RDKit
//! contributors). It owns no energy kernel: the distance constraints are the
//! distance-geometry error function's
//! ([`DistanceViolations`]), the CrystalFF M6 torsions and the sp2
//! planarity terms are `ff::potential` kernels. The minimizer is the crate's
//! one L-BFGS, [`crate::optimize::minimize_lbfgs_rms`], as for RDKit's own
//! BFGS.

use crate::conformer::distgeom::{
    BoundsMatrix, DistanceViolations, ImproperConstraint, TorsionConstraint,
};
use crate::ff::compile::ExplicitTerms;
use crate::ff::ir::Params;
use crate::ff::potential::improper::ImproperDistance;
use crate::ff::potential::{Accumulator, Potential, Potentials};

/// Second-stage (3D) objective: distance constraints from the bounds matrix,
/// the experimental (CrystalFF M6) and flat-ring basic-knowledge torsions and
/// the sp2 planarity terms —
/// RDKit's `construct3DForceField`, which together carry the torsion bias
/// plus the bonded skeleton.
///
/// Only the distance constraints are distance-geometry terms. The torsions and
/// the planarity terms are force-field kernels, priced by `ff::potential`:
///
/// * the M6 torsion `Σₘ Vₘ (1 + sₘ cos mφ)` is LAMMPS `dihedral_style fourier`
///   (`dihedral periodic` here) with `kₘ = Vₘ`, `nₘ = m` and phase 0° for
///   `sₘ = +1`, 180° for `sₘ = −1`;
/// * the planarity term `10 · h²`, `h` the height of the sp2 centre over the
///   plane of its three neighbours (RDKit's out-of-plane term at
///   `oobForceScalingFactor = 10`), is
///   [`ImproperDistance`] with `K₂ = 10`, `K₄ = 0`.
pub(super) struct TorsionRefinement {
    /// The distance constraints of the bounds matrix, in 3D.
    distances: DistanceViolations,
    /// The M6 torsions, as one `dihedral periodic` kernel (empty when there
    /// is none).
    torsions: Potentials,
    /// The sp2 planarity terms.
    planarity: ImproperDistance,
}

/// Force constant of the planarity term (RDKit `oobForceScalingFactor`).
const IMPROPER_FORCE: f64 = 10.0;

impl TorsionRefinement {
    /// Build over the bounds (distance constraints), experimental torsions, and
    /// improper (sp2 planarity) constraints.
    pub(super) fn build<'a>(
        bounds: &BoundsMatrix,
        torsions: impl IntoIterator<Item = &'a TorsionConstraint>,
        impropers: &[ImproperConstraint],
    ) -> Self {
        let mut fourier = ExplicitTerms::new("dihedral", "periodic");
        let mut any_torsion = false;
        for t in torsions {
            any_torsion = true;
            let mut row = Params::new();
            for m in 0..6 {
                row.set(&format!("k{}", m + 1), t.force_constants[m]);
                row.set(&format!("periodicity{}", m + 1), (m + 1) as f64);
                let phase = if t.signs[m] < 0 { 180.0 } else { 0.0 };
                row.set(&format!("phase{}", m + 1), phase);
            }
            fourier = fourier.term(&t.atoms, row);
        }
        let torsions = if any_torsion {
            fourier
                .compile()
                .expect("the built-in `dihedral periodic` style prices any six-term row")
        } else {
            Potentials::new()
        };
        // RDKit packs improper atoms as [n0, center, n2, n3]: index 1 is the
        // centre, which `ImproperDistance` takes first.
        let planarity = impropers.iter().fold(ImproperDistance::new(), |k, im| {
            k.term(
                [im.atoms[1], im.atoms[0], im.atoms[2], im.atoms[3]],
                IMPROPER_FORCE,
                0.0,
            )
        });
        Self {
            distances: DistanceViolations::new(bounds, 3),
            torsions,
            planarity,
        }
    }

    /// Energy + gradient over a flat `n*3` coordinate buffer.
    pub(super) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut energy = self.distances.energy_grad(p, grad);
        // Experimental torsions and planarity: forces are −gradient, so they
        // are tallied into one buffer and subtracted once.
        let mut acc = Accumulator::new(p.len());
        for kernel in [&self.torsions as &dyn Potential, &self.planarity] {
            kernel.evaluate(p, &mut acc);
        }
        energy += acc.energy();
        for (g, f) in grad.iter_mut().zip(acc.forces()) {
            *g -= f;
        }
        energy
    }
}
