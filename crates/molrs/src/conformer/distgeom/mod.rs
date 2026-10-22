//! Distance-geometry constraint generation (ETKDGv3), a faithful port of
//! RDKit's bounds-matrix builder + smoothing + experimental-torsion knowledge.
//!
//! A port of RDKit's ETKDGv3 constraint generation (BSD-3, Copyright (C) Greg Landrum / Sereina Riniker and other
//! RDKit contributors). It produces, for a molecular graph:
//!
//!   * a smoothed **bounds matrix** identical (< 1e-3 Å) to
//!     `rdkit.Chem.rdDistGeom.GetMoleculeBoundsMatrix`,
//!   * **experimental torsion** preferences (CrystalFF M6),
//!   * **chiral** volume constraints,
//!   * **improper** (out-of-plane) constraints,
//!
//! and the error function an embedding minimizes against them
//! ([`ViolationEnergy`]: distance-bound and chiral-volume violations and the
//! fourth-dimension penalty).
//!
//! ## References
//! - Blaney & Dixon, *Rev. Comput. Chem.* **5**, 299 (1994) — bounds smoothing.
//! - Crippen & Havel, *Distance Geometry and Molecular Conformation* (1988).
//! - Riniker & Landrum, *J. Chem. Inf. Model.* **55**, 2562 (2015) — ETKDG.
//! - Wang, Witek, Landrum, Riniker, *J. Chem. Inf. Model.* **60**, 2044 (2020)
//!   — ETKDGv3 (small rings + macrocycles).
//!
//! ## Faithfulness boundary (honest scope)
//! - **Bounds + triangle smoothing**: full port; numerically matches RDKit.
//! - **Chiral / improper / flat-ring knowledge**: full port of the
//!   `findChiralSets` + basic-knowledge logic. Chirality sign is taken from
//!   the input 3D coordinates (molrs has no RDKit `ChiralTag`).
//! - **Experimental torsions**: full port. The complete ETKDGv3 CrystalFF
//!   three-table set (v2 ++ small-rings ++ macrocycles) is matched by the core
//!   SMARTS engine (`molrs::perceive::smarts`), reproducing RDKit
//!   `getExperimentalTorsions` (first-match-wins, one torsion per rotatable
//!   bond). See `torsion_prefs` for the precise boundary. Tetrangle smoothing
//!   is omitted because RDKit's reference matrix does not apply it.

mod basic_knowledge_torsions;
mod bounds;
mod bounds_matrix;
mod chirality;
mod mol_features;
mod torsion_prefs;
mod torsion_tables;
mod triangle_smoothing;
mod violation_energy;

use molrs::core::Atomistic;
use molrs::core::MolRsError;

pub use bounds_matrix::BoundsMatrix;
pub use chirality::{ChiralConstraint, ChiralSign, ImproperConstraint};
pub use torsion_prefs::TorsionConstraint;
pub(crate) use violation_energy::{DistanceViolations, ViolationEnergy, chiral_volume};

/// The full ETKDGv3 constraint set consumed by the embedding stage (spec 04).
pub struct DgConstraints {
    /// Smoothed distance-bounds matrix.
    pub bounds: BoundsMatrix,
    /// Experimental (CrystalFF) torsion preferences (partial — see module docs).
    pub experimental_torsions: Vec<TorsionConstraint>,
    /// Flat sp2-ring planarising torsions (basic knowledge), applied in the
    /// same second stage as the experimental ones.
    pub flat_ring_torsions: Vec<TorsionConstraint>,
    /// Chiral volume constraints.
    pub chiral: Vec<ChiralConstraint>,
    /// Improper / out-of-plane constraints.
    pub improper: Vec<ImproperConstraint>,
}

impl DgConstraints {
    /// The complete ETKDGv3 constraint set for `mol`: topological bounds
    /// (triangle-smoothed in place), experimental torsions, knowledge terms,
    /// chiral and improper constraints.
    pub fn from_graph(mol: &Atomistic) -> Result<Self, MolRsError> {
        if mol.n_atoms() == 0 {
            return Err(MolRsError::validation("molecule has no atoms"));
        }
        let p = mol_features::perceive_dg_features(mol)?;

        let mut bounds = bounds::set_topol_bounds(&p);
        triangle_smoothing::smooth_bounds(&mut bounds)?;

        Ok(Self {
            bounds,
            experimental_torsions: torsion_prefs::assign_experimental_torsions(mol, &p),
            flat_ring_torsions: p.flat_ring_torsions(),
            chiral: p.chiral_constraints(mol),
            improper: p.improper_constraints(),
        })
    }
}
