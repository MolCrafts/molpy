//! ETKDGv3 "basic knowledge" torsion terms.
//!
//! Ported from RDKit's `getExperimentalTorsions` basic-knowledge branch
//! (`$RDBASE/Code/GraphMol/ForceFieldHelpers/CrystalFF/TorsionPreferences.cpp`,
//! BSD-3, Copyright (C) 2017-2023 Sereina Riniker and other RDKit
//! contributors).
//!
//! Basic knowledge adds, on top of the topological bounds:
//!   1. **flat-ring torsions** — every 4-atom path `i-j-k-l` inside a 4-, 5- or
//!      6-membered ring whose four atoms are all sp2 gets a stiff
//!      `V = V2·(1 - cos 2x)` term (force constant 100, sign `-1` on the m=2
//!      component) that planarises the ring;
//!   2. **inversion / out-of-plane** terms for sp2 C/N/O centres — emitted as
//!      `ImproperConstraint`s in `super::chirality`.
//!
//! The 1-3 angle "knowledge" is already realised by the angle-based 1-3
//! distance bounds in `super::bounds` (RDKit applies its 1-3 angle knowledge
//! through `set13Bounds`, not a separate term), so this module only owns the
//! flat-ring proper torsions.

use super::mol_features::DgFeatures;
use super::torsion_prefs::TorsionConstraint;
use molrs::perceive::Hybridization;

impl DgFeatures {
    /// The flat sp2-ring planarising torsions (RDKit basic-knowledge ring
    /// loop), as M6 terms with only the `m = 2` component populated
    /// (`V = V2·(1 − cos 2x)`). Rings smaller than 4 or larger than 6 are
    /// skipped, exactly as RDKit.
    pub fn flat_ring_torsions(&self) -> Vec<TorsionConstraint> {
        flat_ring_torsions(self)
    }
}

fn flat_ring_torsions(p: &DgFeatures) -> Vec<TorsionConstraint> {
    let mut out = Vec::new();
    let mut done_bonds: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    for ring in &p.ring_idx {
        let rsize = ring.len();
        if !(4..=6).contains(&rsize) {
            continue;
        }
        for i in 0..rsize {
            let aid1 = ring[i];
            let aid2 = ring[(i + 1) % rsize];
            let aid3 = ring[(i + 2) % rsize];
            let aid4 = ring[(i + 3) % rsize];
            let bond_key = if aid2 < aid3 {
                (aid2, aid3)
            } else {
                (aid3, aid2)
            };
            let all_sp2 = [aid1, aid2, aid3, aid4]
                .iter()
                .all(|&a| p.atoms[a].hybridization == Hybridization::Sp2);
            if all_sp2 && !done_bonds.contains(&bond_key) {
                done_bonds.insert(bond_key);
                let mut signs = [1i8; 6];
                signs[1] = -1; // MMFF sign for m = 2
                let mut fconsts = [0.0; 6];
                fconsts[1] = 100.0; // strong flat-ring planarisation
                out.push(TorsionConstraint {
                    atoms: [aid1, aid2, aid3, aid4],
                    signs,
                    force_constants: fconsts,
                });
            }
        }
    }
    out
}
