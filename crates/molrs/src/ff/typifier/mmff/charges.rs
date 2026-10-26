//! MMFF94 partial-charge model.
//!
//! Faithful port of `MMFFMolProperties::getMMFFBondType` and
//! `MMFFMolProperties::computeMMFFCharges` from RDKit
//! `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`, plus the
//! sign / canonicalization rule of `MMFFChgCollection::getMMFFChgParams`
//! from `Code/ForceField/MMFF/Params.h` (BSD-3, RDKit contributors).
//!
//! Reference: Halgren, T. A. "MMFF.V", J. Comput. Chem. 1996, 17, 616-641.
//! Eq. 15 (p. 622): `q = (1 - M*v)*q0 + v*sum(qF_nbr) + sum(bci)`.

use crate::core::BondOrder;
use crate::ff::params::mmff::{mmff_chg, mmff_pbci, mmff_prop};
use crate::perceive::mmff_aromaticity::MmffTopology;

const EPS: f64 = 1.0e-8;

/// RDKit `getMMFFBondType`: returns 1 when a single bond joins two atom
/// types that are both `sbmb` or both `arom`, else 0.
pub(crate) fn mmff_bond_type(topo: &MmffTopology, types: &[u8], a: usize, b: usize) -> u8 {
    let order = match topo.bond_order(a, b) {
        Some(o) => o,
        None => return 0,
    };
    let pa = match mmff_prop(types[a]) {
        Some(p) => p,
        None => return 0,
    };
    let pb = match mmff_prop(types[b]) {
        Some(p) => p,
        None => return 0,
    };
    // RDKit tests `bond->getBondType() == Bond::SINGLE`. After MMFF
    // aromaticity perception, aromatic ring bonds have type AROMATIC (not
    // SINGLE), so they never get bond type 1 — only a genuine single bond
    // joining two sbmb/arom atoms does (e.g. the inter-ring bond in biphenyl).
    let is_single = order == BondOrder::Single;
    if is_single && ((pa.sbmb != 0 && pb.sbmb != 0) || (pa.arom != 0 && pb.arom != 0)) {
        1
    } else {
        0
    }
}

/// `getMMFFChgParams` contribution: returns the signed bci, or `None` if the
/// pair is not tabulated (caller falls back to the pbci difference).
fn chg_contribution(bond_type: u8, i_type: u8, j_type: u8) -> Option<f64> {
    let (can_i, can_j, sign) = if i_type > j_type {
        (j_type, i_type, 1.0)
    } else {
        (i_type, j_type, -1.0)
    };
    mmff_chg(bond_type, can_i, can_j).map(|c| sign * c.bci)
}

/// Compute MMFF formal charges (the "q0" pre-distribution charges).
///
/// Faithful port of the first loop of `computeMMFFCharges`.
fn compute_formal_charges(topo: &MmffTopology, types: &[u8]) -> Vec<f64> {
    let n = topo.n_atoms();
    let mut fchg = vec![0.0f64; n];

    for idx in 0..n {
        let atom_type = types[idx];
        let mut f = 0.0;
        match atom_type {
            // O2CM / SM — charge shared across terminal O/S on the neighbour
            32 | 72 => {
                for &nbr in &topo.nbrs[idx] {
                    let nbr_type = types[nbr];
                    let mut n_sec_n = 0i32;
                    let mut n_term_os = 0i32;
                    for &nbr2 in &topo.nbrs[nbr] {
                        if topo.atno[nbr2] == 7 && topo.degree(nbr2) == 2 && !topo.is_aromatic[nbr2]
                        {
                            n_sec_n += 1;
                        }
                        if (topo.atno[nbr2] == 8 || topo.atno[nbr2] == 16) && topo.degree(nbr2) == 1
                        {
                            n_term_os += 1;
                        }
                    }
                    if topo.atno[nbr] == 16 && n_term_os == 2 && n_sec_n == 1 {
                        n_sec_n = 0;
                    }
                    if topo.atno[nbr] == 6 && n_term_os > 0 {
                        f = if n_term_os == 1 {
                            -1.0
                        } else {
                            -((n_term_os - 1) as f64) / (n_term_os as f64)
                        };
                        break;
                    }
                    if nbr_type == 45 && n_term_os == 3 {
                        f = -1.0 / 3.0;
                        break;
                    }
                    if nbr_type == 25 && n_term_os > 0 {
                        f = if n_term_os == 1 {
                            0.0
                        } else {
                            -((n_term_os - 1) as f64) / (n_term_os as f64)
                        };
                        break;
                    }
                    if nbr_type == 18 && n_term_os > 0 {
                        f = if (n_sec_n + n_term_os) == 2 {
                            0.0
                        } else {
                            -(((n_sec_n + n_term_os) - 2) as f64) / (n_term_os as f64)
                        };
                        break;
                    }
                    if nbr_type == 73 && n_term_os > 0 {
                        f = if n_term_os == 1 {
                            0.0
                        } else {
                            -((n_term_os - 1) as f64) / (n_term_os as f64)
                        };
                        break;
                    }
                    if nbr_type == 77 && n_term_os > 0 {
                        f = -(1.0 / (n_term_os as f64));
                        break;
                    }
                }
            }
            // N5M — shared over the type-76 nitrogens in the same 5-ring
            76 => {
                if let Some(ring) = topo.ring_idx.iter().find(|r| r.contains(&idx)) {
                    let n76 = ring.iter().filter(|&&a| types[a] == 76).count();
                    if n76 > 0 {
                        f = -(1.0 / n76 as f64);
                    }
                }
            }
            // conjugated cationic 5-ring nitrogens
            55 | 56 | 81 => {
                f = topo.formal_charge[idx] as f64;
                let mut conj = vec![false; n];
                conj[idx] = true;
                let mut n_conj = 1usize;
                let mut old = 0usize;
                while n_conj > old {
                    old = n_conj;
                    for a in 0..n {
                        if !conj[a] {
                            continue;
                        }
                        for &nbr in &topo.nbrs[a] {
                            let nt = types[nbr];
                            if nt != 57 && nt != 80 {
                                continue;
                            }
                            for &nbr2 in &topo.nbrs[nbr] {
                                let nt2 = types[nbr2];
                                if nt2 != 55 && nt2 != 56 && nt2 != 81 {
                                    continue;
                                }
                                if !conj[nbr2] {
                                    conj[nbr2] = true;
                                    f += topo.formal_charge[nbr2] as f64;
                                    n_conj += 1;
                                }
                            }
                        }
                    }
                }
                if n_conj > 0 {
                    f /= n_conj as f64;
                }
            }
            // isonitrile carbon → +1 on the triple-bonded N(42)
            61 => {
                for &nbr in &topo.nbrs[idx] {
                    if types[nbr] == 42 {
                        f = 1.0;
                    }
                }
            }
            // simple +1
            34 | 49 | 51 | 54 | 58 | 92 | 93 | 94 | 97 => f = 1.0,
            // simple +2
            87 | 95 | 96 | 98 | 99 => f = 2.0,
            // simple +3
            88 => f = 3.0,
            // simple -1
            35 | 62 | 89 | 90 | 91 => f = -1.0,
            _ => {}
        }
        fchg[idx] = f;
    }
    fchg
}

/// Compute MMFF partial charges. Faithful port of the second loop of
/// `computeMMFFCharges`.
pub(crate) fn compute_partial_charges(topo: &MmffTopology, types: &[u8]) -> Vec<f64> {
    let n = topo.n_atoms();
    let fchg = compute_formal_charges(topo, types);
    let mut pchg = vec![0.0f64; n];

    for idx in 0..n {
        let atom_type = types[idx];
        let prop = match mmff_prop(atom_type) {
            Some(p) => p,
            None => continue,
        };
        let pbci = match mmff_pbci(atom_type) {
            Some(p) => p,
            None => continue,
        };
        let mut q0 = fchg[idx];
        let m = prop.crd as f64;
        let v = pbci.fcadj;
        let mut sum_formal = 0.0;
        let mut sum_partial = 0.0;

        if v.abs() < EPS {
            for &nbr in &topo.nbrs[idx] {
                let nbr_fc = fchg[nbr];
                if nbr_fc < 0.0 {
                    q0 += nbr_fc / (2.0 * topo.degree(nbr) as f64);
                }
            }
        }
        if atom_type == 62 {
            for &nbr in &topo.nbrs[idx] {
                let nbr_fc = fchg[nbr];
                if nbr_fc > 0.0 {
                    q0 -= nbr_fc / 2.0;
                }
            }
        }
        for &nbr in &topo.nbrs[idx] {
            let nbr_type = types[nbr];
            let bond_type = mmff_bond_type(topo, types, idx, nbr);
            let contrib = match chg_contribution(bond_type, atom_type, nbr_type) {
                Some(b) => b,
                None => {
                    let pi = pbci.pbci;
                    let pj = mmff_pbci(nbr_type).map(|p| p.pbci).unwrap_or(0.0);
                    pi - pj
                }
            };
            sum_partial += contrib;
            sum_formal += fchg[nbr];
        }
        pchg[idx] = (1.0 - m * v) * q0 + v * sum_formal + sum_partial;
    }
    pchg
}
