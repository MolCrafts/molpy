//! Topological distance bounds (1-2 / 1-3 / 1-4 / 1-5 + vdW lower bounds).
//!
//! Port of RDKit's `setTopolBounds` and its helpers
//! (`$RDBASE/Code/GraphMol/DistGeomHelpers/BoundsMatrixBuilder.cpp`, BSD-3,
//! Copyright (C) 2004-2025 Greg Landrum and other RDKit contributors), plus
//! the `compute13Dist` / `compute14DistCis` / `compute14DistTrans` /
//! `compute15Dist*` geometry helpers from `$RDBASE/Code/Geometry/Utils.h`.
//!
//! Scope: the organic main-group subset (H, C, N, O, F, S, Cl, Br, P, I) that
//! `super::perceive` can type. The ring/fused-ring/macrocycle special cases of
//! `set14Bounds` are ported for the common cases; the full bond-stereo (E/Z)
//! and `useMacrocycle14config` branches are noted where partial (see `mod.rs`).

// These lints are relaxed deliberately: this module is a line-by-line port of
// RDKit's C++ bounds builder, and we keep the original control-flow structure
// (mirrored `if` branches, many-parameter helpers, index loops over two
// arrays) so the port stays auditable against the reference source.
#![allow(
    clippy::if_same_then_else,
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::nonminimal_bool
)]

use std::collections::HashMap;

use super::bounds_matrix::BoundsMatrix;
use super::mol_features::{DgFeatures, PerceivedAtom};
use crate::ff::typifier::uff::{atom_label, bond_rest_length as uff_rest_length};
use molrs::core::Element;
use molrs::perceive::Hybridization;

const DIST12_DELTA: f64 = 0.01;
const DIST13_TOL: f64 = 0.04;
const GEN_DIST_TOL: f64 = 0.06;
const DIST15_TOL: f64 = 0.08;
const VDW_SCALE_15: f64 = 0.7;
const MAX_UPPER: f64 = 1000.0;

/// Cis/trans flag recorded for a 1-4 path, used by `set15Bounds`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path14Type {
    Cis,
    Trans,
    Other,
}

/// One recorded 1-4 path (three consecutive bond indices) and its geometry.
#[derive(Clone, Copy, Debug)]
struct Path14 {
    bid1: usize,
    bid2: usize,
    bid3: usize,
    ptype: Path14Type,
}

/// Per-build scratch state, mirroring RDKit's `ComputedData`.
struct Computed {
    /// Bond rest length per bond index.
    bond_lengths: Vec<f64>,
    /// `bond_adj[(b1,b2)]` = the shared atom index of two adjacent bonds.
    bond_adj: HashMap<(usize, usize), usize>,
    /// `bond_angles[(b1,b2)]` = angle (rad) at the shared atom.
    bond_angles: HashMap<(usize, usize), f64>,
    paths14: Vec<Path14>,
    cis_paths: std::collections::HashSet<(usize, usize, usize)>,
    trans_paths: std::collections::HashSet<(usize, usize, usize)>,
    set15: std::collections::HashSet<(usize, usize)>,
}

impl Computed {
    fn new(n_bonds: usize) -> Self {
        Self {
            bond_lengths: vec![0.0; n_bonds],
            bond_adj: HashMap::new(),
            bond_angles: HashMap::new(),
            paths14: Vec::new(),
            cis_paths: std::collections::HashSet::new(),
            trans_paths: std::collections::HashSet::new(),
            set15: std::collections::HashSet::new(),
        }
    }

    fn angle(&self, b1: usize, b2: usize) -> Option<f64> {
        let key = if b1 < b2 { (b1, b2) } else { (b2, b1) };
        self.bond_angles.get(&key).copied()
    }
    fn set_angle(&mut self, b1: usize, b2: usize, v: f64) {
        let key = if b1 < b2 { (b1, b2) } else { (b2, b1) };
        self.bond_angles.insert(key, v);
    }
    fn adj(&self, b1: usize, b2: usize) -> Option<usize> {
        let key = if b1 < b2 { (b1, b2) } else { (b2, b1) };
        self.bond_adj.get(&key).copied()
    }
    fn set_adj(&mut self, b1: usize, b2: usize, aid: usize) {
        let key = if b1 < b2 { (b1, b2) } else { (b2, b1) };
        self.bond_adj.insert(key, aid);
    }
}

/// A bond with stable index, endpoints, perceived order/aromaticity.
#[derive(Clone, Copy)]
struct BondRec {
    a: usize,
    b: usize,
    order: f64,
    aromatic: bool,
}

/// Indexed bond list + per-atom incident-bond lists for a perceived molecule.
struct BondIndex {
    bonds: Vec<BondRec>,
    atom_bonds: Vec<Vec<usize>>,
    /// `(min,max) -> bond index`.
    pair_to_bond: HashMap<(usize, usize), usize>,
}

fn build_bond_index(p: &DgFeatures) -> BondIndex {
    let n = p.atoms.len();
    let mut bonds = Vec::new();
    let mut atom_bonds = vec![Vec::new(); n];
    let mut pair_to_bond = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    for i in 0..n {
        for &j in &p.adj[i] {
            let key = if i < j { (i, j) } else { (j, i) };
            if !seen.insert(key) {
                continue;
            }
            let bid = bonds.len();
            bonds.push(BondRec {
                a: key.0,
                b: key.1,
                order: p.bond_order(key.0, key.1),
                aromatic: p.is_aromatic_bond(key.0, key.1),
            });
            atom_bonds[key.0].push(bid);
            atom_bonds[key.1].push(bid);
            pair_to_bond.insert(key, bid);
        }
    }
    BondIndex {
        bonds,
        atom_bonds,
        pair_to_bond,
    }
}

/// Breadth-first topological distance matrix (number of bonds between atoms),
/// as `f64` with [`f64::INFINITY`] for unreachable pairs: one
/// [`Topology::distances`](crate::core::Topology::distances) row per atom.
fn topo_distances(p: &DgFeatures) -> Vec<Vec<f64>> {
    let edges: Vec<[usize; 2]> = p
        .adj
        .iter()
        .enumerate()
        .flat_map(|(i, nbrs)| nbrs.iter().filter(move |&&j| i < j).map(move |&j| [i, j]))
        .collect();
    let mut topo = crate::core::Topology::with_atoms(p.adj.len());
    topo.add_bonds(&edges);
    (0..p.adj.len())
        .map(|source| {
            topo.distances(source)
                .into_iter()
                .map(|d| if d < 0 { f64::INFINITY } else { d as f64 })
                .collect()
        })
        .collect()
}

// ── geometry helpers (RDGeom::Utils) ───────────────────────────────────────

fn compute13_dist(d1: f64, d2: f64, angle: f64) -> f64 {
    (d1 * d1 + d2 * d2 - 2.0 * d1 * d2 * angle.cos())
        .max(0.0)
        .sqrt()
}

fn compute14_dist_cis(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64) -> f64 {
    let dx = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy = d3 * a23.sin() - d1 * a12.sin();
    (dx * dx + dy * dy).sqrt()
}

fn compute14_dist_trans(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64) -> f64 {
    let dx = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy = d3 * a23.sin() + d1 * a12.sin();
    (dx * dx + dy * dy).sqrt()
}

// 1-5 helpers, ported verbatim from BoundsMatrixBuilder.cpp.
fn clampc(c: f64) -> f64 {
    c.clamp(-1.0, 1.0)
}
fn c15_cis_cis(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() - d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = clampc((d3 - d2 * a23.cos() + d1 * (a12 + a23).cos()) / d14);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 - a143)
}
fn c15_cis_trans(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() - d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = clampc((d3 - d2 * a23.cos() + d1 * (a12 + a23).cos()) / d14);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 + a143)
}
fn c15_trans_trans(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() + d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = clampc((d3 - d2 * a23.cos() + d1 * (a12 - a23).cos()) / d14);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 + a143)
}
fn c15_trans_cis(d1: f64, d2: f64, d3: f64, d4: f64, a12: f64, a23: f64, a34: f64) -> f64 {
    let dx14 = d2 - d3 * a23.cos() - d1 * a12.cos();
    let dy14 = d3 * a23.sin() + d1 * a12.sin();
    let d14 = (dx14 * dx14 + dy14 * dy14).sqrt();
    let cval = clampc((d3 - d2 * a23.cos() + d1 * (a12 - a23).cos()) / d14);
    let a143 = cval.acos();
    compute13_dist(d14, d4, a34 - a143)
}

// ── bound setters ───────────────────────────────────────────────────────────

/// RDKit `_checkAndSetBounds`: conservative tightening of an existing bound.
fn check_and_set_bounds(mmat: &mut BoundsMatrix, i: usize, j: usize, lb: f64, ub: f64) {
    let clb = mmat.lower(i, j);
    let cub = mmat.upper(i, j);
    if clb <= DIST12_DELTA {
        mmat.set_lower(i, j, lb);
    } else if lb < clb && lb > DIST12_DELTA {
        mmat.set_lower(i, j, lb);
    }
    if cub >= MAX_UPPER {
        mmat.set_upper(i, j, ub);
    } else if ub > cub && ub < MAX_UPPER {
        mmat.set_upper(i, j, ub);
    }
}

fn is_larger_sp2(p: &DgFeatures, i: usize) -> bool {
    let a = &p.atoms[i];
    a.element.z() > 13
        && a.hybridization == Hybridization::Sp2
        && p.rings.is_atom_in_ring(p.atom_ids[i])
}

/// The bond order RDKit's rest-length formula uses: 1.5 for an aromatic
/// bond, else the graph's integral order (`Bond::getBondTypeAsDouble`). An
/// amide C–N is not special-cased: RDKit 2026.03's `set12Bounds` gives
/// acetanilide's C(=O)–N the order-1 `C_R`–`N_R` length, 1.4222 Å.
fn effective_bond_order(order: f64, aromatic: bool) -> f64 {
    if aromatic { 1.5 } else { order }
}

/// The 1-2 bound: the UFF rest length when both atoms have a UFF type
/// ([`uff_rest_length`]), else RDKit's untyped fallback `(vw1 + vw2) / 2`.
fn bond_rest_length(a: &PerceivedAtom, b: &PerceivedAtom, order: f64) -> f64 {
    let label = |x: &PerceivedAtom| {
        atom_label(
            x.element,
            x.hybridization,
            x.aromatic || x.conjugated,
            x.total_valence,
        )
    };
    (order > 0.0)
        .then(|| uff_rest_length(&label(a), &label(b), order))
        .flatten()
        .unwrap_or_else(|| 0.5 * (rvdw(a.element.z()) + rvdw(b.element.z())))
}

/// Van der Waals radius (Å) as RDKit's `PeriodicTable::getRvdw` gives it.
///
/// These differ from [`molrs::core::Element::vdw_radius`] (Bondi-style): RDKit ships
/// its own table in `atomic_data`, and `setLowerBoundVDW` / `set15Bounds`
/// depend on the exact values, so they are transcribed here.
fn rvdw(z: u8) -> f64 {
    match z {
        1 => 1.2,
        5 => 1.8,
        6 => 1.7,
        7 => 1.6,
        8 => 1.55,
        9 => 1.5,
        14 => 2.1,
        15 => 1.95,
        16 => 1.8,
        17 => 1.8,
        35 => 1.9,
        53 => 2.1,
        // RDKit default for unlisted elements is 2.0.
        _ => 2.0,
    }
}

fn set_12_bounds(p: &DgFeatures, bi: &BondIndex, comp: &mut Computed, mmat: &mut BoundsMatrix) {
    for (bid, b) in bi.bonds.iter().enumerate() {
        let eff = effective_bond_order(b.order, b.aromatic);
        let bl = bond_rest_length(&p.atoms[b.a], &p.atoms[b.b], eff);
        comp.bond_lengths[bid] = bl;
        mmat.set_upper(b.a, b.b, bl + DIST12_DELTA);
        mmat.set_lower(b.a, b.b, bl - DIST12_DELTA);
    }
}

/// Ring angle for an sp2/sp3 atom in a ring of `ring_size` (RDKit `_setRingAngle`).
fn ring_angle(hyb: Hybridization, ring_size: usize) -> f64 {
    let pi = std::f64::consts::PI;
    if (hyb == Hybridization::Sp2 && ring_size <= 8) || ring_size == 3 || ring_size == 4 {
        pi * (1.0 - 2.0 / ring_size as f64)
    } else if hyb == Hybridization::Sp3 {
        if ring_size == 5 {
            104.0_f64.to_radians()
        } else {
            109.5_f64.to_radians()
        }
    } else {
        120.0_f64.to_radians()
    }
}

fn set_13_bound_at_angle(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &Computed,
    mmat: &mut BoundsMatrix,
    aid1: usize,
    aid: usize,
    aid3: usize,
    angle: f64,
) {
    let bid1 = bi.pair_to_bond[&pair(aid1, aid)];
    let bid2 = bi.pair_to_bond[&pair(aid, aid3)];
    let dl = compute13_dist(comp.bond_lengths[bid1], comp.bond_lengths[bid2], angle);
    let mut dist_tol = DIST13_TOL;
    if is_larger_sp2(p, aid1) {
        dist_tol *= 2.0;
    }
    if is_larger_sp2(p, aid) {
        dist_tol *= 2.0;
    }
    if is_larger_sp2(p, aid3) {
        dist_tol *= 2.0;
    }
    let du = dl + dist_tol;
    let dl = dl - dist_tol;
    check_and_set_bounds(mmat, aid1, aid3, dl, du);
}

fn pair(i: usize, j: usize) -> (usize, usize) {
    if i < j { (i, j) } else { (j, i) }
}

fn set_13_bounds(p: &DgFeatures, bi: &BondIndex, comp: &mut Computed, mmat: &mut BoundsMatrix) {
    let n = p.atoms.len();
    let mut visited = vec![0usize; n];
    let mut angle_taken = vec![0.0_f64; n];
    let mut done_paths: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    // Rings first, smallest first.
    let mut rings = p.ring_idx.clone();
    rings.sort_by_key(|r| r.len());
    for ring in &rings {
        let rsize = ring.len();
        let mut aid1 = ring[rsize - 1];
        for i in 0..rsize {
            let aid2 = ring[i];
            let aid3 = if i == rsize - 1 { ring[0] } else { ring[i + 1] };
            let bid1 = bi.pair_to_bond[&pair(aid1, aid2)];
            let bid2 = bi.pair_to_bond[&pair(aid2, aid3)];
            let id1 = (bid1, bid2);
            let id2 = (bid2, bid1);
            if !done_paths.contains(&id1) && !done_paths.contains(&id2) {
                let angle = ring_angle(p.atoms[aid2].hybridization, rsize);
                set_13_bound_at_angle(p, bi, comp, mmat, aid1, aid2, aid3, angle);
                comp.set_angle(bid1, bid2, angle);
                comp.set_adj(bid1, bid2, aid2);
                visited[aid2] += 1;
                angle_taken[aid2] += angle;
                done_paths.insert(id1);
                done_paths.insert(id2);
            }
            aid1 = aid2;
        }
    }

    let pi = std::f64::consts::PI;
    for aid2 in 0..n {
        let deg = p.atoms[aid2].degree;
        let n13 = deg * deg.saturating_sub(1) / 2;
        if n13 == visited[aid2] {
            continue;
        }
        let ahyb = p.atoms[aid2].hybridization;
        let incident = &bi.atom_bonds[aid2];
        if visited[aid2] >= 1 {
            for ii in 0..incident.len() {
                let bid1 = incident[ii];
                let aid1 = other_atom(&bi.bonds[bid1], aid2);
                for &bid2 in incident.iter().take(ii) {
                    let aid3 = other_atom(&bi.bonds[bid2], aid2);
                    if comp.angle(bid1, bid2).is_none() {
                        let angle = if ahyb == Hybridization::Sp2 {
                            (2.0 * pi - angle_taken[aid2])
                                / (n13.saturating_sub(visited[aid2])).max(1) as f64
                        } else if ahyb == Hybridization::Sp3 {
                            let mut a = 109.5_f64.to_radians();
                            if p.rings.is_atom_in_ring(p.atom_ids[aid2])
                                && atom_in_ring_of_size(p, aid2, 3)
                            {
                                a = 116.0_f64.to_radians();
                            } else if atom_in_ring_of_size(p, aid2, 4) {
                                a = 112.0_f64.to_radians();
                            }
                            a
                        } else if deg == 5 {
                            105.0_f64.to_radians()
                        } else if deg == 6 {
                            135.0_f64.to_radians()
                        } else {
                            120.0_f64.to_radians()
                        };
                        set_13_bound_at_angle(p, bi, comp, mmat, aid1, aid2, aid3, angle);
                        comp.set_angle(bid1, bid2, angle);
                        comp.set_adj(bid1, bid2, aid2);
                        angle_taken[aid2] += angle;
                        visited[aid2] += 1;
                    }
                }
            }
        } else {
            for ii in 0..incident.len() {
                let bid1 = incident[ii];
                let aid1 = other_atom(&bi.bonds[bid1], aid2);
                for &bid2 in incident.iter().take(ii) {
                    let aid3 = other_atom(&bi.bonds[bid2], aid2);
                    let angle = match ahyb {
                        Hybridization::Sp => pi,
                        Hybridization::Sp2 => 2.0 * pi / 3.0,
                        Hybridization::Sp3 => 109.5_f64.to_radians(),
                        Hybridization::Sp3d => 105.0_f64.to_radians(),
                        Hybridization::Sp3d2 => 90.0_f64.to_radians(),
                        Hybridization::S | Hybridization::Other => 120.0_f64.to_radians(),
                    };
                    if deg <= 4 {
                        set_13_bound_at_angle(p, bi, comp, mmat, aid1, aid2, aid3, angle);
                    } else {
                        let dmax = comp.bond_lengths[bid1] + comp.bond_lengths[bid2];
                        check_and_set_bounds(mmat, aid1, aid3, 1.0, dmax * 1.2);
                    }
                    comp.set_angle(bid1, bid2, angle);
                    comp.set_adj(bid1, bid2, aid2);
                    angle_taken[aid2] += angle;
                    visited[aid2] += 1;
                }
            }
        }
    }
}

fn other_atom(b: &BondRec, a: usize) -> usize {
    if b.a == a { b.b } else { b.a }
}

fn atom_in_ring_of_size(p: &DgFeatures, idx: usize, size: usize) -> bool {
    p.ring_idx
        .iter()
        .any(|r| r.len() == size && r.contains(&idx))
}

// ── 1-4 bounds ──────────────────────────────────────────────────────────────

/// RDKit `_isCarbonyl`: a C with degree > 2 double-bonded to an O or N.
fn is_carbonyl(p: &DgFeatures, at: usize) -> bool {
    if p.atoms[at].element != Element::C || p.atoms[at].degree <= 2 {
        return false;
    }
    p.adj[at].iter().any(|&nb| {
        let o = p.bond_order(at, nb);
        o >= 1.75 && matches!(p.atoms[nb].element, Element::O | Element::N)
    })
}

/// Implicit-H count proxy for an atom (degree minus heavy-neighbour count is
/// not available; with explicit Hs we count bonded H atoms).
fn n_hs(p: &DgFeatures, at: usize) -> usize {
    p.adj[at]
        .iter()
        .filter(|&&nb| p.atoms[nb].element.z() == 1)
        .count()
}

/// RDKit `_checkAmideEster14`: pattern 1-2-3=4 where 3 is a carbonyl C
/// double-bonded to O/N (4), bnd1 (1-2) single, and 2 is O or secondary-amide
/// N. Returns true if the ordered path (atm1,atm2,atm3,atm4 / bnd1,bnd3)
/// matches.
fn check_amide_ester_14(
    p: &DgFeatures,
    bond1_order: f64,
    bond3_order: f64,
    atm2: usize,
    atm3: usize,
    atm4: usize,
) -> bool {
    let a2 = p.atoms[atm2].element.z();
    let a3 = p.atoms[atm3].element.z();
    let a4 = p.atoms[atm4].element.z();
    a3 == 6
        && bond3_order >= 1.75
        && (a4 == 8 || a4 == 7)
        && (bond1_order - 1.0).abs() < 0.01
        && (a2 == 8 || (a2 == 7 && n_hs(p, atm2) == 1))
}

/// RDKit `_checkAmideEster15`: pattern where atm2 is O (or NH1), bnd1 single,
/// atm3 is a carbonyl C with bnd3 single.
fn check_amide_ester_15(
    p: &DgFeatures,
    bond1_order: f64,
    bond3_order: f64,
    atm2: usize,
    atm3: usize,
) -> bool {
    let a2 = p.atoms[atm2].element.z();
    let o_or_nh1 = a2 == 8 || (a2 == 7 && n_hs(p, atm2) == 1);
    o_or_nh1
        && (bond1_order - 1.0).abs() < 0.01
        && p.atoms[atm3].element.z() == 6
        && (bond3_order - 1.0).abs() < 0.01
        && is_carbonyl(p, atm3)
}

/// Set a single 1-4 bound from three consecutive bonds, recording the path.
#[allow(clippy::too_many_arguments)]
fn set_one_14(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &mut Computed,
    mmat: &mut BoundsMatrix,
    topo: &[Vec<f64>],
    bid1: usize,
    bid2: usize,
    bid3: usize,
    in_ring_size: Option<usize>,
    middle_is_ring: bool,
) {
    let atm2 = match comp.adj(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let atm3 = match comp.adj(bid2, bid3) {
        Some(a) => a,
        None => return,
    };
    let aid1 = other_atom(&bi.bonds[bid1], atm2);
    let aid4 = other_atom(&bi.bonds[bid3], atm3);

    // genuine 1-4 contact check
    if topo[aid1.max(aid4)][aid1.min(aid4)] < 2.9 {
        return;
    }
    let bl1 = comp.bond_lengths[bid1];
    let bl2 = comp.bond_lengths[bid2];
    let bl3 = comp.bond_lengths[bid3];
    let ba12 = match comp.angle(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let ba23 = match comp.angle(bid2, bid3) {
        Some(a) => a,
        None => return,
    };

    let hyb2 = p.atoms[atm2].hybridization;
    let hyb3 = p.atoms[atm3].hybridization;
    let b2 = &bi.bonds[bid2];

    let mut dl;
    let mut du;
    let mut ptype = Path14Type::Other;

    if let Some(rsize) = in_ring_size {
        // _setInRing14Bounds: prefer cis for small sp2-sp2 in-ring paths.
        let mut prefer_cis = false;
        if rsize <= 8 && hyb2 == Hybridization::Sp2 && hyb3 == Hybridization::Sp2 {
            prefer_cis = true;
        }
        if prefer_cis {
            dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23) - GEN_DIST_TOL;
            du = dl + 2.0 * GEN_DIST_TOL;
            ptype = Path14Type::Cis;
            comp.cis_paths.insert((bid1, bid2, bid3));
            comp.cis_paths.insert((bid3, bid2, bid1));
        } else {
            dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
            du = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
            if du < dl {
                std::mem::swap(&mut du, &mut dl);
            }
            if (du - dl).abs() < DIST12_DELTA {
                dl -= GEN_DIST_TOL;
                du += GEN_DIST_TOL;
            }
        }
    } else if middle_is_ring {
        // share-ring-bond → same handling as in-ring with rsize=0 (no cis pref).
        dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
        du = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
        if du < dl {
            std::mem::swap(&mut du, &mut dl);
        }
        if (du - dl).abs() < DIST12_DELTA {
            dl -= GEN_DIST_TOL;
            du += GEN_DIST_TOL;
        }
    } else {
        // _setChain14Bounds (no stereo / amide-trans forcing; ETKDG default
        // lets amides roam cis↔trans).
        let order2 = b2.order;
        if order2 >= 1.75 {
            // double middle bond
            if bi.bonds[bid1].order >= 1.75 || bi.bonds[bid3].order >= 1.75 {
                dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23) - GEN_DIST_TOL;
                du = dl + 2.0 * GEN_DIST_TOL;
                ptype = Path14Type::Cis;
                comp.cis_paths.insert((bid1, bid2, bid3));
                comp.cis_paths.insert((bid3, bid2, bid1));
            } else {
                dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
                du = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
                if (du - dl).abs() < DIST12_DELTA {
                    dl -= GEN_DIST_TOL;
                    du += GEN_DIST_TOL;
                }
            }
        } else if p.atoms[atm2].element == Element::S && p.atoms[atm3].element == Element::S {
            // *S-S* torsion ≈ 90°
            let d3d = compute14_dist_3d(bl1, bl2, bl3, ba12, ba23, std::f64::consts::FRAC_PI_2);
            dl = d3d - GEN_DIST_TOL;
            du = dl + 2.0 * GEN_DIST_TOL;
        } else {
            // SINGLE middle bond. RDKit `GetMoleculeBoundsMatrix` runs with
            // `forceTransAmides = true`, so amide/ester paths are pinned to a
            // single cis/trans conformation rather than allowed to roam.
            let ao1 = bi.bonds[bid1].order;
            let ao3 = bi.bonds[bid3].order;
            let amide14 = check_amide_ester_14(p, ao1, ao3, atm2, atm3, aid4)
                || check_amide_ester_14(p, ao3, ao1, atm3, atm2, aid1);
            let amide15 = check_amide_ester_15(p, ao1, ao3, atm2, atm3)
                || check_amide_ester_15(p, ao3, ao1, atm3, atm2);
            if amide14 {
                // forceTransAmides: secondary-amide H → trans, else cis.
                let sec_amide_h = (p.atoms[aid1].element.z() == 1
                    && p.atoms[atm2].element.z() == 7
                    && p.atoms[atm2].degree == 3
                    && n_hs(p, atm2) == 1)
                    || (p.atoms[aid4].element.z() == 1
                        && p.atoms[atm3].element.z() == 7
                        && p.atoms[atm3].degree == 3
                        && n_hs(p, atm3) == 1);
                if sec_amide_h {
                    dl = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
                    ptype = Path14Type::Trans;
                    comp.trans_paths.insert((bid1, bid2, bid3));
                    comp.trans_paths.insert((bid3, bid2, bid1));
                } else {
                    dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
                    ptype = Path14Type::Cis;
                    comp.cis_paths.insert((bid1, bid2, bid3));
                    comp.cis_paths.insert((bid3, bid2, bid1));
                }
                du = dl;
                dl -= GEN_DIST_TOL;
                du += GEN_DIST_TOL;
            } else if amide15 {
                // forceTransAmides: secondary-amide H → cis, else trans.
                let sec_amide_h = (p.atoms[aid1].element.z() == 1
                    && p.atoms[atm2].element.z() == 7
                    && p.atoms[atm2].degree == 3
                    && n_hs(p, atm2) == 1)
                    || (p.atoms[aid4].element.z() == 1
                        && p.atoms[atm3].element.z() == 7
                        && p.atoms[atm3].degree == 3
                        && n_hs(p, atm3) == 1);
                if sec_amide_h {
                    dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
                    ptype = Path14Type::Cis;
                    comp.cis_paths.insert((bid1, bid2, bid3));
                    comp.cis_paths.insert((bid3, bid2, bid1));
                } else {
                    dl = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
                    ptype = Path14Type::Trans;
                    comp.trans_paths.insert((bid1, bid2, bid3));
                    comp.trans_paths.insert((bid3, bid2, bid1));
                }
                du = dl;
                dl -= GEN_DIST_TOL;
                du += GEN_DIST_TOL;
            } else {
                dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
                du = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
            }
        }
    }

    if (du - dl).abs() < DIST12_DELTA {
        dl -= GEN_DIST_TOL;
        du += GEN_DIST_TOL;
    }
    check_and_set_bounds(mmat, aid1, aid4, dl, du);
    comp.paths14.push(Path14 {
        bid1,
        bid2,
        bid3,
        ptype,
    });
}

fn compute14_dist_3d(d1: f64, d2: f64, d3: f64, a12: f64, a23: f64, tor: f64) -> f64 {
    // p1 = (d1 cos a12, d1 sin a12, 0); p4 rotated about x by tor.
    let p1 = [d1 * a12.cos(), d1 * a12.sin(), 0.0];
    let p4y = d3 * a23.sin();
    let p4 = [d2 - d3 * a23.cos(), p4y * tor.cos(), p4y * tor.sin()];
    let dx = p4[0] - p1[0];
    let dy = p4[1] - p1[1];
    let dz = p4[2] - p1[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn set_14_bounds(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &mut Computed,
    mmat: &mut BoundsMatrix,
    topo: &[Vec<f64>],
) {
    let mut ring_bond_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut done_paths: std::collections::HashSet<(usize, usize, usize)> =
        std::collections::HashSet::new();

    // bond rings
    let bond_rings: Vec<Vec<usize>> = p
        .ring_idx
        .iter()
        .map(|ring| {
            let rsize = ring.len();
            (0..rsize)
                .map(|i| bi.pair_to_bond[&pair(ring[i], ring[(i + 1) % rsize])])
                .collect()
        })
        .collect();

    for bring in &bond_rings {
        let rsize = bring.len();
        if rsize < 3 {
            continue;
        }
        let mut bid1 = bring[rsize - 1];
        for i in 0..rsize {
            let bid2 = bring[i];
            let bid3 = bring[(i + 1) % rsize];
            ring_bond_pairs.insert((bid1, bid2));
            ring_bond_pairs.insert((bid2, bid1));
            done_paths.insert((bid1, bid2, bid3));
            done_paths.insert((bid3, bid2, bid1));
            if rsize > 5 {
                set_one_14(p, bi, comp, mmat, topo, bid1, bid2, bid3, Some(rsize), true);
            } else {
                record_14_path(comp, p, bi, bid1, bid2, bid3);
            }
            bid1 = bid2;
        }
    }

    for bid2 in 0..bi.bonds.len() {
        let aid2 = bi.bonds[bid2].a;
        let aid3 = bi.bonds[bid2].b;
        for &bid1 in &bi.atom_bonds[aid2] {
            if bid1 == bid2 {
                continue;
            }
            for &bid3 in &bi.atom_bonds[aid3] {
                if bid3 == bid2 {
                    continue;
                }
                if done_paths.contains(&(bid1, bid2, bid3))
                    || done_paths.contains(&(bid3, bid2, bid1))
                {
                    continue;
                }
                let in_ring = ring_bond_pairs.contains(&(bid1, bid2))
                    || ring_bond_pairs.contains(&(bid2, bid1))
                    || ring_bond_pairs.contains(&(bid2, bid3))
                    || ring_bond_pairs.contains(&(bid3, bid2));
                let middle_ring = n_bond_rings(p, bi, bid2) > 0;
                let b1_ring = n_bond_rings(p, bi, bid1) > 0;
                let b3_ring = n_bond_rings(p, bi, bid3) > 0;
                if in_ring {
                    // two in same ring: 0-180 unless sp2-sp2 → trans flat
                    set_two_in_same_ring_14(p, bi, comp, mmat, topo, bid1, bid2, bid3);
                } else if (b1_ring && middle_ring) || (middle_ring && b3_ring) {
                    // two-in-different-ring → _setInRing14Bounds(ringSize=0)
                    set_one_14(p, bi, comp, mmat, topo, bid1, bid2, bid3, Some(0), true);
                } else if middle_ring {
                    // share-ring-bond → also _setInRing14Bounds(ringSize=0)
                    set_one_14(p, bi, comp, mmat, topo, bid1, bid2, bid3, Some(0), true);
                } else {
                    set_one_14(p, bi, comp, mmat, topo, bid1, bid2, bid3, None, false);
                }
            }
        }
    }
}

fn n_bond_rings(p: &DgFeatures, bi: &BondIndex, bid: usize) -> usize {
    let b = &bi.bonds[bid];
    // count rings whose consecutive atoms include this bond
    p.ring_idx
        .iter()
        .filter(|ring| {
            let rsize = ring.len();
            (0..rsize).any(|i| {
                let x = ring[i];
                let y = ring[(i + 1) % rsize];
                (x == b.a && y == b.b) || (x == b.b && y == b.a)
            })
        })
        .count()
}

#[allow(clippy::too_many_arguments)]
fn set_two_in_same_ring_14(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &mut Computed,
    mmat: &mut BoundsMatrix,
    topo: &[Vec<f64>],
    bid1: usize,
    bid2: usize,
    bid3: usize,
) {
    let atm2 = match comp.adj(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let atm3 = match comp.adj(bid2, bid3) {
        Some(a) => a,
        None => return,
    };
    let aid1 = other_atom(&bi.bonds[bid1], atm2);
    let aid4 = other_atom(&bi.bonds[bid3], atm3);
    if topo[aid1.max(aid4)][aid1.min(aid4)] < 2.9 {
        return;
    }
    if bi.pair_to_bond.contains_key(&pair(aid1, atm3))
        || bi.pair_to_bond.contains_key(&pair(aid4, atm2))
    {
        return;
    }
    let bl1 = comp.bond_lengths[bid1];
    let bl2 = comp.bond_lengths[bid2];
    let bl3 = comp.bond_lengths[bid3];
    let ba12 = match comp.angle(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let ba23 = match comp.angle(bid2, bid3) {
        Some(a) => a,
        None => return,
    };
    let hyb2 = p.atoms[atm2].hybridization;
    let hyb3 = p.atoms[atm3].hybridization;
    let mut dl;
    let mut du;
    let mut ptype = Path14Type::Other;
    if hyb2 == Hybridization::Sp2 && hyb3 == Hybridization::Sp2 {
        dl = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
        du = dl;
        dl -= GEN_DIST_TOL;
        du += GEN_DIST_TOL;
        ptype = Path14Type::Trans;
        comp.trans_paths.insert((bid1, bid2, bid3));
        comp.trans_paths.insert((bid3, bid2, bid1));
    } else {
        dl = compute14_dist_cis(bl1, bl2, bl3, ba12, ba23);
        du = compute14_dist_trans(bl1, bl2, bl3, ba12, ba23);
        if du < dl {
            std::mem::swap(&mut du, &mut dl);
        }
        if (du - dl).abs() < DIST12_DELTA {
            dl -= GEN_DIST_TOL;
            du += GEN_DIST_TOL;
        }
    }
    check_and_set_bounds(mmat, aid1, aid4, dl, du);
    comp.paths14.push(Path14 {
        bid1,
        bid2,
        bid3,
        ptype,
    });
}

fn record_14_path(
    comp: &mut Computed,
    p: &DgFeatures,
    bi: &BondIndex,
    bid1: usize,
    bid2: usize,
    bid3: usize,
) {
    let atm2 = match comp.adj(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let atm3 = match comp.adj(bid2, bid3) {
        Some(a) => a,
        None => return,
    };
    let hyb2 = p.atoms[atm2].hybridization;
    let hyb3 = p.atoms[atm3].hybridization;
    let ptype = if hyb2 == Hybridization::Sp2 && hyb3 == Hybridization::Sp2 {
        comp.cis_paths.insert((bid1, bid2, bid3));
        comp.cis_paths.insert((bid3, bid2, bid1));
        Path14Type::Cis
    } else {
        Path14Type::Other
    };
    let _ = bi;
    comp.paths14.push(Path14 {
        bid1,
        bid2,
        bid3,
        ptype,
    });
}

// ── 1-5 bounds ──────────────────────────────────────────────────────────────

fn extend_14_path_to_15(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &mut Computed,
    mmat: &mut BoundsMatrix,
    topo: &[Vec<f64>],
    bid1: usize,
    bid2: usize,
    bid3: usize,
    ptype: Path14Type,
) {
    let aid2 = match comp.adj(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let aid1 = other_atom(&bi.bonds[bid1], aid2);
    let aid3 = match comp.adj(bid2, bid3) {
        Some(a) => a,
        None => return,
    };
    let aid4 = other_atom(&bi.bonds[bid3], aid3);
    let d1 = comp.bond_lengths[bid1];
    let d2 = comp.bond_lengths[bid2];
    let d3 = comp.bond_lengths[bid3];
    let ang12 = match comp.angle(bid1, bid2) {
        Some(a) => a,
        None => return,
    };
    let ang23 = match comp.angle(bid2, bid3) {
        Some(a) => a,
        None => return,
    };

    for i in 0..bi.bonds.len() {
        if comp.adj(bid3, i) != Some(aid4) {
            continue;
        }
        let aid5 = other_atom(&bi.bonds[i], aid4);
        if topo[aid1.max(aid5)][aid1.min(aid5)] < 3.9 {
            continue;
        }
        if aid1 == aid5 {
            continue;
        }
        let already = comp.set15.contains(&pair(aid1, aid5));
        if !(mmat.lower(aid1, aid5) < DIST12_DELTA || already) {
            continue;
        }
        let d4 = comp.bond_lengths[i];
        let ang34 = match comp.angle(bid3, i) {
            Some(a) => a,
            None => continue,
        };
        let path_id = (bid2, bid3, i);
        let mut dl;
        let mut du;
        match ptype {
            Path14Type::Cis => {
                if comp.cis_paths.contains(&path_id) {
                    dl = c15_cis_cis(d1, d2, d3, d4, ang12, ang23, ang34);
                    du = dl + DIST15_TOL;
                    dl -= DIST15_TOL;
                } else if comp.trans_paths.contains(&path_id) {
                    dl = c15_cis_trans(d1, d2, d3, d4, ang12, ang23, ang34);
                    du = dl + DIST15_TOL;
                    dl -= DIST15_TOL;
                } else {
                    dl = c15_cis_cis(d1, d2, d3, d4, ang12, ang23, ang34) - DIST15_TOL;
                    du = c15_cis_trans(d1, d2, d3, d4, ang12, ang23, ang34) + DIST15_TOL;
                }
            }
            Path14Type::Trans => {
                if comp.cis_paths.contains(&path_id) {
                    dl = c15_trans_cis(d1, d2, d3, d4, ang12, ang23, ang34);
                    du = dl + DIST15_TOL;
                    dl -= DIST15_TOL;
                } else if comp.trans_paths.contains(&path_id) {
                    dl = c15_trans_trans(d1, d2, d3, d4, ang12, ang23, ang34);
                    du = dl + DIST15_TOL;
                    dl -= DIST15_TOL;
                } else {
                    dl = c15_trans_cis(d1, d2, d3, d4, ang12, ang23, ang34) - DIST15_TOL;
                    du = c15_trans_trans(d1, d2, d3, d4, ang12, ang23, ang34) + DIST15_TOL;
                }
            }
            Path14Type::Other => {
                if comp.cis_paths.contains(&path_id) {
                    dl = c15_cis_cis(d4, d3, d2, d1, ang34, ang23, ang12) - DIST15_TOL;
                    du = c15_cis_trans(d4, d3, d2, d1, ang34, ang23, ang12) + DIST15_TOL;
                } else if comp.trans_paths.contains(&path_id) {
                    dl = c15_trans_cis(d4, d3, d2, d1, ang34, ang23, ang12) - DIST15_TOL;
                    du = c15_trans_trans(d4, d3, d2, d1, ang34, ang23, ang12) + DIST15_TOL;
                } else {
                    let vw1 = rvdw(p.atoms[aid1].element.z());
                    let vw5 = rvdw(p.atoms[aid5].element.z());
                    dl = VDW_SCALE_15 * (vw1 + vw5);
                    du = -1.0;
                }
            }
        }
        if du < 0.0 {
            du = MAX_UPPER;
        }
        check_and_set_bounds(mmat, aid1, aid5, dl, du);
        comp.set15.insert(pair(aid1, aid5));
    }
}

fn set_15_bounds(
    p: &DgFeatures,
    bi: &BondIndex,
    comp: &mut Computed,
    mmat: &mut BoundsMatrix,
    topo: &[Vec<f64>],
) {
    let paths = comp.paths14.clone();
    for path in paths {
        extend_14_path_to_15(
            p, bi, comp, mmat, topo, path.bid1, path.bid2, path.bid3, path.ptype,
        );
        extend_14_path_to_15(
            p, bi, comp, mmat, topo, path.bid3, path.bid2, path.bid1, path.ptype,
        );
    }
}

// ── vdW lower bounds ─────────────────────────────────────────────────────────

fn set_lower_bound_vdw(p: &DgFeatures, mmat: &mut BoundsMatrix, topo: &[Vec<f64>]) {
    let n = p.atoms.len();
    for i in 1..n {
        let vw1 = rvdw(p.atoms[i].element.z());
        for j in 0..i {
            let vw2 = rvdw(p.atoms[j].element.z());
            if mmat.lower(i, j) < DIST12_DELTA {
                let d = topo[i][j];
                let lb = if d == 4.0 {
                    VDW_SCALE_15 * (vw1 + vw2)
                } else if d == 5.0 {
                    (VDW_SCALE_15 + 0.5 * (1.0 - VDW_SCALE_15)) * (vw1 + vw2)
                } else {
                    vw1 + vw2
                };
                mmat.set_lower(i, j, lb);
            }
        }
    }
}

/// Build the full topological bounds matrix for `p` (RDKit `setTopolBounds`
/// with `set15bounds=true, scaleVDW=false`).
pub fn set_topol_bounds(p: &DgFeatures) -> BoundsMatrix {
    let n = p.atoms.len();
    let mut mmat = BoundsMatrix::new(n, 0.0);
    // initBoundsMat(min=0, max=1000)
    for i in 1..n {
        for j in 0..i {
            mmat.set_upper(i, j, MAX_UPPER);
            mmat.set_lower(i, j, 0.0);
        }
    }
    let bi = build_bond_index(p);
    let topo = topo_distances(p);
    let mut comp = Computed::new(bi.bonds.len());

    set_12_bounds(p, &bi, &mut comp, &mut mmat);
    set_13_bounds(p, &bi, &mut comp, &mut mmat);
    set_14_bounds(p, &bi, &mut comp, &mut mmat, &topo);
    set_15_bounds(p, &bi, &mut comp, &mut mmat, &topo);
    set_lower_bound_vdw(p, &mut mmat, &topo);
    mmat
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Element;

    fn atom(element: Element, hybridization: Hybridization) -> PerceivedAtom {
        PerceivedAtom {
            element,
            hybridization,
            aromatic: false,
            conjugated: false,
            degree: 0,
            total_valence: 0.0,
        }
    }

    #[test]
    fn effective_order_is_one_and_a_half_when_aromatic() {
        assert_eq!(effective_bond_order(1.0, true), 1.5);
        assert_eq!(effective_bond_order(2.0, false), 2.0);
    }

    /// A typed bond gets the UFF rest length; one with no usable order gets
    /// RDKit's van der Waals guess.
    #[test]
    fn a_typed_bond_gets_the_uff_rest_length_and_an_unordered_one_the_vdw_guess() {
        let c = atom(Element::C, Hybridization::Sp3);
        let r_cc = bond_rest_length(&c, &c, 1.0);
        assert_eq!(Some(r_cc), uff_rest_length("C_3", "C_3", 1.0));
        assert!(
            bond_rest_length(&c, &c, 2.0) < r_cc,
            "a double bond is shorter"
        );
        assert_eq!(bond_rest_length(&c, &c, 0.0), 0.5 * (rvdw(6) + rvdw(6)));
    }

    #[test]
    fn rvdw_is_rdkits_table_with_a_two_angstrom_default() {
        assert_eq!(rvdw(6), 1.7);
        assert_eq!(rvdw(1), 1.2);
        assert_eq!(rvdw(26), 2.0);
    }
}
