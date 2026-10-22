//! MMFF94 parameter resolution: equivalence-level table lookups, type codes,
//! and the empirical-rule fallbacks used when explicit parameters are absent.
//!
//! **This is the one correct implementation of MMFF's context rules in molrs**,
//! and everything that needs a bond / angle / torsion type or a force constant
//! comes through here. It is not an energy file: energies come from the
//! generic typifier → `ForceField` → kernel path, which reproduces RDKit on
//! 11/11 fixtures, term by term. The resolver is what the *typifier* runs.
//!
//! What makes these rules irreducible to a `(type_i, type_j, …) → params` table
//! — and hence the reason MMFF's kernels are
//! [`ParamSource::PerInstance`](crate::ff::ir::ParamSource::PerInstance):
//!
//! * the type codes depend on **topology**, not just atom types — aromaticity
//!   ([`bond_type`]), 3-/4-membered-ring membership ([`angle_type`]),
//!   4-/5-ring torsions ([`torsion_type`]);
//! * lookups **degrade** through four equivalence levels ([`eq_level`]) before
//!   they give up;
//! * and when they do give up, the parameters are **invented on the spot** from
//!   covalent radii and electronegativities ([`bond_empirical`],
//!   [`angle_empirical`], [`torsion_empirical`]).
//!
//! The typifier resolves all of it per interaction and bakes the numbers into
//! Frame columns; the kernels read those columns.
//!
//! Ported from RDKit (BSD-3, Paolo Tosco / RDKit contributors):
//! - `Code/ForceField/MMFF/Params.h` (the `operator()` / `getMMFF*Params`
//!   equivalence-level lookups).
//! - `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp`
//!   (`getMMFFAngleType` / `getMMFFBondType` / `getMMFFStretchBendType` /
//!   `getMMFFTorsionType`, and the `getMMFF*EmpiricalRuleParams` fallbacks).
//!
//! Reference: Halgren, T. A. *J. Comput. Chem.* 1996, 17, 490-641 (MMFF.I-V).
//!
//! # Why this file is not in `ff/params/`
//!
//! `ff/params/` is the home of *tables*, and this file holds none — it is an
//! algorithm (`bond_type` / `angle_type` / `torsion_type`, the four-level
//! equivalence degradation, the empirical rules). It reads
//! [`crate::ff::params::mmff`] from one module away and belongs where it is
//! used, with the typifier front end. Its name says what it does: it
//! *resolves*.

use super::charges::mmff_bond_type;
use super::properties::MmffVariant;
use crate::core::BondOrder;
use crate::ff::params::mmff::{
    mmff_angle, mmff_bndk, mmff_bond, mmff_cov_rad_pau_ele, mmff_def, mmff_dfsb,
    mmff_herschbach_laurie, mmff_oop, mmff_oop_s, mmff_prop, mmff_stbn, mmff_tor, mmff_tor_s,
};
use crate::perceive::mmff_aromaticity::MmffTopology;

// MMFF's degree → radian conversion, from the crate-level `constants` module so
// the `potential` kernels share one definition. (The MMFF tables store reference
// angles in degrees; molrs is radians internally.)

#[inline]
fn is_zero(x: f64) -> bool {
    x.abs() < 1.0e-8
}

/// MMFF resolved bond parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BondParams {
    pub r0: f64,
    pub kb: f64,
}

/// MMFF resolved angle parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AngleParams {
    pub theta0: f64,
    pub ka: f64,
}

/// MMFF resolved stretch-bend force constants (`kba_ijk`, `kba_kji`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct StretchBendParams {
    pub kba_ijk: f64,
    pub kba_kji: f64,
}

/// MMFF resolved torsion Fourier coefficients.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TorParams {
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

// vdW is deliberately absent from this resolver. MMFF's van der Waals
// parameters are a genuine 95-row per-atom-type table, so `pair/mmff_vdw` is a
// real `ParamSource::TypeRows` style: the rows live in the parameter XML and
// `pair_mmff_vdw_constructor` (`ff/potential/pair/mmff.rs`) applies the combining rules to
// them. Resolving them a second time here would be the duplication this module's
// own history is a warning about.

// --- periodic-table rows (AtomTyper.cpp) ---------------------------------

fn periodic_row(atno: u8) -> u8 {
    match atno {
        3..=10 => 1,
        11..=18 => 2,
        19..=36 => 3,
        37..=54 => 4,
        _ => 0,
    }
}

fn periodic_row_hl(atno: u8) -> u8 {
    let mut row = match atno {
        2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        _ => 0,
    };
    if (21..=30).contains(&atno) || (39..=48).contains(&atno) {
        row *= 10;
    }
    row
}

// --- bond type (single sbmb/arom pair) -----------------------------------

/// RDKit `getMMFFBondType`. Delegates to the already-ported charge helper.
///
/// **1 only for a bond that is `SINGLE` and joins two `sbmb`/`arom` atom types.**
/// After MMFF aromaticity perception an aromatic ring bond is `AROMATIC`, never
/// `SINGLE`, so a benzene ring bond is bond type **0** — the inter-ring bond of
/// biphenyl is the shape that gets a 1. Reading the raw bond order instead (and
/// calling ~1.5 "aromatic → 1") inverts the rule, and on a Kekulé input labels
/// the two halves of a six-fold-symmetric ring differently.
pub(crate) fn bond_type(topo: &MmffTopology, types: &[u8], i: usize, j: usize) -> u8 {
    mmff_bond_type(topo, types, i, j)
}

// --- ring-size helpers (AtomTyper.cpp) -----------------------------------

/// RDKit `isAngleInRingOfSize3or4`.
fn angle_ring_size(topo: &MmffTopology, i: usize, j: usize, k: usize) -> u8 {
    if topo.bond_order(i, j).is_none() || topo.bond_order(j, k).is_none() {
        return 0;
    }
    if topo.bond_order(k, i).is_some() {
        return 3;
    }
    // 4-ring: a shared neighbour of i and k (other than j).
    for &a in &topo.nbrs[i] {
        if a == j {
            continue;
        }
        if topo.nbrs[k].iter().any(|&b| b != j && b == a) {
            return 4;
        }
    }
    0
}

/// RDKit `isTorsionInRingOfSize4or5`.
fn torsion_ring_size(topo: &MmffTopology, i: usize, j: usize, k: usize, l: usize) -> u8 {
    if topo.bond_order(i, j).is_none()
        || topo.bond_order(j, k).is_none()
        || topo.bond_order(k, l).is_none()
    {
        return 0;
    }
    if topo.bond_order(l, i).is_some() {
        return 4;
    }
    for &a in &topo.nbrs[i] {
        if a == j {
            continue;
        }
        if topo.nbrs[l].iter().any(|&b| b != k && b == a) {
            return 5;
        }
    }
    0
}

// --- angle / stretch-bend / torsion type codes (AtomTyper.cpp) -----------

/// RDKit `getMMFFAngleType`.
///
/// Needs the **topology**, not just the two bond types: [`angle_ring_size`]
/// promotes an angle inside a 3- or 4-membered ring, overwriting the bond-type
/// sum with the ring size (cyclopropane's C-C-C angles are type 3). A classifier
/// whose arguments are only `(bt_ij, bt_jk)` cannot express that at any input —
/// ring membership is not among them.
pub(crate) fn angle_type(topo: &MmffTopology, types: &[u8], i: usize, j: usize, k: usize) -> u8 {
    let bts = bond_type(topo, types, i, j) + bond_type(topo, types, j, k);
    let mut at = bts;
    let size = angle_ring_size(topo, i, j, k);
    if size != 0 {
        at = size;
        if bts != 0 {
            at += bts + size - 2;
        }
    }
    at
}

/// RDKit `getMMFFStretchBendType`: the stretch-bend class (SBT 0–11) of an
/// angle of type `angle_type` whose i–j bond has type `bt1` and j–k bond type
/// `bt2`. For the classes that tell the two bonds apart (angle types 1, 5
/// and 7), which bond is the type-1 bond depends on the order `bt1`, `bt2`
/// are given in.
pub(crate) fn stretch_bend_type(angle_type: u8, bt1: u8, bt2: u8) -> u8 {
    match angle_type {
        1 => {
            if bt1 != 0 || bt1 == bt2 {
                1
            } else {
                2
            }
        }
        2 => 3,
        4 => 4,
        3 => 5,
        5 => {
            if bt1 != 0 || bt1 == bt2 {
                6
            } else {
                7
            }
        }
        6 => 8,
        7 => {
            if bt1 != 0 || bt1 == bt2 {
                9
            } else {
                10
            }
        }
        8 => 11,
        _ => 0,
    }
}

/// RDKit `getMMFFTorsionType`. Returns `(principal, secondary)`.
///
/// Same disease as [`angle_type`] if you try to key it off bond types alone: the
/// 4-/5-membered-ring promotions ([`torsion_ring_size`]) need the topology.
pub(crate) fn torsion_type(
    topo: &MmffTopology,
    types: &[u8],
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> (u8, u8) {
    let bt_ij = bond_type(topo, types, i, j);
    let bt_jk = bond_type(topo, types, j, k);
    let bt_kl = bond_type(topo, types, k, l);
    let jk_single = topo.bond_order(j, k) == Some(BondOrder::Single);
    let mut tt = bt_jk;
    let mut second = 0u8;
    if bt_jk == 0 && jk_single && (bt_ij == 1 || bt_kl == 1) {
        tt = 2;
    }
    let size = torsion_ring_size(topo, i, j, k, l);
    if size == 4 && topo.bond_order(i, k).is_none() && topo.bond_order(j, l).is_none() {
        second = tt;
        tt = 4;
    } else if size == 5 && (types[i] == 1 || types[j] == 1 || types[k] == 1 || types[l] == 1) {
        second = tt;
        tt = 5;
    }
    (tt, second)
}

// --- equivalence-level lookups (Params.h operator()) ---------------------

/// RDKit `MMFFDefCollection::operator()` equivalence-level lookup.
///
/// `MMFF_DEF` (generated from RDKit `defaultMMFFDef`) covers every MMFF
/// atom type 1-82 / 87-99, so the 5-stage equivalence search for aromatic /
/// charged-N / metal types reaches its wild-card defaults uniformly. Falls
/// back to the type itself for any key absent from the table (RDKit returns a
/// null params pointer there, which the callers below treat as "no match").
fn eq_level(atom_type: u8, level: usize) -> u8 {
    mmff_def(atom_type)
        .map(|d| d.eq_level[level])
        .unwrap_or(atom_type)
}

/// RDKit `MMFFAngleCollection::operator()` (5-stage equivalence search).
fn angle_lookup(angle_type: u8, i: u8, j: u8, k: u8) -> Option<AngleParams> {
    for iter in 0..4 {
        let (mut ci, mut ck) = (eq_level(i, iter), eq_level(k, iter));
        if ci > ck {
            std::mem::swap(&mut ci, &mut ck);
        }
        if let Some(a) = mmff_angle(angle_type, ci, j, ck) {
            return Some(AngleParams {
                theta0: a.theta0,
                ka: a.ka,
            });
        }
    }
    None
}

/// RDKit `MMFFOopCollection::operator()` (5-stage equivalence search).
///
/// RDKit selects the whole table at construction
/// (`defaultMMFFsOop` vs `defaultMMFFOop`) based on `isMMFFs`. The `_S` table
/// shares every key with the base table, so we look up `_S` first under
/// [`MmffVariant::Mmff94s`] and fall back to the base table for safety.
///
/// Returns the **canonical key of the row that matched** (peripherals degraded to
/// the level that hit, then sorted ascending; the centre stays in second place)
/// together with its `koop`. The key is what the typifier stamps on the improper
/// as its `type` label, so the label names the row the number actually came from.
fn out_of_plane_lookup(variant: MmffVariant, i: u8, j: u8, k: u8, l: u8) -> Option<(String, f64)> {
    for iter in 0..4 {
        let mut ikl = [eq_level(i, iter), eq_level(k, iter), eq_level(l, iter)];
        ikl.sort_unstable();
        let hit = match variant {
            MmffVariant::Mmff94s => mmff_oop_s(ikl[0], j, ikl[1], ikl[2])
                .or_else(|| mmff_oop(ikl[0], j, ikl[1], ikl[2])),
            MmffVariant::Mmff94 => mmff_oop(ikl[0], j, ikl[1], ikl[2]),
        };
        if let Some(o) = hit {
            return Some((format!("{}_{}_{}_{}", ikl[0], j, ikl[1], ikl[2]), o.koop));
        }
    }
    None
}

/// RDKit `MMFFTorCollection::getMMFFTorParams`. Returns `(secondary, params)`,
/// `secondary` telling whether the row was found only on the restart under the
/// secondary torsion type.
///
/// Like the Oop collection, RDKit picks `defaultMMFFsTor` vs `defaultMMFFTor`
/// wholesale on `isMMFFs`; the `_S` table shares every key with the base
/// table, so under [`MmffVariant::Mmff94s`] we query `_S` first and fall back
/// to the base table.
fn torsion_lookup(
    variant: MmffVariant,
    tor_type: (u8, u8),
    i: u8,
    j: u8,
    k: u8,
    l: u8,
) -> Option<(bool, TorParams)> {
    let mut iter: i32 = 0;
    let mut max_iter = 5i32;
    let mut can_tor = tor_type.0;
    let mut secondary = false;
    // Mirrors the RDKit `while` guard: keep iterating while we have not yet
    // found a hit (we return on the first), with the special last-resort
    // restart when torType is (5, secondary).
    loop {
        let cont = (iter < max_iter) || (iter == 4 && tor_type.0 == 5 && tor_type.1 != 0);
        if !cont {
            break;
        }
        if max_iter == 5 && iter == 4 {
            max_iter = 4;
            iter = 0;
            can_tor = tor_type.1;
            secondary = true;
        }
        let (mut i_wild, mut l_wild) = (iter as usize, iter as usize);
        if iter == 1 {
            i_wild = 1;
            l_wild = 3;
        } else if iter == 2 {
            i_wild = 3;
            l_wild = 1;
        }
        let mut ci = eq_level(i, i_wild);
        let mut cj = j;
        let mut ck = k;
        let mut cl = eq_level(l, l_wild);
        if cj > ck {
            std::mem::swap(&mut cj, &mut ck);
            std::mem::swap(&mut ci, &mut cl);
        } else if cj == ck && ci > cl {
            std::mem::swap(&mut ci, &mut cl);
        }
        let hit = match variant {
            MmffVariant::Mmff94s => {
                mmff_tor_s(can_tor, ci, cj, ck, cl).or_else(|| mmff_tor(can_tor, ci, cj, ck, cl))
            }
            MmffVariant::Mmff94 => mmff_tor(can_tor, ci, cj, ck, cl),
        };
        if let Some(t) = hit {
            return Some((
                secondary,
                TorParams {
                    v1: t.v1,
                    v2: t.v2,
                    v3: t.v3,
                },
            ));
        }
        iter += 1;
    }
    None
}

// --- bond stretch (explicit + empirical) ---------------------------------

/// RDKit `getMMFFBondStretchParams` + `getMMFFBondStretchEmpiricalRuleParams`.
pub(crate) fn bond_params(
    topo: &MmffTopology,
    types: &[u8],
    i: usize,
    j: usize,
) -> Option<BondParams> {
    let bt = bond_type(topo, types, i, j);
    let (ti, tj) = (types[i].min(types[j]), types[i].max(types[j]));
    if let Some(b) = mmff_bond(bt, ti, tj) {
        return Some(BondParams { r0: b.r0, kb: b.kb });
    }
    bond_empirical(topo, i, j)
}

/// RDKit `getMMFFBondStretchEmpiricalRuleParams` (MMFF.V eq. 18/19 + HL rule).
fn bond_empirical(topo: &MmffTopology, i: usize, j: usize) -> Option<BondParams> {
    let (an1, an2) = (topo.atno[i], topo.atno[j]);
    let cr1 = mmff_cov_rad_pau_ele(an1)?;
    let cr2 = mmff_cov_rad_pau_ele(an2)?;
    let c = if an1 == 1 || an2 == 1 { 0.050 } else { 0.085 };
    let n = 1.4;
    let r0 = cr1.r0 + cr2.r0 - c * (cr1.chi - cr2.chi).abs().powf(n);
    let kb = if let Some(bndk) = mmff_bndk(an1, an2) {
        let coeff = bndk.r0 / r0;
        let coeff2 = coeff * coeff;
        bndk.kb * coeff2 * coeff2 * coeff2
    } else {
        let hl = mmff_herschbach_laurie(periodic_row_hl(an1), periodic_row_hl(an2))?;
        10f64.powf(-(r0 - hl.a_ij) / hl.d_ij)
    };
    Some(BondParams { r0, kb })
}

// --- angle bend (explicit + empirical) -----------------------------------

/// RDKit `getMMFFAngleBendParams`.
pub(crate) fn angle_params(
    topo: &MmffTopology,
    types: &[u8],
    i: usize,
    j: usize,
    k: usize,
) -> Option<AngleParams> {
    let at = angle_type(topo, types, i, j, k);
    let explicit = angle_lookup(at, types[i], types[j], types[k]);
    match explicit {
        Some(p) if !is_zero(p.ka) => Some(p),
        _ => {
            let b1 = bond_params(topo, types, i, j)?;
            let b2 = bond_params(topo, types, j, k)?;
            angle_empirical(topo, types, explicit, (&b1, &b2), [i, j, k])
        }
    }
}

/// RDKit `getMMFFAngleBendEmpiricalRuleParams` (MMFF.V eq. 20 + Table VI).
fn angle_empirical(
    topo: &MmffTopology,
    types: &[u8],
    old: Option<AngleParams>,
    bonds: (&BondParams, &BondParams),
    idx: [usize; 3],
) -> Option<AngleParams> {
    let [i, j, k] = idx;
    let (b1, b2) = bonds;
    let atno = [topo.atno[i], topo.atno[j], topo.atno[k]];
    let ring = angle_ring_size(topo, i, j, k);
    let prop_j = mmff_prop(types[j])?;

    let theta0 = if let Some(o) = old {
        o.theta0
    } else {
        let mut t = 120.0;
        match prop_j.crd {
            4 => t = 109.45,
            2 => {
                if atno[1] == 8 {
                    t = 105.0;
                } else if prop_j.linh == 1 {
                    t = 180.0;
                }
            }
            3 if prop_j.val == 3 && prop_j.mltb == 0 => {
                t = if atno[1] == 7 { 107.0 } else { 92.0 };
            }
            _ => {}
        }
        if ring == 3 {
            t = 60.0;
        } else if ring == 4 {
            t = 90.0;
        }
        t
    };

    // Table VI (MMFF.V p.628) Z and C constants.
    let zc = |an: u8| -> (f64, f64) {
        match an {
            1 => (1.395, 0.0),
            6 => (2.494, 1.016),
            7 => (2.711, 1.113),
            8 => (3.045, 1.337),
            9 => (2.847, 0.0),
            14 => (2.350, 0.811),
            15 => (2.350, 1.068),
            16 => (2.980, 1.249),
            17 => (2.909, 1.078),
            35 => (3.017, 0.0),
            53 => (3.086, 0.0),
            _ => (0.0, 0.0),
        }
    };
    let (z0, _) = zc(atno[0]);
    let (_, c1) = zc(atno[1]);
    let (z2, _) = zc(atno[2]);

    let mut beta = 1.75;
    let r0ij = b1.r0;
    let r0jk = b2.r0;
    let d = (r0ij - r0jk) * (r0ij - r0jk) / ((r0ij + r0jk) * (r0ij + r0jk));
    let theta0_rad = theta0.to_radians();
    if ring == 4 {
        beta *= 0.85;
    } else if ring == 3 {
        beta *= 0.05;
    }
    let ka = beta * z0 * c1 * z2 / ((r0ij + r0jk) * theta0_rad * theta0_rad * (2.0 * d).exp());
    Some(AngleParams { theta0, ka })
}

// --- stretch-bend (explicit + dfsb fallback) -----------------------------

/// RDKit `getMMFFStretchBendParams`. Returns the resolved params plus the
/// two bond rest lengths and the angle theta0 (needed by the energy term).
pub(crate) fn stretch_bend_params(
    topo: &MmffTopology,
    types: &[u8],
    i: usize,
    j: usize,
    k: usize,
) -> Option<(StretchBendParams, f64, f64, f64)> {
    let prop_j = mmff_prop(types[j])?;
    if prop_j.linh != 0 {
        return None;
    }
    let b1 = bond_params(topo, types, i, j)?;
    let b2 = bond_params(topo, types, j, k)?;
    let angle = angle_params(topo, types, i, j, k)?;

    let bt1 = bond_type(topo, types, i, j);
    let bt2 = bond_type(topo, types, j, k);
    let at = angle_type(topo, types, i, j, k);
    let (ti, tk) = (types[i], types[k]);
    let sbt = stretch_bend_type(
        at,
        if ti <= tk { bt1 } else { bt2 },
        if ti < tk { bt2 } else { bt1 },
    );

    // explicit STBN lookup with the same swap convention as RDKit.
    let (swap, stbn) = stretch_bend_lookup(sbt, bt1, bt2, ti, types[j], tk);
    let params = if let Some(s) = stbn {
        if swap {
            StretchBendParams {
                kba_ijk: s.1,
                kba_kji: s.0,
            }
        } else {
            StretchBendParams {
                kba_ijk: s.0,
                kba_kji: s.1,
            }
        }
    } else {
        // default-by-period-row (mmff_dfsb)
        let (swap_d, d) = dfsb_lookup(topo.atno[i], topo.atno[j], topo.atno[k]);
        let d = d?;
        if swap_d {
            StretchBendParams {
                kba_ijk: d.1,
                kba_kji: d.0,
            }
        } else {
            StretchBendParams {
                kba_ijk: d.0,
                kba_kji: d.1,
            }
        }
    };
    if is_zero(params.kba_ijk) && is_zero(params.kba_kji) {
        return None;
    }
    Some((params, b1.r0, b2.r0, angle.theta0))
}

/// RDKit `MMFFStbnCollection::getMMFFStbnParams`. Returns `(swap, (ijk,kji))`.
fn stretch_bend_lookup(
    sbt: u8,
    bt1: u8,
    bt2: u8,
    i: u8,
    j: u8,
    k: u8,
) -> (bool, Option<(f64, f64)>) {
    let (mut ci, mut ck) = (i, k);
    let swap = if i > k {
        std::mem::swap(&mut ci, &mut ck);
        true
    } else if i == k {
        bt1 < bt2
    } else {
        false
    };
    let p = mmff_stbn(sbt, ci, j, ck).map(|s| (s.kba_ijk, s.kba_kji));
    (swap, p)
}

/// RDKit `MMFFDfsbCollection::getMMFFDfsbParams`. Returns `(swap, (ijk,kji))`.
fn dfsb_lookup(an_i: u8, an_j: u8, an_k: u8) -> (bool, Option<(f64, f64)>) {
    let (mut r1, mut r3) = (periodic_row(an_i), periodic_row(an_k));
    let swap = if r1 > r3 {
        std::mem::swap(&mut r1, &mut r3);
        true
    } else {
        false
    };
    let p = mmff_dfsb(r1, periodic_row(an_j), r3).map(|d| (d.kba_ijk, d.kba_kji));
    (swap, p)
}

// --- out-of-plane --------------------------------------------------------

/// RDKit `getMMFFOopBendParams` (no empirical fallback; term excluded if absent).
///
/// `j` is the trigonal centre. Returns `(canonical key, koop)` — `None` when MMFF
/// defines no out-of-plane term for the centre, which is how the typifier knows
/// to skip it.
pub(crate) fn out_of_plane_params(
    variant: MmffVariant,
    types: &[u8],
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> Option<(String, f64)> {
    out_of_plane_lookup(variant, types[i], types[j], types[k], types[l])
}

// --- torsion (explicit + empirical) --------------------------------------

/// Which rule produced a torsion's `(v1, v2, v3)`, carrying every input that
/// rule reads beyond the principal torsion type and the four atom types.
///
/// Whether the principal pass hits is itself fixed by `(principal, i, j, k, l)`
/// and the variant, so a label that names the source together with those five
/// names exactly one parameter set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TorSource {
    /// A table row under the principal torsion type.
    Principal,
    /// No row under the principal type; a row found on the restart under this
    /// secondary torsion type.
    Secondary(u8),
    /// No row under either type: the MMFF.V empirical rules, which read the j–k
    /// bond class (the atom types of `j` and `k` fix the rest — element,
    /// `crd` / `val` / `pilp` / `mltb` / `linh` and the aromatic-type flag).
    Empirical(JkBond),
}

/// The j–k bond as the empirical torsion rules tell it apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum JkBond {
    /// SINGLE.
    Single,
    /// DOUBLE.
    Double,
    /// AROMATIC between two aromatic MMFF types on perceived-aromatic atoms —
    /// rule (b).
    Aromatic,
    /// Anything else (triple, or an aromatic bond rule (b) does not accept):
    /// the rules treat these alike.
    Other,
}

/// The empirical rules' reading of the j–k bond — the one place it is derived,
/// so the parameters and the typifier's label cannot disagree about it.
fn jk_bond(topo: &MmffTopology, types: &[u8], j: usize, k: usize) -> JkBond {
    match topo.bond_order(j, k) {
        Some(BondOrder::Single) => JkBond::Single,
        Some(BondOrder::Double) => JkBond::Double,
        Some(BondOrder::Aromatic)
            if crate::ff::params::mmff::mmff_is_arom(types[j])
                && crate::ff::params::mmff::mmff_is_arom(types[k])
                && topo.is_aromatic[j]
                && topo.is_aromatic[k] =>
        {
            JkBond::Aromatic
        }
        _ => JkBond::Other,
    }
}

/// RDKit `getMMFFTorsionParams`, with the [`TorSource`] the numbers came from.
/// The params are `None` when all coefficients vanish (or the empirical rules
/// know neither central type).
pub(crate) fn torsion_params(
    variant: MmffVariant,
    topo: &MmffTopology,
    types: &[u8],
    i: usize,
    j: usize,
    k: usize,
    l: usize,
) -> (TorSource, Option<TorParams>) {
    let tt = torsion_type(topo, types, i, j, k, l);
    let (source, p) = match torsion_lookup(variant, tt, types[i], types[j], types[k], types[l]) {
        Some((false, p)) => (TorSource::Principal, Some(p)),
        Some((true, p)) => (TorSource::Secondary(tt.1), Some(p)),
        None => {
            let bond = jk_bond(topo, types, j, k);
            (
                TorSource::Empirical(bond),
                torsion_empirical(topo, types, j, k, bond),
            )
        }
    };
    let p = p.filter(|p| !(is_zero(p.v1) && is_zero(p.v2) && is_zero(p.v3)));
    (source, p)
}

/// RDKit `getMMFFTorsionEmpiricalRuleParams` (MMFF.V rules a-h, p.632), for
/// the j–k bond class `bond` ([`jk_bond`]).
fn torsion_empirical(
    topo: &MmffTopology,
    types: &[u8],
    j: usize,
    k: usize,
    bond: JkBond,
) -> Option<TorParams> {
    let jp = mmff_prop(types[j])?;
    let kp = mmff_prop(types[k])?;
    let atno = [topo.atno[j], topo.atno[k]];

    let mut u = [0.0f64; 2];
    let mut v = [0.0f64; 2];
    let mut w = [0.0f64; 2];
    for (idx, &an) in atno.iter().enumerate() {
        match an {
            6 => {
                u[idx] = 2.0;
                v[idx] = 2.12;
            }
            7 => {
                u[idx] = 2.0;
                v[idx] = 1.5;
            }
            8 => {
                u[idx] = 2.0;
                v[idx] = 0.2;
                w[idx] = 2.0;
            }
            14 => {
                u[idx] = 1.25;
                v[idx] = 1.22;
            }
            15 => {
                u[idx] = 1.25;
                v[idx] = 2.40;
            }
            16 => {
                u[idx] = 1.25;
                v[idx] = 0.49;
                w[idx] = 8.0;
            }
            _ => {}
        }
    }

    let n_jk = ((jp.crd as i32 - 1) * (kp.crd as i32 - 1)) as f64;
    let mut tor = TorParams {
        v1: 0.0,
        v2: 0.0,
        v3: 0.0,
    };
    let (beta, pi_jk);
    let row = |a: u8| periodic_row(a);

    if jp.linh != 0 || kp.linh != 0 {
        // rule (a): all zero
    } else if bond == JkBond::Aromatic {
        // rule (b)
        beta = if (jp.val == 3 && kp.val == 4) || (jp.val == 4 && kp.val == 3) {
            3.0
        } else {
            6.0
        };
        pi_jk = if jp.pilp == 0 && kp.pilp == 0 {
            0.5
        } else {
            0.3
        };
        tor.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
    } else if bond == JkBond::Double {
        // rule (c)
        beta = 6.0;
        pi_jk = if jp.mltb == 2 && kp.mltb == 2 {
            1.0
        } else {
            0.4
        };
        tor.v2 = beta * pi_jk * (u[0] * u[1]).sqrt();
    } else if jp.crd == 4 && kp.crd == 4 {
        // rule (d)
        tor.v3 = (v[0] * v[1]).sqrt() / n_jk;
    } else if jp.crd == 4 && kp.crd != 4 {
        // rule (e)
        if (kp.crd == 3 && ((kp.val == 4 || kp.val == 34) || kp.mltb != 0))
            || (kp.crd == 2 && (kp.val == 3 || kp.mltb != 0))
        {
            // zero
        } else {
            tor.v3 = (v[0] * v[1]).sqrt() / n_jk;
        }
    } else if kp.crd == 4 && jp.crd != 4 {
        // rule (f)
        if (jp.crd == 3 && ((jp.val == 4 || jp.val == 34) || jp.mltb != 0))
            || (jp.crd == 2 && (jp.val == 3 || jp.mltb != 0))
        {
            // zero
        } else {
            tor.v3 = (v[0] * v[1]).sqrt() / n_jk;
        }
    } else if (bond == JkBond::Single && jp.mltb != 0 && kp.mltb != 0)
        || (jp.mltb != 0 && kp.pilp != 0)
        || (jp.pilp != 0 && kp.mltb != 0)
    {
        // rule (g)
        if jp.pilp != 0 && kp.pilp != 0 {
            // case (1): zero
        } else if jp.pilp != 0 && kp.mltb != 0 {
            // case (2)
            let pi = if jp.mltb == 1 {
                0.5
            } else if row(atno[0]) == 2 && row(atno[1]) == 2 {
                0.3
            } else {
                0.15
            };
            tor.v2 = 6.0 * pi * (u[0] * u[1]).sqrt();
        } else if kp.pilp != 0 && jp.mltb != 0 {
            // case (3)
            let pi = if kp.mltb == 1 {
                0.5
            } else if row(atno[0]) == 2 && row(atno[1]) == 2 {
                0.3
            } else {
                0.15
            };
            tor.v2 = 6.0 * pi * (u[0] * u[1]).sqrt();
        } else if (jp.mltb == 1 || kp.mltb == 1) && (atno[0] != 6 || atno[1] != 6) {
            // case (4)
            tor.v2 = 6.0 * 0.4 * (u[0] * u[1]).sqrt();
        } else {
            // case (5)
            tor.v2 = 6.0 * 0.15 * (u[0] * u[1]).sqrt();
        }
    } else {
        // rule (h)
        if (atno[0] == 8 || atno[0] == 16) && (atno[1] == 8 || atno[1] == 16) {
            tor.v2 = -(w[0] * w[1]).sqrt();
        } else {
            tor.v3 = (v[0] * v[1]).sqrt() / n_jk;
        }
    }
    Some(tor)
}
