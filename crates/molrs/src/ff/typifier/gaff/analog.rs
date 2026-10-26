//! GAFF bonds and angles the table lacks, estimated exactly as parmchk2
//! estimates them.
//!
//! parmchk2's `chk_bond` visits every bond `i-j` (`i < j`, atom order), names
//! it with its two types in order, and for a name no row holds tries a row over
//! equivalent types (`EQUA`), then the cheapest row over corresponding types
//! (`CORR`). Nothing else: its empirical bond formula needs the bond's length,
//! which only a row could give, so a bond neither step reaches is written with
//! a zero length marked `ATTN, need revision` — here, a missing term.
//!
//! `chk_angle` visits every angle `i-j-k` (`i ≠ k`), names it with its ends in
//! order, and tries: a row over equivalent types; the cheapest row over
//! corresponding types scoring at most `THRESHOLD_BA`; the empirical angle of
//! its own types; the empirical angle of equivalent types; the cheapest
//! empirical angle of corresponding types. The empirical angle (Wang et al.
//! 2004, Eq. 5) takes θ₀ as the mean of the `A-B-A` and `C-B-C` rows and the
//! two bond lengths from the bond rows — the table's, and those estimated
//! before it in the molecule — so it is reached only where those exist.
//!
//! A `CORR` row scores, per atom, `w·column + w·force-column`: bond length and
//! force at either end of a bond (`bl`, `blf`); at an angle's end `ba`, `baf`,
//! at its vertex `cba`, `cbaf` times `WEIGHT_BA_CTR` — the `PARMCHK.DAT`
//! columns in parmchk2's order (`bl blf cba cbaf ba baf ctor tor`), a blank
//! one at its `DEFAULT_*`, a `CORR` line with no columns at 0. A substitution
//! across atom-type groups costs `WEIGHT_GROUP`, and a bond also pays
//! `equtype_penalty`. A name estimated once is reused for the rest of the
//! molecule.
//!
//! Reproduced as written (AmberTools `parmchk2.c` `chk_bond`, `chk_angle`,
//! `empangle`, `read_parmchk_parm`), including the order it adds penalties in
//! (which decides between equal analogs), and checked against parmchk2
//! (AmberTools 26.1) on 127 molecules and on 800 type graphs built to need
//! bond and angle estimates: every frcmod bond and angle row — parameters,
//! analog, penalty, and every `ATTN` — agrees.

use std::collections::{BTreeSet, HashMap};

use molrs::core::NodeId;
use molrs::core::constants::parmchk2_radians;

use crate::ff::ir::Params;
use crate::ff::params::{EmpiricalTable, ParmTable, ParmchkTable, ParmchkWeights};
use crate::ff::typifier::estimate::empirical::angle_k;
use crate::ff::typifier::{EstimateMethod, Provenance};

/// A bond or angle parmchk2 estimated: its parameters (`k`, `r0` / `theta0`),
/// and how it reached them.
#[derive(Debug, Clone)]
pub(super) struct AnalogEstimate {
    pub params: Params,
    pub provenance: Provenance,
}

/// A bond row: its two types, `K` and `r₀`.
#[derive(Debug, Clone, Copy)]
struct BondRow {
    names: [&'static str; 2],
    k: f64,
    r0: f64,
}

/// An angle row: its three types (vertex in the middle), `K` and θ₀ (deg).
#[derive(Debug, Clone, Copy)]
struct AngleRow {
    names: [&'static str; 3],
    k: f64,
    theta0: f64,
}

/// One `PARMCHK.DAT` candidate: the type, its eight scored columns in
/// parmchk2's order (`bl blf cba cbaf ba baf ctor tor`), blanks defaulted, and
/// its kind (0 the type itself, 1 `EQUA`, 2 `CORR`).
#[derive(Debug, Clone, Copy)]
struct Corr {
    to: &'static str,
    columns: [f64; 8],
    kind: u8,
}

const BL: usize = 0;
const BLF: usize = 1;
const CBA: usize = 2;
const CBAF: usize = 3;
const BA: usize = 4;
const BAF: usize = 5;

/// parmchk2's bond and angle estimates for one molecule, by name.
pub(super) struct Analogs {
    bonds: HashMap<[&'static str; 2], AnalogEstimate>,
    angles: HashMap<[&'static str; 3], AnalogEstimate>,
}

impl Analogs {
    /// Run `chk_bond` then `chk_angle` over the molecule (`order`, each atom's
    /// `neighbours`, `type_of`).
    pub(super) fn new(
        order: &[NodeId],
        neighbours: &[Vec<NodeId>],
        type_of: &HashMap<NodeId, &'static str>,
        table: ParmTable,
        parmchk: &ParmchkTable,
        empirical: EmpiricalTable,
    ) -> Self {
        let index: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let bonded: Vec<BTreeSet<usize>> = neighbours
            .iter()
            .map(|around| around.iter().map(|a| index[a]).collect())
            .collect();
        let types: Vec<&'static str> = order.iter().map(|a| type_of[a]).collect();
        let mut search = Search {
            parmchk,
            empirical,
            bonds: table
                .bonds
                .iter()
                .map(|r| BondRow {
                    names: [table.name_of(r.i), table.name_of(r.j)],
                    k: r.force_constant,
                    r0: r.length,
                })
                .collect(),
            angles: table
                .angles
                .iter()
                .map(|r| AngleRow {
                    names: [table.name_of(r.i), table.name_of(r.j), table.name_of(r.k)],
                    k: r.force_constant,
                    theta0: r.angle_deg,
                })
                .collect(),
            table_bonds: table.bonds.len(),
            table_angles: table.angles.len(),
        };
        let mut out = Self {
            bonds: HashMap::new(),
            angles: HashMap::new(),
        };
        let n = order.len();
        for i in 0..n {
            for &j in bonded[i].iter().filter(|&&j| j > i) {
                let mut name = [types[i], types[j]];
                name.sort_unstable();
                if search.find_bond(name, search.bonds.len()).is_some() {
                    continue;
                }
                let found = search.bond([types[i], types[j]]);
                let (k, r0) = found
                    .as_ref()
                    .map_or((0.0, 0.0), |(row, _)| (row.k, row.r0));
                search.bonds.push(BondRow { names: name, k, r0 });
                if let Some((row, provenance)) = found {
                    out.bonds.insert(
                        name,
                        AnalogEstimate {
                            params: Params::from_iter([("k", row.k), ("r0", row.r0)]),
                            provenance,
                        },
                    );
                }
            }
        }
        // parmchk2's loop order: i, then j, then k, every index ascending.
        for i in 0..n {
            for &j in &bonded[i] {
                for &k in bonded[j].iter().filter(|&&k| k != i) {
                    let [a, c] = if types[i] <= types[k] {
                        [types[i], types[k]]
                    } else {
                        [types[k], types[i]]
                    };
                    let name = [a, types[j], c];
                    if search.find_angle(name, search.angles.len()).is_some() {
                        continue;
                    }
                    let found = search.angle([types[i], types[j], types[k]], name);
                    let (kk, theta0) = found
                        .as_ref()
                        .map_or((0.0, 0.0), |(row, _)| (row.k, row.theta0));
                    search.angles.push(AngleRow {
                        names: name,
                        k: kk,
                        theta0,
                    });
                    if let Some((row, provenance)) = found {
                        out.angles.insert(
                            name,
                            AnalogEstimate {
                                params: Params::from_iter([("k", row.k), ("theta0", row.theta0)]),
                                provenance,
                            },
                        );
                    }
                }
            }
        }
        out
    }

    /// parmchk2's estimate for the bond of types `pair` (either order).
    pub(super) fn bond(&self, pair: [&'static str; 2]) -> Option<&AnalogEstimate> {
        let mut name = pair;
        name.sort_unstable();
        self.bonds.get(&name)
    }

    /// parmchk2's estimate for the angle of types `triple` (either way).
    pub(super) fn angle(&self, triple: [&'static str; 3]) -> Option<&AnalogEstimate> {
        let [a, b, c] = triple;
        self.angles.get(&if a <= c { [a, b, c] } else { [c, b, a] })
    }
}

/// The rows a molecule's bonds and angles are searched in.
struct Search<'a> {
    parmchk: &'a ParmchkTable,
    empirical: EmpiricalTable,
    /// The table's rows, then every row estimated since.
    bonds: Vec<BondRow>,
    angles: Vec<AngleRow>,
    /// How many of `bonds` / `angles` are the table's.
    table_bonds: usize,
    table_angles: usize,
}

impl Search<'_> {
    fn weights(&self) -> &ParmchkWeights {
        &self.parmchk.weights
    }

    /// The first of the first `n` bond rows named `a-b` either way.
    fn find_bond(&self, [a, b]: [&str; 2], n: usize) -> Option<usize> {
        self.bonds[..n]
            .iter()
            .position(|r| r.names == [a, b] || r.names == [b, a])
    }

    /// The first of the first `n` angle rows named `a-b-c` either way.
    fn find_angle(&self, [a, b, c]: [&str; 3], n: usize) -> Option<usize> {
        self.angles[..n]
            .iter()
            .position(|r| r.names == [a, b, c] || r.names == [c, b, a])
    }

    fn equivalents(&self, ty: &'static str) -> Vec<&'static str> {
        let mut out = vec![ty];
        if let Some(block) = self.parmchk.get(ty) {
            out.extend(block.equivalent.iter().copied());
        }
        out
    }

    fn correspondents(&self, ty: &'static str) -> Vec<Corr> {
        let w = self.weights();
        let defaults = [
            w.default_bond_length,
            w.default_bond_force,
            w.default_angle_centre,
            w.default_angle_centre_force,
            w.default_angle,
            w.default_angle_force,
            w.default_torsion_centre,
            w.default_torsion,
        ];
        let mut out = vec![Corr {
            to: ty,
            columns: [0.0; 8],
            kind: 0,
        }];
        if let Some(block) = self.parmchk.get(ty) {
            out.extend(block.equivalent.iter().map(|&e| Corr {
                to: e,
                columns: [0.0; 8],
                kind: 1,
            }));
            for c in block.corresponding {
                let p = c.penalties;
                // A CORR line with no columns reads as zeros; a written -1 is
                // the column's default.
                let columns = if p[8] < 0.0 {
                    [0.0; 8]
                } else {
                    std::array::from_fn(|i| if p[i] < 0.0 { defaults[i] } else { p[i] })
                };
                out.push(Corr {
                    to: c.to,
                    columns,
                    kind: 2,
                });
            }
        }
        out
    }

    fn group(&self, ty: &str) -> Option<i32> {
        self.parmchk.get(ty).map(|b| b.group)
    }

    fn atomic_number(&self, ty: &str) -> Option<u8> {
        self.parmchk.get(ty).map(|b| b.atomic_number)
    }

    fn equtype(&self, ty: &str) -> i32 {
        self.parmchk.get(ty).map_or(0, |b| b.equivalent_flag)
    }

    /// `equtype_penalty` (see the torsion search).
    fn equtype_penalty(&self, a: &str, b: &str, ca: &str, cb: &str) -> f64 {
        let w = self.weights().weight_equivalent;
        let (n1, n2, n3, n4) = (
            self.equtype(a),
            self.equtype(b),
            self.equtype(ca),
            self.equtype(cb),
        );
        if n1 == 0 && n2 == 0 {
            return 0.0;
        }
        let (t1, t2) = (n1.abs() + n2.abs(), n3.abs() + n4.abs());
        if (t2 == 3 && t1 != 3) || (t1 == 3 && t2 != 3) {
            return w;
        }
        if n1 + n2 == 0 && n3 < 0 && n4 < 0 {
            return 0.5 * w;
        }
        0.0
    }

    /// `chk_bond` steps 2–3 for the bond of `types` (molecule order).
    fn bond(&self, [ti, tj]: [&'static str; 2]) -> Option<(BondRow, Provenance)> {
        let w = self.weights();
        let n = self.table_bonds;
        for (m, &e1) in self.equivalents(ti).iter().enumerate() {
            for (q, &e2) in self.equivalents(tj).iter().enumerate() {
                if m == 0 && q == 0 {
                    continue;
                }
                if let Some(at) = self.find_bond([e1, e2], n) {
                    let row = self.bonds[at];
                    return Some((row, Provenance::analogy(0.0, row.names.join("-"))));
                }
            }
        }
        let mut best: Option<(usize, f64)> = None;
        let (ci, cj) = (self.correspondents(ti), self.correspondents(tj));
        for c1 in &ci {
            let s1 = c1.columns[BL] * w.weight_bond_length + c1.columns[BLF] * w.weight_bond_force;
            for c2 in &cj {
                if c1.kind <= 1 && c2.kind <= 1 {
                    continue;
                }
                // parmchk2's own order of additions: `score1 + score2`, then the
                // surcharges (it decides which of two equal analogs wins).
                let s2 =
                    c2.columns[BL] * w.weight_bond_length + c2.columns[BLF] * w.weight_bond_force;
                let mut score = s1 + s2;
                if self.group(c1.to) != self.group(c2.to) {
                    score += w.weight_group;
                }
                score += self.equtype_penalty(ti, tj, c1.to, c2.to);
                if best.is_some_and(|(_, b)| score >= b) {
                    continue;
                }
                if let Some(at) = self.find_bond([c1.to, c2.to], n) {
                    best = Some((at, score));
                }
            }
        }
        best.map(|(at, score)| {
            let row = self.bonds[at];
            (row, Provenance::analogy(score, row.names.join("-")))
        })
    }

    /// The score of a `CORR` triple substituted into an angle.
    fn angle_score(&self, c1: &Corr, c2: &Corr, c3: &Corr) -> f64 {
        let w = self.weights();
        let end = |c: &Corr| c.columns[BA] * w.weight_angle + c.columns[BAF] * w.weight_angle_force;
        let vertex = (c2.columns[CBA] * w.weight_angle + c2.columns[CBAF] * w.weight_angle_force)
            * w.weight_angle_centre;
        let mut score = end(c1) + vertex + end(c3) + w.weight_group;
        let g = self.group(c1.to);
        if self.group(c2.to) == g && self.group(c3.to) == g {
            score -= w.weight_group;
        }
        score
    }

    /// `empangle`: the empirical angle of types `t1-t2-t3` with the elements
    /// `z1`, `z2`, `z3`, from the `t1-t2-t1` and `t3-t2-t3` rows and the
    /// `t1-t2`, `t2-t3` bond rows known so far, or `None` where one is missing.
    fn empangle(&self, [t1, t2, t3]: [&str; 3], z: [Option<u8>; 3]) -> Option<(f64, f64)> {
        let all = self.angles.len();
        let a1 = self.find_angle([t1, t2, t1], all)?;
        let a2 = self.find_angle([t3, t2, t3], all)?;
        let theta0 = 0.5 * (self.angles[a1].theta0 + self.angles[a2].theta0);
        if theta0 <= 0.0 {
            return None;
        }
        let length = |a: &str, b: &str| {
            self.find_bond([a, b], self.bonds.len())
                .map_or(0.0, |at| self.bonds[at].r0)
        };
        let (b1, b2) = (length(t1, t2), length(t2, t3));
        if b1 == 0.0 || b2 == 0.0 {
            return None;
        }
        // A factor upstream has no row for reads as 0, as parmchk2's table does.
        let factor = |z: Option<u8>, centre: bool| {
            z.and_then(|z| self.empirical.angle(z))
                .map_or(0.0, |row| if centre { row.c } else { row.z_factor })
        };
        let k = angle_k(
            factor(z[0], false),
            factor(z[1], true),
            factor(z[2], false),
            b1,
            b2,
            parmchk2_radians(theta0),
        );
        Some((k, theta0))
    }

    /// `chk_angle` steps 2–6 for the angle of `types` (molecule order),
    /// named `name`.
    fn angle(
        &self,
        [ti, tj, tk]: [&'static str; 3],
        name: [&'static str; 3],
    ) -> Option<(AngleRow, Provenance)> {
        let w = self.weights();
        let n = self.table_angles;
        let (ei, ej, ek) = (
            self.equivalents(ti),
            self.equivalents(tj),
            self.equivalents(tk),
        );
        // A row over equivalent types; the first found wins.
        for (m, &e4) in ej.iter().enumerate() {
            for (q, &e5) in ei.iter().enumerate() {
                for (o, &e6) in ek.iter().enumerate() {
                    if m == 0 && q == 0 && o == 0 {
                        continue;
                    }
                    if let Some(at) = self.find_angle([e5, e4, e6], n) {
                        let row = self.angles[at];
                        return Some((row, Provenance::analogy(0.0, row.names.join("-"))));
                    }
                }
            }
        }
        let (ci, cj, ck) = (
            self.correspondents(ti),
            self.correspondents(tj),
            self.correspondents(tk),
        );
        // The cheapest row over corresponding types, within THRESHOLD_BA.
        let mut best: Option<(usize, f64)> = None;
        for c1 in &ci {
            for c2 in &cj {
                for c3 in &ck {
                    if c1.kind <= 1 && c2.kind <= 1 && c3.kind <= 1 {
                        continue;
                    }
                    let score = self.angle_score(c1, c2, c3);
                    if best.is_some_and(|(_, b)| score >= b) || score > w.threshold_angle {
                        continue;
                    }
                    if let Some(at) = self.find_angle([c1.to, c2.to, c3.to], n) {
                        best = Some((at, score));
                    }
                }
            }
        }
        if let Some((at, score)) = best {
            let row = self.angles[at];
            return Some((row, Provenance::analogy(score, row.names.join("-"))));
        }
        let empirical = |k: f64, theta0: f64, penalty: f64, analog: [&str; 3]| {
            (
                AngleRow {
                    names: name,
                    k,
                    theta0,
                },
                Provenance {
                    penalty,
                    method: EstimateMethod::Empirical,
                    analog: analog.join("-"),
                },
            )
        };
        // The empirical angle of its own types.
        let z = [ti, tj, tk].map(|t| self.atomic_number(t));
        if let Some((k, theta0)) = self.empangle(name, z) {
            return Some(empirical(k, theta0, 0.0, name));
        }
        // The empirical angle of equivalent types; the first found wins.
        for (m, &e4) in ej.iter().enumerate() {
            for (q, &e5) in ei.iter().enumerate() {
                for (o, &e6) in ek.iter().enumerate() {
                    if m == 0 && q == 0 && o == 0 {
                        continue;
                    }
                    let types = [e5, e4, e6];
                    let z = types.map(|t| self.atomic_number(t));
                    if let Some((k, theta0)) = self.empangle(types, z) {
                        return Some(empirical(k, theta0, 0.0, types));
                    }
                }
            }
        }
        // The cheapest empirical angle of corresponding types.
        let mut best: Option<(f64, f64, f64, [&str; 3])> = None;
        for c1 in &ci {
            for c2 in &cj {
                for c3 in &ck {
                    if c1.kind <= 1 && c2.kind <= 1 && c3.kind <= 1 {
                        continue;
                    }
                    let score = self.angle_score(c1, c2, c3);
                    if best.is_some_and(|(_, _, b, _)| score >= b) {
                        continue;
                    }
                    let types = [c1.to, c2.to, c3.to];
                    let z = types.map(|t| self.atomic_number(t));
                    if let Some((k, theta0)) = self.empangle(types, z) {
                        best = Some((k, theta0, score, types));
                    }
                }
            }
        }
        best.map(|(k, theta0, score, types)| empirical(k, theta0, score, types))
    }
}
