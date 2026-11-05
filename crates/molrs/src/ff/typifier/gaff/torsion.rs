//! GAFF torsions the table lacks, estimated exactly as parmchk2 estimates them.
//!
//! parmchk2's `chk_torsion` visits every torsion `i-j-k-l` of the molecule
//! with `j < k` (atom order), names it canonically (the inner pair in
//! alphabetical order), and for a name no row holds tries, in order: a
//! specific row over equivalent types (`EQUA`), the wildcard row `X-j-k-X`
//! (a parameter, not an estimate: nothing is written), a wildcard row over
//! equivalent inner types, the cheapest specific row over corresponding types
//! (`CORR`, scored), and the cheapest wildcard row over corresponding inner
//! types. A name estimated once is reused for the rest of the molecule. The
//! search runs over the atoms in molecule order, so — as for impropers — one
//! name can be estimated differently in two molecules.
//!
//! Reproduced as written (AmberTools `parmchk2.c` `chk_torsion`, `torsion`,
//! `equtype_penalty`, `read_parmchk_parm`): the inner atoms' `CORR` score is
//! `(ctor or DEFAULT_TOR_CTR)·FRACT1 + similarity·FRACT2`, weighted by
//! `WEIGHT_TOR_CTR`; an outer atom's is `tor`, or `DEFAULT_TOR` where blank;
//! a `CORR` line with no columns scores 0 everywhere. Checked against
//! parmchk2 (AmberTools 26.1) on 127 molecules under GAFF and GAFF2: every
//! frcmod torsion row (terms, analog, penalty) agrees.

use std::collections::{BTreeSet, HashMap};

use molrs::core::NodeId;

use crate::ff::params::{ParmDihedralRow, ParmTable, ParmchkTable};
use crate::ff::typifier::Provenance;

/// The wildcard atom type.
const X: &str = "X";

/// One torsion row with its four slot names (`X` a wildcard).
#[derive(Debug, Clone)]
struct Row {
    names: [&'static str; 4],
    /// The row's index in the table, whose `more_terms` chain is the torsion.
    first: usize,
}

/// A torsion parmchk2 estimated: the table rows it copied (one per cosine
/// term) and how it reached them.
#[derive(Debug, Clone)]
pub(super) struct TorsionEstimate {
    pub rows: Vec<&'static ParmDihedralRow>,
    pub provenance: Provenance,
}

/// A `PARMCHK.DAT` substitution candidate: the type, its outer (`tor`) and
/// inner (`ctor`) torsion scores, and its kind (0 itself, 1 `EQUA`, 2 `CORR`).
type Corr = (&'static str, f64, f64, u8);

/// The canonical name of a torsion: inner pair in order; on a tie, the ends.
pub(super) fn canonical(types: [&str; 4]) -> [&str; 4] {
    let [mut a, mut b, mut c, mut d] = types;
    if b > c {
        (a, b, c, d) = (d, c, b, a);
    } else if b == c && a > d {
        std::mem::swap(&mut a, &mut d);
    }
    [a, b, c, d]
}

/// Torsion rows, indexed by name: the first row each name, read either way,
/// belongs to.
#[derive(Debug, Clone, Default)]
struct Rows {
    rows: Vec<Row>,
    first: HashMap<[&'static str; 4], usize>,
}

impl Rows {
    /// The table's rows, in table order.
    fn of(table: ParmTable) -> Self {
        let mut rows = Self::default();
        for (first, row) in table.dihedrals.iter().enumerate() {
            rows.push(Row {
                names: [row.i, row.j, row.k, row.l]
                    .map(|slot| slot.map_or(X, |ty| table.name_of(ty))),
                first,
            });
        }
        rows
    }

    fn push(&mut self, row: Row) {
        let at = self.rows.len();
        let [a, b, c, d] = row.names;
        self.first.entry([a, b, c, d]).or_insert(at);
        self.first.entry([d, c, b, a]).or_insert(at);
        self.rows.push(row);
    }

    /// The table index of the first row named `names` either way.
    fn find(&self, names: [&'static str; 4]) -> Option<usize> {
        self.first.get(&names).map(|&at| self.rows[at].first)
    }
}

/// The cosine terms of the torsion whose first row is `first`.
fn chain(table: ParmTable, first: usize) -> Vec<&'static ParmDihedralRow> {
    let mut out = vec![&table.dihedrals[first]];
    let mut at = first;
    while table.dihedrals[at].more_terms {
        at += 1;
        out.push(&table.dihedrals[at]);
    }
    out
}

/// One molecule's torsions as parmchk2 and tleap resolve what the table's
/// specific rows miss: parmchk2's estimates, by canonical name, and the
/// table's wildcard rows.
pub(super) struct Torsions {
    table: ParmTable,
    rows: Rows,
    estimates: HashMap<[&'static str; 4], TorsionEstimate>,
}

impl Torsions {
    /// Run parmchk2's torsion search over the molecule (`order`, each atom's
    /// `neighbours`, `type_of`).
    pub(super) fn new(
        order: &[NodeId],
        neighbours: &[Vec<NodeId>],
        type_of: &HashMap<NodeId, &'static str>,
        table: ParmTable,
        parmchk: &ParmchkTable,
    ) -> Self {
        let index: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, &a)| (a, i)).collect();
        let bonded: Vec<BTreeSet<usize>> = neighbours
            .iter()
            .map(|around| around.iter().map(|a| index[a]).collect())
            .collect();
        let types: Vec<&'static str> = order.iter().map(|a| type_of[a]).collect();
        let search = Search {
            parmchk,
            rows: Rows::of(table),
        };
        let mut known = search.rows.clone();
        let mut estimates = HashMap::new();
        let n = order.len();
        // parmchk2's loop order: i, then j < k, then l, every index ascending.
        for i in 0..n {
            for &j in &bonded[i] {
                for &k in bonded[j].iter().filter(|&&k| k > j && k != i) {
                    for &l in bonded[k].iter().filter(|&&l| l != j) {
                        let quartet = [types[i], types[j], types[k], types[l]];
                        let name = canonical(quartet);
                        if known.find(name).is_some() {
                            continue;
                        }
                        // `None`: the wildcard row covers it, or nothing does
                        // (parmchk2 writes a zero barrier marked "ATTN, need
                        // revision"; the typifier reports the term missing).
                        // Either way the name is settled for this molecule.
                        let found = search.run(quartet, name);
                        known.push(Row {
                            names: name,
                            first: found.as_ref().map_or(usize::MAX, |(first, _)| *first),
                        });
                        if let Some((first, provenance)) = found {
                            estimates.insert(
                                name,
                                TorsionEstimate {
                                    rows: chain(table, first),
                                    provenance,
                                },
                            );
                        }
                    }
                }
            }
        }
        Self {
            table,
            rows: search.rows,
            estimates,
        }
    }

    /// parmchk2's estimate for the torsion canonically named `name`.
    pub(super) fn estimate(&self, name: &[&'static str; 4]) -> Option<&TorsionEstimate> {
        self.estimates.get(name)
    }

    /// The wildcard row `X-b-c-X` (either way), as its rows.
    pub(super) fn general(
        &self,
        b: &'static str,
        c: &'static str,
    ) -> Option<Vec<&'static ParmDihedralRow>> {
        self.rows
            .find([X, b, c, X])
            .map(|first| chain(self.table, first))
    }
}

/// The table and substitution data one molecule's torsions are searched in.
struct Search<'a> {
    parmchk: &'a ParmchkTable,
    /// The table's rows (not the ones estimated since).
    rows: Rows,
}

impl Search<'_> {
    fn equivalents(&self, ty: &'static str) -> Vec<&'static str> {
        let mut out = vec![ty];
        if let Some(block) = self.parmchk.get(ty) {
            out.extend(block.equivalent.iter().copied());
        }
        out
    }

    fn correspondents(&self, ty: &'static str) -> Vec<Corr> {
        let w = &self.parmchk.weights;
        let mut out: Vec<Corr> = vec![(ty, 0.0, 0.0, 0)];
        if let Some(block) = self.parmchk.get(ty) {
            out.extend(block.equivalent.iter().map(|&e| (e, 0.0, 0.0, 1)));
            for c in block.corresponding {
                let p = c.penalties;
                // A CORR line with no columns reads as zeros.
                let (tor, ctor) = if p[8] < 0.0 {
                    (0.0, 0.0)
                } else {
                    let tor = if p[7] < 0.0 { w.default_torsion } else { p[7] };
                    let ctor = if p[6] < 0.0 {
                        w.default_torsion_centre
                    } else {
                        p[6]
                    };
                    (
                        tor,
                        ctor * w.default_fraction_1 + p[8] * w.default_fraction_2,
                    )
                };
                out.push((c.to, tor, ctor, 2));
            }
        }
        out
    }

    fn group(&self, ty: &str) -> Option<i32> {
        self.parmchk.get(ty).map(|b| b.group)
    }

    fn equtype(&self, ty: &str) -> i32 {
        self.parmchk.get(ty).map_or(0, |b| b.equivalent_flag)
    }

    /// `equtype_penalty`: replacing a conjugated pair by an unpaired one (or
    /// the reverse), or `cc/cd`-phase by `cd/cd`-phase.
    fn equtype_penalty(&self, j: &str, k: &str, cj: &str, ck: &str) -> f64 {
        let w = self.parmchk.weights.weight_equivalent;
        let (n1, n2, n3, n4) = (
            self.equtype(j),
            self.equtype(k),
            self.equtype(cj),
            self.equtype(ck),
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

    fn label(&self, first: usize) -> String {
        self.rows.rows[first].names.join("-")
    }

    /// Steps 2–6 of `chk_torsion` for the torsion `quartet` (molecule order),
    /// canonically `names`: the first row of the torsion it copies, and how.
    /// `None` when step 3 covers it (no estimate) or nothing does.
    fn run(
        &self,
        quartet: [&'static str; 4],
        names: [&'static str; 4],
    ) -> Option<(usize, Provenance)> {
        let w = &self.parmchk.weights;
        let [ti, tj, tk, tl] = quartet;
        let rows = &self.rows;

        // Step 2: a specific row over equivalent types; the first found wins.
        let (ej, ek, ei, el) = (
            self.equivalents(tj),
            self.equivalents(tk),
            self.equivalents(ti),
            self.equivalents(tl),
        );
        for (m, &e5) in ej.iter().enumerate() {
            for (n, &e6) in ek.iter().enumerate() {
                for (p, &e7) in ei.iter().enumerate() {
                    for (q, &e8) in el.iter().enumerate() {
                        if m == 0 && n == 0 && p == 0 && q == 0 {
                            continue;
                        }
                        if let Some(first) = rows.find([e7, e5, e6, e8]) {
                            return Some((first, Provenance::analogy(0.0, self.label(first))));
                        }
                    }
                }
            }
        }

        // Step 3: the wildcard row of the canonical inner pair covers it.
        if rows.find([X, names[1], names[2], X]).is_some() {
            return None;
        }

        // Step 4: a wildcard row over equivalent inner types; the last found wins.
        let mut found = None;
        for (n, &e6) in ej.iter().enumerate() {
            for (p, &e7) in ek.iter().enumerate() {
                if n == 0 && p == 0 {
                    continue;
                }
                if let Some(first) = rows.find([X, e6, e7, X]) {
                    found = Some(first);
                }
            }
        }
        if let Some(first) = found {
            return Some((first, Provenance::analogy(0.0, self.label(first))));
        }

        // Step 5: the cheapest specific row over corresponding types.
        let (ci, cj, ck, cl) = (
            self.correspondents(ti),
            self.correspondents(tj),
            self.correspondents(tk),
            self.correspondents(tl),
        );
        let mut best: Option<(usize, f64)> = None;
        for c1 in &ci {
            for c2 in &cj {
                for c3 in &ck {
                    for c4 in &cl {
                        if c1.3 <= 1 && c2.3 <= 1 && c3.3 <= 1 && c4.3 <= 1 {
                            continue;
                        }
                        let mut score = c1.1
                            + c2.2 * w.weight_torsion_centre
                            + c3.2 * w.weight_torsion_centre
                            + c4.1
                            + w.weight_group
                            + self.equtype_penalty(tj, tk, c2.0, c3.0);
                        let g = self.group(c1.0);
                        if g.is_some()
                            && self.group(c2.0) == g
                            && self.group(c3.0) == g
                            && self.group(c4.0) == g
                        {
                            score -= w.weight_group;
                        }
                        if best.is_some_and(|(_, b)| score >= b) {
                            continue;
                        }
                        if let Some(first) = rows.find([c1.0, c2.0, c3.0, c4.0]) {
                            best = Some((first, score));
                        }
                    }
                }
            }
        }
        if let Some((first, score)) = best {
            return Some((first, Provenance::analogy(score, self.label(first))));
        }

        // Step 6: the cheapest wildcard row over corresponding inner types.
        let mut best: Option<(usize, f64)> = None;
        for c2 in &cj {
            for c3 in &ck {
                if c2.3 <= 1 && c3.3 <= 1 {
                    continue;
                }
                let mut score = c2.2 * w.weight_torsion_centre
                    + c3.2 * w.weight_torsion_centre
                    + self.equtype_penalty(tj, tk, c2.0, c3.0);
                if self.group(c2.0) != self.group(c3.0) {
                    score += w.weight_group;
                }
                if best.is_some_and(|(_, b)| score >= b) {
                    continue;
                }
                if let Some(first) = rows.find([X, c2.0, c3.0, X]) {
                    best = Some((first, score));
                }
            }
        }
        best.map(|(first, score)| (first, Provenance::analogy(score, self.label(first))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_torsion_is_named_inner_pair_first_in_order() {
        assert_eq!(
            canonical(["hc", "c3", "ca", "ca"]),
            ["hc", "c3", "ca", "ca"]
        );
        assert_eq!(
            canonical(["ca", "ca", "c3", "hc"]),
            ["hc", "c3", "ca", "ca"]
        );
        assert_eq!(
            canonical(["os", "c3", "c3", "hc"]),
            ["hc", "c3", "c3", "os"]
        );
    }
}
