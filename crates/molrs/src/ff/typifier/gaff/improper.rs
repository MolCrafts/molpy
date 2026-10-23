//! GAFF impropers exactly as AmberTools builds them: `parmchk2` estimates, then
//! tleap matches and orders.
//!
//! An AMBER improper is decided in two programs, and both decisions shape the
//! energy:
//!
//! 1. **parmchk2** (`chk_improper`) visits every atom whose type `PARMCHK.DAT`
//!    flags as an improper centre, in atom order, with its first three bonded
//!    neighbours in bond order, and writes a frcmod row for every type quartet
//!    the table lacks: an equivalent-type row, a wildcard row, an
//!    equivalent-type wildcard row, a corresponding-type row, a
//!    corresponding-type wildcard row, or the 1.1 kcal/mol default — the first
//!    of those six steps that answers. The search runs on the peripherals **in
//!    bond order**, not sorted, so one quartet can be estimated differently in
//!    two molecules (aspirin's ester carbon reaches `c3-o -c -oh`, ethyl
//!    acetate's the `X -X -c -o` amide term); a quartet estimated once in a
//!    molecule is reused for the rest of it, and is itself a row the later
//!    searches can copy (indole's `ca-cc-cd-ha` copies the `ca-ca-ca-ha` its
//!    benzene ring was given just before).
//! 2. **tleap** visits every atom with three or more neighbours and every
//!    triple of them, looks the triple up in the parameter sets — the unit's own
//!    (the terms it has already used), the frcmod, then `gaff.dat` /
//!    `gaff2.dat` — keeping the most specific row, and adds an improper only
//!    where a row exists. It orders the four atoms by that row
//!    (`ParmSetImproperOrderAtoms`): the row's own slot order for a row read
//!    from the table, sorted order for one already in the unit's set, then
//!    wildcard slots and same-type slots by atom index. The centre is always
//!    third.
//!
//! Both programs are reproduced as written (AmberTools `parmchk2.c`
//! `chk_improper`, LEaP `unitio.c` / `parmSet.c`), including two quirks of
//! parmchk2's: an `X` in a peripheral slot costs 3, not `PARMCHK.DAT`'s
//! `WEIGHT_X 10` (the `WEIGHT_X` prefix also matches the `WEIGHT_X3` line, which
//! is read last), and a `CORR` line without a ninth column scores 0 as an
//! improper analog. Checked against antechamber + parmchk2 + tleap (AmberTools
//! 26.1) on 127 molecules under GAFF and GAFF2: every frcmod improper row
//! (barrier, analog, penalty) and every prmtop improper (atoms, order,
//! barrier) agree.

use std::collections::HashMap;

use crate::core::NeighborKinds;
use molrs::core::Atomistic;
use molrs::core::NodeId;

use crate::ff::ir::Params;
use crate::ff::params::{ParmTable, ParmchkTable, ParmchkType};
use crate::ff::typifier::Provenance;

/// parmchk2's improper default: `1.1 kcal/mol`, phase 180°, periodicity 2.
///
/// Not a row of any table — it is the value parmchk2 falls back on when nothing
/// matches, and it prints no penalty for it ("a default is not a substitution, so
/// there is nothing to score").
const DEFAULT_IMPROPER: (f64, f64, u32) = (1.1, 180.0, 2);

/// The wildcard atom type.
const X: &str = "X";

/// What parmchk2 charges an `X` in a peripheral slot (see the module doc).
const WEIGHT_X: f64 = 3.0;

/// One improper of the molecule, as tleap builds it.
#[derive(Debug, Clone)]
pub(super) struct ImproperTerm {
    /// The four atoms in AMBER order, centre third.
    pub atoms: [NodeId; 4],
    /// Their atom types, in the same order.
    pub types: [String; 4],
    /// `k`, `periodicity`, `phase` (degrees).
    pub params: Params,
    /// How parmchk2 estimated it, when it did; `None` for a table row.
    pub estimate: Option<Provenance>,
}

/// One improper parameter row: four type slots (`X` a wildcard), the cosine
/// term, and the estimate that produced it (frcmod rows only).
#[derive(Debug, Clone)]
struct Row {
    names: [String; 4],
    force: f64,
    phase: f64,
    periodicity: f64,
    estimate: Option<Provenance>,
}

impl Row {
    fn wildcards(&self) -> usize {
        self.names.iter().filter(|n| *n == X).count()
    }

    fn label(&self) -> String {
        self.names.join("-")
    }
}

/// The impropers of `graph` under `table`, as antechamber's parmchk2 and tleap
/// would build them. `type_of` gives every atom's GAFF type.
pub(super) fn impropers(
    graph: &Atomistic,
    table: ParmTable,
    parmchk: &ParmchkTable,
    type_of: &HashMap<NodeId, &'static str>,
) -> Vec<ImproperTerm> {
    let order: Vec<NodeId> = graph.atoms().map(|(id, _)| id).collect();
    let index: HashMap<NodeId, usize> = order.iter().enumerate().map(|(i, &a)| (a, i)).collect();
    let neighbours: Vec<Vec<NodeId>> = order
        .iter()
        .map(|&a| {
            graph
                .neighbors(a, NeighborKinds::BONDS)
                .map(|(n, _)| n)
                .collect()
        })
        .collect();
    let written: Vec<Row> = table
        .impropers
        .iter()
        .map(|row| Row {
            names: [row.i, row.j, row.k, row.l]
                .map(|slot| slot.map_or(X, |ty| table.name_of(ty)).to_owned()),
            force: row.barrier,
            phase: row.phase_deg,
            periodicity: f64::from(row.periodicity),
            estimate: None,
        })
        .collect();

    let frcmod = Parmchk2 {
        parmchk,
        rows: written.iter().map(normalised).collect(),
    }
    .estimate(&order, &neighbours, type_of);
    tleap(&order, &neighbours, &index, type_of, &frcmod, &written)
}

// ---------------------------------------------------------------------------
// parmchk2
// ---------------------------------------------------------------------------

/// A table row as parmchk2 reads it: a wildcard-free row's peripherals sorted,
/// a one-wildcard row's two concrete peripherals sorted in place (`X` stays
/// where it is), a two-wildcard row as written.
fn normalised(row: &Row) -> Row {
    let [mut a, mut b, c, mut d] = row.names.clone();
    let wild = row.wildcards();
    if wild == 0 {
        if a > b {
            std::mem::swap(&mut a, &mut b);
        }
        if a > d {
            std::mem::swap(&mut a, &mut d);
        }
        if b > d {
            std::mem::swap(&mut b, &mut d);
        }
    }
    if wild == 1 || (wild == 2 && c == X) {
        if row.names[0] == X && b > d {
            std::mem::swap(&mut b, &mut d);
        }
        if row.names[1] == X && a > d {
            std::mem::swap(&mut a, &mut d);
        }
        if row.names[3] == X && a > b {
            std::mem::swap(&mut a, &mut b);
        }
    }
    Row {
        names: [a, b, c, d],
        ..row.clone()
    }
}

/// One substitution candidate of a `PARMCHK.DAT` block: the type, its
/// improper score, and its kind (0 the type itself, 1 `EQUA`, 2 `CORR`).
type Corr = (&'static str, f64, u8);

/// parmchk2's improper estimator over one table.
struct Parmchk2<'a> {
    parmchk: &'a ParmchkTable,
    /// The table's rows, normalised, then every row estimated so far.
    rows: Vec<Row>,
}

impl Parmchk2<'_> {
    fn block(&self, ty: &str) -> Option<&'static ParmchkType> {
        self.parmchk.get(ty)
    }

    /// The type itself, then its `EQUA` types.
    fn equivalents(&self, ty: &'static str) -> Vec<&'static str> {
        let mut out = vec![ty];
        if let Some(block) = self.block(ty) {
            out.extend(block.equivalent.iter().copied());
        }
        out
    }

    /// The type itself, its `EQUA` types, then its `CORR` types with their
    /// improper scores (the ninth column; absent scores 0).
    fn correspondents(&self, ty: &'static str) -> Vec<Corr> {
        let mut out: Vec<Corr> = vec![(ty, 0.0, 0)];
        if let Some(block) = self.block(ty) {
            out.extend(block.equivalent.iter().map(|&e| (e, 0.0, 1)));
            out.extend(
                block
                    .corresponding
                    .iter()
                    .map(|c| (c.to, c.penalties[8].max(0.0), 2)),
            );
        }
        out
    }

    fn group(&self, ty: &str) -> Option<i32> {
        self.block(ty).map(|b| b.group)
    }

    fn same_group(&self, types: [&str; 4]) -> bool {
        let g = self.group(types[0]);
        g.is_some() && types[1..].iter().all(|&t| self.group(t) == g)
    }

    /// The penalty of the `X` slots of `row`.
    fn wildcard_score(&self, row: &Row) -> f64 {
        self.add_wildcards(0.0, row)
    }

    /// `score` plus the penalty of each `X` slot of `row`, added one by one.
    fn add_wildcards(&self, score: f64, row: &Row) -> f64 {
        let w = &self.parmchk.weights;
        row.names
            .iter()
            .enumerate()
            .filter(|(_, n)| *n == X)
            .fold(score, |score, (slot, _)| {
                score
                    + if slot == 2 {
                        w.weight_wildcard_centre
                    } else {
                        WEIGHT_X
                    }
            })
    }

    /// Does a wildcard `row` match `names` slot by slot?
    fn general_match(row: &Row, names: [&str; 4]) -> bool {
        row.wildcards() > 0 && (0..4).all(|i| row.names[i] == X || row.names[i] == names[i])
    }

    /// The frcmod improper rows of the molecule: one per estimated quartet, in
    /// the order parmchk2 writes them.
    fn estimate(
        mut self,
        order: &[NodeId],
        neighbours: &[Vec<NodeId>],
        type_of: &HashMap<NodeId, &'static str>,
    ) -> Vec<Row> {
        let table_rows = self.rows.len();
        for (atom, around) in order.iter().zip(neighbours) {
            let centre = type_of[atom];
            if around.len() < 3 || !self.block(centre).is_some_and(|b| b.improper) {
                continue;
            }
            // parmchk2's slots: the first three neighbours, the centre third.
            let unsorted = [
                type_of[&around[0]],
                type_of[&around[1]],
                centre,
                type_of[&around[2]],
            ];
            let mut sorted = [unsorted[0], unsorted[1], unsorted[3]];
            sorted.sort_unstable();
            let names = [sorted[0], sorted[1], centre, sorted[2]];

            // Step 1: a row (table or already estimated) for exactly these names.
            if self.rows.iter().any(|row| row.names == names) {
                continue;
            }
            let (force, phase, periodicity, provenance) =
                match self.search(&self.rows, unsorted, names) {
                    Some((row, provenance)) => (row.force, row.phase, row.periodicity, provenance),
                    None => {
                        let (k, phase, n) = DEFAULT_IMPROPER;
                        (k, phase, f64::from(n), Provenance::wildcard(0.0, ""))
                    }
                };
            self.rows.push(Row {
                names: names.map(str::to_owned),
                force,
                phase,
                periodicity,
                estimate: Some(provenance),
            });
        }
        self.rows.split_off(table_rows)
    }

    /// Steps 2–6 of `chk_improper` for one improper: the row they settle on
    /// and how, over `rows` — the table's and those estimated so far.
    /// `unsorted` is in the atoms' bond order, `names` sorted.
    fn search<'r>(
        &self,
        rows: &'r [Row],
        unsorted: [&'static str; 4],
        names: [&str; 4],
    ) -> Option<(&'r Row, Provenance)> {
        let w = &self.parmchk.weights;
        let equa = unsorted.map(|t| self.equivalents(t));

        // Step 2: a wildcard-free row over equivalent types.
        for (m, &e1) in equa[0].iter().enumerate() {
            for (n, &e2) in equa[1].iter().enumerate() {
                for (p, &e3) in equa[2].iter().enumerate() {
                    for (q, &e4) in equa[3].iter().enumerate() {
                        if m == 0 && n == 0 && p == 0 && q == 0 {
                            continue;
                        }
                        if let Some(row) = rows
                            .iter()
                            .find(|r| r.wildcards() == 0 && r.names == [e1, e2, e3, e4])
                        {
                            return Some((row, Provenance::analogy(0.0, row.label())));
                        }
                    }
                }
            }
        }

        // Step 3: a wildcard row over the sorted names, cheapest first.
        let mut best: Option<(&Row, f64)> = None;
        for row in rows.iter().filter(|r| Self::general_match(r, names)) {
            let score = self.wildcard_score(row);
            if best.is_none_or(|(_, b)| score < b) {
                best = Some((row, score));
            }
        }
        if let Some((row, score)) = best {
            return Some((row, Provenance::wildcard(score, row.label())));
        }

        // Step 4: a wildcard row over equivalent types; the first row matching a
        // combination is that combination's only candidate.
        let mut best: Option<(&Row, f64)> = None;
        for (m, &e1) in equa[0].iter().enumerate() {
            for (n, &e2) in equa[1].iter().enumerate() {
                for (p, &e3) in equa[2].iter().enumerate() {
                    for (q, &e4) in equa[3].iter().enumerate() {
                        if m == 0 && n == 0 && p == 0 && q == 0 {
                            continue;
                        }
                        if let Some(row) = rows
                            .iter()
                            .find(|r| Self::general_match(r, [e1, e2, e3, e4]))
                        {
                            let score = self.wildcard_score(row);
                            if best.is_none_or(|(_, b)| score < b) {
                                best = Some((row, score));
                            }
                        }
                    }
                }
            }
        }
        if let Some((row, score)) = best {
            return Some((row, Provenance::analogy(score, row.label())));
        }

        let corr = unsorted.map(|t| self.correspondents(t));
        let substituted = |c: [&Corr; 4]| c.iter().any(|&&(_, _, kind)| kind > 1);
        // parmchk2 adds WEIGHT_GROUP and takes it back off for one group; the
        // additions are made in its order, so equal analogs tie as they do there.
        let grouped = |c: [&Corr; 4], score: f64| {
            let score = score + w.weight_group;
            if self.same_group([c[0].0, c[1].0, c[2].0, c[3].0]) {
                score - w.weight_group
            } else {
                score
            }
        };

        // Step 5: a wildcard-free row over corresponding types; the centre's
        // score is weighted by WEIGHT_IMPROPER.
        let mut best: Option<(&Row, f64)> = None;
        for c1 in &corr[0] {
            for c2 in &corr[1] {
                for c3 in &corr[2] {
                    for c4 in &corr[3] {
                        let c = [c1, c2, c3, c4];
                        if !substituted(c) {
                            continue;
                        }
                        let score = grouped(c, c1.1 + c2.1 + c3.1 * w.weight_improper + c4.1);
                        if best.is_some_and(|(_, b)| score >= b) {
                            continue;
                        }
                        if let Some(row) = rows
                            .iter()
                            .find(|r| r.wildcards() == 0 && r.names == [c1.0, c2.0, c3.0, c4.0])
                        {
                            best = Some((row, score));
                        }
                    }
                }
            }
        }
        if let Some((row, score)) = best {
            return Some((row, Provenance::analogy(score, row.label())));
        }

        // Step 6: a wildcard row over corresponding types; the centre's score
        // takes WEIGHT_IMPROPER as a surcharge, and the first row matching a
        // combination is its only candidate.
        let mut best: Option<(&Row, f64)> = None;
        for c1 in &corr[0] {
            for c2 in &corr[1] {
                for c3 in &corr[2] {
                    for c4 in &corr[3] {
                        let c = [c1, c2, c3, c4];
                        if !substituted(c) {
                            continue;
                        }
                        let base = grouped(c, c1.1 + c2.1 + (c3.1 + w.weight_improper) + c4.1);
                        if let Some(row) = rows
                            .iter()
                            .find(|r| Self::general_match(r, [c1.0, c2.0, c3.0, c4.0]))
                        {
                            let score = self.add_wildcards(base, row);
                            if best.is_none_or(|(_, b)| score < b) {
                                best = Some((row, score));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(row, score)| (row, Provenance::analogy(score, row.label())))
    }
}

// ---------------------------------------------------------------------------
// tleap
// ---------------------------------------------------------------------------

/// A row as LEaP stores it: peripherals in its canonical order (wildcards
/// first, then alphabetical) and the permutation back to the order written.
#[derive(Debug, Clone)]
struct Stored {
    types: [String; 4],
    order: [usize; 4],
    row: Row,
}

/// `zParmSetOrderImproperAtoms`: sort the three peripherals, `X` first, and
/// track where each slot came from.
fn stored(row: &Row) -> Stored {
    // A wildcard sorts before every type.
    let key = |name: &str| {
        if name == X {
            String::new()
        } else {
            name.to_owned()
        }
    };
    let mut a = [
        row.names[0].clone(),
        row.names[1].clone(),
        row.names[3].clone(),
    ];
    let mut order = [0usize, 1, 2, 3];
    let swap01 = |a: &mut [String; 3], order: &mut [usize; 4]| {
        a.swap(0, 1);
        order.swap(0, 1);
    };
    let swap12 = |a: &mut [String; 3], order: &mut [usize; 4]| {
        a.swap(1, 2);
        order.swap(1, 3);
    };
    if key(&a[0]) > key(&a[1]) {
        swap01(&mut a, &mut order);
    }
    if key(&a[1]) > key(&a[2]) {
        swap12(&mut a, &mut order);
    }
    if key(&a[0]) > key(&a[1]) {
        swap01(&mut a, &mut order);
    }
    if key(&a[1]) > key(&a[2]) {
        swap12(&mut a, &mut order);
    }
    let [p0, p1, p3] = a;
    Stored {
        types: [p0, p1, row.names[2].clone(), p3],
        order,
        row: row.clone(),
    }
}

impl Stored {
    fn generality(&self) -> usize {
        self.types.iter().filter(|t| *t == X).count()
    }

    fn wild(&self) -> bool {
        self.types[0] == X
    }

    /// `zbParmSetMatchImproper` against a query in canonical order.
    fn matches(&self, q: &[&str; 4]) -> bool {
        let t = &self.types;
        if t[2] != q[2] {
            return false;
        }
        match self.generality() {
            0 => t[0] == q[0] && t[1] == q[1] && t[3] == q[3],
            1 => {
                if t[1] == q[0] {
                    t[3] == q[1] || t[3] == q[3]
                } else {
                    t[1] == q[1] && t[3] == q[3]
                }
            }
            2 => t[3] == q[0] || t[3] == q[1] || t[3] == q[3],
            _ => true,
        }
    }
}

/// `ParmSetImproperOrderAtoms`: put `atoms` (with `types`, in loop order) into
/// the slots of `row`, then order wildcard and same-type peripherals by
/// `index`.
fn order_atoms(
    row: &Stored,
    mut types: [&str; 4],
    mut atoms: [NodeId; 4],
    index: &HashMap<NodeId, usize>,
) -> ([NodeId; 4], [String; 4]) {
    let p = &row.types;
    let swap = |types: &mut [&str; 4], atoms: &mut [NodeId; 4], a: usize, b: usize| {
        types.swap(a, b);
        atoms.swap(a, b);
    };
    if p[0] != X && p[0] != types[0] {
        if p[0] == types[1] && p[1] != types[1] {
            swap(&mut types, &mut atoms, 1, 0);
        } else if p[0] == types[3] && p[3] != types[3] {
            swap(&mut types, &mut atoms, 3, 0);
        }
    }
    if p[1] != X && p[1] != types[1] {
        if p[1] == types[0] && p[0] == X {
            swap(&mut types, &mut atoms, 1, 0);
        } else if p[1] == types[3] && p[3] != types[3] {
            swap(&mut types, &mut atoms, 3, 1);
        }
    }
    if p[3] != X && p[3] != types[3] {
        if p[3] == types[0] && p[0] == X {
            swap(&mut types, &mut atoms, 3, 0);
        } else if p[3] == types[1] && p[1] == X {
            swap(&mut types, &mut atoms, 3, 1);
        }
    }
    let mut t = types;
    let mut a = atoms;
    for k in 0..4 {
        t[row.order[k]] = types[k];
        a[row.order[k]] = atoms[k];
    }
    let peripheral = [0usize, 1, 3];
    // Wildcard slots by atom index.
    for (n, &i) in peripheral.iter().enumerate() {
        if p[i] != X {
            continue;
        }
        let best = peripheral[n + 1..]
            .iter()
            .copied()
            .filter(|&j| p[j] == X)
            .fold(None, |best: Option<usize>, j| {
                let current = best.map_or(index[&a[i]], |b| index[&a[b]]);
                (index[&a[j]] < current).then_some(j).or(best)
            });
        if let Some(j) = best {
            t.swap(i, j);
            a.swap(i, j);
        }
    }
    // Same-type peripherals by atom index.
    for (n, &i) in peripheral.iter().enumerate() {
        let best = peripheral[n + 1..]
            .iter()
            .copied()
            .filter(|&j| t[j] == t[i])
            .fold(None, |best: Option<usize>, j| {
                let current = best.map_or(index[&a[i]], |b| index[&a[b]]);
                (index[&a[j]] < current).then_some(j).or(best)
            });
        if let Some(j) = best {
            t.swap(i, j);
            a.swap(i, j);
        }
    }
    (a, t.map(str::to_owned))
}

/// tleap's impropers: every triple of neighbours of every atom with three or
/// more, wherever the unit's terms, the frcmod or the table has a row.
fn tleap(
    order: &[NodeId],
    neighbours: &[Vec<NodeId>],
    index: &HashMap<NodeId, usize>,
    type_of: &HashMap<NodeId, &'static str>,
    frcmod: &[Row],
    table: &[Row],
) -> Vec<ImproperTerm> {
    let frcmod: Vec<Stored> = frcmod.iter().map(stored).collect();
    let table: Vec<Stored> = table.iter().map(stored).collect();
    let mut unit: Vec<Stored> = Vec::new();
    let mut out = Vec::new();
    for (&centre, around) in order.iter().zip(neighbours) {
        let n = around.len();
        for i0 in 0..n {
            for i1 in i0 + 1..n {
                for i2 in i1 + 1..n {
                    let atoms = [around[i0], around[i1], centre, around[i2]];
                    let types = atoms.map(|a| type_of[&a]);
                    let mut q = [types[0], types[1], types[3]];
                    q.sort_unstable();
                    let query = [q[0], q[1], types[2], q[2]];

                    let mut best: Option<&Stored> = None;
                    for set in [&unit, &frcmod, &table] {
                        for row in set.iter().filter(|r| r.matches(&query)) {
                            let better = match best {
                                None => true,
                                Some(b) => {
                                    (b.wild() && !row.wild()) || row.generality() < b.generality()
                                }
                            };
                            if better {
                                best = Some(row);
                            }
                        }
                        if best.is_some_and(|b| !b.wild()) {
                            break;
                        }
                    }
                    let Some(row) = best.cloned() else {
                        continue;
                    };
                    let (atoms, types) = order_atoms(&row, types, atoms, index);
                    // The term joins the unit's set in its canonical order.
                    if !unit
                        .iter()
                        .any(|u| u.types == row.types && u.row.force == row.row.force)
                    {
                        unit.push(Stored {
                            order: [0, 1, 2, 3],
                            ..row.clone()
                        });
                    }
                    out.push(ImproperTerm {
                        atoms,
                        types,
                        params: Params::from_iter([
                            ("k", row.row.force),
                            ("periodicity", row.row.periodicity),
                            ("phase", row.row.phase),
                        ]),
                        estimate: row.row.estimate.clone(),
                    });
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(names: [&str; 4]) -> Row {
        Row {
            names: names.map(str::to_owned),
            force: 1.1,
            phase: 180.0,
            periodicity: 2.0,
            estimate: None,
        }
    }

    /// parmchk2 reads `ca-ca-ca-c3` as `c3-ca-ca-ca` and leaves `X -X -c -o`
    /// as written.
    #[test]
    fn parmchk2_sorts_a_specific_rows_peripherals_on_read() {
        assert_eq!(
            normalised(&row(["ca", "ca", "ca", "c3"])).names,
            ["c3", "ca", "ca", "ca"]
        );
        assert_eq!(
            normalised(&row(["X", "X", "c", "o"])).names,
            ["X", "X", "c", "o"]
        );
        assert_eq!(
            normalised(&row(["X", "os", "c", "o"])).names,
            ["X", "o", "c", "os"]
        );
    }

    /// LEaP stores `ca-ca-ca-c3` sorted and remembers that the `c3` was
    /// written fourth.
    #[test]
    fn leap_stores_a_row_sorted_with_the_way_back() {
        let s = stored(&row(["ca", "ca", "ca", "c3"]));
        assert_eq!(s.types, ["c3", "ca", "ca", "ca"]);
        assert_eq!(s.order, [3, 0, 2, 1]);
    }
}
