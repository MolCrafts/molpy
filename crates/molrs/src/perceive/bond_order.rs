//! Bond orders from connectivity alone — antechamber's bond-order perception.

use std::collections::HashMap;

use crate::core::Atomistic;
use crate::core::BondNumber;
use crate::core::MolRsError;
use crate::core::NodeId;
use crate::core::PropValue;
use crate::core::keys;

use super::valence::atomic_number;

/// `define.h`'s `PSCUTOFF`: penalties above it are never tried as valence states.
const PSCUTOFF: i32 = 10;

/// `APS.DAT`'s `*` — a valence the row does not allow.
const NONE: i32 = 9999;

/// Which `APS.DAT` line prefix a row carries — the environment it applies in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApsEnvironment {
    /// `APS`: any atom none of the special environments claims.
    Plain,
    /// `APSCO2`: a trivalent carbon with two or more terminal O/S.
    Co2,
    /// `APSPO2`: a tetravalent phosphorus with exactly two terminal O/S.
    Po2,
    /// `APSPO3`: a tetravalent phosphorus with three or more terminal O/S.
    Po3,
    /// `APSSO2`: a tetravalent sulfur with exactly two terminal O/S.
    So2,
    /// `APSSO3`: a tetravalent sulfur with exactly three terminal O/S.
    So3,
    /// `APSSO4`: a tetravalent sulfur with four terminal O/S.
    So4,
    /// `APSN1-`: a terminal nitrogen on a two-connected nitrogen (azide end).
    N1Minus,
    /// `APSN2+`: a two-connected nitrogen with a terminal C/N neighbour.
    N2Plus,
    /// `APSNO2`: a three-connected nitrogen with two or more terminal O/S.
    No2,
    /// `APSN3+`: a three-connected nitrogen with one terminal O/S (an N-oxide).
    N3Plus,
    /// `APSO1-`: a terminal oxygen on a three-connected N with no other terminal O/S.
    O1Minus,
    /// `APSS1-`: the sulfur twin of [`ApsEnvironment::O1Minus`].
    S1Minus,
    /// `APSC1+`: a terminal carbon on a two-connected nitrogen (isonitrile).
    C1Plus,
}

/// One `APS.DAT` row: element, connection count (`None` for `*`), and the
/// penalty of total valence 0 … 7 ([`NONE`] for `*`).
struct ApsRow {
    environment: ApsEnvironment,
    z: u8,
    connections: Option<usize>,
    aps: [i32; 8],
}

const fn row(
    environment: ApsEnvironment,
    z: u8,
    connections: Option<usize>,
    aps: [i32; 8],
) -> ApsRow {
    ApsRow {
        environment,
        z,
        connections,
        aps,
    }
}

/// `APS.DAT` of AmberTools 26.1, every row in file order (the first row that
/// applies wins, so the order is part of the table).
#[rustfmt::skip]
static APS: &[ApsRow] = &[
    row(ApsEnvironment::Plain,   1,  None,    [64, 0, 64, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   9,  None,    [64, 0, 64, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   17, None,    [64, 0, 64, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   35, None,    [64, 0, 64, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   53, None,    [64, 0, 64, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   6,  Some(1), [NONE, NONE, NONE, 1, 0, 32, NONE, NONE]),
    row(ApsEnvironment::C1Plus,  6,  Some(1), [NONE, NONE, NONE, 0, 1, 32, NONE, NONE]),
    row(ApsEnvironment::Plain,   6,  None,    [NONE, NONE, 64, 32, 0, 32, 64, NONE]),
    row(ApsEnvironment::Plain,   14, None,    [NONE, NONE, NONE, NONE, 0, NONE, NONE, NONE]),
    row(ApsEnvironment::Co2,     6,  Some(3), [NONE, NONE, NONE, NONE, 32, 0, 32, NONE]),
    row(ApsEnvironment::Plain,   7,  Some(1), [NONE, NONE, 3, 0, 32, NONE, NONE, NONE]),
    row(ApsEnvironment::N1Minus, 7,  Some(1), [NONE, NONE, 0, 0, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   7,  Some(2), [NONE, NONE, 4, 0, 2, NONE, NONE, NONE]),
    row(ApsEnvironment::N2Plus,  7,  Some(2), [NONE, NONE, NONE, 1, 0, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   7,  Some(3), [NONE, NONE, 32, 0, 1, 2, NONE, NONE]),
    row(ApsEnvironment::No2,     7,  Some(3), [NONE, NONE, NONE, 64, 32, 0, 32, NONE]),
    row(ApsEnvironment::N3Plus,  7,  Some(3), [NONE, NONE, NONE, 1, 0, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   7,  Some(4), [NONE, NONE, NONE, 64, 0, 64, NONE, NONE]),
    row(ApsEnvironment::Plain,   8,  Some(1), [NONE, 1, 0, 64, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::O1Minus, 8,  Some(1), [NONE, 0, 1, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   8,  Some(2), [NONE, 32, 0, 64, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   15, Some(1), [NONE, NONE, 2, 0, 32, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   15, Some(2), [NONE, NONE, 4, 0, 2, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   15, Some(3), [NONE, NONE, 32, 0, 1, 2, NONE, NONE]),
    row(ApsEnvironment::Plain,   15, Some(4), [NONE, NONE, NONE, 64, 1, 0, 32, NONE]),
    row(ApsEnvironment::Po2,     15, Some(4), [NONE, NONE, NONE, NONE, NONE, 32, 0, 32]),
    row(ApsEnvironment::Po3,     15, Some(4), [NONE, NONE, NONE, NONE, NONE, NONE, 32, 0]),
    row(ApsEnvironment::Plain,   16, Some(1), [NONE, 2, 0, 64, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::S1Minus, 16, Some(1), [NONE, 0, 1, NONE, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   16, Some(2), [NONE, 2, 0, 64, NONE, NONE, NONE, NONE]),
    row(ApsEnvironment::Plain,   16, Some(3), [NONE, NONE, NONE, 1, 0, 2, 2, NONE]),
    row(ApsEnvironment::Plain,   16, Some(4), [NONE, NONE, NONE, NONE, 4, 2, 0, NONE]),
    row(ApsEnvironment::So2,     16, Some(4), [NONE, NONE, NONE, NONE, NONE, NONE, 0, 32]),
    row(ApsEnvironment::So3,     16, Some(4), [NONE, NONE, NONE, NONE, NONE, NONE, 32, 0]),
    row(ApsEnvironment::So4,     16, Some(4), [NONE, NONE, NONE, NONE, NONE, NONE, 32, 0]),
    row(ApsEnvironment::Plain,   28, Some(5), [NONE, NONE, NONE, NONE, NONE, 1, 0, 1]),
];

/// `bondtype`'s `avH`: the score of a capping hydrogen.
const CAP_H: [i32; 8] = [64, 0, 64, NONE, NONE, NONE, NONE, NONE];

/// Write the bond orders antechamber's `bondtype -j full` judges from the
/// connectivity onto `mol`, in place.
///
/// Every judged bond gets `bond_number` (1 / 2 / 3) and the localized
/// `bond_type` it implies, replacing whatever the input stated — aromatic
/// markings included; perceive aromaticity afterwards if it is wanted. The bonds
/// of a residue no valence state closes are left as they were (see
/// [`perceive_bond_orders`]).
///
/// # Arguments
///
/// * `mol` — the molecule, every hydrogen drawn; on return it carries the
///   judged Kekulé structure.
///
/// # Errors
///
/// [`MolRsError`] when a judged class cannot be written (a bond column holding
/// the key under a foreign dtype); `mol` may then be partly rewritten.
pub fn assign_bond_orders(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let bond_ids: Vec<_> = mol.bond_ids().collect();
    let judged = perceive_bond_orders(mol);
    for (bid, order) in bond_ids.into_iter().zip(judged) {
        if let Some(order) = order {
            let number = BondNumber::from_code(u32::from(order));
            mol.set_bond_class(bid, number.implied_type(), number)?;
        }
    }
    Ok(())
}

/// Judge the bond orders of `mol` from its connectivity, as antechamber's
/// `bondtype -j full` does.
///
/// `antechamber` (its default `-j 4`) runs `bondtype -j full` before it types a
/// single atom: whatever bond orders the input file states, it throws them away
/// and re-derives a Kekulé structure from the elements and the connectivity. The
/// atom types it then assigns — GAFF's `cc` / `cd`, `ce` / `cf`, `nc` / `nd`
/// colouring above all — follow *that* structure. Where a molecule has more than
/// one (azulene, cyclooctatetraene, every polycyclic aromatic), the one antechamber
/// settles on is fixed by its search, not by chemistry, so matching antechamber
/// means running its search. This function is that search.
///
/// # The algorithm (`bondtype.c`)
///
/// 1. **Atomic penalty scores** (`assignav`). Every atom gets a row of `APS.DAT`:
///    the penalty of each total valence 0–7 for its element and connection count,
///    with special rows for carboxylate / phosphate / sulfonate / nitro centres,
///    `N1-` / `N2+` (azides, diazo), `N3+` / `O1-` / `S1-` (N-oxides) and `C1+`
///    (isonitriles). The valence of penalty 0 is the atom's best valence. An atom
///    no row covers (B, Se, metals) has no score, and its bonds are *frozen* at
///    their input order.
/// 2. **Valence states** (`process`, `ncsu-penalties.c`). The state of best
///    valences is tried first, then every state that raises atoms to valences of
///    penalty 1 … `PSCUTOFF` (10), in order of total penalty, and within one
///    penalty in the order of Stockmal's partition enumeration and of the
///    lexicographic combinations of the atoms (in atom order) holding each
///    penalty.
/// 3. **Bond orders for one state** (`judgebt`, `jbo_induce`, `jbo_iteration`).
///    Orders that are forced are induced (an atom with one undecided bond takes
///    all its remaining valence on it; an atom whose undecided bonds equal its
///    remaining valence takes them all single). When nothing is forced, the first
///    undecided bond *in bond order* is tried single, then double, then triple,
///    inducing after each, and a violation backtracks to the snapshot taken at the
///    start of that sweep. The first state that closes with every valence spent
///    wins.
///
/// The answer therefore depends on the order of the atoms (the valence states,
/// the induction sweep) and of the bonds (the trial order) — exactly as
/// antechamber's does on the file it reads. A molecule built from a mol2 file
/// keeps the file's order, so molrs and antechamber see the same input.
///
/// Each residue (`res_id`) is judged on its own, as `bondtype` does: a bond to
/// another residue is single, and its far end stands in the residue as a capping
/// hydrogen. (`bondtype` also carries the valence states of one residue into the
/// next — its state table is never reset — which reads past its arrays; molrs
/// judges each residue from a fresh table.)
///
/// The graph must carry every hydrogen, as antechamber's input does: the scores
/// are keyed on the drawn connection count.
///
/// # Provenance
///
/// A transcription of AmberTools' `antechamber/bondtype.c` (`assignav`, `main`'s
/// residue set-up, `process`, `judgebt`, `jbo_induce`, `jbo_iteration`,
/// `jbo_score`) and `ncsu-penalties.c`, with the `APS.DAT` rows of AmberTools
/// 26.1 transcribed in a private table. Checked against `bondtype -j full` of AmberTools
/// 26.1 bond for bond.
///
/// # Arguments
///
/// * `mol` — the molecule, every hydrogen drawn. Its bond orders are read only
///   for the bonds `bondtype` freezes (those of an atom `APS.DAT` does not
///   score), which keep their stated number (single when it states none).
///
/// # Returns
///
/// One entry per bond, in [`Atomistic::bonds`] order: the judged order (1, 2
/// or 3), or `None` for every bond of a residue no valence state up to
/// `PSCUTOFF` gives a consistent structure (where `bondtype` warns "The assigned
/// bond types may be wrong" and keeps the file's bond types).
pub fn perceive_bond_orders(mol: &Atomistic) -> Vec<Option<u8>> {
    let graph = Graph::new(mol);
    let mut out: Vec<Option<u8>> = vec![None; graph.bonds.len()];
    if graph.bonds.is_empty() {
        return out;
    }
    let scores = graph.scores();

    // Residues in order of appearance, a new one at every change of `res_id`
    // (antechamber's `res[]`; a residue that reappears later is judged again).
    let mut residues: Vec<Option<u64>> = Vec::new();
    let mut previous: Option<Option<u64>> = None;
    for res in &graph.res {
        if previous != Some(*res) {
            residues.push(*res);
            previous = Some(*res);
        }
    }
    let single_residue = residues.len() == 1;
    for (k, (i, j)) in graph.bonds.iter().enumerate() {
        if graph.res[*i] != graph.res[*j] {
            out[k] = Some(1);
        }
    }
    for res in residues {
        let members: Vec<usize> = (0..graph.z.len())
            .filter(|a| single_residue || graph.res[*a] == res)
            .collect();
        let mut residue = Residue::new(&graph, &scores, &members);
        if let Some(orders) = residue.judge() {
            for (k, order) in orders {
                out[k] = u8::try_from(order).ok();
            }
        }
    }
    out
}

/// The molecule in antechamber's index space: atom `i` is the `i`-th atom of
/// [`Atomistic::atoms`], bond `k` the `k`-th of [`Atomistic::bonds`], and each
/// atom's neighbours are listed in the order its bonds appear (`atom[].con[]`
/// as `rac` / `rmol2` build it).
struct Graph {
    z: Vec<u8>,
    con: Vec<Vec<usize>>,
    bonds: Vec<(usize, usize)>,
    /// The stated order of each bond, read only for frozen bonds.
    stated: Vec<i32>,
    res: Vec<Option<u64>>,
}

impl Graph {
    fn new(mol: &Atomistic) -> Self {
        let mut ids: Vec<NodeId> = Vec::with_capacity(mol.n_atoms());
        let mut z = Vec::with_capacity(mol.n_atoms());
        let mut res = Vec::with_capacity(mol.n_atoms());
        for (id, atom) in mol.atoms() {
            ids.push(id);
            // A wildcard atom (Z = 0) has no `APS.DAT` row, so like B or a
            // metal its bonds are frozen at their stated order.
            z.push(atomic_number(&atom));
            res.push(
                atom.get(keys::RES_ID)
                    .and_then(PropValue::as_f64)
                    .map(|v| v.round() as u64),
            );
        }
        let rows = mol.atom_table();
        let mut con = vec![Vec::new(); ids.len()];
        let mut bonds = Vec::new();
        let mut stated = Vec::new();
        for (_, bond) in mol.bonds() {
            let (Some(i), Some(j)) = (rows.row(bond.nodes[0]), rows.row(bond.nodes[1])) else {
                continue;
            };
            con[i].push(j);
            con[j].push(i);
            bonds.push((i, j));
            let number = BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER)).count();
            stated.push(if (1..=3).contains(&number) {
                number as i32
            } else {
                1
            });
        }
        Self {
            z,
            con,
            bonds,
            stated,
            res,
        }
    }

    fn degree(&self, a: usize) -> usize {
        self.con[a].len()
    }

    /// A one-connected O or S.
    fn terminal_chalcogen(&self, a: usize) -> bool {
        matches!(self.z[a], 8 | 16) && self.degree(a) == 1
    }

    /// `assignav`: the `APS.DAT` row of every atom.
    fn scores(&self) -> Vec<[i32; 8]> {
        (0..self.z.len()).map(|a| self.score(a)).collect()
    }

    fn score(&self, i: usize) -> [i32; 8] {
        let (z, degree) = (self.z[i], self.degree(i));
        let terminal = |a: usize| usize::from(self.terminal_chalcogen(a));
        let around = |a: usize| self.con[a].iter().map(|n| terminal(*n)).sum::<usize>();

        let co2 = if z == 6 && degree == 3 { around(i) } else { 0 };
        let po = if z == 15 && degree == 4 { around(i) } else { 0 };
        let so = if z == 16 && degree == 4 { around(i) } else { 0 };
        let n1 = z == 7 && degree == 1 && {
            let n = self.con[i][0];
            self.z[n] == 7 && self.degree(n) == 2
        };
        let n2 = z == 7
            && degree == 2
            && self.con[i]
                .iter()
                .any(|n| matches!(self.z[*n], 6 | 7) && self.degree(*n) == 1);
        let no2 = if z == 7 && degree == 3 { around(i) } else { 0 };
        let n3 = z == 7 && degree == 3 && no2 < 2 && no2 > 0;
        // A terminal O / S on a three-connected N that carries no second one.
        let x1 = |element: u8| {
            z == element && degree == 1 && {
                let n = self.con[i][0];
                self.z[n] == 7 && self.degree(n) == 3 && around(n) <= 1
            }
        };
        let o1 = x1(8);
        let s1 = x1(16);
        let c1 = z == 6 && degree == 1 && {
            let n = self.con[i][0];
            self.z[n] == 7 && self.degree(n) == 2
        };

        let plain =
            co2 < 2 && po < 2 && so < 2 && !n1 && !n2 && no2 < 2 && !n3 && !o1 && !s1 && !c1;
        let applies = |environment: ApsEnvironment| match environment {
            // Every line starts with `APS`, so with no special environment every row
            // is a candidate; the plain rows come first for each element.
            _ if plain => true,
            ApsEnvironment::Plain => false,
            ApsEnvironment::Co2 => co2 >= 2,
            ApsEnvironment::Po2 => po == 2,
            ApsEnvironment::Po3 => po > 2,
            ApsEnvironment::So2 => so == 2,
            ApsEnvironment::So3 => so == 3,
            ApsEnvironment::So4 => so == 4,
            ApsEnvironment::N1Minus => n1,
            ApsEnvironment::N2Plus => n2,
            ApsEnvironment::No2 => no2 >= 2,
            ApsEnvironment::N3Plus => n3,
            ApsEnvironment::O1Minus => o1,
            ApsEnvironment::S1Minus => s1,
            ApsEnvironment::C1Plus => c1,
        };
        APS.iter()
            .find(|r| {
                applies(r.environment) && r.z == z && r.connections.is_none_or(|c| c == degree)
            })
            .map_or([NONE; 8], |r| r.aps)
    }
}

/// The best valence of a score row — the *last* valence of penalty 0, as
/// `assignav`'s loop leaves it — or `None` when no valence has penalty 0.
fn best_valence(aps: &[i32; 8]) -> Option<i32> {
    (0..8).rev().find(|v| aps[*v] == 0).map(|v| v as i32)
}

/// One bond of a residue (`bond2[]`).
#[derive(Clone)]
struct ResBond {
    i: usize,
    j: usize,
    /// The bond's current order; `-1` while undecided.
    order: i32,
    frozen: bool,
    /// The bond of the molecule it stands for; `None` for a capping bond.
    of: Option<usize>,
}

/// One residue being judged: `bondtype`'s `atom2[]`, `bond2[]`, `av2[]` …
struct Residue {
    con: Vec<Vec<usize>>,
    aps: Vec<[i32; 8]>,
    /// `apstype2[]`: whether the atom has a valence of penalty 0.
    scored: Vec<bool>,
    best: Vec<i32>,
    bonds: Vec<ResBond>,
    /// Per atom: its bonds, in bond order.
    incident: Vec<Vec<usize>>,
    max_aps: i32,
    // The search state.
    decided: Vec<bool>,
    valence_left: Vec<i32>,
    bonds_left: Vec<i32>,
}

impl Residue {
    fn new(graph: &Graph, scores: &[[i32; 8]], members: &[usize]) -> Self {
        let mut local: HashMap<usize, usize> = HashMap::new();
        for (l, g) in members.iter().enumerate() {
            local.insert(*g, l);
        }
        let mut con: Vec<Vec<usize>> = members
            .iter()
            .map(|g| {
                graph.con[*g]
                    .iter()
                    .filter_map(|n| local.get(n).copied())
                    .collect()
            })
            .collect();
        let mut aps: Vec<[i32; 8]> = members.iter().map(|g| scores[*g]).collect();
        let mut best: Vec<i32> = aps.iter().map(|a| best_valence(a).unwrap_or(0)).collect();
        let mut scored: Vec<bool> = aps.iter().map(|a| best_valence(a).is_some()).collect();

        let max_aps = aps
            .iter()
            .flat_map(|a| a.iter())
            .filter(|p| **p <= PSCUTOFF)
            .copied()
            .max()
            .unwrap_or(0)
            .max(0);

        let mut bonds: Vec<ResBond> = Vec::new();
        for (k, (i, j)) in graph.bonds.iter().enumerate() {
            if let (Some(&li), Some(&lj)) = (local.get(i), local.get(j)) {
                bonds.push(ResBond {
                    i: li,
                    j: lj,
                    order: graph.stated[k],
                    frozen: false,
                    of: Some(k),
                });
            }
        }
        // A bond to another residue: its far end stands in as a capping H.
        for (i, j) in &graph.bonds {
            let (inside, far_is_j) = match (local.get(i), local.get(j)) {
                (Some(&li), None) => (li, true),
                (None, Some(&lj)) => (lj, false),
                _ => continue,
            };
            let h = con.len();
            con.push(vec![inside]);
            con[inside].push(h);
            aps.push(CAP_H);
            best.push(1);
            scored.push(true);
            let (bi, bj) = if far_is_j { (inside, h) } else { (h, inside) };
            bonds.push(ResBond {
                i: bi,
                j: bj,
                order: 1,
                frozen: false,
                of: None,
            });
        }

        // Pre-defined bond types: every bond of an unscored atom is frozen at
        // its stated order, and an atom whose bonds are all frozen keeps the
        // valence they sum to.
        for b in &mut bonds {
            if !scored[b.i] || !scored[b.j] {
                b.frozen = true;
            }
        }
        for a in 0..con.len() {
            let mut total = 0;
            let mut frozen = 0;
            for b in &bonds {
                if b.i == a || b.j == a {
                    total += b.order;
                    frozen += usize::from(b.frozen);
                }
            }
            if scored[a] {
                if con[a].len() != frozen {
                    continue;
                }
                aps[a] = [NONE; 8];
            }
            if let Ok(v) = usize::try_from(total)
                && v < 8
            {
                aps[a][v] = 0;
            }
            best[a] = total;
        }

        let n = con.len();
        let n_bonds = bonds.len();
        let mut incident = vec![Vec::new(); n];
        for (k, b) in bonds.iter().enumerate() {
            incident[b.i].push(k);
            incident[b.j].push(k);
        }
        Self {
            con,
            aps,
            scored,
            best,
            bonds,
            incident,
            max_aps,
            decided: vec![false; n_bonds],
            valence_left: vec![0; n],
            bonds_left: vec![0; n],
        }
    }

    /// `process` + `judgebt`: the orders of the first valence state that closes,
    /// as `(molecule bond, order)` pairs.
    fn judge(&mut self) -> Option<Vec<(usize, i32)>> {
        if self.attempt(&[]) {
            return Some(self.result());
        }
        if self.max_aps <= 0 {
            return None;
        }
        // `points[p]`: every (atom, valence) of penalty p, atoms in order.
        let max = self.max_aps as usize;
        let mut points: Vec<Vec<(usize, i32)>> = vec![Vec::new(); max + 1];
        for (a, row) in self.aps.iter().enumerate() {
            for (v, p) in row.iter().enumerate() {
                if *p > 0 && *p <= self.max_aps {
                    points[*p as usize].push((a, v as i32));
                }
            }
        }
        let sizes: Vec<usize> = (1..=max).map(|p| points[p].len()).collect();
        for goal in 1..=max {
            let closed = penalty_states(max, &sizes, goal, &mut |pairs| {
                let state: Vec<(usize, i32)> = pairs
                    .iter()
                    .map(|(set, element)| points[set + 1][*element])
                    .collect();
                self.attempt(&state)
            });
            if closed {
                return Some(self.result());
            }
        }
        None
    }

    fn result(&self) -> Vec<(usize, i32)> {
        self.bonds
            .iter()
            .filter_map(|b| b.of.map(|k| (k, b.order)))
            .collect()
    }

    /// One valence state of `judgebt`: the best valences, with `state`'s
    /// (atom, valence) pairs applied in order on top. Whether it closes.
    fn attempt(&mut self, state: &[(usize, i32)]) -> bool {
        let mut valence = self.best.clone();
        for (a, v) in state {
            valence[*a] = *v;
        }
        let n = self.con.len();
        let mut connections: Vec<i32> = self.con.iter().map(|c| c.len() as i32).collect();
        self.valence_left = valence;
        self.bonds_left = connections.clone();
        for k in 0..self.bonds.len() {
            let b = &mut self.bonds[k];
            if b.frozen {
                self.decided[k] = true;
                connections[b.i] -= 1;
                connections[b.j] -= 1;
                self.valence_left[b.i] -= b.order;
                self.valence_left[b.j] -= b.order;
                self.bonds_left[b.i] = connections[b.i];
                self.bonds_left[b.j] = connections[b.j];
            } else {
                b.order = -1;
                self.decided[k] = false;
            }
        }
        self.induce();
        let mut failed = self.decided.iter().any(|d| !d) && self.iterate() > 0;
        if (0..n).any(|a| self.valence_left[a] != 0 || self.bonds_left[a] != 0) {
            failed = true;
        }
        !failed
    }

    /// `jbo_induce`: settle every forced bond, sweeping until a sweep settles
    /// nothing.
    fn induce(&mut self) {
        loop {
            let mut settled = 0;
            for i in 0..self.con.len() {
                let forced = (self.bonds_left[i] == 1 && self.valence_left[i] > 0)
                    || (self.bonds_left[i] == self.valence_left[i] && self.valence_left[i] != 0);
                if !forced {
                    continue;
                }
                for n in 0..self.con[i].len() {
                    let other = self.con[i][n];
                    // The first undecided bond joining the two, in bond order.
                    let Some(&k) = self.incident[i].iter().find(|k| {
                        !self.decided[**k] && {
                            let b = &self.bonds[**k];
                            b.i == other || b.j == other
                        }
                    }) else {
                        continue;
                    };
                    let order = self.valence_left[i] / self.bonds_left[i];
                    self.bonds[k].order = order;
                    self.decided[k] = true;
                    self.valence_left[i] -= order;
                    self.valence_left[other] -= order;
                    self.bonds_left[i] -= 1;
                    self.bonds_left[other] -= 1;
                    settled += 1;
                }
            }
            if settled == 0 {
                return;
            }
        }
    }

    /// `jbo_score`: scored atoms whose remaining valence cannot be spent.
    fn violations(&self) -> usize {
        (0..self.con.len())
            .filter(|a| self.scored[*a])
            .filter(|a| {
                (self.bonds_left[*a] == 0 && self.valence_left[*a] != 0)
                    || self.bonds_left[*a] > self.valence_left[*a]
            })
            .count()
    }

    fn undecided(&self) -> bool {
        self.bonds.iter().any(|b| b.order == -1)
    }

    /// `jbo_iteration`: try the undecided bonds single, double, triple in bond
    /// order. Returns the violation count it ends on (0 = closed).
    fn iterate(&mut self) -> usize {
        let mut score = 0;
        while self.undecided() {
            // The snapshot every failed trial in this sweep returns to.
            let orders: Vec<i32> = self.bonds.iter().map(|b| b.order).collect();
            let decided = self.decided.clone();
            let valence = self.valence_left.clone();
            let left = self.bonds_left.clone();
            let restore = |me: &mut Self| {
                for (b, o) in me.bonds.iter_mut().zip(&orders) {
                    b.order = *o;
                }
                me.decided.clone_from(&decided);
                me.valence_left.clone_from(&valence);
                me.bonds_left.clone_from(&left);
            };
            for k in 0..self.bonds.len() {
                if self.decided[k] {
                    continue;
                }
                let (i, j) = (self.bonds[k].i, self.bonds[k].j);
                for order in 1..=3 {
                    if order > 1 {
                        restore(self);
                    }
                    self.valence_left[i] -= order;
                    self.valence_left[j] -= order;
                    self.bonds_left[i] -= 1;
                    self.bonds_left[j] -= 1;
                    self.bonds[k].order = order;
                    self.decided[k] = true;
                    self.induce();
                    score = self.violations();
                    if score == 0 && !self.undecided() {
                        return 0;
                    }
                    if score == 0 {
                        break;
                    }
                    if order == 3 {
                        // Give up on this state: orders back, valences as left.
                        for (b, o) in self.bonds.iter_mut().zip(&orders) {
                            b.order = *o;
                        }
                        self.decided.clone_from(&decided);
                        return score;
                    }
                }
            }
        }
        score
    }
}

/// The visitor [`penalty_states`] hands each valence state to.
type Visit<'a> = dyn FnMut(&[(usize, usize)]) -> bool + 'a;

/// `ncsu_penalties` for every `goal`: each way to pick, from sets `1 … n` of
/// sizes `sizes`, elements whose set numbers sum to `goal` — the valence
/// states of total penalty `goal`, in `bondtype`'s order. `visit` gets each as
/// `(set index, element index)` pairs and returns `true` to stop.
fn penalty_states(n: usize, sizes: &[usize], goal: usize, visit: &mut Visit<'_>) -> bool {
    let mut partitions = IntPartition::new(n, goal);
    let mut combinations: Vec<Combination> = (0..n).map(|_| Combination::default()).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    while let Some(tpl) = partitions.next_partition() {
        if (0..goal).any(|i| tpl[i] > sizes[i]) {
            continue;
        }
        let mut last = 0;
        for i in 0..goal {
            if tpl[i] != 0 {
                combinations[i].setup(sizes[i], tpl[i]);
                combinations[i].next_combination();
                last = i;
            }
        }
        loop {
            pairs.clear();
            for i in 0..goal {
                if tpl[i] != 0 {
                    for l in 0..combinations[i].k {
                        pairs.push((i, combinations[i].c[l]));
                    }
                }
            }
            if visit(&pairs) {
                return true;
            }
            let mut i = 0;
            while i < goal {
                if tpl[i] != 0 {
                    if combinations[i].next_combination() {
                        break;
                    }
                    if i == last {
                        break;
                    }
                    combinations[i].next_combination();
                }
                i += 1;
            }
            if i == last && combinations[i].is_null() {
                break;
            }
        }
    }
    false
}

/// `combination_t`: the `k`-combinations of `0 … n-1` in lexicographic order.
#[derive(Default)]
struct Combination {
    n: usize,
    k: usize,
    c: Vec<usize>,
}

impl Combination {
    fn setup(&mut self, n: usize, k: usize) {
        self.n = n;
        self.k = k;
        self.c = vec![n; k];
    }

    fn is_null(&self) -> bool {
        self.c[0] == self.n
    }

    /// Advance; `false` (and back to null) after the last combination.
    fn next_combination(&mut self) -> bool {
        if self.is_null() {
            for (i, c) in self.c.iter_mut().enumerate() {
                *c = i;
            }
            return true;
        }
        // The rightmost position not at its maximum moves up one, and every
        // position after it follows consecutively.
        let k = self.k;
        let Some(r) = (0..k).rev().find(|r| self.c[*r] != self.n - k + r) else {
            self.c[0] = self.n;
            return false;
        };
        self.c[r] += 1;
        for i in r + 1..k {
            self.c[i] = self.c[i - 1] + 1;
        }
        true
    }
}

/// `int_part_t`: the solutions of `c[0]·1 + … + c[K-1]·K = N` in part-count
/// form, in the order of Stockmal's Algorithm 95 (CACM 5(6), 1962).
struct IntPartition {
    k: usize,
    n: usize,
    c: Vec<usize>,
}

impl IntPartition {
    fn new(k: usize, n: usize) -> Self {
        let mut c = vec![0; k];
        c[0] = n + 1;
        Self { k, n, c }
    }

    fn next_partition(&mut self) -> Option<&[usize]> {
        if self.k == 1 && self.c[0] == self.n {
            self.c[0] = self.n + 1;
            return None;
        }
        if self.c[0] == self.n + 1 {
            // The first partition: N·1.
            self.c[0] = self.n;
            for i in 1..self.k {
                self.c[i] = 0;
            }
            return Some(&self.c);
        }
        let mut j = 1;
        let mut a = self.c[0];
        loop {
            if a > j {
                self.c[j] += 1;
                self.c[0] = a - j - 1;
                for i in 1..j {
                    self.c[i] = 0;
                }
                return Some(&self.c);
            }
            if j + 1 == self.k {
                self.c[0] = self.n + 1;
                return None;
            }
            a += (j + 1) * self.c[j];
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondOrder, Element};
    use crate::props;

    /// A molecule as a mol2 file lists it: atoms by element, bonds in file
    /// order, every bond stated single (what `antechamber` reads).
    fn mol2(elements: &[Element], bonds: &[(usize, usize)]) -> Atomistic {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = elements
            .iter()
            .map(|e| mol.add_atom(props! { keys::ATOMIC_NUMBER => *e }).unwrap())
            .collect();
        for (i, j) in bonds {
            // `add_bond` states a bond with no `bond_type` single.
            mol.add_bond(ids[*i], ids[*j], props! {}).unwrap();
        }
        mol
    }

    /// The heavy-atom part of `perceive_bond_orders`: the first `n` bonds.
    fn judged(mol: &Atomistic, n: usize) -> Vec<Option<u8>> {
        perceive_bond_orders(mol)[..n].to_vec()
    }

    fn all(orders: &[u8]) -> Vec<Option<u8>> {
        orders.iter().map(|o| Some(*o)).collect()
    }

    /// Azulene as the GAFF benchmark's mol2 lists it. `bondtype -j full`
    /// (AmberTools 26.1) gives 1 2 1 8 7 8 7 7 2 1 2 — single / double once
    /// the five-ring's aromatic 7 / 8 are read back as 1 / 2.
    fn azulene() -> Atomistic {
        let mut elements = vec![Element::C; 10];
        elements.extend([Element::H; 8]);
        mol2(
            &elements,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (3, 7),
                (7, 8),
                (8, 9),
                (0, 9),
                (0, 10),
                (1, 11),
                (2, 12),
                (4, 13),
                (5, 14),
                (6, 15),
                (8, 16),
                (9, 17),
            ],
        )
    }

    /// Cyclooctatetraene as the benchmark lists it; `bondtype` puts the first
    /// double bond on C2–C3, not on C1–C2 where the SMILES `C1=CC=CC=CC=C1`
    /// drew it.
    fn cyclooctatetraene() -> Atomistic {
        let mut elements = vec![Element::C; 8];
        elements.extend([Element::H; 8]);
        mol2(
            &elements,
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (0, 7),
                (0, 8),
                (1, 9),
                (2, 10),
                (3, 11),
                (4, 12),
                (5, 13),
                (6, 14),
                (7, 15),
            ],
        )
    }

    #[test]
    fn azulene_takes_antechambers_kekule_structure() {
        assert_eq!(
            judged(&azulene(), 11),
            all(&[1, 2, 1, 2, 1, 2, 1, 1, 2, 1, 2])
        );
    }

    #[test]
    fn cyclooctatetraene_takes_antechambers_kekule_structure() {
        assert_eq!(
            judged(&cyclooctatetraene(), 8),
            all(&[1, 2, 1, 2, 1, 2, 1, 2])
        );
    }

    #[test]
    fn the_input_orders_do_not_move_the_answer() {
        // The other Kekulé structure, stated: `bondtype -j full` ignores it.
        let mut mol = cyclooctatetraene();
        let bonds: Vec<_> = mol.bonds().map(|(b, _)| b).collect();
        for (k, b) in bonds.iter().take(8).enumerate() {
            let (t, n) = if k % 2 == 0 {
                (BondOrder::Double, BondNumber::Double)
            } else {
                (BondOrder::Single, BondNumber::Single)
            };
            mol.set_bond_class(*b, t, n).unwrap();
        }
        assert_eq!(judged(&mol, 8), all(&[1, 2, 1, 2, 1, 2, 1, 2]));
    }

    #[test]
    fn methyl_azide_closes_in_a_raised_valence_state() {
        // No structure closes at the best valences; `bondtype` reports
        // "valence state (4) with penalty (4)": C-N-N#N.
        let mol = mol2(
            &[
                Element::C,
                Element::N,
                Element::N,
                Element::N,
                Element::H,
                Element::H,
                Element::H,
            ],
            &[(0, 1), (1, 2), (2, 3), (0, 4), (0, 5), (0, 6)],
        );
        assert_eq!(perceive_bond_orders(&mol), all(&[1, 1, 3, 1, 1, 1]));
    }

    #[test]
    fn tropylium_closes_in_no_valence_state() {
        // Seven trivalent ring carbons cannot share out three double bonds;
        // `bondtype` warns "the assigned bond types may be wrong".
        let mut elements = vec![Element::C; 7];
        elements.extend([Element::H; 7]);
        let mut bonds: Vec<(usize, usize)> = (0..6).map(|i| (i, i + 1)).collect();
        bonds.push((0, 6));
        bonds.extend((0..7).map(|i| (i, i + 7)));
        let mol = mol2(&elements, &bonds);
        assert!(perceive_bond_orders(&mol).iter().all(Option::is_none));
    }

    #[test]
    fn a_bond_between_residues_is_single_and_each_side_is_capped() {
        // Butadiene cut between its two residues: each half judges as a vinyl
        // group capped by a hydrogen, so both C=C survive.
        let mut mol = mol2(
            &[
                Element::C,
                Element::C,
                Element::C,
                Element::C,
                Element::H,
                Element::H,
                Element::H,
                Element::H,
                Element::H,
                Element::H,
            ],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (0, 4),
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (3, 9),
            ],
        );
        let ids: Vec<NodeId> = mol.atoms().map(|(a, _)| a).collect();
        for (i, res) in [1, 1, 2, 2, 1, 1, 1, 2, 2, 2].into_iter().enumerate() {
            mol.set_atom(ids[i], keys::RES_ID, PropValue::Int(res))
                .unwrap();
        }
        assert_eq!(judged(&mol, 3), all(&[2, 1, 2]));
    }

    #[test]
    fn find_bond_orders_writes_the_judged_structure() {
        let mut out = azulene();
        assign_bond_orders(&mut out).unwrap();
        let numbers: Vec<u32> = out
            .bonds()
            .map(|(b, _)| out.bond_number(b).unwrap().code())
            .take(3)
            .collect();
        assert_eq!(numbers, vec![1, 2, 1]);
        let (b, _) = out.bonds().nth(1).unwrap();
        assert_eq!(out.bond_type(b).unwrap(), BondOrder::Double);
    }

    #[test]
    fn valence_states_are_enumerated_as_ncsu_penalties_does() {
        // `ncsu-penalties.h`'s own example: N = 3, sizes (2, 3, 1), goal 2.
        let mut seen = Vec::new();
        penalty_states(3, &[2, 3, 1], 2, &mut |pairs| {
            seen.push(pairs.to_vec());
            false
        });
        assert_eq!(
            seen,
            vec![
                vec![(0, 0), (0, 1)],
                vec![(1, 0)],
                vec![(1, 1)],
                vec![(1, 2)],
            ]
        );
    }
}
