//! Kekulé assignment: a legal localized bond number for every aromatic bond.

use std::collections::HashMap;

use super::bcc_bond_class::{
    AROMATIC_DOUBLE, AROMATIC_SINGLE, AROMATIC_UNRESOLVED, SINGLE, TRIPLE,
};
use crate::core::Atomistic;
use crate::core::MolRsError;
use crate::core::keys;
use crate::core::keys::BCC_BOND_TYPE;
use crate::core::{BondNumber, BondOrder};
use crate::core::{NodeId, RelationId};
use crate::core::{PropValue, Props};
use crate::perceive::perceive_rings;

use super::valence::atomic_number;

/// A valence-state penalty that is never worth paying: used for the valences an
/// element's `APS.DAT` row leaves blank (`*`).
const FORBIDDEN: u32 = 1000;

/// Assign a legal localized [`BondNumber`] to every aromatic bond, in place.
///
/// An aromatic ring states no single/double phase of its own; a Kekulé
/// structure is the matching that gives each aromatic atom at most one double
/// bond while every valence stays legal. This writes one; the search minimises
/// AmberTools' `APS.DAT` valence-state penalty, so the BCC bond typing of
/// [`assign_bcc_bond_types`](crate::perceive::assign_bcc_bond_types), which
/// types an aromatic bond by its Kekulé phase, reads the same structure.
///
/// This is the Kekulé standardization the standard defines: **ensure every
/// aromatic subgraph carries a legal localized integer**. It **only**
/// kekulizes: it does not perceive aromaticity, so a molecule whose aromatic
/// bonds are not yet marked is left unchanged. That is the division of labour
/// — perception decides *which* bonds are aromatic, this decides *what phase*
/// they take, and neither calls the other.
///
/// The guarantees, and where each comes from:
///
/// * **Local** — only bonds whose class is `Aromatic` are written. A
///   non-aromatic bond is not read for a decision and not touched.
/// * **Conservative** — an aromatic bond that already states a number keeps it
///   when the whole system is already consistent; nothing is renumbered for its
///   own sake.
/// * **Complete** — on success every aromatic bond has `Single` or `Double`.
/// * **Deterministic** — degenerate assignments are real (benzene has two), so
///   the tie-break is part of the answer. It is keyed on **atom** order alone,
///   so permuting bonds or swapping endpoints cannot move it.
/// * **Transactional** — a system with no legal assignment leaves *no* bond
///   changed, rather than a ring half-assigned. Returns `false` in that case.
///
/// Returns whether every aromatic bond came out with a legal number.
///
/// # Errors
///
/// [`MolRsError`] when a chosen number cannot be written (a bond column
/// holding the key under a foreign dtype).
pub fn assign_kekule_bond_orders(mol: &mut Atomistic) -> Result<bool, MolRsError> {
    // Nothing to assign without an aromatic bond. Checked on the bonds
    // directly: `BondGraph::new` perceives rings and implicit hydrogens for
    // every atom, which is the whole cost of this call.
    if !has_aromatic_marking(mol) {
        return Ok(true);
    }
    let graph = BondGraph::new(mol);

    // Conservative: a legal assignment already on the graph is *the* answer.
    // Re-deriving one would rewrite the phase an input stated for itself, so a
    // file that round-trips would come back with its bonds renumbered.
    let numbers: Vec<BondNumber> = graph
        .bond_ids
        .iter()
        .map(|bid| mol.bond_number(*bid))
        .collect::<Result<_, MolRsError>>()?;
    let stated: Vec<bool> = numbers
        .iter()
        .enumerate()
        .map(|(k, number)| graph.aromatic[k] && *number == BondNumber::Double)
        .collect();
    let all_stated = numbers
        .iter()
        .enumerate()
        .filter(|(k, _)| graph.aromatic[*k])
        .all(|(_, number)| *number != BondNumber::Unknown);
    if all_stated && valences_are_satisfiable(&graph, &stated) {
        return Ok(true);
    }

    let doubles = kekulize(&graph);

    // Transactionality: decide first, write second. An aromatic atom that ends
    // up in no double bond and had no lone pair to give is the signal that no
    // legal assignment exists — the search returns its least-bad answer, and
    // only a completeness check can tell the two apart.
    let mut assignment: Vec<(RelationId, BondNumber)> = Vec::new();
    for (k, bid) in graph.bond_ids.iter().enumerate() {
        if !graph.aromatic[k] {
            continue;
        }
        assignment.push((
            *bid,
            if doubles[k] {
                BondNumber::Double
            } else {
                BondNumber::Single
            },
        ));
    }
    if !valences_are_satisfiable(&graph, &doubles) {
        return Ok(false);
    }

    for (bid, number) in assignment {
        let class = mol.bond_type(bid)?;
        mol.set_bond_class(bid, class, number)?;
    }
    Ok(true)
}

/// Does every aromatic atom end this assignment with a satisfiable valence?
///
/// The kekulizer minimises a valence-state penalty rather than failing, so a
/// ring with no legal assignment still comes back with *an* answer. This is the
/// check that tells "the best assignment" from "a legal one": an aromatic atom
/// whose element demands a π bond, that got none, and whose penalty says the
/// resulting valence is forbidden.
fn valences_are_satisfiable(graph: &BondGraph, doubles: &[bool]) -> bool {
    // Same totals the search scored against: implicit hydrogens count toward
    // both, or this guard passes a structure the penalty would have rejected.
    let mut valence: Vec<i32> = graph.implicit_h.iter().map(|h| *h as i32).collect();
    let mut degree: Vec<usize> = graph.implicit_h.iter().map(|h| *h as usize).collect();
    for (k, (i, j)) in graph.ends.iter().copied().enumerate() {
        let w = if graph.aromatic[k] {
            if doubles[k] { 2 } else { 1 }
        } else {
            (graph.order[k].round() as i32).clamp(SINGLE, TRIPLE)
        };
        valence[i] += w;
        valence[j] += w;
        degree[i] += 1;
        degree[j] += 1;
    }
    graph
        .z
        .iter()
        .enumerate()
        .filter(|(i, _)| graph.aromatic_atom[*i])
        .all(|(i, _)| valence_penalty(graph.scored_as_z[i], degree[i], valence[i]) < FORBIDDEN)
}

/// Does any bond already claim to be aromatic?
///
/// Perception is only run when the answer is no; a graph that already carries
/// aromatic markings keeps them, so a caller's own aromaticity model wins over
/// ours.
pub(super) fn has_aromatic_marking(mol: &Atomistic) -> bool {
    mol.bonds().any(|(_, bond)| aromatic_marking(&bond.props))
}

/// Is this bond marked aromatic by any of the three markings we accept: an
/// `is_aromatic` prop, an `order` of 1.5, or a [`BCC_BOND_TYPE`] of 7/8/10 (the
/// aromatic single, the aromatic double, and the unresolved `ar` precursor)?
///
/// The last of the three is what makes perception idempotent, and what resolves a
/// caller-supplied type 10 into 7 or 8. It reads **our own** key: the bond's
/// `keys::TYPE` is the caller's, and a LAMMPS bond-type id that happened to be 7
/// must not make a bond aromatic.
fn aromatic_marking(props: &Props) -> bool {
    BondOrder::from_prop(props.get(keys::BOND_TYPE)).is_aromatic()
        || props
            .get(BCC_BOND_TYPE)
            .and_then(PropValue::as_f64)
            .is_some_and(|t| {
                let t = t.round() as i32;
                matches!(t, AROMATIC_SINGLE | AROMATIC_DOUBLE | AROMATIC_UNRESOLVED)
            })
}

/// The molecule flattened into the dense index space the rules are written in:
/// atom index `i` is the `i`-th atom of [`Atomistic::atoms`], bond index `k` the
/// `k`-th bond of [`Atomistic::bonds`].
///
/// antechamber's rules read `atomicnum` and `connum` (the *degree*, hydrogens
/// included) off exactly this shape, so building it once keeps the rule bodies a
/// faithful transcription rather than a translation.
pub(super) struct BondGraph {
    /// Per bond: its handle, in bond index order.
    pub(super) bond_ids: Vec<RelationId>,
    /// Bond handle -> bond index.
    pub(super) bond_index: HashMap<RelationId, usize>,
    /// Per bond: its two endpoint atom indices.
    pub(super) ends: Vec<(usize, usize)>,
    /// Per bond: the input bond order.
    pub(super) order: Vec<f64>,
    /// Per bond: whether the input marks it aromatic.
    pub(super) aromatic: Vec<bool>,
    /// Per atom: atomic number (0 for a wildcard atom).
    pub(super) z: Vec<u8>,
    /// Per atom: hydrogens implied by valence but not drawn in the graph.
    pub(super) implicit_h: Vec<u32>,
    /// Per atom: the element the π-demand penalty scores it as — the real
    /// element for a neutral atom, its isoelectronic neighbour for a charged
    /// one. Read *only* by the aromatic assignment's penalty.
    pub(super) scored_as_z: Vec<u8>,
    /// Per atom: neighbour atom indices.
    pub(super) adj: Vec<Vec<usize>>,
    /// Per atom: whether it carries an aromatic bond.
    pub(super) aromatic_atom: Vec<bool>,
    /// The SSSR rings, as atom indices.
    pub(super) rings: Vec<Vec<usize>>,
}

impl BondGraph {
    /// Flatten a molecule. A wildcard atom (`z = 0`) matches no rule; its bonds
    /// fall through to their plain bond order.
    pub(super) fn new(mol: &Atomistic) -> Self {
        // Atom index = the atom table's row, the order `atom_ids` yields.
        let atom_ids: Vec<NodeId> = mol.atom_ids().collect();
        let rows = mol.atom_table();

        let z: Vec<u8> = mol.atoms().map(|(_, atom)| atomic_number(&atom)).collect();

        // The element `APS.DAT` is *scored* as, for the π-demand search only.
        //
        // A formal charge changes how many bonds an atom wants, and the penalty
        // table is keyed on the element alone, so a charged atom is scored as
        // its isoelectronic neighbour: a carbocation like boron, a carbanion
        // like nitrogen. Without it tropylium's C+ is handed a double bond and
        // reaches valence 4, which the table accepts.
        //
        // This is a **scoring proxy inside the aromatic assignment**, nothing
        // more. It never leaves this struct, `z` above stays the real atomic
        // number, and no general valence or element semantics change. It should
        // become an explicit charged-aromatic-atom rule; scoring by proxy is
        // the shortest correct step, not the final shape.
        let implicit_h: Vec<u32> = atom_ids
            .iter()
            .map(|aid| crate::perceive::n_implicit_hydrogens(mol, *aid).unwrap_or(0))
            .collect();

        let scored_as_z: Vec<u8> = atom_ids
            .iter()
            .zip(&z)
            .map(|(aid, &zi)| {
                let charge = mol.get_atom(*aid).map_or(0, |atom| atom.formal_charge());
                let shifted = i32::from(zi) - charge;
                if zi == 0 || shifted <= 0 {
                    zi
                } else {
                    shifted.min(i32::from(u8::MAX)) as u8
                }
            })
            .collect();

        let n = atom_ids.len();
        let mut bond_ids = Vec::new();
        let mut ends = Vec::new();
        let mut order = Vec::new();
        let mut aromatic = Vec::new();
        let mut adj = vec![Vec::new(); n];
        let mut aromatic_atom = vec![false; n];

        for (bid, bond) in mol.bonds() {
            let (Some(i), Some(j)) = (rows.row(bond.nodes[0]), rows.row(bond.nodes[1])) else {
                continue;
            };
            let is_aromatic = aromatic_marking(&bond.props);
            // An aromatic bond may not have a number yet — that is what this
            // module is about to decide — so it enters the valence sum as the
            // one bond every bond is at least.
            let o = BondNumber::from_prop(bond.props.get(keys::BOND_NUMBER))
                .count()
                .max(1) as f64;

            adj[i].push(j);
            adj[j].push(i);
            if is_aromatic {
                aromatic_atom[i] = true;
                aromatic_atom[j] = true;
            }
            bond_ids.push(bid);
            ends.push((i, j));
            order.push(o);
            aromatic.push(is_aromatic);
        }

        let ring_set = perceive_rings(mol);
        let rings = ring_set
            .rings()
            .iter()
            .map(|ring| ring.iter().filter_map(|aid| rows.row(*aid)).collect())
            .collect();

        let bond_index = bond_ids.iter().enumerate().map(|(k, b)| (*b, k)).collect();
        Self {
            bond_ids,
            bond_index,
            ends,
            order,
            aromatic,
            z,
            implicit_h,
            scored_as_z,
            adj,
            aromatic_atom,
            rings,
        }
    }

    /// Number of bonds on an atom — antechamber's `connum`. Hydrogens count.
    /// Drawn neighbours — antechamber's `connum`.
    ///
    /// Deliberately *not* including implicit hydrogens: the `ATOMTYPE_*.DEF`
    /// rules are transcribed against this count, and an acetate written without
    /// its formal charge relies on the unfilled valence to read as a terminal
    /// oxygen rather than a hydroxyl.
    pub(super) fn degree(&self, atom: usize) -> usize {
        self.adj[atom].len()
    }

    /// Drawn neighbours **plus** the hydrogens the graph implies but does not
    /// draw — the count `valence_penalty` must see.
    ///
    /// The penalty is keyed on `(element, degree, valence)`, so a hydrogen the
    /// graph never drew is a bond the table never sees. Without it a
    /// pyrrole-type N and a pyridine-type N are the same atom to it — degree 2,
    /// valence 2 — imidazole's two candidate Kekulé structures tie, and the
    /// tie-break puts the double on the N already carrying a hydrogen.
    fn total_degree(&self, atom: usize) -> usize {
        self.adj[atom].len() + self.implicit_h[atom] as usize
    }
}

/// Derive a Kekulé structure for the aromatic subsystem: which aromatic bonds are
/// the double ones.
///
/// The aromatic bonds are the only unknowns — every other bond keeps its input
/// order — so this is a **matching** problem: each aromatic atom either takes one
/// double bond or none, and no two doubles may share an atom. Among the legal
/// matchings we minimise the total *valence-state penalty*, the same objective
/// AmberTools minimises (`APS.DAT`, transcribed in [`valence_penalty`]). That is
/// what makes pyridine's N take a double while pyrrole's N does not, and what keeps
/// thiophene's S out of the matching.
///
/// # Tie-break
///
/// Degenerate minima are real (benzene has two; imidazolium has two that type its
/// two nitrogens differently), so the tie-break is part of the answer, not an
/// implementation detail. Aromatic bonds are visited ordered by
/// `(min endpoint asc, max endpoint desc)`, doubles tried before singles, and the
/// first minimum-penalty solution wins. This is calibrated to reproduce
/// AmberTools25 on every aromatic molecule of the antechamber oracle
/// (benzene, toluene, phenol, aniline, fluorobenzene, pyridine, imidazole,
/// imidazolium, thiophene), and — unlike antechamber's own search — it depends on
/// **atom** order only, so permuting the input's bonds cannot move it.
pub(super) fn kekulize(graph: &BondGraph) -> Vec<bool> {
    let n_bonds = graph.ends.len();
    let mut doubles = vec![false; n_bonds];

    let mut aromatic: Vec<usize> = (0..n_bonds).filter(|k| graph.aromatic[*k]).collect();
    if aromatic.is_empty() {
        return doubles;
    }
    aromatic.sort_by_key(|k| {
        let (i, j) = graph.ends[*k];
        (i.min(j), std::cmp::Reverse(i.max(j)))
    });

    // sigma[a]: the bond-order sum already committed at atom `a` — every
    // non-aromatic bond at its input order, plus one per aromatic bond (each is at
    // least single). A matched atom ends one higher.
    // Seeded with the implicit hydrogens: a bond the graph never drew is still
    // a bond the atom's valence has spent, and `valence_penalty` reads the
    // total.
    let mut sigma: Vec<i32> = graph.implicit_h.iter().map(|h| *h as i32).collect();
    for (k, (i, j)) in graph.ends.iter().copied().enumerate() {
        let w = if graph.aromatic[k] {
            1
        } else {
            (graph.order[k].round() as i32).clamp(SINGLE, TRIPLE)
        };
        sigma[i] += w;
        sigma[j] += w;
    }

    // Atoms touched by the search, and how many of their aromatic bonds are still
    // undecided — an atom's penalty is payable as soon as that count hits zero.
    let mut pending = vec![0usize; graph.z.len()];
    for k in &aromatic {
        let (i, j) = graph.ends[*k];
        pending[i] += 1;
        pending[j] += 1;
    }

    let mut search = Kekule {
        graph,
        aromatic: &aromatic,
        sigma,
        pending,
        matched: vec![false; graph.z.len()],
        best: None,
        best_doubles: Vec::new(),
        chosen: Vec::new(),
    };
    search.walk(0, 0);

    for k in search.best_doubles {
        doubles[k] = true;
    }
    doubles
}

/// The backtracking state of [`kekulize`].
struct Kekule<'a> {
    /// The molecule being kekulized.
    graph: &'a BondGraph,
    /// Aromatic bond indices, in visit order.
    aromatic: &'a [usize],
    /// Per atom: bond-order sum already committed (see [`kekulize`]).
    sigma: Vec<i32>,
    /// Per atom: aromatic bonds not yet decided.
    pending: Vec<usize>,
    /// Per atom: already carries its double bond.
    matched: Vec<bool>,
    /// Penalty of the best complete assignment so far.
    best: Option<u32>,
    /// The double bonds of that assignment.
    best_doubles: Vec<usize>,
    /// The double bonds of the assignment under construction.
    chosen: Vec<usize>,
}

impl Kekule<'_> {
    /// Visit aromatic bond `pos`, having already paid `penalty`.
    ///
    /// An atom's valence penalty is charged the moment its last aromatic bond is
    /// decided, which lets a branch be abandoned as soon as it can no longer beat
    /// the incumbent. Pruning on `>=` (not `>`) is what makes the first
    /// minimum-penalty solution the winner and keeps the tie-break deterministic.
    fn walk(&mut self, pos: usize, penalty: u32) {
        if self.best.is_some_and(|best| penalty >= best) {
            return;
        }
        let Some(&k) = self.aromatic.get(pos) else {
            self.best = Some(penalty);
            self.best_doubles = self.chosen.clone();
            return;
        };
        let (i, j) = self.graph.ends[k];

        for as_double in [true, false] {
            if as_double && (self.matched[i] || self.matched[j]) {
                continue;
            }
            if as_double {
                self.matched[i] = true;
                self.matched[j] = true;
                self.chosen.push(k);
            }
            self.pending[i] -= 1;
            self.pending[j] -= 1;
            let settled = self.settle(i) + self.settle(j);

            self.walk(pos + 1, penalty.saturating_add(settled));

            self.pending[i] += 1;
            self.pending[j] += 1;
            if as_double {
                self.matched[i] = false;
                self.matched[j] = false;
                self.chosen.pop();
            }
        }
    }

    /// The penalty an atom owes now that a bond of its has been decided: zero until
    /// its last aromatic bond is placed, then the cost of the valence it landed on.
    fn settle(&self, atom: usize) -> u32 {
        if self.pending[atom] > 0 {
            return 0;
        }
        let valence = self.sigma[atom] + i32::from(self.matched[atom]);
        // `total_degree`, not `degree`: `sigma` is seeded with the implicit
        // hydrogens, so the degree handed to the same penalty must count them
        // too. The acceptance guard already scores this way — scoring the
        // search differently lets it pick a structure the guard then rejects.
        valence_penalty(
            self.graph.scored_as_z[atom],
            self.graph.total_degree(atom),
            valence,
        )
    }
}

/// The penalty of an atom sitting at a given total valence — AmberTools' `APS.DAT`,
/// restricted to the elements that can carry an aromatic bond.
///
/// The numbers are the table's, not ours: they are what rank pyrrole's
/// three-valent N (0) above its four-valent alternative (1), and what forbid
/// thiophene's S from reaching valence 3 (64). `FORBIDDEN` stands in for the
/// table's `*` — a valence the element does not have.
///
/// Only the generic rows are transcribed. The context-specific rows (`APSNO2`,
/// `APSN3+`, …) re-rank valences for centres that are *already* decided by their
/// substituents, and cannot change which Kekulé structure wins for any ring.
fn valence_penalty(z: u8, degree: usize, valence: i32) -> u32 {
    let row: &[(i32, u32)] = match (z, degree) {
        (6, 1) => &[(3, 1), (4, 0), (5, 32)],
        (6, _) => &[(2, 64), (3, 32), (4, 0), (5, 32), (6, 64)],
        (7, 1) => &[(2, 3), (3, 0), (4, 32)],
        (7, 2) => &[(2, 4), (3, 0), (4, 2)],
        (7, 3) => &[(2, 32), (3, 0), (4, 1), (5, 2)],
        (7, _) => &[(3, 64), (4, 0), (5, 64)],
        (8, 1) => &[(1, 1), (2, 0), (3, 64)],
        (8, _) => &[(1, 32), (2, 0), (3, 64)],
        (15, 1) => &[(2, 2), (3, 0), (4, 32)],
        (15, 2) => &[(2, 4), (3, 0), (4, 2)],
        (15, 3) => &[(2, 32), (3, 0), (4, 1), (5, 2)],
        (15, _) => &[(3, 64), (4, 1), (5, 0), (6, 32)],
        (16, 1 | 2) => &[(1, 2), (2, 0), (3, 64)],
        (16, 3) => &[(3, 1), (4, 0), (5, 2), (6, 2)],
        (16, _) => &[(4, 4), (5, 2), (6, 0)],
        // Everything else (H, halogens, metals, the wildcard) has one valence
        // and nothing to choose; it never constrains a matching.
        _ => return 0,
    };
    row.iter()
        .find(|(v, _)| *v == valence)
        .map_or(FORBIDDEN, |(_, penalty)| *penalty)
}

#[cfg(all(test, feature = "smiles"))]
mod tests {
    use super::*;
    use crate::io::smiles::SmilesIr;

    /// Imidazole exactly as the antechamber oracle states it (case `imidazole`,
    /// SMILES `c1cnc[nH]1`): ring C0 C1 N2 C3 N4 with **implicit** hydrogens —
    /// N2 carries none (pyridine-type), N4 carries one (pyrrole-type).
    ///
    /// The hydrogens must stay implicit: drawn explicitly, `total_degree` and
    /// `degree` agree and the tie-break under test cannot be reached.
    fn imidazole() -> Atomistic {
        SmilesIr::parse("c1cnc[nH]1")
            .expect("parse")
            .to_atomistic()
            .expect("to_atomistic")
    }

    /// Ring bonds as `((i, j), is_double)`, endpoints low-high, sorted.
    fn ring_doubles(mol: &Atomistic) -> Vec<((usize, usize), bool)> {
        let rows = mol.atom_table();
        let mut out: Vec<((usize, usize), bool)> = mol
            .bonds()
            .map(|(id, _)| {
                let ends = mol.get_bond(id).expect("endpoints").nodes;
                let (a, b) = (ends[0], ends[1]);
                let (a, b) = (rows.row(a).unwrap(), rows.row(b).unwrap());
                (
                    (a.min(b), a.max(b)),
                    mol.bond_number(id).unwrap() == BondNumber::Double,
                )
            })
            .collect();
        out.sort();
        out
    }

    #[test]
    fn the_fixture_keeps_its_hydrogens_implicit() {
        // Guards the two tests below: were `to_atomistic` to start drawing
        // hydrogens, `total_degree` would equal `degree` and they would pass
        // against either implementation without testing anything.
        let mol = imidazole();
        assert_eq!(mol.n_atoms(), 5, "hydrogens must not be drawn");
        let implicit: Vec<u32> = mol
            .atoms()
            .map(|(id, _)| crate::perceive::n_implicit_hydrogens(&mol, id).unwrap_or(0))
            .collect();
        assert_eq!(
            implicit,
            vec![1, 1, 0, 1, 1],
            "N2 pyridine-type, N4 pyrrole-type"
        );
    }

    #[test]
    fn the_search_and_the_acceptance_guard_count_degree_the_same_way() {
        // The guard in `kekule_is_legal` seeds its degree with the implicit
        // hydrogens ("implicit hydrogens count toward both, or this guard
        // passes a structure the penalty would have rejected"). `settle` scores
        // the search against the same penalty, so it must count them too —
        // that is what `total_degree` is, and wiring it is what keeps the two
        // sides of that comment true.
        //
        // Both feed `valence_penalty`, whose rows differ by degree: a
        // pyrrole-type N reads row (7, 2) on the drawn count and (7, 3) on the
        // total, which price valence 2 at 4 vs 32 and valence 5 at
        // FORBIDDEN vs 2.
        let mol = imidazole();
        let graph = BondGraph::new(&mol);
        let pyrrole_n = 4;
        assert_eq!(graph.degree(pyrrole_n), 2, "drawn neighbours only");
        assert_eq!(
            graph.total_degree(pyrrole_n),
            3,
            "drawn neighbours plus the implicit hydrogen"
        );
        assert_ne!(
            valence_penalty(7, graph.degree(pyrrole_n), 2),
            valence_penalty(7, graph.total_degree(pyrrole_n), 2),
            "the two counts must reach different penalty rows, or this is moot"
        );
    }

    #[test]
    fn imidazole_kekulizes_the_way_antechamber_does() {
        let mut mol = imidazole();
        assert!(assign_kekule_bond_orders(&mut mol).unwrap());
        // AmberTools25 oracle `bcc_bond_types`: (0,1) and (2,3) are type 8
        // (aromatic double); (1,2), (3,4), (4,0) are type 7 (aromatic single).
        assert_eq!(
            ring_doubles(&mol),
            vec![
                ((0, 1), true),
                ((0, 4), false),
                ((1, 2), false),
                ((2, 3), true),
                ((3, 4), false),
            ],
        );
    }

    #[test]
    fn the_pyrrole_nitrogen_takes_no_double() {
        let mut mol = imidazole();
        assert!(assign_kekule_bond_orders(&mut mol).unwrap());
        // N4 already spends a valence on an undrawn hydrogen, so a double here
        // makes it five-valent. This is what `BondGraph::total_degree` exists
        // for: with the drawn-neighbour `degree`, N4 and N2 both look like
        // (degree 2, valence 2) to `valence_penalty`, the two Kekule structures
        // tie, and the double lands on the nitrogen carrying the hydrogen.
        for ((i, j), is_double) in ring_doubles(&mol) {
            if i == 4 || j == 4 {
                assert!(!is_double, "double placed on the pyrrole N: ({i}, {j})");
            }
        }
    }

    #[test]
    fn a_ring_free_chain_keeps_every_bond_number() {
        // No aromatic bond, so nothing to assign: the call succeeds and every
        // localized number -- single, double and triple alike -- is the one
        // the input stated.
        let mut mol = SmilesIr::parse("C=CC#CCCCC")
            .expect("parse")
            .to_atomistic()
            .expect("to_atomistic");
        let numbers = |m: &Atomistic| -> Vec<BondNumber> {
            m.bonds()
                .map(|(id, _)| m.bond_number(id).unwrap())
                .collect()
        };
        let before = numbers(&mol);
        assert!(before.contains(&BondNumber::Double) && before.contains(&BondNumber::Triple));

        assert!(assign_kekule_bond_orders(&mut mol).unwrap());

        assert_eq!(numbers(&mol), before);
    }
}
