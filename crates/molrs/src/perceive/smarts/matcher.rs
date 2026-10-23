//! Backtracking subgraph-isomorphism matcher (non-uniquified), with
//! recursive-SMARTS evaluation.
//!
//! Ported (semantics only) from RDKit's substructure matcher under BSD-3:
//! `Code/GraphMol/Substruct/SubstructMatch.cpp`. Match semantics follow
//! `GetSubstructMatches(uniquify=False)`: every distinct query-atom →
//! mol-atom embedding is reported once, ordered by query-atom index, and the
//! enumeration order tracks RDKit's (query atom 0 outer loop, depth-first).
//!
//! The algorithm is a straightforward depth-first backtracking VF2-style
//! match. The query is assumed connected (true for all ETKDG torsion
//! patterns); each query atom after the first connects to an already-placed
//! query atom, so candidates are generated from the neighbourhood of the
//! anchor's image.

use crate::core::Atomistic;

use crate::core::NeighborKinds;
use crate::core::NodeId;

use super::compile::QueryGraph;
use super::predicate::{BondFacts, RecursiveEval, SmartsTarget};
use super::{MatchOptions, SmartsMatch};

/// Resolve bond facts between two molecule atoms, if they are bonded.
fn bond_facts(ctx: &SmartsTarget, a: NodeId, b: NodeId) -> Option<BondFacts> {
    let mol = ctx.mol;
    for (other, bid) in mol.neighbors(a, NeighborKinds::BONDS) {
        if other == b {
            // §11.2: `:` reads the bond *class*. §11.3: an explicit `-`/`=`/`#`
            // reads the localized integer. Two questions, two fields — never a
            // number doing both jobs.
            const LIVE: &str = "neighbors yields live bonds";
            let aromatic = mol.bond_type(bid).expect(LIVE).is_aromatic();
            let order = mol.bond_number(bid).expect(LIVE).count().max(1) as f64;
            let in_ring = ctx.rings.is_bond_in_ring(bid);
            return Some(BondFacts {
                order,
                aromatic,
                in_ring,
            });
        }
    }
    None
}

/// Adapter giving the AST a way to evaluate recursive `$(...)` subpatterns.
struct RecursiveEvaluator<'g> {
    recursives: &'g [QueryGraph],
}

impl RecursiveEval for RecursiveEvaluator<'_> {
    fn eval_recursive(&self, sub_index: usize, ctx: &SmartsTarget, id: NodeId) -> bool {
        let sub = &self.recursives[sub_index];
        // The recursive subpattern matches iff it has at least one embedding
        // whose first query atom maps to `id` (RDKit roots `$(...)` at the
        // candidate atom).
        let mut found = false;
        enumerate_matches(sub, ctx, Some(id), &mut |_assign| {
            found = true;
            false // stop at the first hit
        });
        found
    }
}

/// Enumerate all embeddings of `query` into the molecule described by `ctx`.
///
/// `root_fix`, when `Some(id)`, constrains query atom 0 to map to `id`
/// (used for recursive SMARTS). `visit` is called for every complete match
/// with the assignment vector (indexed by query-atom); returning `false`
/// stops the enumeration early.
///
/// Query atoms are placed in index order, so "already placed" is "lower
/// index": the anchor of atom `q` is its lowest-indexed neighbour below `q`,
/// and the mol atoms in use are exactly `assign[..depth]`. Neither needs a
/// map, and the search allocates only the two vectors below, once.
fn enumerate_matches(
    query: &QueryGraph,
    ctx: &SmartsTarget,
    root_fix: Option<NodeId>,
    visit: &mut dyn FnMut(&[NodeId]) -> bool,
) {
    let n = query.atoms.len();
    if n == 0 {
        return;
    }
    let rec = RecursiveEvaluator {
        recursives: &query.recursives,
    };
    let mut assign: Vec<Option<NodeId>> = vec![None; n];
    let mut full: Vec<NodeId> = Vec::with_capacity(n);
    backtrack(query, ctx, &rec, root_fix, 0, &mut assign, &mut full, visit);
}

/// The earliest-placed query atom `qa` is bonded to, i.e. its lowest-indexed
/// neighbour below `qa`. A compiled query is one connected pattern numbered
/// in writing order, so every non-root atom has one.
fn anchor_of(query: &QueryGraph, qa: usize) -> Option<usize> {
    query
        .bonds
        .iter()
        .filter_map(|b| {
            let other = if b.a == qa {
                b.b
            } else if b.b == qa {
                b.a
            } else {
                return None;
            };
            (other < qa).then_some(other)
        })
        .min()
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    query: &QueryGraph,
    ctx: &SmartsTarget,
    rec: &dyn RecursiveEval,
    root_fix: Option<NodeId>,
    depth: usize,
    assign: &mut [Option<NodeId>],
    full: &mut Vec<NodeId>,
    visit: &mut dyn FnMut(&[NodeId]) -> bool,
) -> bool {
    if depth == query.atoms.len() {
        full.clear();
        full.extend(
            assign
                .iter()
                .map(|a| a.expect("every query atom is placed at a leaf")),
        );
        return visit(full);
    }

    match (depth, root_fix) {
        (0, Some(id)) => try_place(query, ctx, rec, root_fix, depth, id, assign, full, visit),
        (0, None) => {
            for (id, _) in ctx.mol.atoms() {
                if !try_place(query, ctx, rec, root_fix, depth, id, assign, full, visit) {
                    return false;
                }
            }
            true
        }
        _ => {
            // Candidates come from the anchor's image neighbourhood.
            let anchor = anchor_of(query, depth).expect("non-root query atom must have an anchor");
            let anchor_img = assign[anchor].expect("anchor must be assigned");
            // Bonds only: a port (or any other 2-ary kind) is not a bond.
            for (cand, _) in ctx.mol.neighbors(anchor_img, NeighborKinds::BONDS) {
                if !try_place(query, ctx, rec, root_fix, depth, cand, assign, full, visit) {
                    return false;
                }
            }
            true
        }
    }
}

/// Try `cand` as the image of query atom `depth`, recursing on success.
/// Returns `false` only when the visitor asked to stop.
#[allow(clippy::too_many_arguments)]
fn try_place(
    query: &QueryGraph,
    ctx: &SmartsTarget,
    rec: &dyn RecursiveEval,
    root_fix: Option<NodeId>,
    depth: usize,
    cand: NodeId,
    assign: &mut [Option<NodeId>],
    full: &mut Vec<NodeId>,
    visit: &mut dyn FnMut(&[NodeId]) -> bool,
) -> bool {
    if assign[..depth].contains(&Some(cand)) {
        return true;
    }
    // Atom primitive must match.
    if !query.atoms[depth].query.eval(ctx, cand, rec) {
        return true;
    }
    // Every query bond from this atom to an already-placed atom must be
    // satisfied by an actual molecule bond matching the bond query.
    for qb in &query.bonds {
        let other = if qb.a == depth {
            qb.b
        } else if qb.b == depth {
            qb.a
        } else {
            continue;
        };
        if other >= depth {
            continue;
        }
        let other_img = assign[other].expect("earlier query atoms are placed");
        match bond_facts(ctx, cand, other_img) {
            Some(facts) if qb.query.eval(&facts) => {}
            _ => return true,
        }
    }

    assign[depth] = Some(cand);
    let keep_going = backtrack(query, ctx, rec, root_fix, depth + 1, assign, full, visit);
    assign[depth] = None;
    keep_going
}

/// Find every non-uniquified embedding of `query` in `mol`.
pub fn find(query: &QueryGraph, mol: &Atomistic, options: MatchOptions<'_>) -> Vec<SmartsMatch> {
    let ctx = match options.labels {
        Some(labels) => SmartsTarget::new(mol).with_labels(labels),
        None => SmartsTarget::new(mol),
    };
    find_in_target(query, &ctx, options.root, options.limit)
}

/// Match against a target already perceived for this molecular graph.
pub(crate) fn find_in_target(
    query: &QueryGraph,
    ctx: &SmartsTarget<'_>,
    root: Option<NodeId>,
    limit: Option<usize>,
) -> Vec<SmartsMatch> {
    let Some(limit) = limit else {
        let mut out = Vec::new();
        enumerate_matches(query, ctx, root, &mut |assign| {
            out.push(SmartsMatch {
                atoms: assign.to_vec(),
            });
            true
        });
        return out;
    };
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    enumerate_matches(query, ctx, root, &mut |assign| {
        out.push(SmartsMatch {
            atoms: assign.to_vec(),
        });
        out.len() < limit
    });
    out
}

/// Whether at least one embedding exists.
pub fn has_match(query: &QueryGraph, mol: &Atomistic, mut options: MatchOptions<'_>) -> bool {
    options.limit = Some(1);
    !find(query, mol, options).is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondNumber, BondOrder, Element, keys};
    use crate::perceive::smarts::MatchOptions;
    use crate::props;

    /// Ethanol without hydrogens: C0–C1–O2, single bonds.
    fn ethanol() -> Atomistic {
        let mut mol = Atomistic::new();
        let c0 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c1 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        mol.add_bond(c0, c1, props! {}).unwrap();
        mol.add_bond(c1, o, props! {}).unwrap();
        mol
    }

    fn matches(smarts: &str, mol: &Atomistic) -> Vec<Vec<usize>> {
        let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        let q = super::super::compile::compile(smarts).unwrap();
        find(&q, mol, MatchOptions::default())
            .into_iter()
            .map(|m| {
                m.atoms
                    .iter()
                    .map(|a| ids.iter().position(|id| id == a).unwrap())
                    .collect()
            })
            .collect()
    }

    /// A port records a joining site on a real bond; it is not a bond. The
    /// hydroxyl O of `C–C–O–H` with a port on its O–H valence is still
    /// two-connected with one hydrogen, and the H still one-connected.
    #[test]
    fn a_port_is_not_counted_as_a_bond() {
        use crate::core::PortKind;
        let mut mol = ethanol();
        let o = mol.atoms().map(|(id, _)| id).nth(2).expect("O");
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(o, h, props! {}).unwrap();
        mol.add_port(o, h, PortKind::Left, "", BondNumber::Single)
            .expect("a bonded H is a legal handle");

        assert_eq!(matches("[OX2H1]", &mol), vec![vec![2]]);
        assert_eq!(matches("[HX1]O", &mol), vec![vec![3, 2]]);
        assert_eq!(
            matches("OC", &mol),
            vec![vec![2, 1]],
            "no path through the port"
        );
    }

    /// `H<n>` and `X<n>` count the hydrogens a SMILES leaves implicit as
    /// they count drawn ones: ethanol read from `CCO` matches what its
    /// explicit-hydrogen drawing matches.
    #[test]
    fn hydrogen_counts_include_implicit_hydrogens() {
        let mol = crate::io::smiles::SmilesIr::parse("CCO")
            .expect("parse")
            .to_atomistic()
            .expect("to_atomistic");
        assert_eq!(mol.n_atoms(), 3, "the hydrogens stay implicit");
        assert_eq!(matches("[OH]", &mol), vec![vec![2]]);
        assert_eq!(matches("[OX2H1]", &mol), vec![vec![2]]);
        assert_eq!(matches("[CX4H3]", &mol), vec![vec![0]]);
        assert_eq!(matches("[CX4H2]", &mol), vec![vec![1]]);
        assert_eq!(
            matches("[CD1]", &mol),
            vec![vec![0]],
            "D stays the drawn degree"
        );
    }

    #[test]
    fn every_embedding_is_reported_in_query_atom_order() {
        let mol = ethanol();
        // C–C matches both directions; C–O only one way round.
        assert_eq!(matches("CC", &mol), vec![vec![0, 1], vec![1, 0]]);
        assert_eq!(matches("CO", &mol), vec![vec![1, 2]]);
        assert_eq!(matches("CCO", &mol), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn an_atom_is_never_used_twice_in_one_embedding() {
        let mol = ethanol();
        // A three-membered ring cannot embed into a chain.
        assert!(matches("C1CO1", &mol).is_empty());
    }

    #[test]
    fn bond_primitives_are_checked_against_the_molecule() {
        let mut mol = ethanol();
        let bonds: Vec<_> = mol.bonds().map(|(id, _)| id).collect();
        mol.set_bond_class(bonds[1], BondOrder::Double, BondNumber::Double)
            .unwrap();
        assert_eq!(matches("C=O", &mol).len(), 1);
        assert!(matches("C-O", &mol).is_empty());
        assert_eq!(matches("C~O", &mol).len(), 1, "`~` is any bond");
    }

    /// `^<n>` reads the perceived hybridization: acetaldehyde's carbonyl
    /// carbon and oxygen are sp², its methyl carbon sp³.
    #[test]
    fn hybridization_primitives_read_the_perceived_hybridization() {
        let mut mol = ethanol();
        let bonds: Vec<_> = mol.bonds().map(|(id, _)| id).collect();
        mol.set_bond_class(bonds[1], BondOrder::Double, BondNumber::Double)
            .unwrap();
        assert_eq!(matches("[C^2]", &mol), [[1]]);
        assert_eq!(matches("[C^3]", &mol), [[0]]);
        assert_eq!(matches("[O^2]", &mol), [[2]]);
        assert!(matches("[*^1]", &mol).is_empty());
        assert!(super::super::compile::compile("[C^7]").is_err());
    }

    /// Outside brackets, `a` is any aromatic atom and `A` any aliphatic one,
    /// as inside them.
    #[test]
    fn bare_aromatic_and_aliphatic_atoms_are_the_bracket_primitives() {
        let mol = ethanol();
        assert_eq!(matches("A", &mol).len(), 3);
        assert!(matches("a", &mol).is_empty());
        assert_eq!(matches("O~A", &mol), matches("O~[A]", &mol));
    }

    #[test]
    fn a_root_pin_and_a_limit_narrow_the_enumeration() {
        let mol = ethanol();
        let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        let q = super::super::compile::compile("C").unwrap();
        let rooted = find(
            &q,
            &mol,
            MatchOptions {
                root: Some(ids[1]),
                ..MatchOptions::default()
            },
        );
        assert_eq!(rooted.len(), 1);
        assert_eq!(rooted[0].atoms, vec![ids[1]]);
        let limited = find(
            &q,
            &mol,
            MatchOptions {
                limit: Some(1),
                ..MatchOptions::default()
            },
        );
        assert_eq!(limited.len(), 1);
        assert!(has_match(&q, &mol, MatchOptions::default()));
    }

    #[test]
    fn a_recursive_primitive_is_rooted_at_the_candidate() {
        let mol = ethanol();
        // The carbon bonded to oxygen, and only that one.
        assert_eq!(matches("[C;$(CO)]", &mol), vec![vec![1]]);
    }
}
