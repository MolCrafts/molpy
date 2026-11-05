//! Stored stereochemical reference orders and their rebasing during graph edits.
//!
//! A tetrahedral `"CW"` / `"CCW"` on an atom is the clock sense of its
//! [reference order](tetrahedral_reference), seen from the first site. An
//! `"E"` / `"Z"` on a double bond relates the [priority
//! substituent](priority_substituent) of each end. This module is the one
//! definition of both references and of the parity a reordering applies, so
//! perception, line notation, copying, joining and hydrogen operations cannot
//! drift apart on what a stored label means.

use std::collections::{HashMap, HashSet};

use crate::core::bond_order::carries_ez;
use crate::core::{Graph, GraphAccess, MolGraph, MolRsError, NodeId, RelationId, keys};

/// One site of the tetrahedral reference order.
///
/// [`TetrahedralSite::Hydrogen`] is the implicit hydrogen of a stated
/// `h_count` of 1, not an atom of the graph. An explicit hydrogen atom is
/// [`TetrahedralSite::Atom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TetrahedralSite {
    Atom(NodeId),
    Hydrogen,
}

/// The `h_count` an atom states, or `None` when the key is absent.
///
/// Absent is not zero. A bracket atom that omitted the count stores zero; an
/// organic-subset atom stores nothing.
pub(crate) fn stated_hydrogen_count(mol: &impl Graph, id: NodeId) -> Option<i32> {
    mol.graph()
        .get_node(id)
        .ok()
        .and_then(|atom| atom.get_int(keys::H_COUNT))
}

/// The order a tetrahedral `"CW"` / `"CCW"` is read against.
///
/// Four bond neighbors and a missing or zero `h_count` stand in bond
/// (adjacency) order. Three bond neighbors and `h_count` 1 put the implicit
/// hydrogen first. Any other count, including four neighbors together with a
/// non-zero `h_count`, is not a reference this function names.
pub(crate) fn tetrahedral_reference(mol: &impl Graph, id: NodeId) -> Option<[TetrahedralSite; 4]> {
    let graph = mol.graph();
    let bonds = graph.bonds_kind()?;
    let heavy: Vec<NodeId> = graph
        .neighbor_relations(id)
        .filter_map(|(kind, _, neighbor)| (kind == bonds).then_some(neighbor))
        .collect();
    match (heavy.len(), stated_hydrogen_count(mol, id)) {
        (4, None | Some(0)) => Some([
            TetrahedralSite::Atom(heavy[0]),
            TetrahedralSite::Atom(heavy[1]),
            TetrahedralSite::Atom(heavy[2]),
            TetrahedralSite::Atom(heavy[3]),
        ]),
        (3, Some(1)) => Some([
            TetrahedralSite::Hydrogen,
            TetrahedralSite::Atom(heavy[0]),
            TetrahedralSite::Atom(heavy[1]),
            TetrahedralSite::Atom(heavy[2]),
        ]),
        _ => None,
    }
}

/// A surviving alkene's geometrical reference before a graph edit.
pub(crate) struct AlkeneStereo {
    pub(crate) bond: RelationId,
    pub(crate) ends: [NodeId; 2],
    references: [NodeId; 2],
    opposite: bool,
}

impl AlkeneStereo {
    /// Capture E/Z bonds whose endpoints survive deletion. The reference
    /// includes atom-row ties, so all surviving alkenes are captured.
    pub(crate) fn capture(
        mol: &impl Graph,
        removed: &HashSet<NodeId>,
    ) -> Result<Vec<Self>, MolRsError> {
        let graph = mol.graph();
        let mut result = Vec::new();
        let Some(bonds) = graph.bonds_kind() else {
            return Ok(result);
        };
        for (bond, relation) in graph.relations(bonds) {
            let opposite = match relation.props.get_str(keys::STEREO) {
                Some("E") => true,
                Some("Z") => false,
                _ => continue,
            };
            let ends = [relation.nodes[0], relation.nodes[1]];
            if ends.iter().any(|end| removed.contains(end)) {
                continue;
            }
            if !carries_ez(&relation.props) {
                return Err(MolRsError::validation(format!(
                    "bond {} stereo: E/Z requires a double bond",
                    u64::from(bond)
                )));
            }
            let mut references = ends;
            for side in 0..2 {
                let subs = Self::substituents(graph, bond, ends[side], ends[1 - side])?;
                references[side] =
                    priority_substituent(graph, &subs).expect("validated nonempty list");
            }
            result.push(Self {
                bond,
                ends,
                references,
                opposite,
            });
        }
        Ok(result)
    }

    /// Put incoming atoms in the departing substituents' geometrical slots.
    pub(crate) fn substitute(
        &mut self,
        mol: &impl Graph,
        replacement: &HashMap<(NodeId, NodeId), NodeId>,
    ) -> Result<(), MolRsError> {
        let graph = mol.graph();
        for side in 0..2 {
            let subs = Self::substituents(graph, self.bond, self.ends[side], self.ends[1 - side])?;
            for sub in subs {
                if let Some(&incoming) = replacement.get(&(self.ends[side], sub))
                    && graph.get_node(incoming)?.atomic_number().is_none()
                {
                    return Err(MolRsError::validation(format!(
                        "bond {} stereo: substituents must carry atomic numbers",
                        u64::from(self.bond)
                    )));
                }
            }
            if let Some(&incoming) = replacement.get(&(self.ends[side], self.references[side])) {
                self.references[side] = incoming;
            }
        }
        Ok(())
    }

    /// Restate the saved geometry against the graph's current references.
    pub(crate) fn label_after(&self, mol: &impl Graph) -> Result<&'static str, MolRsError> {
        let graph = mol.graph();
        let mut opposite = self.opposite;
        for side in 0..2 {
            let subs = Self::substituents(graph, self.bond, self.ends[side], self.ends[1 - side])?;
            let current = priority_substituent(graph, &subs).expect("validated nonempty list");
            opposite ^= current != self.references[side];
        }
        Ok(if opposite { "E" } else { "Z" })
    }

    fn substituents(
        graph: &MolGraph,
        bond: RelationId,
        end: NodeId,
        other_end: NodeId,
    ) -> Result<Vec<NodeId>, MolRsError> {
        let bonds = graph.bonds_kind();
        let subs: Vec<_> = graph
            .neighbor_relations(end)
            .filter_map(|(kind, _, other)| {
                (Some(kind) == bonds && other != other_end).then_some(other)
            })
            .collect();
        if !(1..=2).contains(&subs.len()) || subs.len() == 2 && subs[0] == subs[1] {
            return Err(MolRsError::validation(format!(
                "bond {} stereo: E/Z requires one or two distinct explicit substituents on each end",
                u64::from(bond)
            )));
        }
        for &sub in &subs {
            if graph.get_node(sub)?.atomic_number().is_none() {
                return Err(MolRsError::validation(format!(
                    "bond {} stereo: substituents must carry atomic numbers",
                    u64::from(bond)
                )));
            }
        }
        Ok(subs)
    }
}

/// The E/Z reference substituent: highest atomic number, then earliest node
/// row. Callers supply live atoms with atomic numbers, including wildcard 0.
pub(crate) fn priority_substituent(mol: &impl Graph, subs: &[NodeId]) -> Option<NodeId> {
    let graph = mol.graph();
    subs.iter().copied().max_by_key(|&substituent| {
        let atom = graph.get_node(substituent).expect("a substituent is live");
        let z = atom
            .atomic_number()
            .expect("stereo substituents have atomic numbers");
        let row = graph
            .node_table()
            .row(substituent)
            .expect("a substituent is live");
        (z, std::cmp::Reverse(row))
    })
}

/// Whether `permutation` — the old index of each new position — has odd
/// parity, which is what flips a clock sense.
pub(crate) fn odd_permutation(permutation: &[usize]) -> bool {
    let inversions: usize = permutation
        .iter()
        .enumerate()
        .map(|(i, index)| {
            permutation[i + 1..]
                .iter()
                .filter(|other| *other < index)
                .count()
        })
        .sum();
    inversions % 2 == 1
}

/// A clock sense restated after the old neighbor indices take `permutation`
/// order. An implicit hydrogen fixed first in both orders can be omitted.
pub(crate) fn permuted_label(clockwise: bool, permutation: &[usize]) -> &'static str {
    if clockwise ^ odd_permutation(permutation) {
        "CW"
    } else {
        "CCW"
    }
}

/// `label` after the stated implicit hydrogen of a three-neighbour centre is
/// drawn as its fourth bond: the first reference site becomes the last.
pub(crate) fn label_after_drawing_hydrogen(label: &str) -> &'static str {
    permuted_label(label == "CW", &[1, 2, 3, 0])
}

/// `label` after the explicit hydrogen in bond slot `at` of a four-neighbour
/// centre becomes its stated implicit hydrogen: that site moves to the front.
pub(crate) fn label_after_undrawing_hydrogen(label: &str, at: usize) -> &'static str {
    let permutation: Vec<usize> = std::iter::once(at)
        .chain((0..4).filter(|&index| index != at))
        .collect();
    permuted_label(label == "CW", &permutation)
}

impl MolGraph {
    /// Declared clock sense and its explicit neighbors in adjacency order.
    /// The implicit hydrogen, when stated, occupies an unchanged first slot.
    ///
    /// # Errors
    ///
    /// A declared label whose centre has no [`tetrahedral_reference`], or
    /// whose four sites are not distinct, cannot be interpreted.
    pub(super) fn tetrahedral_neighbors(
        &self,
        center: NodeId,
    ) -> Result<Option<(bool, Vec<NodeId>)>, MolRsError> {
        let props = self.get_node(center)?;
        let clockwise = match props.get_str(keys::STEREO) {
            Some("CW") => true,
            Some("CCW") => false,
            _ => return Ok(None),
        };
        let reference = tetrahedral_reference(self, center).ok_or_else(|| {
            MolRsError::validation(format!(
                "atom {} declares tetrahedral stereo without a four-site reference \
                 (four bond neighbours, or three and one stated hydrogen)",
                u64::from(center)
            ))
        })?;
        let neighbors: Vec<NodeId> = reference
            .iter()
            .filter_map(|site| match site {
                TetrahedralSite::Atom(id) => Some(*id),
                TetrahedralSite::Hydrogen => None,
            })
            .collect();
        if neighbors
            .iter()
            .enumerate()
            .any(|(index, neighbor)| neighbors[..index].contains(neighbor))
        {
            return Err(MolRsError::validation(format!(
                "atom {} declares tetrahedral stereo with a repeated neighbour",
                u64::from(center)
            )));
        }
        Ok(Some((clockwise, neighbors)))
    }

    /// Labels for a copy whose adjacency is reconstructed in bond-table row
    /// order. Swap-removing rows does not reorder the source adjacency.
    ///
    /// A label is read against adjacency order. When that order is already
    /// the bond-row order there is nothing to restate, and the label passes
    /// through unchanged whether or not its centre can be interpreted — a
    /// fragment body whose descriptor sites are not atoms yet cannot be.
    /// Only an actual reordering needs the centre interpreted, so only then
    /// is a centre without four distinct sites refused.
    pub(super) fn bond_row_stereo(&self) -> Result<Vec<(NodeId, &'static str)>, MolRsError> {
        let mut labels = Vec::new();
        if self.node_table().entity_column(keys::STEREO).is_none() {
            return Ok(labels);
        }
        for center in self.node_ids() {
            if !self.node_table().has(center, keys::STEREO) {
                continue;
            }
            let clockwise = match self.get_node(center)?.get_str(keys::STEREO) {
                Some("CW") => true,
                Some("CCW") => false,
                _ => continue,
            };
            let Some(bonds) = self.bonds_kind() else {
                continue;
            };
            let mut order: Vec<(usize, usize)> = self
                .neighbor_relations(center)
                .filter(|&(kind, _, _)| kind == bonds)
                .enumerate()
                .map(|(index, (_, id, _))| {
                    (
                        index,
                        self.relation_row(bonds, id).expect("a live bond has a row"),
                    )
                })
                .collect();
            order.sort_unstable_by_key(|&(_, row)| row);
            let permutation: Vec<usize> = order.into_iter().map(|(index, _)| index).collect();
            if permutation.iter().copied().eq(0..permutation.len()) {
                continue;
            }
            self.tetrahedral_neighbors(center)?;
            labels.push((center, permuted_label(clockwise, &permutation)));
        }
        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Atomistic, Element};
    use crate::props;

    #[test]
    fn drawing_and_undrawing_the_hydrogen_are_inverse_label_transitions() {
        for label in ["CW", "CCW"] {
            let drawn = label_after_drawing_hydrogen(label);
            assert_eq!(label_after_undrawing_hydrogen(drawn, 3), label);
        }
        // From the front, three other sites are crossed: odd, so the sense flips.
        assert_eq!(label_after_drawing_hydrogen("CW"), "CCW");
        assert_eq!(label_after_undrawing_hydrogen("CW", 0), "CW");
        assert_eq!(label_after_undrawing_hydrogen("CW", 1), "CCW");
    }

    #[test]
    fn parity_counts_inversions() {
        assert!(!odd_permutation(&[0, 1, 2, 3]));
        assert!(odd_permutation(&[1, 0, 2, 3]));
        assert!(odd_permutation(&[1, 2, 3, 0]));
        assert!(!odd_permutation(&[1, 0, 3, 2]));
        assert_eq!(permuted_label(true, &[1, 0, 2, 3]), "CCW");
        assert_eq!(permuted_label(false, &[1, 0, 3, 2]), "CCW");
    }

    /// A fragment-stage centre (three sites, descriptor not yet an atom)
    /// passes through a row-faithful copy unchanged, and is refused only
    /// when restating would require interpreting it.
    #[test]
    fn uninterpretable_label_passes_through_only_when_rows_are_unchanged() {
        let mut mol = Atomistic::new();
        let unrelated = [
            mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                .unwrap(),
            mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
                .unwrap(),
        ];
        let filler = mol.add_bond(unrelated[0], unrelated[1], props! {}).unwrap();
        let center = mol
            .add_atom(props! {
                keys::ATOMIC_NUMBER => Element::C,
                keys::H_COUNT => 1,
                keys::STEREO => "CW",
            })
            .unwrap();
        for element in [Element::F, Element::Cl] {
            let neighbor = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => element })
                .unwrap();
            mol.add_bond(center, neighbor, props! {}).unwrap();
        }
        let frame = mol.to_frame().unwrap();
        let stereo = frame["atoms"]
            .get(keys::STEREO)
            .and_then(|column| column.get::<String>())
            .unwrap();
        assert_eq!(stereo[[2]], "CW");

        // Deleting the unrelated bond swap-removes the last bond into row 0,
        // so the centre's rows are now permuted against its adjacency.
        mol.del_bond(filler).unwrap();
        let error = mol.to_frame().unwrap_err().to_string();
        assert!(error.contains("without a four-site reference"), "{error}");
    }
}
