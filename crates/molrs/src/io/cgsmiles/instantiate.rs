//! Expansion of one `CGsmiles` resolution level into the next.
//!
//! One step, one rule: **every node of the level-k graph is instantiated as a
//! disjoint copy of its fragment graph, carrying its own descriptor lists**
//! (R4.10 of `.claude/specs/cgsmiles-01c-fragments.md` § Domain basis). The
//! copies are appended in the parent's index order, each body's own edges are
//! remapped by the copy's offset with their orders untouched, and every copied
//! node records the parent it came from.
//!
//! What this step deliberately does *not* do is create an edge **between** two
//! copies (R4.20): a descriptor belongs to the table it was written in and is
//! paired at its own resolution step, the subject of the next spec in the
//! chain (`.claude/specs/cgsmiles-01d-resolve.md`). Expansion that guessed a
//! bond here would leave that step nothing to pair and no way to tell a guess
//! from a written edge.
//!
//! Membership is implicit and positional (R5.1): a copied node belongs to the
//! parent whose index it carries, and nothing else records the grouping.

use std::collections::BTreeMap;

use crate::io::cgsmiles::{CgEdge, CgGraph, EdgeOrigin};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};

/// Build the level `parent` denotes, one disjoint copy of a fragment graph per
/// node of `parent`.
///
/// `defs` is a borrowed view of the fragment table that resolves `parent`'s
/// names, holding only the coarse-graph bodies — an atomistic body cannot
/// reach this function *by type*. `input` is the whole `CGsmiles` string, read
/// only to span a diagnostic into it.
///
/// # Errors
///
/// [`SmilesErrorKind::CgBuild`] naming the fragment, when a name of `parent`
/// is absent from `defs`. That is a reader bug rather than a bad input —
/// `check_coverage` has already run over the same pair — so it is reported by
/// value, identically in debug and release, never as a panic and never as
/// [`SmilesErrorKind::CgUndefinedFragment`], which would tell the user their
/// string is wrong when it is not.
pub(super) fn instantiate(
    parent: &CgGraph,
    defs: &BTreeMap<&str, &CgGraph>,
    input: &str,
) -> Result<CgGraph, SmilesError> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    for (index, source) in parent.nodes.iter().enumerate() {
        let Some(body) = defs.get(source.name.as_str()) else {
            let reason = format!("no definition for fragment '{}'", source.name);
            let kind = SmilesErrorKind::CgBuild(reason);
            return Err(SmilesError::new(
                kind,
                source.span,
                input,
                Notation::CgSmiles,
            ));
        };
        let offset = nodes.len();
        for node in &body.nodes {
            let mut copy = node.clone();
            copy.parent = Some(index);
            nodes.push(copy);
        }
        for edge in &body.edges {
            edges.push(CgEdge {
                i: edge.i + offset,
                j: edge.j + offset,
                order: edge.order,
                span: edge.span,
                // A body's edges are what the table wrote; only resolution
                // derives an edge, and it runs after instantiation.
                origin: EdgeOrigin::Written,
            });
        }
    }
    Ok(CgGraph { nodes, edges })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;

    use crate::io::cgsmiles::{CgBondOrder, CgEdge, CgGraph, CgNode, EdgeOrigin};
    use crate::io::smiles::{BondingDescriptor, DescriptorKind, SmilesErrorKind, Span};

    // Every expectation here is hand-derived from R4.10 ("every node of the
    // level-k graph is instantiated as a disjoint copy of its fragment graph,
    // carrying its own descriptor lists"), R4.20 (descriptors stay in their own
    // table, so no inter-copy edge is created here) and R5.1 (membership is
    // positional) of `.claude/specs/cgsmiles-01c-fragments.md` § Domain basis.
    // No external program produced any value below.

    /// The `CGsmiles` text the hand-built graphs below stand for (F8).
    /// `instantiate` reads it only to span a diagnostic.
    const F8: &str = "{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
                      {#PEO=[>]COC[<],#PE=[>]CC[<]}";

    // -- hand-built fixtures ------------------------------------------------

    /// An unlabelled descriptor of `kind`, with no bond order written.
    fn descriptor(kind: DescriptorKind) -> BondingDescriptor {
        BondingDescriptor {
            kind,
            label: String::new(),
            order: None,
        }
    }

    /// A coarse node named `name`, carrying `kinds` and no parent.
    fn node(name: &str, kinds: &[DescriptorKind]) -> CgNode {
        CgNode {
            name: name.to_owned(),
            charge: None,
            annotations: Vec::new(),
            descriptors: kinds.iter().copied().map(descriptor).collect(),
            parent: None,
            span: Span::new(0, 0),
        }
    }

    /// A single edge between `i` and `j`.
    fn edge(i: usize, j: usize) -> CgEdge {
        CgEdge {
            i,
            j,
            order: CgBondOrder::Single,
            span: Span::new(0, 0),
            origin: EdgeOrigin::Written,
        }
    }

    /// F8's base graph, `{[#B1][#B2][#B1]}`.
    fn f8_parent() -> CgGraph {
        CgGraph {
            nodes: vec![node("B1", &[]), node("B2", &[]), node("B1", &[])],
            edges: vec![edge(0, 1), edge(1, 2)],
        }
    }

    /// A two-node body `[>][#NAME][#NAME][<]`, the shape of F8's `#B1`.
    fn two_node_body(name: &str) -> CgGraph {
        CgGraph {
            nodes: vec![
                node(name, &[DescriptorKind::Right]),
                node(name, &[DescriptorKind::Left]),
            ],
            edges: vec![edge(0, 1)],
        }
    }

    /// Node names in index order.
    fn names(graph: &CgGraph) -> Vec<&str> {
        graph.nodes.iter().map(|n| n.name.as_str()).collect()
    }

    /// `parent` of every node, in index order.
    fn parents(graph: &CgGraph) -> Vec<Option<usize>> {
        graph.nodes.iter().map(|n| n.parent).collect()
    }

    /// `(i, j, order)` of every edge, in order.
    fn edges(graph: &CgGraph) -> Vec<(usize, usize, CgBondOrder)> {
        graph.edges.iter().map(|e| (e.i, e.j, e.order)).collect()
    }

    /// The descriptor kinds each node carries, in index order.
    fn kinds(graph: &CgGraph) -> Vec<Vec<DescriptorKind>> {
        graph
            .nodes
            .iter()
            .map(|n| n.descriptors.iter().map(|d| d.kind).collect())
            .collect()
    }

    // -- F8: one copy of the body per referencing node (R4.10) --------------

    #[test]
    fn test_instantiate_copies_one_body_per_parent_node_in_index_order() {
        let b1 = two_node_body("PEO");
        let b2 = two_node_body("PE");
        let defs = BTreeMap::from([("B1", &b1), ("B2", &b2)]);
        let level = instantiate(&f8_parent(), &defs, F8).expect("F8 level 1 must build");
        assert_eq!(names(&level), vec!["PEO", "PEO", "PE", "PE", "PEO", "PEO"]);
    }

    #[test]
    fn test_instantiate_sets_parent_to_the_index_of_the_source_node() {
        let b1 = two_node_body("PEO");
        let b2 = two_node_body("PE");
        let defs = BTreeMap::from([("B1", &b1), ("B2", &b2)]);
        let level = instantiate(&f8_parent(), &defs, F8).expect("F8 level 1 must build");
        assert_eq!(
            parents(&level),
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]
        );
    }

    #[test]
    fn test_instantiate_remaps_intra_fragment_edges_by_the_copy_offset() {
        let b1 = two_node_body("PEO");
        let b2 = two_node_body("PE");
        let defs = BTreeMap::from([("B1", &b1), ("B2", &b2)]);
        let level = instantiate(&f8_parent(), &defs, F8).expect("F8 level 1 must build");
        assert_eq!(
            edges(&level),
            vec![
                (0, 1, CgBondOrder::Single),
                (2, 3, CgBondOrder::Single),
                (4, 5, CgBondOrder::Single),
            ]
        );
    }

    /// R4.20: pairing descriptors across copies is 01d's job, so expansion
    /// creates no edge between two copies.
    #[test]
    fn test_instantiate_creates_no_edge_between_two_copies() {
        let b1 = two_node_body("PEO");
        let b2 = two_node_body("PE");
        let defs = BTreeMap::from([("B1", &b1), ("B2", &b2)]);
        let level = instantiate(&f8_parent(), &defs, F8).expect("F8 level 1 must build");
        let crossing: Vec<(usize, usize)> = level
            .edges
            .iter()
            .filter(|e| level.nodes[e.i].parent != level.nodes[e.j].parent)
            .map(|e| (e.i, e.j))
            .collect();
        assert!(crossing.is_empty(), "inter-copy edges were {crossing:?}");
    }

    #[test]
    fn test_instantiate_clones_the_body_descriptors_onto_every_copy() {
        let b1 = two_node_body("PEO");
        let b2 = two_node_body("PE");
        let defs = BTreeMap::from([("B1", &b1), ("B2", &b2)]);
        let level = instantiate(&f8_parent(), &defs, F8).expect("F8 level 1 must build");
        assert_eq!(
            kinds(&level),
            vec![
                vec![DescriptorKind::Right],
                vec![DescriptorKind::Left],
                vec![DescriptorKind::Right],
                vec![DescriptorKind::Left],
                vec![DescriptorKind::Right],
                vec![DescriptorKind::Left],
            ]
        );
    }

    /// `parent` is overwritten, never inherited: one body used by two nodes
    /// yields two copies whose parents differ.
    #[test]
    fn test_instantiate_gives_two_uses_of_one_body_different_parents() {
        let body = two_node_body("PEO");
        let defs = BTreeMap::from([("B1", &body)]);
        let parent = CgGraph {
            nodes: vec![node("B1", &[]), node("B1", &[])],
            edges: vec![edge(0, 1)],
        };
        let level = instantiate(&parent, &defs, "{[#B1][#B1]}").expect("level 1 must build");
        assert_eq!(
            parents(&level),
            vec![Some(0), Some(0), Some(1), Some(1)],
            "one body used twice must yield copies with different parents"
        );
    }

    // -- internal invariant -------------------------------------------------

    /// A name missing from `defs` is a reader bug, not user input: coverage
    /// has already run. It is reported by value — never a panic, and never
    /// `CgUndefinedFragment`.
    #[test]
    fn test_instantiate_reports_a_name_absent_from_defs_as_a_build_error() {
        let b1 = two_node_body("PEO");
        let defs = BTreeMap::from([("B1", &b1)]);
        let err = instantiate(&f8_parent(), &defs, F8)
            .expect_err("a name absent from defs must be refused");
        assert!(
            matches!(&err.kind, SmilesErrorKind::CgBuild(name) if name.contains("B2")),
            "kind was {:?}",
            err.kind
        );
    }
}
