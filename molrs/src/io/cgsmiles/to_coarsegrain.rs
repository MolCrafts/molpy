//! The coarsest `CGsmiles` level read as a bead graph: one [`CoarseGrain`]
//! bead per node of `levels[0]`, one CG bond per edge.
//!
//! This is the third conversion of a [`CgSmilesIr`], beside
//! [`to_atomistic`](CgSmilesIr::to_atomistic) (the whole molecule) and
//! [`templates`](CgSmilesIr::templates) (one template per atomistic
//! definition). Both of those need a fragment table; this one needs only the
//! base block, so a base-only string such as `{[#1][#1][#1][#4]}` is its main
//! input. Its main consumer is a bead-group pattern written as notation
//! instead of built bead by bead.

use crate::io::cgsmiles::CgSmilesIr;
use crate::io::cgsmiles::templates::cg_build;
use crate::io::smiles::SmilesError;
use molrs::core::CoarseGrain;
use molrs::core::NodeId;
use molrs::core::Props;
use molrs::core::keys;

impl CgSmilesIr {
    /// Read the coarsest level, `levels[0]`, as a [`CoarseGrain`].
    ///
    /// One bead per node, added in node order, so bead row *k* is node *k*;
    /// its only property is `bead_type`, set to the fragment name written
    /// after `#`. One CG bond per edge, added in edge order with its endpoints
    /// in the order `(i, j)` the edge records.
    ///
    /// **Level 0 only.** No fragment table, no descriptor pair and no deeper
    /// level is read, so a base-only string converts and
    /// [`CgNotExpandable`](crate::io::smiles::SmilesErrorKind::CgNotExpandable)
    /// is never raised. Block 0 is exactly what the base block wrote: the
    /// reader appends derived edges only to levels below it.
    ///
    /// **No geometry.** A line notation states topology, so no bead carries
    /// `x` / `y` / `z`, `mass` or `charge`, and the result has no bead
    /// membership. [`CoarseGrain::center`](crate::core::CoarseGrain::center) therefore refuses it; a
    /// pattern graph exists to be matched, not centred.
    ///
    /// # What is dropped
    ///
    /// * [`CgEdge::order`](crate::io::cgsmiles::CgEdge::order): a multiplicity
    ///   says how many atomistic bonds the edge becomes on expansion; between
    ///   beads an edge of any multiplicity is one connection, and a CG bond
    ///   has no order column. Every bond's property bag is empty.
    /// * [`CgEdge::span`](crate::io::cgsmiles::CgEdge::span) and
    ///   [`CgEdge::origin`](crate::io::cgsmiles::CgEdge::origin).
    /// * [`CgNode::charge`](crate::io::cgsmiles::CgNode::charge) (a partial
    ///   charge in `e`),
    ///   [`CgNode::annotations`](crate::io::cgsmiles::CgNode::annotations),
    ///   [`CgNode::descriptors`](crate::io::cgsmiles::CgNode::descriptors),
    ///   [`CgNode::parent`](crate::io::cgsmiles::CgNode::parent) and
    ///   [`CgNode::span`](crate::io::cgsmiles::CgNode::span). Carrying any of
    ///   them needs a column decision for CG beads that has not been made.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgBuild`](crate::io::smiles::SmilesErrorKind::CgBuild),
    /// spanned at the whole string, when the IR breaks a reader invariant:
    /// `levels` is empty, an edge endpoint is out of range (naming the edge
    /// index and the bad endpoint), or adding a CG bond fails (wrapping the
    /// underlying error). None is reachable from the output of
    /// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse); only a
    /// hand-built or hand-edited IR gets there. The graph is built locally,
    /// so an error leaves nothing half-built behind.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::cgsmiles::CgSmilesIr;
    ///
    /// // Four beads of types 1, 1, 1, 4 in a chain.
    /// let ir = CgSmilesIr::parse("{[#1][#1][#1][#4]}")?;
    /// let cg = ir.to_coarsegrain()?;
    ///
    /// assert_eq!(cg.n_beads(), 4);
    /// assert_eq!(cg.n_bonds(), 3);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_coarsegrain(&self) -> Result<CoarseGrain, SmilesError> {
        let Some(level) = self.levels.first() else {
            return Err(cg_build(
                self.span,
                "IR has no levels (level 0 is the base block)".to_owned(),
            ));
        };

        let mut cg = CoarseGrain::new();
        let beads: Vec<NodeId> = level
            .nodes
            .iter()
            .map(|node| cg.add_bead(Props::new().with(keys::BEAD_TYPE, node.name.as_str())))
            .collect::<Result<_, _>>()
            .map_err(|e| cg_build(self.span, format!("level 0 bead: {e}")))?;

        for (k, edge) in level.edges.iter().enumerate() {
            let (Some(&a), Some(&b)) = (beads.get(edge.i), beads.get(edge.j)) else {
                let bad = if edge.i >= beads.len() {
                    edge.i
                } else {
                    edge.j
                };
                return Err(cg_build(
                    self.span,
                    format!(
                        "level 0 edge {k} ({}, {}) names node {bad}, but the level has {} nodes",
                        edge.i,
                        edge.j,
                        beads.len()
                    ),
                ));
            };
            cg.add_bond(a, b, Props::new()).map_err(|e| {
                cg_build(
                    self.span,
                    format!("level 0 edge {k}: adding the CG bond failed: {e}"),
                )
            })?;
        }

        Ok(cg)
    }
}

#[cfg(test)]
mod tests {
    use crate::io::cgsmiles::CgSmilesIr;
    use crate::io::smiles::SmilesErrorKind;
    use molrs::core::CoarseGrain;
    use molrs::core::NodeId;
    use molrs::core::keys;

    // Every expected value below is hand-derived from the notation of the
    // input string (spec `backmap-primitives-06-cgsmiles`, § Testing
    // strategy): block 0 writes one bead per `[#NAME]` in reading order and
    // one edge per adjacency or `|n` chaining bond. No external program
    // produced any value here.

    // -- helpers ------------------------------------------------------------

    /// The IR of a `CGsmiles` string that must parse.
    fn parsed(text: &str) -> CgSmilesIr {
        CgSmilesIr::parse(text).unwrap_or_else(|e| panic!("{text:?} must parse: {e}"))
    }

    /// The bead graph of a `CGsmiles` string that must parse and convert.
    fn converted(text: &str) -> CoarseGrain {
        parsed(text)
            .to_coarsegrain()
            .unwrap_or_else(|e| panic!("{text:?} must convert: {e}"))
    }

    /// Bead handles in row order.
    fn rows(cg: &CoarseGrain) -> Vec<NodeId> {
        cg.bead_ids().collect()
    }

    /// `bead_type` of every bead, in row order.
    fn bead_types(cg: &CoarseGrain) -> Vec<String> {
        rows(cg)
            .into_iter()
            .map(|id| {
                cg.get_bead(id)
                    .expect("a live bead")
                    .get_str(keys::BEAD_TYPE)
                    .expect("every bead carries bead_type")
                    .to_owned()
            })
            .collect()
    }

    /// Every CG bond as the row indices of its two endpoints, as stored.
    fn bond_rows(cg: &CoarseGrain) -> Vec<(usize, usize)> {
        let rows = rows(cg);
        let row = |id: NodeId| {
            rows.iter()
                .position(|&r| r == id)
                .expect("a bond endpoint is a live bead")
        };
        cg.bonds()
            .map(|(_, bond)| (row(bond.nodes[0]), row(bond.nodes[1])))
            .collect()
    }

    /// The payload of a `CgBuild` refusal; any other outcome fails the test.
    fn cg_build_reason(ir: &CgSmilesIr) -> String {
        match ir.to_coarsegrain() {
            Err(e) => match e.kind {
                SmilesErrorKind::CgBuild(reason) => reason,
                other => panic!("expected CgBuild, got {other:?}"),
            },
            Ok(cg) => panic!("expected CgBuild, got {} beads", cg.n_beads()),
        }
    }

    // -- goldens ------------------------------------------------------------

    #[test]
    fn base_only_chain_becomes_four_beads_and_three_bonds() {
        // `{[#1][#1][#1][#4]}`: four nodes in reading order, three adjacency
        // edges between neighbours.
        let cg = converted("{[#1][#1][#1][#4]}");

        assert_eq!(bead_types(&cg), ["1", "1", "1", "4"]);
        assert_eq!(bond_rows(&cg), [(0, 1), (1, 2), (2, 3)]);
        for (_, bead) in cg.beads() {
            for key in ["x", "y", "z"] {
                assert!(
                    !bead.contains_key(key),
                    "a line notation states no geometry, yet a bead carries {key:?}"
                );
            }
        }
    }

    #[test]
    fn f2_is_read_at_level_zero() {
        // `[#OH][#PEO]|3[#OH]`: OH, three chained PEO copies, OH — five
        // nodes, four edges (OH–PEO, two `|3` chaining bonds, PEO–OH).
        let cg = converted("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}");

        assert_eq!(bead_types(&cg), ["OH", "PEO", "PEO", "PEO", "OH"]);
        assert_eq!(bond_rows(&cg), [(0, 1), (1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn three_block_string_reads_levels_zero_only() {
        // Level 0 writes B1–B2–B1; level 1 expands each B* into two beads, so
        // reading the last level would give six PEO/PE beads instead.
        let ir = parsed(
            "{[#B1][#B2][#B1]}.\
             {#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
             {#PEO=[>]COC[<],#PE=[>]CC[<]}",
        );
        assert_eq!(ir.levels.len(), 2);
        assert_eq!(ir.levels[1].nodes.len(), 6);

        let cg = ir.to_coarsegrain().expect("a parsed IR converts");

        assert_eq!(bead_types(&cg), ["B1", "B2", "B1"]);
        assert_eq!(bond_rows(&cg), [(0, 1), (1, 2)]);
    }

    #[test]
    fn edge_multiplicity_is_not_recorded() {
        // `=` writes multiplicity 2: two atomistic bonds on expansion, one
        // connection between the beads.
        let cg = converted("{[#SC3]=[#SC3]}");

        assert_eq!(cg.n_bonds(), 1);
        let (_, bond) = cg.bonds().next().expect("one bond");
        assert!(
            bond.props.is_empty(),
            "a CG bond has no order column, got {:?}",
            bond.props
        );
    }

    // -- refusals -----------------------------------------------------------

    #[test]
    fn out_of_range_edge_endpoint_is_cg_build_naming_the_edge() {
        let mut ir = parsed("{[#A][#B]}");
        ir.levels[0].edges[0].j = 5;

        let reason = cg_build_reason(&ir);

        assert!(reason.contains("edge 0"), "reason names edge 0: {reason:?}");
        assert!(
            reason.contains('5'),
            "reason names the endpoint: {reason:?}"
        );
    }

    #[test]
    fn empty_levels_is_cg_build() {
        let mut ir = parsed("{[#A][#B]}");
        ir.levels.clear();

        cg_build_reason(&ir);
    }
}
