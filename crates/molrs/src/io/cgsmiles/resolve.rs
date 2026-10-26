//! Descriptor resolution: pairing the bonding descriptors of one `CGsmiles`
//! level with the descriptors of the entities its edges join.
//!
//! A parsed level knows that two beads are bonded; it does not know *through
//! which written descriptor*. This step answers that once, at parse time: for
//! every edge, in parse order, and for each of the bonds its multiplicity
//! stands for, the first free compatible pair of ports wins, both ports are
//! consumed and the match is recorded on the IR as a [`ResolvedPair`]. An edge
//! that no free pair can form is refused
//! ([`SmilesErrorKind::CgUnmatchableEdge`]) rather than dropped; ports still
//! free when a level is done are dropped rather than forced, and the freed
//! valence of an atomistic port is a later hydrogen-repletion step's business.
//!
//! Two descriptors are compatible when, after flipping `<` against `>` on one
//! of them, the whole descriptor agrees: the same pairing role, the same
//! label, and the same **effective** bond order — no symbol written is the
//! same order as an explicit `-`, so `[$]` pairs `-[$]` while `=[$]` pairs
//! neither.
//!
//! **Edges are paired in parse order** — the order the notation formed them,
//! with a ring-closure edge where its closing marker was read — while the
//! `CGsmiles` reference implementation (the Python package
//! `gruenewald-lab/CGsmiles`, cited in full below) iterates its graph
//! library's adjacency order instead: for
//! `{[#A]1[#B][#C]1}` that is (0,1), (1,2), (0,2) here against (0,1), (0,2),
//! (1,2) there. Because pairing is greedy and consumes ports as it goes, a
//! ring-bearing string can therefore consume different ports than the
//! reference does. What is contracted is the graph the expansion builds, not
//! which port of a bead a given edge took.
//!
//! Resolving an intermediate level also creates work for the next one: each
//! pair it forms joins two child nodes, and that bond is appended to the level
//! below as a [`Derived`](EdgeOrigin::Derived) edge — after every written
//! edge, so indices already handed out stay stable — before that level is
//! resolved in turn.
//!
//! Rules R4.6, R4.9, R4.11–R4.17, R4.19 and R4.20 of
//! `.claude/specs/cgsmiles-01d-resolve.md` § Domain basis. The compatibility
//! rule is the `BigSMILES` one — Lin, T.-S. et al., *BigSMILES: A
//! Structurally-Based Line Notation for Describing Macromolecules*, ACS Cent.
//! Sci. **5**, 1523–1531 (2019), DOI 10.1021/acscentsci.9b00476 — as the
//! `CGsmiles` reference implementation applies it
//! (github.com/gruenewald-lab/CGsmiles at commit `910c9ee`,
//! `resolve.py:14-59`), and the bond-kind precedence is the Daylight Theory
//! Manual's, *SMILES* § 3.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use crate::io::cgsmiles::parser::CgParser;
use crate::io::cgsmiles::{
    CgBondOrder, CgEdge, CgFragmentDef, CgGraph, CgSmilesIr, EdgeOrigin, FragmentBody, PairEnd,
    ResolvedPair,
};
use crate::io::smiles::{BondKind, BondingDescriptor, DescriptorKind, FragmentIr};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::atom_keys;
use molrs::core::Atomistic;
use molrs::core::NodeId;

/// Pair the bonding descriptors of every level of `ir`, coarsest level first.
///
/// `input` is the whole `CGsmiles` string, read only so that a refusal carries
/// the text, a span into it and [`Notation::CgSmiles`]. `ir` is mutated in two
/// places and no others: [`pairs`](CgSmilesIr::pairs) gains one list per
/// level, and a level gains the [`Derived`](EdgeOrigin::Derived) edges the
/// level above induced in it.
///
/// A level's ports come from whichever side of it holds descriptors: for an
/// intermediate level, the child nodes the next level down instantiated, in
/// written order; for the last level, the descriptor map of each node's
/// atomistic body. A base-only string has neither — no fragment table means no
/// ports at all — and is left alone, its written edges intact and its one pair
/// list empty, rather than refused.
///
/// A private pipeline step, sibling of `instantiate` and `validate_ir`: an
/// unresolved [`CgSmilesIr`] is not a value this module hands out.
///
/// # Errors
///
/// [`SmilesErrorKind::CgUnmatchableEdge`] naming the level and the edge, when
/// a written edge has no free pair of compatible descriptors left to form it
/// (R4.15) — spanned at that edge, which for a derived edge is a copy of the
/// span of the coarse bond that induced it.
///
/// [`SmilesErrorKind::CgBuild`] for a reader invariant the input cannot break:
/// a name of the last level absent from its fragment table, a coarse body in
/// that table, or an intermediate pair whose end is not a child node.
///
/// [`SmilesErrorKind::CgLastBlockNotAtomistic`] boxing the fragment
/// converter's own error, for a last-block body that *parses* as fragment
/// SMILES and then fails to build — a ring closure the body never matches
/// (`{[#A][#B]}.{#A=[$]C1CC,#B=[$]C}`), which the body parser does not check.
/// The diagnostic is the parser's: re-based through the body's offset in the
/// whole input, so the caret lands on the offending token of the string the
/// caller passed in, and boxed, except
/// [`SmilesErrorKind::AtomAnnotationUnsupported`], which propagates
/// un-wrapped.
///
/// # Panics
///
/// If [`pairs`](CgSmilesIr::pairs) is shorter than
/// [`levels`](CgSmilesIr::levels). The parser builds the two together — one
/// empty list per level — and this step is reached through no other path.
pub(super) fn resolve(ir: &mut CgSmilesIr, input: &str) -> Result<(), SmilesError> {
    // A base-only string writes edges between beads that offer no ports at
    // all. There is nothing to pair, and nothing to refuse.
    if ir.fragments.is_empty() {
        return Ok(());
    }
    let mut cache = FragmentCache::default();
    for level in 0..ir.levels.len() {
        let is_last = level + 1 == ir.levels.len();
        let mut ports = if is_last {
            last_level_ports(&ir.levels[level], &ir.fragments[level], &mut cache, input)?
        } else {
            PortTable::of_children(&ir.levels[level], &ir.levels[level + 1])
        };
        let pairs = resolve_level(&ir.levels[level], &mut ports, level, input)?;
        if !is_last {
            let induced = derived_edges(&pairs, &ir.levels[level], level, input)?;
            ir.levels[level + 1].edges.extend(induced);
        }
        ir.pairs[level] = pairs;
    }
    Ok(())
}

/// Pair every edge of one level, in parse order.
///
/// `level` is the index of `graph` among the IR's levels, carried only so a
/// refusal can name it.
///
/// # Errors
///
/// [`SmilesErrorKind::CgUnmatchableEdge`] at the first edge whose next bond
/// finds no free compatible pair.
fn resolve_level(
    graph: &CgGraph,
    ports: &mut PortTable,
    level: usize,
    input: &str,
) -> Result<Vec<ResolvedPair>, SmilesError> {
    let mut pairs = Vec::new();
    for (edge, written) in graph.edges.iter().enumerate() {
        // An edge of order n is n separate bonds between the two instances,
        // never one multiple bond (R6.2): each consumes its own port pair.
        for bond in 0..usize::from(written.order.multiplicity()) {
            let Some((src, dst, kind)) = ports.take_first_compatible(written.i, written.j) else {
                let unmatchable = SmilesErrorKind::CgUnmatchableEdge { level, edge };
                return Err(SmilesError::new(
                    unmatchable,
                    written.span,
                    input,
                    Notation::CgSmiles,
                ));
            };
            pairs.push(ResolvedPair {
                edge,
                bond,
                src,
                dst,
                kind,
            });
        }
    }
    Ok(pairs)
}

/// The edges the pairs of an intermediate level induce in the level below.
///
/// One edge per pair, between the two child nodes the pair joined,
/// [`CgBondOrder::Single`] because one pair is one bond, and spanned at the
/// coarse edge that induced it — the only text that names that bond.
///
/// # Errors
///
/// [`SmilesErrorKind::CgBuild`] if a pair of an intermediate level carries an
/// end that is not a child node: the port table of such a level offers
/// [`PairEnd::Sub`] ends only, so that state is a reader bug rather than a bad
/// input.
fn derived_edges(
    pairs: &[ResolvedPair],
    graph: &CgGraph,
    level: usize,
    input: &str,
) -> Result<Vec<CgEdge>, SmilesError> {
    let mut edges = Vec::with_capacity(pairs.len());
    for (pair, resolved) in pairs.iter().enumerate() {
        let span = graph.edges[resolved.edge].span;
        let (PairEnd::Sub { node: i, .. }, PairEnd::Sub { node: j, .. }) =
            (&resolved.src, &resolved.dst)
        else {
            let reason = format!("pair {pair} of level {level} is not a pair of child nodes");
            let kind = SmilesErrorKind::CgBuild(reason);
            return Err(SmilesError::new(kind, span, input, Notation::CgSmiles));
        };
        edges.push(CgEdge {
            i: *i,
            j: *j,
            order: CgBondOrder::Single,
            span,
            origin: EdgeOrigin::Derived { level, pair },
        });
    }
    Ok(edges)
}

/// The ports of the **last** level: the descriptors written on each node's
/// atomistic body, in the order the one walker visited them.
///
/// The entity of this level is the **atom** (R4.12): a node contributes one
/// entity per port-carrying atom of its body, in walker order, and an atom
/// written with two descriptors offers both of them as one entity. Grouping a
/// whole body as one entity instead would scan (source port, target port)
/// rather than (source atom, target atom, source descriptor, target
/// descriptor), which pairs a different port of a multi-port body — a
/// different molecule, not a different labelling.
///
/// The map is in visit order, so one atom's descriptors are contiguous in it
/// and the grouping is a scan over consecutive equal [`NodeId`]s. A port keeps
/// indexing the instance's flat map, so a [`PairEnd::Body`] is unaffected by
/// how the ports are grouped.
///
/// The conversion runs once per fragment *definition*, not once per node: that
/// is what `cache` is for.
///
/// # Errors
///
/// [`SmilesErrorKind::CgBuild`] when a node names a fragment the table does
/// not define, when that definition's body is a coarse graph (the last table
/// is atomistic by position, checked at parse), or when the converted body
/// does not hold an atom its own descriptor map names.
///
/// [`SmilesErrorKind::CgLastBlockNotAtomistic`] boxing the fragment
/// converter's own error — an unmatched ring closure inside a body — re-based
/// through [`CgParser::rebase`] onto the body's offset in `input`, the same
/// treatment the parser gives a body that does not even parse.
fn last_level_ports(
    graph: &CgGraph,
    defs: &BTreeMap<String, CgFragmentDef>,
    cache: &mut FragmentCache,
    input: &str,
) -> Result<PortTable, SmilesError> {
    let mut instances = Vec::with_capacity(graph.nodes.len());
    for (instance, node) in graph.nodes.iter().enumerate() {
        let span = node.span;
        let build = |reason: String| {
            SmilesError::new(
                SmilesErrorKind::CgBuild(reason),
                span,
                input,
                Notation::CgSmiles,
            )
        };
        let Some(def) = defs.get(&node.name) else {
            return Err(build(format!("no definition for fragment '{}'", node.name)));
        };
        let FragmentBody::Smiles(body) = &def.body else {
            return Err(build(format!(
                "fragment '{}' of the last level has a coarse body",
                node.name
            )));
        };
        // A body's diagnostic is body-local, and converting one catches what
        // parsing it did not (a ring closure the body never matches). It gets
        // the parser's own treatment — span shifted to the body's offset in
        // the entry `#NAME=body`, kind boxed — so one failure class reaches
        // the reader one way whichever step found it.
        let offset = def.span.start + 1 + def.name.len() + 1;
        let (mol, map) = cache
            .get_or_build(&node.name, body)
            .map_err(|e| CgParser::rebase(e, offset, input))?;
        let mut entities: Vec<Vec<Port>> = Vec::new();
        let mut entity: Vec<Port> = Vec::new();
        let mut current: Option<NodeId> = None;
        for (port, (atom, descriptor)) in map.iter().enumerate() {
            let props = mol
                .get_atom(*atom)
                .map_err(|e| build(format!("port {port} of '{}': {e}", node.name)))?;
            // Walker order, so a change of atom closes the entity the previous
            // atom's descriptors made.
            if current.is_some_and(|held| held != *atom) {
                entities.push(std::mem::take(&mut entity));
            }
            current = Some(*atom);
            entity.push(Port {
                descriptor: descriptor.clone(),
                end: PairEnd::Body { instance, port },
                // What `Builder` stamped on a freshly converted body can only
                // have come from the notation: no perception has run on it.
                aromatic: atom_keys::is_aromatic(&props),
                consumed: false,
            });
        }
        if !entity.is_empty() {
            entities.push(entity);
        }
        instances.push(entities);
    }
    Ok(PortTable { instances })
}

/// One free-or-consumed joining site offered to the pairing scan.
///
/// A copy, not a borrow of the IR: the descriptor lists a
/// [`PairEnd`] indexes are never mutated, because removing a consumed
/// descriptor would invalidate every port index already recorded.
struct Port {
    /// The descriptor as written: kind, label and annotated order.
    descriptor: BondingDescriptor,
    /// How a [`ResolvedPair`] names this port.
    end: PairEnd,
    /// Whether the notation wrote the port's atom aromatic. Always `false` at
    /// an intermediate level, whose ports are beads rather than atoms.
    aromatic: bool,
    /// Whether an earlier bond already took this port (R4.13). The local
    /// consumed bitmap lives here, one flag per port copy.
    consumed: bool,
}

/// The ports one level offers, grouped the way the pairing scan reads them:
/// per instance, then per entity, then in written order.
///
/// An *entity* is whatever carries descriptors at this level — a child node at
/// an intermediate level, one atom of the node's atomistic body at the last
/// one — and it is a grouping level of its own because the scan iterates
/// entities before it iterates the ports within one (R4.12).
struct PortTable {
    /// Indexed by node of the level being resolved.
    instances: Vec<Vec<Vec<Port>>>,
}

impl PortTable {
    /// The ports of an **intermediate** level: the descriptors its children
    /// carry one level down, grouped by the parent that instantiated them.
    ///
    /// `graph` is the level being resolved and `children` the level below it,
    /// whose nodes each name their [`parent`](crate::io::cgsmiles::CgNode::parent).
    /// A child naming no parent, or a parent index no node of `graph` has,
    /// offers this level nothing and is skipped: instantiation sets a parent on
    /// every copy it makes, so neither case arises from a parsed value.
    fn of_children(graph: &CgGraph, children: &CgGraph) -> Self {
        let mut instances: Vec<Vec<Vec<Port>>> = graph.nodes.iter().map(|_| Vec::new()).collect();
        for (node, child) in children.nodes.iter().enumerate() {
            let Some(parent) = child.parent.filter(|p| *p < instances.len()) else {
                // A child with no parent belongs to no instance of this level
                // and offers it nothing; instantiation sets the parent on
                // every copy it makes.
                continue;
            };
            let ports = child
                .descriptors
                .iter()
                .enumerate()
                .map(|(port, descriptor)| Port {
                    descriptor: descriptor.clone(),
                    end: PairEnd::Sub { node, port },
                    aromatic: false,
                    consumed: false,
                })
                .collect();
            instances[parent].push(ports);
        }
        Self { instances }
    }

    /// Take the first free compatible pair of ports between instances `i` and
    /// `j`, consume both, and report the ends and the bond kind they resolve
    /// to.
    ///
    /// First by the scan order of R4.12 — source entities in written order,
    /// target entities in written order, then the ports within each — and
    /// greedy: the first pair wins and nothing backtracks. `None` when no free
    /// compatible pair is left, which is what makes an edge unmatchable.
    fn take_first_compatible(
        &mut self,
        i: usize,
        j: usize,
    ) -> Option<(PairEnd, PairEnd, BondKind)> {
        let (src, dst) = self.scan(i, j)?;
        let source = &self.instances[i][src.0][src.1];
        let target = &self.instances[j][dst.0][dst.1];
        // An explicit symbol wins; absence between two written-aromatic ports
        // is aromatic (the Daylight rule); everything else is single. The two
        // effective orders are equal by compatibility, so whichever side wrote
        // one wrote this bond's kind.
        let kind = source
            .descriptor
            .order
            .or(target.descriptor.order)
            .unwrap_or(if source.aromatic && target.aromatic {
                BondKind::Aromatic
            } else {
                BondKind::Single
            });
        let ends = (source.end.clone(), target.end.clone());
        self.instances[i][src.0][src.1].consumed = true;
        self.instances[j][dst.0][dst.1].consumed = true;
        Some((ends.0, ends.1, kind))
    }

    /// The `(entity, port)` coordinates of the first free compatible pair
    /// between instances `i` and `j`, without consuming anything.
    fn scan(&self, i: usize, j: usize) -> Option<((usize, usize), (usize, usize))> {
        for (se, source) in self.instances[i].iter().enumerate() {
            for (de, target) in self.instances[j].iter().enumerate() {
                for (sp, src) in source.iter().enumerate() {
                    if src.consumed {
                        continue;
                    }
                    for (dp, dst) in target.iter().enumerate() {
                        if !dst.consumed && compatible(&src.descriptor, &dst.descriptor) {
                            return Some(((se, sp), (de, dp)));
                        }
                    }
                }
            }
        }
        None
    }
}

/// Whether two written descriptors may form a bond (R4.6).
///
/// The rule is the whole descriptor, flipped: `$` pairs `$`, `>` pairs `<` and
/// neither pairs the other; the labels must be equal character for character,
/// so `[$a]` and `[$b]` are different classes and `[$a]` and `[$]` are too;
/// and the **effective** orders must agree, an unwritten symbol counting as
/// [`BondKind::Single`], so `[$]` pairs `-[$]` and refuses `=[$]` (R4.9: the
/// order is compared every time, never inherited from a first occurrence).
///
/// [`DescriptorKind::Shared`] pairs with [`DescriptorKind::Shared`], which
/// R4.6 states and [`flip`] therefore says; no string reaches this with one,
/// because `validate_ir` refuses the squash operator before `resolve` runs.
fn compatible(a: &BondingDescriptor, b: &BondingDescriptor) -> bool {
    flip(a.kind) == b.kind
        && a.label == b.label
        && a.order.unwrap_or(BondKind::Single) == b.order.unwrap_or(BondKind::Single)
}

/// The one kind a descriptor of `kind` may pair with: the two roles of the
/// AB-type operator exchange, and the two self-pairing kinds are their own
/// partner.
///
/// Total and an **involution** — `flip(flip(k)) == k` for every kind — which is
/// what makes [`compatible`] symmetric in its two arguments however the scan
/// orders them.
fn flip(kind: DescriptorKind) -> DescriptorKind {
    match kind {
        DescriptorKind::Symmetric => DescriptorKind::Symmetric,
        DescriptorKind::Left => DescriptorKind::Right,
        DescriptorKind::Right => DescriptorKind::Left,
        DescriptorKind::Shared => DescriptorKind::Shared,
    }
}

/// One converted fragment body: the heavy-atom graph exactly as `Builder`
/// wrote it and its port map in walker order (see [`FragmentCache`]).
pub(super) type ConvertedBody = (Atomistic, Vec<(NodeId, BondingDescriptor)>);

/// One converted atomistic body per fragment **definition**, with its port
/// map.
///
/// A per-call local, built and dropped inside one call of [`resolve`] or of
/// the lowest-level expansion behind
/// [`CgSmilesIr::to_atomistic`](crate::io::cgsmiles::CgSmilesIr::to_atomistic):
/// never a field of the IR (which would make a value of parse results own a
/// second representation of its own bodies), never a `static`, a
/// `thread_local` or a cross-call memo. Within one call a definition is
/// converted once however many instances name it.
///
/// It is also the **only** route from this module to
/// [`FragmentIr::to_atomistic`](crate::io::smiles::FragmentIr::to_atomistic): no
/// other code under `cgsmiles/` calls the converter, so there is one walker
/// over a fragment body and one place a converted body can come from.
///
/// # Invariant
///
/// An entry is the body exactly as the one walker wrote it — **no perception
/// has run on it**. Its `is_aromatic` stamps therefore reflect only what the
/// notation wrote, which is the property the bond-kind precedence reads; a
/// Kekulé-spelled ring carries none of them.
#[derive(Default)]
pub(super) struct FragmentCache {
    /// Converted bodies keyed by the fragment name written after `#`.
    entries: BTreeMap<String, ConvertedBody>,
}

impl FragmentCache {
    /// The converted body and port map of the fragment called `name`,
    /// converting `body` on the first call and handing back the stored value
    /// on every later one.
    ///
    /// `body` is the fragment definition's parsed SMILES; it is read only when
    /// the name is absent, so two definitions that share a name — which the
    /// reader refuses at parse — could not collide here either.
    ///
    /// The returned map is the port order: index *p* of the map is port *p* of
    /// every instance of this definition, so a [`PairEnd::Body`] and an
    /// [`NodeId`] cannot disagree.
    ///
    /// # Errors
    ///
    /// Whatever
    /// [`FragmentIr::to_atomistic`](crate::io::smiles::FragmentIr::to_atomistic)
    /// returns for `body` — an unmatched ring closure inside it, say. The
    /// error is the converter's own, carrying a span into the whole
    /// `CGsmiles` string; callers holding that string re-stamp it into
    /// [`Notation::CgSmiles`].
    pub(super) fn get_or_build(
        &mut self,
        name: &str,
        body: &FragmentIr,
    ) -> Result<&ConvertedBody, SmilesError> {
        match self.entries.entry(name.to_owned()) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => Ok(entry.insert(body.to_atomistic()?)),
        }
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use crate::io::cgsmiles::fixtures::{body, fragment, pair};
    use crate::io::cgsmiles::parser::CgParser;
    use crate::io::cgsmiles::{CgBondOrder, CgSmilesIr, EdgeOrigin, PairEnd, ResolvedPair};
    use crate::io::smiles::{
        BondKind, DescriptorKind, Notation, SmilesError, SmilesErrorKind, Span,
    };

    // Every expectation below is hand-derived from § Domain basis of
    // `.claude/specs/cgsmiles-01d-resolve.md` — R4.6 (whole-descriptor flip,
    // exact label, effective order), R4.11 (one bond per multiplicity), R4.12
    // (source entities × target entities × source ports × target ports, first
    // free compatible pair wins), R4.13 (both descriptors consumed), R4.14
    // (explicit symbol > written-aromatic pair > single), R4.15 (an
    // unmatchable edge is an error) and R4.16 (edges in parse order) — and
    // from the port lists the one walker `FragmentIr::to_atomistic` returns. No
    // external program produced any value here.

    // -- helpers ------------------------------------------------------------

    /// The resolved IR of a `CGsmiles` string that must parse and resolve.
    fn resolved(text: &str) -> CgSmilesIr {
        CgSmilesIr::parse(text)
            .unwrap_or_else(|e| panic!("CgSmilesIr::parse({text:?}) failed: {e}"))
    }

    /// The error a `CGsmiles` string must be refused with.
    fn err_of(text: &str) -> SmilesError {
        CgSmilesIr::parse(text)
            .err()
            .unwrap_or_else(|| panic!("CgSmilesIr::parse({text:?}) was accepted"))
    }

    /// The IR as the syntax / instantiation / validation stages leave it,
    /// before `resolve` runs: `CgParser::parse` is those three stages and
    /// nothing else.
    fn unresolved(text: &str) -> CgSmilesIr {
        CgParser::new(text)
            .parse()
            .unwrap_or_else(|e| panic!("CgParser::parse({text:?}) failed: {e}"))
    }

    /// An end at an intermediate level: the child node carrying the port.
    fn sub(node: usize, port: usize) -> PairEnd {
        PairEnd::Sub { node, port }
    }

    /// Every `(instance, port)` a list of pairs consumes, sorted. Panics on a
    /// `Sub` end, so a caller asserting last-level ends also asserts they are
    /// last-level ends.
    fn body_ends(pairs: &[ResolvedPair]) -> Vec<(usize, usize)> {
        let mut ends = Vec::new();
        for resolved in pairs {
            for end in [&resolved.src, &resolved.dst] {
                match end {
                    PairEnd::Body { instance, port } => ends.push((*instance, *port)),
                    PairEnd::Sub { .. } => panic!("expected a last-level end, got {end:?}"),
                }
            }
        }
        ends.sort();
        ends
    }

    // -- F2: an OH-capped PEO trimer (R4.12, R4.13) -------------------------

    /// `{[#OH][#PEO]|3[#OH]}` is OH0-PEO1-PEO2-PEO3-OH4 with edges (0,1),
    /// (1,2), (2,3), (3,4). `[$]COC[$]` offers two ports — 0 on its first
    /// carbon, 1 on its last — so consumption alone forces head-to-tail
    /// pairing: edge (OH0,PEO1) takes PEO1's port 0, and edge (PEO1,PEO2)
    /// finds it consumed and takes port 1.
    #[test]
    fn test_f2_resolves_four_head_to_tail_pairs() {
        let ir = resolved("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}");
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Single),
                pair(1, 0, body(1, 1), body(2, 0), BondKind::Single),
                pair(2, 0, body(2, 1), body(3, 0), BondKind::Single),
                pair(3, 0, body(3, 1), body(4, 0), BondKind::Single),
            ]
        );
    }

    /// The eight descriptors F2 writes — one on each `[$]O`, two on each
    /// `[$]COC[$]` — are consumed exactly once each, leaving no free port.
    #[test]
    fn test_f2_consumes_every_port_exactly_once() {
        let ir = resolved("{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}");
        assert_eq!(
            body_ends(&ir.pairs[0]),
            vec![
                (0, 0),
                (1, 0),
                (1, 1),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (4, 0)
            ]
        );
    }

    // -- F3: Martini benzene (R4.14, R4.16) ---------------------------------

    /// `{[#TC5]1[#TC5][#TC5]1}` forms edges in parse order (0,1), (1,2),
    /// (0,2) — the ring-closure edge where its marker was read. Every port of
    /// `[$]cc[$]` is a written-aromatic carbon and no edge writes a symbol, so
    /// all three inter-bead bonds are aromatic (R4.14).
    #[test]
    fn test_f3_resolves_three_aromatic_pairs_in_parse_order() {
        let ir = resolved("{[#TC5]1[#TC5][#TC5]1}.{#TC5=[$]cc[$]}");
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Aromatic),
                pair(1, 0, body(1, 1), body(2, 0), BondKind::Aromatic),
                pair(2, 0, body(0, 1), body(2, 1), BondKind::Aromatic),
            ]
        );
    }

    // -- F4: a coarse double edge is two single bonds (R4.11, R6.2) ---------

    /// `{[#SC3]=[#SC3]}` is **one** edge of multiplicity 2, so it is resolved
    /// twice — bond 0 and bond 1 — and each attempt takes the next free
    /// compatible port pair. Neither bond is a double bond: the coarse order
    /// is a count, the resolved kind is chemistry.
    #[test]
    fn test_f4_double_edge_resolves_to_two_single_pairs() {
        let ir = resolved("{[#SC3]=[#SC3]}.{#SC3=[$]CCC[$]}");
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Single),
                pair(0, 1, body(0, 1), body(1, 1), BondKind::Single),
            ]
        );
    }

    // -- F5: polystyrene, the scan order made visible (R4.6, R4.12) ---------

    /// `{[#BB]([#PH])|2}` is BB0(PH1), BB2(PH3) with edges in parse order
    /// (0,1), (2,3), (0,2): the `|` copy replays the branch before the bond
    /// that chains it on. `[>]CC[<][$]` offers ports 0 `>` on C0, 1 `<` on C1
    /// and 2 `$` on C1; `[$]c1ccccc1` offers one `$`.
    ///
    /// Edge (BB0,PH1): `>` vs `$` no, `<` vs `$` no, `$` vs `$` pair.
    /// Edge (BB2,PH3): the same three steps on the second copy.
    /// Edge (BB0,BB2): `>` vs BB2's `>` no, `>` vs BB2's `<` pair.
    #[test]
    fn test_f5_resolves_three_pairs_in_parse_order() {
        let ir = resolved("{[#BB]([#PH])|2}.{#BB=[>]CC[<][$],#PH=[$]c1ccccc1}");
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 2), body(1, 0), BondKind::Single),
                pair(1, 0, body(2, 2), body(3, 0), BondKind::Single),
                pair(2, 0, body(0, 0), body(2, 1), BondKind::Single),
            ]
        );
    }

    /// Two ports stay free — BB0's `<` (port 1) and BB2's `>` (port 0) — and
    /// are dropped rather than forced (R4.17). Asserted as absence from every
    /// pair, never as a reduced pair count.
    #[test]
    fn test_f5_leaves_the_two_incompatible_ports_unpaired() {
        let ir = resolved("{[#BB]([#PH])|2}.{#BB=[>]CC[<][$],#PH=[$]c1ccccc1}");
        let ends = body_ends(&ir.pairs[0]);
        assert!(!ends.contains(&(0, 1)), "ends were {ends:?}");
        assert!(!ends.contains(&(2, 0)), "ends were {ends:?}");
    }

    // -- F6: a tripeptide, `>` pairs only with `<` (R4.6) -------------------

    /// `[>]NCC(=O)[<]` offers port 0 `>` on its nitrogen and port 1 `<` on the
    /// carbonyl carbon; `[>]NC(C)C(=O)[<]` the same two roles. Each edge
    /// therefore pairs the source's `>` with the target's `<`.
    #[test]
    fn test_f6_resolves_two_left_right_pairs() {
        let ir = resolved("{[#GLY][#ALA][#GLY]}.{#GLY=[>]NCC(=O)[<],#ALA=[>]NC(C)C(=O)[<]}");
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 0), body(1, 1), BondKind::Single),
                pair(1, 0, body(1, 0), body(2, 1), BondKind::Single),
            ]
        );
    }

    /// The chain's two termini keep their unpaired ports: GLY0's `<` and
    /// GLY2's `>`.
    #[test]
    fn test_f6_leaves_the_chain_termini_unpaired() {
        let ir = resolved("{[#GLY][#ALA][#GLY]}.{#GLY=[>]NCC(=O)[<],#ALA=[>]NC(C)C(=O)[<]}");
        let ends = body_ends(&ir.pairs[0]);
        assert!(!ends.contains(&(0, 1)), "ends were {ends:?}");
        assert!(!ends.contains(&(2, 0)), "ends were {ends:?}");
    }

    // -- F7: labels are matched exactly (R4.6) ------------------------------

    /// Dichlorotoluene over three beads: `$a` pairs only `$a`, `$b` only `$b`
    /// and the unlabelled `$` only the unlabelled `$`, so the three edges have
    /// no freedom at all. Every port atom is a written-aromatic carbon and no
    /// symbol is written, so every kind is aromatic.
    #[test]
    fn test_f7_resolves_three_label_forced_pairs() {
        let ir = resolved(
            "{[#SC4]1[#SX3][#SX3A]1}.{#SC4=Cc[$a]c[$],#SX3=Clc[$a]c[$b],#SX3A=Clc[$b]c[$]}",
        );
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, body(0, 0), body(1, 0), BondKind::Aromatic),
                pair(1, 0, body(1, 1), body(2, 0), BondKind::Aromatic),
                pair(2, 0, body(0, 1), body(2, 1), BondKind::Aromatic),
            ]
        );
    }

    /// Flipping `SX3A`'s `$b` to `$c` leaves edge 1 — `[#SX3]`–`[#SX3A]` —
    /// with a free `$b` on one side and `$c`, `$` on the other. The reference
    /// implementation drops such an edge silently; molrs refuses it, spanned
    /// at the level-0 edge that cannot be formed: `[#SX3A]` occupies bytes
    /// 14..21 of the string (`{[#SC4]1[#SX3]` is fourteen bytes).
    #[test]
    fn test_f7_with_a_flipped_label_refuses_the_edge_it_breaks() {
        let text = "{[#SC4]1[#SX3][#SX3A]1}.{#SC4=Cc[$a]c[$],#SX3=Clc[$a]c[$b],#SX3A=Clc[$c]c[$]}";
        let err = err_of(text);
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgUnmatchableEdge { level: 0, edge: 1 }
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span, Span::new(14, 21));
    }

    // -- F8: three resolutions, and the edges one level induces in the next --

    /// F8 — beads of blocks, blocks of beads, beads of atoms.
    const F8: &str = "{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
                      {#PEO=[>]COC[<],#PE=[>]CC[<]}";

    /// Syntax, instantiation and validation pair nothing: they hand `resolve`
    /// one empty pair list per level.
    #[test]
    fn test_f8_carries_no_pairs_before_resolution() {
        let ir = unresolved(F8);
        assert_eq!(ir.pairs, vec![Vec::<ResolvedPair>::new(), Vec::new()]);
    }

    /// Every edge the parser and the instantiation step build is written
    /// notation; only `resolve` may tag an edge `Derived`.
    #[test]
    fn test_f8_edges_are_all_written_before_resolution() {
        let ir = unresolved(F8);
        let origins: Vec<&EdgeOrigin> = ir
            .levels
            .iter()
            .flat_map(|level| level.edges.iter().map(|edge| &edge.origin))
            .collect();
        assert_eq!(origins, vec![&EdgeOrigin::Written; 5]);
    }

    /// Level 0 is resolved against the **children** of its beads, so both ends
    /// are `Sub`: `[>][#PEO][#PEO][<]` gives bead 0 a `>` on child 0 and a `<`
    /// on child 1, and edge (0,1) pairs child 0's `>` with bead 1's `<`-child,
    /// which is child 3.
    #[test]
    fn test_f8_level_zero_pairs_the_child_nodes_of_its_beads() {
        let ir = resolved(F8);
        assert_eq!(
            ir.pairs[0],
            vec![
                pair(0, 0, sub(0, 0), sub(3, 0), BondKind::Single),
                pair(1, 0, sub(2, 0), sub(5, 0), BondKind::Single),
            ]
        );
    }

    /// A `Sub` end names no instance of its own because it does not have to:
    /// the child's `parent` is the bead, and the bead is the edge endpoint.
    #[test]
    fn test_f8_sub_ends_belong_to_the_beads_their_edge_joins() {
        let ir = resolved(F8);
        for resolved_pair in &ir.pairs[0] {
            let edge = &ir.levels[0].edges[resolved_pair.edge];
            let PairEnd::Sub { node: src, .. } = resolved_pair.src else {
                panic!("level-0 src was {:?}", resolved_pair.src);
            };
            let PairEnd::Sub { node: dst, .. } = resolved_pair.dst else {
                panic!("level-0 dst was {:?}", resolved_pair.dst);
            };
            assert_eq!(ir.levels[1].nodes[src].parent, Some(edge.i));
            assert_eq!(ir.levels[1].nodes[dst].parent, Some(edge.j));
        }
    }

    /// Each level-0 pair induces one edge between the two child nodes it
    /// joined. They are appended after the three written intra-bead edges, in
    /// pair order, so indices already handed out stay stable.
    #[test]
    fn test_f8_appends_one_derived_edge_per_level_zero_pair() {
        let ir = resolved(F8);
        let edges: Vec<(usize, usize, EdgeOrigin)> = ir.levels[1]
            .edges
            .iter()
            .map(|edge| (edge.i, edge.j, edge.origin.clone()))
            .collect();
        assert_eq!(
            edges,
            vec![
                (0, 1, EdgeOrigin::Written),
                (2, 3, EdgeOrigin::Written),
                (4, 5, EdgeOrigin::Written),
                (0, 3, EdgeOrigin::Derived { level: 0, pair: 0 }),
                (2, 5, EdgeOrigin::Derived { level: 0, pair: 1 }),
            ]
        );
    }

    /// One resolved pair makes one bond, so a derived edge is `Single` — a
    /// count, not a placeholder.
    #[test]
    fn test_f8_derived_edges_are_single_bonds() {
        let ir = resolved(F8);
        let orders: Vec<CgBondOrder> = ir.levels[1].edges[3..]
            .iter()
            .map(|edge| edge.order)
            .collect();
        assert_eq!(orders, vec![CgBondOrder::Single, CgBondOrder::Single]);
    }

    /// The only text naming a derived edge is the coarse bond that induced it,
    /// so the derived edge carries that edge's span.
    #[test]
    fn test_f8_derived_edges_copy_the_span_of_the_inducing_edge() {
        let ir = resolved(F8);
        assert_eq!(ir.levels[1].edges[3].span, ir.levels[0].edges[0].span);
        assert_eq!(ir.levels[1].edges[4].span, ir.levels[0].edges[1].span);
    }

    /// Level 1 is the last level, so its ends are `Body` ends into the
    /// atomistic bodies: `[>]COC[<]` offers port 0 on its first carbon and
    /// port 1 on its last. The two derived edges are resolved after the three
    /// written ones, against the ports the written edges left free.
    #[test]
    fn test_f8_level_one_resolves_every_edge_including_the_derived_ones() {
        let ir = resolved(F8);
        assert_eq!(
            ir.pairs[1],
            vec![
                pair(0, 0, body(0, 0), body(1, 1), BondKind::Single),
                pair(1, 0, body(2, 0), body(3, 1), BondKind::Single),
                pair(2, 0, body(4, 0), body(5, 1), BondKind::Single),
                pair(3, 0, body(0, 1), body(3, 0), BondKind::Single),
                pair(4, 0, body(2, 1), body(5, 0), BondKind::Single),
            ]
        );
    }

    // -- a base-only string has nothing to pair against ---------------------

    /// No fragment table means no ports at all: the level is left alone rather
    /// than refused, exactly as 01b's single-block fixtures require.
    #[test]
    fn test_base_only_string_resolves_to_no_pairs() {
        let ir = resolved("{[#PEO][#PEO][#PEO]}");
        assert_eq!(ir.pairs, vec![Vec::<ResolvedPair>::new()]);
    }

    /// Its written edges survive resolution untouched.
    #[test]
    fn test_base_only_string_keeps_its_written_edges() {
        let ir = resolved("{[#PEO][#PEO][#PEO]}");
        assert_eq!(ir.levels[0].edges.len(), 2);
    }

    // -- F9: the bond-kind precedence (R4.14) -------------------------------

    /// No symbol written and both port atoms written aromatic: the Daylight
    /// rule promotes the bond to aromatic.
    #[test]
    fn test_unwritten_bond_between_aromatic_ports_is_aromatic() {
        let ir = resolved("{[#PH][#PH]}.{#PH=[$]c1ccccc1}");
        assert_eq!(ir.pairs[0][0].kind, BondKind::Aromatic);
    }

    /// The same ring spelled Kekulé-style writes no aromatic atom, so the same
    /// skeleton resolves to a single bond. Written aromaticity is what the
    /// rule reads — never what perception would later conclude.
    #[test]
    fn test_unwritten_bond_between_kekule_ports_is_single() {
        let ir = resolved("{[#PH][#PH]}.{#PH=[$]C1=CC=CC=C1}");
        assert_eq!(ir.pairs[0][0].kind, BondKind::Single);
    }

    /// Biphenyl: an explicit `-` on one side beats the aromatic promotion, and
    /// the two descriptors still pair because `None` and `Some(Single)` have
    /// the same *effective* order.
    #[test]
    fn test_explicit_single_between_aromatic_ports_stays_single() {
        let ir = resolved("{[#A][#B]}.{#A=c1ccccc1-[$],#B=[$]c1ccccc1}");
        assert_eq!(ir.pairs[0][0].kind, BondKind::Single);
    }

    /// `=` written on both sides: the explicit symbol is the resolved kind.
    #[test]
    fn test_double_written_on_both_ports_pairs_as_a_double_bond() {
        let ir = resolved("{[#PH][#PH]}.{#PH=c1ccccc1=[$]}");
        assert_eq!(ir.pairs[0][0].kind, BondKind::Double);
    }

    /// The order is compared every time (R4.9): `=[$]` does not pair a bare
    /// `[$]`, whose effective order is single. `[#B]` spans bytes 5..9.
    #[test]
    fn test_double_against_an_unwritten_order_is_unmatchable() {
        let text = "{[#A][#B]}.{#A=C=[$],#B=[$]C}";
        let err = err_of(text);
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgUnmatchableEdge { level: 0, edge: 0 }
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span, Span::new(5, 9));
    }

    // -- must-raise (R4.15) -------------------------------------------------

    /// Labels must match exactly: `$a` and `$b` are different classes, so the
    /// one edge between them can be formed by nothing.
    #[test]
    fn test_ports_differing_only_in_label_are_unmatchable() {
        let text = "{[#A][#B]}.{#A=[$a]C,#B=[$b]C}";
        let err = err_of(text);
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgUnmatchableEdge { level: 0, edge: 0 }
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span, Span::new(5, 9));
    }

    /// The flip is `Left` ↔ `Right`: two `>` are not complementary, however
    /// symmetric the string looks.
    #[test]
    fn test_two_right_descriptors_are_unmatchable() {
        let text = "{[#A][#B]}.{#A=CC[>],#B=[>]O}";
        let err = err_of(text);
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgUnmatchableEdge { level: 0, edge: 0 }
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span, Span::new(5, 9));
    }

    // -- the port order is the walker's order -------------------------------

    /// A port index is a position in the map `FragmentIr::to_atomistic` returns,
    /// and that map is in **visit** order, not text order: in `C([$]O)[>]` the
    /// `$` written inside the branch is visited before the `>` written after
    /// it, and both are anchored on the same atom — the carbon the branch
    /// hangs off.
    #[test]
    fn test_fragment_to_atomistic_map_is_in_walker_order() {
        let ir = fragment("C([$]O)[>]");
        let (mol, ports) = ir.to_atomistic().expect("fragment body must convert");
        assert_eq!(mol.n_atoms(), 2);
        let kinds: Vec<DescriptorKind> = ports.iter().map(|(_, d)| d.kind).collect();
        assert_eq!(
            kinds,
            vec![DescriptorKind::Symmetric, DescriptorKind::Right]
        );
        assert_eq!(ports[0].0, ports[1].0);
    }

    // -- the fragment cache converts one definition once ---------------------

    /// Three instances of one definition are three lookups and one
    /// conversion: the cache is keyed by definition name.
    #[test]
    fn test_fragment_cache_builds_one_entry_for_three_lookups() {
        let body = fragment("[>]COC[<]");
        let mut cache = FragmentCache::default();
        for _ in 0..3 {
            cache
                .get_or_build("PEO", &body)
                .expect("fragment body must convert");
        }
        assert_eq!(cache.entries.len(), 1);
    }

    /// What a lookup hands back is the body and its ports: `[>]COC[<]` is
    /// three heavy atoms with a port on each terminal carbon.
    #[test]
    fn test_fragment_cache_returns_the_body_and_its_ports() {
        let body = fragment("[>]COC[<]");
        let mut cache = FragmentCache::default();
        let (mol, ports) = cache
            .get_or_build("PEO", &body)
            .expect("fragment body must convert");
        assert_eq!(mol.n_atoms(), 3);
        assert_eq!(ports.len(), 2);
    }

    // -- a body that parses and then fails to build -------------------------

    /// `[$]C1CC` parses as a fragment body — the body parser leaves ring
    /// closures unchecked — and fails when resolution converts it. The
    /// converter's kind is kept, boxed inside `CgLastBlockNotAtomistic`, and
    /// the diagnostic is re-based onto the whole `CGsmiles` string: the `#A=`
    /// body starts at byte 15, so its `1` is bytes 19..20 and the caret lands
    /// there rather than on byte 4 of the body.
    #[test]
    fn test_unmatched_ring_closure_in_a_body_is_boxed_and_rebased() {
        let text = "{[#A][#B]}.{#A=[$]C1CC,#B=[$]C}";
        let err = err_of(text);
        assert!(
            matches!(
                err.kind,
                SmilesErrorKind::CgLastBlockNotAtomistic(ref inner)
                    if **inner == SmilesErrorKind::UnmatchedRingClosure(1)
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.span, Span::new(19, 20));
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
    }

    // -- R4.12: the entity of the last level is the atom --------------------

    /// `[$][>]CCC` writes both descriptors on its first carbon, so instance 0
    /// offers one entity of two ports — 0 `$`, 1 `>` — while `[<]CCO[$]`
    /// offers two entities, port 0 `<` on its first carbon and port 1 `$` on
    /// its oxygen. The scan takes target entities before source ports, so the
    /// first free compatible pair is `>` against `<`: port 1 of instance 0
    /// with port 0 of instance 1. Grouping a whole body as one entity would
    /// have tried `$` against `$` — port 0 with port 1 — a different molecule.
    #[test]
    fn test_two_descriptors_on_one_atom_are_scanned_as_one_entity() {
        let ir = resolved("{[#A][#B]}.{#A=[$][>]CCC,#B=[<]CCO[$]}");
        assert_eq!(
            ir.pairs[0],
            vec![pair(0, 0, body(0, 1), body(1, 0), BondKind::Single)]
        );
    }

    // -- `flip` and `compatible`, the pairing rule itself (R4.6) ------------

    /// A descriptor as written: the operator, its label and the bond symbol
    /// beside its bracket.
    fn descriptor(kind: DescriptorKind, label: &str, order: Option<BondKind>) -> BondingDescriptor {
        BondingDescriptor {
            kind,
            label: label.to_owned(),
            order,
        }
    }

    /// `flip` is an involution on the whole kind set, which is what makes
    /// `compatible` symmetric however the scan orders its two arguments.
    #[test]
    fn test_flip_is_an_involution_on_every_kind() {
        for kind in [
            DescriptorKind::Symmetric,
            DescriptorKind::Left,
            DescriptorKind::Right,
            DescriptorKind::Shared,
        ] {
            assert_eq!(flip(flip(kind)), kind, "kind was {kind:?}");
        }
    }

    /// The two roles of the AB-type operator exchange: `<` asks for `>`.
    #[test]
    fn test_flip_exchanges_the_two_ab_type_roles() {
        assert_eq!(flip(DescriptorKind::Left), DescriptorKind::Right);
        assert_eq!(flip(DescriptorKind::Right), DescriptorKind::Left);
    }

    /// The two self-pairing operators are their own partner.
    #[test]
    fn test_flip_keeps_the_self_pairing_kinds() {
        assert_eq!(flip(DescriptorKind::Symmetric), DescriptorKind::Symmetric);
        assert_eq!(flip(DescriptorKind::Shared), DescriptorKind::Shared);
    }

    /// `<` pairs `>` whichever of the two the scan reads first.
    #[test]
    fn test_compatible_pairs_left_with_right_in_both_directions() {
        let left = descriptor(DescriptorKind::Left, "", None);
        let right = descriptor(DescriptorKind::Right, "", None);
        assert!(compatible(&left, &right));
        assert!(compatible(&right, &left));
    }

    /// Two `<` are not complementary: the flip is `Left` ↔ `Right`, never
    /// `Left` ↔ `Left`.
    #[test]
    fn test_compatible_refuses_two_left_descriptors() {
        let left = descriptor(DescriptorKind::Left, "", None);
        assert!(!compatible(&left, &left));
    }

    /// `$` is self-complementary, so two of them pair.
    #[test]
    fn test_compatible_pairs_two_symmetric_descriptors() {
        let symmetric = descriptor(DescriptorKind::Symmetric, "", None);
        assert!(compatible(&symmetric, &symmetric));
    }

    /// `flip` sends `Shared` to itself, so `compatible` says two squash
    /// operators pair — R4.6's own statement. **Unreachable in practice**: no
    /// string carrying `[!]` reaches `resolve`, because `validate_ir` refuses
    /// the squash operator first. The case is pinned here so the involution
    /// stays total rather than acquiring a hole nothing would notice.
    #[test]
    fn test_compatible_pairs_two_shared_descriptors() {
        let shared = descriptor(DescriptorKind::Shared, "", None);
        assert!(compatible(&shared, &shared));
    }

    /// Labels are compared character for character, so `$a` and `$b` are
    /// different classes of the same kind.
    #[test]
    fn test_compatible_refuses_descriptors_differing_only_in_label() {
        let a = descriptor(DescriptorKind::Symmetric, "a", None);
        let b = descriptor(DescriptorKind::Symmetric, "b", None);
        assert!(!compatible(&a, &b));
    }

    /// The effective order has to agree too: an unwritten symbol is single,
    /// so `=[$]` refuses a bare `[$]`.
    #[test]
    fn test_compatible_refuses_a_double_against_an_unwritten_order() {
        let double = descriptor(DescriptorKind::Symmetric, "", Some(BondKind::Double));
        let unwritten = descriptor(DescriptorKind::Symmetric, "", None);
        assert!(!compatible(&double, &unwritten));
    }

    /// Nothing written and an explicit `-` are the same effective order, so
    /// `[$]` pairs `-[$]` — biphenyl's case.
    #[test]
    fn test_compatible_pairs_an_unwritten_order_with_an_explicit_single() {
        let unwritten = descriptor(DescriptorKind::Symmetric, "", None);
        let single = descriptor(DescriptorKind::Symmetric, "", Some(BondKind::Single));
        assert!(compatible(&unwritten, &single));
    }
}
