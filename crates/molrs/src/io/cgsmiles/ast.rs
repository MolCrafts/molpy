//! `CGsmiles` coarse-graph intermediate representation.

use std::collections::BTreeMap;

use crate::io::smiles::{BondKind, BondingDescriptor, FragmentIr, Span};
use crate::op::F;

/// A parsed `CGsmiles` string: the blocks that were written, and the levels
/// they denote.
///
/// A *resolution level* is one view of the same molecule: the coarse level
/// names beads, and a finer level names what each bead is made of. A
/// `CGsmiles` string carries one base block and any number of **fragment
/// blocks** after it, each mapping the names used one level up to the bodies
/// that replace them.
///
/// # Authority
///
/// `fragments[k]` is the **authority** for a fragment's shape: it is what the
/// string wrote. `levels[k + 1]` is the **expansion** that table denotes — one
/// disjoint copy of a body per node of `levels[k]` — a derived representation
/// the reader produces once. Editing one would not update the other, which is
/// why there is no public constructor and no supported mutation: a value of
/// this type is read, not built.
///
/// # Alignment invariant
///
/// A base-only string has no fragment table and one level:
/// `fragments.is_empty() && levels.len() == 1`. Otherwise `fragments.len()` is
/// the number of fragment blocks that were written and
/// `levels.len() == fragments.len()` — one fewer level than one might expect,
/// because the **last** block is atomistic and builds no coarse level. That
/// asymmetry is intentional: levels are graphs of beads, and atoms are not
/// [`CgNode`]s. The last table's bodies stay
/// [`FragmentBody::Smiles`] values with their bonding
/// descriptors intact.
///
/// # A value the reader returns is finished
///
/// Every `CgSmilesIr` returned by
/// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse) is **fully
/// instantiated, validated and resolved**: every name used at a level has a
/// definition, every intermediate table has been expanded into the next level,
/// every bonding descriptor a written edge needs has been paired and recorded
/// in [`pairs`](CgSmilesIr::pairs), and every refusal this version makes
/// (starting with the squash operator `[!]`) has already been raised. The
/// reader hands out no half-resolved state. The fields are `pub`, so the type
/// does not enforce this for a hand-built value — the reader is the only
/// supported source.
#[derive(Debug, Clone, PartialEq)]
pub struct CgSmilesIr {
    /// Resolution levels, coarsest first. Level 0 is the base block that was
    /// written; every later level is the expansion of the table above it.
    pub levels: Vec<CgGraph>,
    /// Fragment tables, coarsest first, each keyed by the fragment name
    /// written after `#`. `fragments[k]` resolves the names of `levels[k]`.
    ///
    /// [`BTreeMap`] rather than a hash map so iteration — and therefore any
    /// diagnostic or example that walks a table — is in name order, the way
    /// the crate already keys name tables.
    pub fragments: Vec<BTreeMap<String, CgFragmentDef>>,
    /// Resolved descriptor pairs, parallel to [`levels`](CgSmilesIr::levels):
    /// `pairs[k]` holds the pairs that satisfy `levels[k].edges`, in
    /// resolution order — edge order first, then bond index within an edge of
    /// multiplicity greater than one.
    ///
    /// A pair names the two *ports* — occurrences of a bonding descriptor,
    /// defined at [`PairEnd`] — that the bond consumed.
    ///
    /// A level with nothing to pair against carries an empty list rather than
    /// a missing one: a base-only string (no fragment table) writes edges
    /// whose endpoints offer no ports at all, and `pairs == vec![vec![]]` is
    /// the honest record of that.
    pub pairs: Vec<Vec<ResolvedPair>>,
    /// Byte range of the whole parsed string.
    pub span: Span,
}

/// One end of a [`ResolvedPair`]: the port that was consumed, and the entity
/// that offered it.
///
/// A **port** is one occurrence of a bonding descriptor — one written `[$]`,
/// `[<]` or `[>]`, the marks that say where a fragment may later be joined to
/// another (see [`BondingDescriptor`]). A port is offered once and consumed by
/// at most one bond, so a fragment written `[$]COC[$]` offers two of them, and
/// a port is addressed by its **index** in the list the entity wrote its
/// descriptors in.
///
/// The two variants are the two kinds of level a pair can be resolved at, and
/// they are separate variants rather than one struct with an optional field
/// because a field whose meaning depends on a sibling field is a shape no
/// reader can check.
#[derive(Debug, Clone, PartialEq)]
pub enum PairEnd {
    /// An end at an **intermediate** level, where the ports of a bead are the
    /// descriptors its child nodes carry one level down.
    ///
    /// `node` indexes `levels[k + 1].nodes` — the child carrying the port —
    /// and `port` indexes that child's [`descriptors`](CgNode::descriptors).
    /// The bead the port belongs to is
    /// [`levels[k + 1].nodes[node].parent`](CgNode::parent), which is also the
    /// endpoint of the pair's edge, so a third field repeating it would be a
    /// second copy of a fact that already has an owner.
    Sub {
        /// Index into `levels[k + 1].nodes` of the child carrying the port.
        node: usize,
        /// Index into that child's [`descriptors`](CgNode::descriptors).
        port: usize,
    },
    /// An end at the **last** level, where the ports of a node are the
    /// descriptors written on its atomistic body — there is no child node to
    /// hang them on.
    ///
    /// `instance` indexes `levels[k].nodes` and `port` indexes the descriptor
    /// map
    /// [`FragmentIr::to_atomistic`](crate::io::smiles::FragmentIr::to_atomistic)
    /// returns for that node's body, whose order is the walker's visit order.
    Body {
        /// Index into `levels[k].nodes` of the node whose body holds the port.
        instance: usize,
        /// Index into that body's descriptor map.
        port: usize,
    },
}

/// One bond a written coarse edge stands for, with the two ports it consumed.
///
/// Pairing is what turns "these two beads are bonded" into "this port of this
/// bead is bonded to that port of that one" — the fact an expansion needs and
/// a coarse edge does not carry.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPair {
    /// Index into `levels[k].edges` of the edge this pair satisfies, in the
    /// **final** edge list: derived edges are appended to a level before that
    /// level is resolved, so this index is valid against the list a reader
    /// holds.
    pub edge: usize,
    /// Which of that edge's [`CgBondOrder::multiplicity`] bonds this is,
    /// 0-based: a coarse edge of order *n* stands for *n* distinct bonds, each
    /// consuming its own pair of ports.
    pub bond: usize,
    /// The port the edge's first endpoint offered.
    pub src: PairEnd,
    /// The port the edge's second endpoint offered.
    pub dst: PairEnd,
    /// The chemistry of the resolved bond: the order the notation wrote on
    /// either descriptor, [`BondKind::Aromatic`] when neither wrote one and
    /// both port atoms are written aromatic, and [`BondKind::Single`]
    /// otherwise.
    ///
    /// Named `kind`, not `order`, so it never reads as
    /// [`CgEdge::order`] — which is a count of bonds, not a bond order.
    pub kind: BondKind,
}

/// Where an edge of a level came from: the notation, or the resolution of the
/// level above it.
///
/// Written and derived edges live in **one** list — a consumer walking a
/// level's edges must see all of them — and this tag is how provenance is
/// recovered.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeOrigin {
    /// The notation wrote this edge: adjacency, a ring closure, or the bond
    /// that chains one `|n` copy to the previous one.
    Written,
    /// Resolution created this edge from a pair of the level above: `pairs`
    /// entry `pair` of level `level` joined two child nodes, and the bond it
    /// stands for is an edge of *this* level.
    ///
    /// Derived edges are appended **after** every written edge of the level, in
    /// pair order, so an index handed out before they arrive stays valid. Each
    /// carries [`CgBondOrder::Single`] — one resolved pair is one bond, and a
    /// coarse order is a count rather than a placeholder — and a copy of the
    /// [`span`](CgEdge::span) of the coarse edge whose pair induced it, the
    /// only text that names that bond.
    ///
    /// Derived edges bypass the parser's simple-graph check on purpose: a
    /// coarse edge of multiplicity *n* ≥ 2 stands for *n* separate bonds
    /// (R6.2) and so induces *n* derived edges between the **same** pair of
    /// child nodes. An instantiated level may therefore carry parallel edges,
    /// which the notation could not have written itself.
    Derived {
        /// Index of the level whose resolution induced this edge.
        level: usize,
        /// Index into `pairs[level]` of the pair that induced it.
        pair: usize,
    },
}

/// One entry of a fragment block: `#PEO=[$]COC[$]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CgFragmentDef {
    /// The fragment name written after `#`, without the sigil.
    pub name: String,
    /// What the name stands for: a coarse graph of the next level's nodes, or
    /// an atomistic SMILES fragment body.
    pub body: FragmentBody,
    /// Byte range of the whole entry — name, `=` and body — in the parsed
    /// input.
    pub span: Span,
}

/// The two shapes a fragment body may take, told apart by the block's
/// **position** rather than by any syntax: the last block's bodies are
/// atomistic, every earlier block's are coarse.
///
/// The notation marks neither (the reference implementation passes its reader
/// a `last_all_atom` flag instead), so a string whose deepest resolution is
/// meant to stay coarse-grained cannot be written — the limitation that
/// positional dispatch buys.
#[derive(Debug, Clone, PartialEq)]
pub enum FragmentBody {
    /// A coarse graph over the *next* level's `[#X]` nodes, as written in an
    /// intermediate block: `#B1=[>][#PEO][#PEO][<]`.
    Graph(CgGraph),
    /// An atomistic OpenSMILES body with bonding descriptors, as written in
    /// the last block: `#PEO=[>]COC[<]`. It is kept as the parser returned
    /// it — never expanded into atoms here.
    Smiles(FragmentIr),
}

/// One resolution level: coarse-grained nodes and the edges between them.
///
/// A coarse level is the parsed form of one block such as `{[#PEO][#PEO]}`:
/// named nodes, typed edges, and a byte [`Span`] back into the input for every
/// one of them. Like every record of the notation, it is a value produced by
/// [`CgSmilesIr::parse`]: the fields are public to read, there is no public
/// constructor, and there is no supported mutation — the same shape as
/// [`SmilesIr`](crate::io::smiles::SmilesIr).
///
/// Both vectors are in **parse order** — nodes in the order their brackets
/// were read, edges in the order the notation formed them, with a ring-closure
/// edge emitted where its closing marker was read. The `CGsmiles` reference
/// implementation (github.com/gruenewald-lab/CGsmiles) iterates its graph
/// library's adjacency order instead, so a ring-bearing string indexes edges
/// differently there.
#[derive(Debug, Clone, PartialEq)]
pub struct CgGraph {
    /// Nodes in parse order; a node is addressed by its index in this vector.
    pub nodes: Vec<CgNode>,
    /// Edges in parse order, each naming two indices into `nodes`.
    pub edges: Vec<CgEdge>,
}

/// One coarse-grained node: `[#PEO]`, `[#A;q=-0.5]`.
///
/// Two occurrences of `[#A]` are two distinct nodes that happen to share a
/// name; the name is a *fragment* name to be resolved later, not an identity.
#[derive(Debug, Clone, PartialEq)]
pub struct CgNode {
    /// The fragment name written after `#`, without the sigil: `PEO`, or `*`
    /// for the wildcard node.
    pub name: String,
    /// Partial charge in `e` (`CGsmiles` `q`, positional slot 2), `None` when
    /// the notation wrote none.
    ///
    /// The unit is the molrs convention — the notation states none — and the
    /// field is a **partial**, not a formal, charge: the formal charge of an
    /// atom is
    /// [`AtomSpec::Bracket`](crate::io::smiles::AtomSpec::Bracket)'s
    /// `charge: Option<i8>`, an integer count of elementary charges written
    /// `[NH4+]`. This one is a fractional force-field charge attached to a
    /// bead.
    pub charge: Option<F>,
    /// Annotations the dialect does not reserve, as written, in written order.
    ///
    /// `[#A;q=-0.5;kind=ether]` leaves `[("kind", "ether")]` here. The
    /// reserved keys never reach this list: `q` is bound to
    /// [`charge`](CgNode::charge); `w` — the mapping weight, dimensionless —
    /// is accepted only at its default `1` and then dropped, any other value
    /// being refused; and `x`, the chirality key, is refused outright. A field
    /// written with no `=` names no key and takes the next positional slot
    /// instead (slot 2 is `q`, slot 3 is `w`), so `[#A;-0.5]` sets the charge
    /// rather than landing here.
    pub annotations: Vec<(String, String)>,
    /// Bonding descriptors written beside this node, in written order.
    ///
    /// A *bonding descriptor* — `[$]`, `[<]`, `[>]` or `[!]`, optionally
    /// labelled as in `[$1]` — marks a site at which this bead may later be
    /// joined to another fragment; [`BondingDescriptor`] carries the glyph,
    /// the label and the bond order of the join.
    ///
    /// A descriptor is a pairing *class*, consumed one occurrence at a time —
    /// which is why the `|n` repeat operator clones the descriptors of the
    /// unit it copies into every copy, while it refuses to replay a ring
    /// marker (a one-shot identity).
    pub descriptors: Vec<BondingDescriptor>,
    /// Index into the *previous* level's `nodes` of the node this one was
    /// instantiated from.
    ///
    /// `None` in `levels[0]` — nothing above the base graph instantiated it —
    /// and `None` in every [`FragmentBody::Graph`] body, which is a template
    /// rather than an instance. It is **set by instantiation, never
    /// inherited**: the copies of one body used by two different nodes carry
    /// the two different parents, not the body's own value.
    pub parent: Option<usize>,
    /// Byte range of the node's own text within the parsed input.
    ///
    /// Spans in a [`CgGraph`] are neither disjoint nor monotonic: the copies
    /// the `|n` repeat operator makes are re-read from the template's bytes,
    /// so every copy **shares** the template's span.
    pub span: Span,
}

/// One coarse edge, joining `nodes[i]` and `nodes[j]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CgEdge {
    /// Index of the first endpoint in [`CgGraph::nodes`].
    pub i: usize,
    /// Index of the second endpoint in [`CgGraph::nodes`].
    pub j: usize,
    /// Multiplicity of the edge, from the bond symbol that formed it, or
    /// [`CgBondOrder::Single`] when the notation wrote no symbol.
    pub order: CgBondOrder,
    /// Byte range of the text at which the edge was formed.
    ///
    /// Which text that is depends on how the edge came about: for a bond
    /// between two adjacent nodes it is the bracket of the *second* node, for
    /// a ring closure it is the closing marker, and for the bond that chains
    /// one `|n` copy to the previous one it is the `|` itself. Like
    /// [`CgNode::span`], a replayed copy shares the template's span.
    ///
    /// A [`Derived`](EdgeOrigin::Derived) edge writes no text of its own, so
    /// it carries a **copy of the span of the coarse edge whose pair induced
    /// it** — the only text that names that bond — which keeps this field
    /// total and points a diagnostic at something the caller wrote.
    pub span: Span,
    /// Whether the notation wrote this edge or resolution derived it.
    pub origin: EdgeOrigin,
}

/// Multiplicity of a coarse edge, as its bond symbol writes it: a
/// dimensionless count of 1..=4 bonds.
///
/// This is deliberately not
/// [`BondKind`](crate::io::smiles::BondKind): that enum also carries
/// `Aromatic`, `Up`, `Down`, `Any` and `Ring`, none of which means anything
/// between two beads, and reusing it would let a stereo bond typecheck its way
/// into a coarse edge. An edge here is a *multiplicity*; the chemistry of a
/// resolved bond is a `BondKind` and is named `kind` where it belongs.
///
/// Order 0 — `.`, the virtual edge the notation permits between virtual
/// particles — is not representable here on purpose:
/// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse) refuses `.` with
/// [`SmilesErrorKind::CgInvalidBondOrder`](crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder)
/// rather than admit an edge that is not an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CgBondOrder {
    /// `-`, and the default when no symbol is written.
    Single,
    /// `=`.
    Double,
    /// `#`.
    Triple,
    /// `$`.
    Quadruple,
}

impl CgBondOrder {
    /// The multiplicity as a number, 1..=4: how many bonds this edge stands
    /// for when the level is expanded into atoms. A pure count, dimensionless.
    pub fn multiplicity(&self) -> u8 {
        match self {
            CgBondOrder::Single => 1,
            CgBondOrder::Double => 2,
            CgBondOrder::Triple => 3,
            CgBondOrder::Quadruple => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Bond-symbol → multiplicity table: `-` 1, `=` 2, `#` 3, `$` 4. (Rule
    // R2.4 of `.claude/specs/cgsmiles-01b-graph.md` § Domain basis.)
    // Hand-written from the cited documentation; no external tool.

    #[test]
    fn test_single_bond_has_multiplicity_one() {
        assert_eq!(CgBondOrder::Single.multiplicity(), 1);
    }

    #[test]
    fn test_double_bond_has_multiplicity_two() {
        assert_eq!(CgBondOrder::Double.multiplicity(), 2);
    }

    #[test]
    fn test_triple_bond_has_multiplicity_three() {
        assert_eq!(CgBondOrder::Triple.multiplicity(), 3);
    }

    #[test]
    fn test_quadruple_bond_has_multiplicity_four() {
        assert_eq!(CgBondOrder::Quadruple.multiplicity(), 4);
    }
}
