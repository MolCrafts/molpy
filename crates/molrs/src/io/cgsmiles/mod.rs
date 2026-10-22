//! `CGsmiles`, the coarse-grained line notation (Grünewald et al., JCTC 2025):
//! parser and intermediate representation.
//!
//! The door is [`CgSmilesIr::parse`]; the molecule a string states (its
//! lowest level expanded into atoms) is
//! `CgSmilesIr::parse(text)?.to_atomistic()`. This module holds the
//! notation's classes: [`CgSmilesIr`] and the records it hands out.
//!
//! `CGsmiles` writes a molecule at a *coarse-grained* resolution: a block
//! `{…}` of named **beads** — single nodes each standing for a whole fragment
//! of the molecule — joined like atoms in SMILES. `{[#PEO][#PEO][#PEO]}` is a
//! three-bead trimer of poly(ethylene oxide); each `#NAME` is a fragment name
//! that a later block resolves. A whole string is a `.`-separated sequence of
//! blocks: block 0 is the coarsest graph, and every block after it is a
//! *fragment table* naming the bodies of the nodes written one level up. This
//! module reads all of them into a [`CgSmilesIr`], and expands each
//! intermediate table into the level it denotes.
//!
//! **Reading builds no atoms.** What the reader produces is the coarse levels,
//! their fragment tables and the descriptor pairing over them — no `Frame`, no
//! `MolGraph`. Turning the lowest level into real atoms and bonds is a second
//! step the caller asks for by name, [`CgSmilesIr::to_atomistic`]. Beside it
//! stands [`CgSmilesIr::templates`], which builds the *pieces* rather than
//! the whole: one instance-free
//! ported [`Atomistic`](crate::core::Atomistic) **template** per definition of the
//! last fragment table, each open valence made explicit as a capping hydrogen
//! carrying a port. Expansion is the molecule the string states; a template is
//! what a builder places, many times, without re-reading the string. The third
//! conversion, [`CgSmilesIr::to_coarsegrain`], needs no fragment table at all:
//! it reads the coarsest level, `levels[0]`, as a
//! [`CoarseGrain`](crate::core::CoarseGrain) bead graph — one
//! bead per node, one CG bond per edge, no coordinates.
//!
//! # What the block grammar says
//!
//! * A node is `[#NAME]`, optionally annotated `[#NAME;q=-0.5;kind=ether]`.
//!   Two `[#A]` occurrences are two distinct nodes sharing a name.
//! * Adjacency is a bond. The symbols `-`, `=`, `#`, `$` write the
//!   multiplicities 1..=4, and the default is 1. One symbol serves every
//!   position it may be written in: between two nodes, before a ring marker,
//!   around a branch, before `|` — where it becomes the bond that joins each
//!   repeated copy to the previous one — and before a bonding-descriptor
//!   bracket, where it becomes the order the descriptor's later pairing will
//!   take.
//! * `( … )` branches and nests, attaching to the node written before `(`.
//! * A bare digit, or `%` and the whole digit run that follows it (`%1` is
//!   marker 1, `%123` is marker 123, up to 65535), opens a ring marker that
//!   its second occurrence closes; a marker number is reusable once closed.
//!   The closure's bond order may be written at *either* end — an order
//!   belongs to the bond, not to the end it was written at — and two
//!   **differing** explicit symbols are a conflict
//!   ([`SmilesErrorKind::RingBondConflict`]), not an override.
//! * `|n` repeats the preceding unit `n` times in total, chaining the copies.
//!   After `)` the unit is the whole branch **including its anchor**, so
//!   `{[#A]([#B][#C])|2}` is six nodes, not four.
//! * Node annotations bind positionally: slot 1 is the `#NAME`, slot 2 is `q`
//!   (partial charge), slot 3 is `w` (mapping weight). `[#A;0.5]` therefore
//!   means `[#A;q=0.5]`, and a fourth positional field has no slot to take.
//!   Keys the table does not reserve are kept verbatim — except `x`, the
//!   dialect's chirality key, which is refused (below).
//!
//! Units: `charge` is a **partial charge in elementary charge units `e`**, not
//! a formal charge (`.claude/notes/science.md`); `w` is dimensionless; a bond
//! order is a dimensionless multiplicity 1..=4.
//!
//! # What the fragment blocks say
//!
//! A block after the first is a table, `{#NAME=body,#NAME=body}`: entries are
//! separated by `,` and each splits on its **first** `=`, so a body may carry
//! further ones (`#A=C=C` is a double bond). A name must be defined exactly
//! once in its table and every name used one level up must be defined —
//! [`SmilesErrorKind::CgDuplicateFragment`] and
//! [`SmilesErrorKind::CgUndefinedFragment`] otherwise. The converse is not an
//! error: a table may define more than the level above it uses, which is how a
//! shared library block is written.
//!
//! **The wildcard bead `[#*]` is an ordinary fragment name**, not a
//! match-anything pattern: `*` is simply a name character, so `{[#*][#A]}` is
//! a base-only string over the two names `*` and `A`. Name coverage treats it
//! like any other name — `{[#*]}.{#A=CC}` is
//! [`SmilesErrorKind::CgUndefinedFragment`] carrying `"*"`, because the table
//! defines `A` and no table defines `*`.
//!
//! **The last block is atomistic, by position.** Its bodies are OpenSMILES
//! fragment bodies, kept as [`FragmentBody::Smiles`] with their bonding
//! descriptors; every earlier block's bodies are coarse graphs over the next
//! level's `[#X]` nodes, kept as [`FragmentBody::Graph`]. Nothing in the
//! notation marks which is which — the reference implementation passes its
//! reader a `last_all_atom` flag instead — so dispatch here is positional and
//! flag-free. The **limitation** that follows is stated rather than worked
//! around: a string whose deepest resolution is meant to stay coarse-grained
//! cannot be written, and a coarse body in the last block is
//! [`SmilesErrorKind::CgLastBlockNotAtomistic`].
//!
//! **Levels and tables do not line up one-to-one, on purpose.** Each
//! intermediate table is expanded into the next level — one disjoint copy of a
//! body per node that names it, with [`CgNode::parent`] pointing back — while
//! the last table builds no level at all, because a level is a graph of beads
//! and atoms are not [`CgNode`]s. A base-only string therefore has no table
//! and one level; otherwise `levels.len() == fragments.len()`.
//!
//! # Refusals of valid notation (choices of this first implementation)
//!
//! None of these is a property of the notation — each is this implementation
//! declining to model something, loudly rather than silently. ("A later
//! `cgsmiles-*` link" names a follow-up specification in
//! `.claude/specs/`, the chain of documents this parser was built from.)
//!
//! * **Bond order 0**, written `.` (`{[#A].[#B]}`): a virtual edge between
//!   virtual particles. Refused with [`SmilesErrorKind::CgInvalidBondOrder`]
//!   until a later `cgsmiles-*` link models virtual particles, which needs a
//!   level-side representation rather than a fifth [`CgBondOrder`] variant.
//! * **A non-default mapping weight `w`**, keyword (`[#A;w=2]`) or positional
//!   (`[#A;0;0.5]`): refused with
//!   [`SmilesErrorKind::CgUnsupportedAnnotation`] until a later `cgsmiles-*`
//!   link models mapping weights, together with the pending
//!   schema-vocabulary spec that decides where a weight column lives. The
//!   default `w = 1` parses, and is dropped rather than retained.
//! * **A chirality annotation `x`** (`[#A;x=S]`): refused with the same
//!   [`SmilesErrorKind::CgUnsupportedAnnotation`], because a coarse-grained
//!   bead has no stereocentre for molrs to place and silently keeping the key
//!   as free text would claim otherwise.
//! * **The squash operator `[!]`** (`{[#A][!]}`), the one bonding descriptor
//!   that puts a single atom in two beads at once rather than marking a site
//!   where two fragments may later be joined: refused with
//!   [`SmilesErrorKind::CgSquashUnsupported`] wherever it is written — base
//!   graph, coarse fragment body or atomistic fragment body, all three caught
//!   by the one structural rule. No specification in the chain lifts it yet;
//!   parsing it as an ordinary descriptor would keep the glyph and lose the
//!   sharing it stands for.
//!
//! # Invariants and caveats
//!
//! * **Alignment.** `fragments[k]` is the authority for a fragment's shape and
//!   `levels[k + 1]` is the expansion it denotes. A base-only string has
//!   `fragments.is_empty() && levels.len() == 1`; otherwise
//!   `levels.len() == fragments.len()`, one level short of the number of
//!   tables because the last one is atomistic.
//! * **Descriptor pairing is finished before the IR is handed out.**
//!   Instantiation copies a body's own edges and nothing else; the step that
//!   pairs the bonding descriptors *between* two copies runs straight after
//!   it, and records every match in [`CgSmilesIr::pairs`]. **Why pairing is
//!   parse-stage while SMILES ring closure is build-stage:** a ring closure is
//!   intra-body and only means anything once atoms exist, so it belongs to the
//!   builder, whereas descriptor pairs are needed at coarse levels that never
//!   become atoms at all — a reader of the coarse graph reads `pairs` without
//!   expanding anything — so they have to be finished before the value is
//!   returned.
//! * **Pairing follows parse order, and only the graph is contracted.** Edges
//!   are paired in the order the notation formed them, while the reference
//!   implementation follows its graph library's adjacency order: for
//!   `{[#A]1[#B][#C]1}` that is (0,1), (1,2), (0,2) here against (0,1), (0,2),
//!   (1,2) there. A ring-bearing string can therefore consume different ports
//!   and, once expanded, number its atoms differently. What is contracted is
//!   the **connectivity**: the expansion
//!   ([`CgSmilesIr::to_atomistic`]) is isomorphic to the reference's and to
//!   what the molpy builder builds from the same string as an unlabelled
//!   graph; atom indices are not part of that contract, and neither are bond
//!   classes. The reference sets order 1.5 whenever both endpoint atoms are
//!   aromatic, over a written symbol as well, so its biphenyl `-[$]` bond is
//!   1.5 where molrs writes [`BondKind::Single`](crate::io::smiles::BondKind)
//!   and its `=`/`=` pair is 1.5 where molrs writes `Double`; molrs follows
//!   the Daylight Theory Manual, *SMILES* § 3.2.2, where a written symbol is
//!   the bond's order.
//! * **Ports left free are dropped.** A descriptor no edge needed creates
//!   neither a bond nor an atom; filling the valence it leaves open at the
//!   atomistic level is a later hydrogen-repletion step the caller composes.
//! * **Replayed spans are shared.** `|n` re-reads the unit's bytes by
//!   rewinding the one scanner over the full input, so every copy carries the
//!   *template's* [`Span`](crate::io::smiles::Span): spans within a level are
//!   neither disjoint nor monotonic, and a diagnostic raised inside a copy
//!   points at the template's text.
//! * **Descriptors are cloned per copy, ring markers are not replayable.** A
//!   bonding descriptor is a pairing *class*, consumed one occurrence at a
//!   time, so three copies of `[$1]` are three ports of class `$1`. A ring
//!   marker is a one-shot identity, so opening or closing one inside a
//!   repeated unit is [`SmilesErrorKind::CgRepeatOnRingMarker`].
//! * **Edges are in parse order**, the ring-closure edge emitted where its
//!   closing marker is read. The reference implementation iterates its graph
//!   library's adjacency order, so a ring-bearing string indexes edges
//!   differently there.
//!
//! # Three reference behaviours this reader does not mirror
//!
//! Each is a place where `CGsmiles @ 910c9ee` accepts or drops something
//! silently and molrs refuses it instead, so the difference is recorded rather
//! than discovered:
//!
//! * two definitions of one name — the reference keeps the first and drops the
//!   rest; this reader raises [`SmilesErrorKind::CgDuplicateFragment`];
//! * an empty block `{}` — the reference's block regular expression skips it;
//!   this reader raises [`SmilesErrorKind::CgEmptyBlock`], in any position;
//! * `|n` written as a body's last token (`{#B1=[#PEO]|4}`) — the reference
//!   crashes there; this reader replays it like any other repeat.
//!
//! # Example
//!
//! Three resolutions: beads of blocks, blocks of beads, beads of atoms. Level
//! 1 is the expansion of the first table — two `[#PEO]` copies per `[#B1]`
//! node, each remembering the node it came from — while the last table's
//! bodies stay atomistic.
//!
//! ```
//! use molrs::io::cgsmiles::{CgSmilesIr, FragmentBody};
//!
//! let ir = CgSmilesIr::parse(
//!     "{[#B1][#B2][#B1]}.\
//!      {#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
//!      {#PEO=[>]COC[<],#PE=[>]CC[<]}",
//! )
//! .unwrap();
//!
//! assert_eq!(ir.levels.len(), 2);
//! assert_eq!(ir.levels[0].nodes.len(), 3);
//! assert_eq!(ir.levels[1].nodes.len(), 6);
//! assert_eq!(ir.levels[1].nodes[3].parent, Some(1));
//! assert!(matches!(ir.fragments[1]["PEO"].body, FragmentBody::Smiles(_)));
//! ```
//!
//! # References
//!
//! Primary source: the `CGsmiles` documentation, cgsmiles.readthedocs.io,
//! *Basic graph description* and *Multiple resolutions*. Reference
//! implementation: `CGsmiles @ 910c9ee`, `read_cgsmiles.py:134` (the
//! bond-order table) and `dialects.py` (the reserved-annotation table).
//! The `CGsmiles` paper — Grünewald, F. et al., J. Chem. Inf. Model. **65**
//! (2025), DOI 10.1021/acs.jcim.5c00064 — is cited for provenance only: it was
//! not readable when this module was written, so no rule above rests on it.
//!
//! [`SmilesErrorKind::CgInvalidBondOrder`]: crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder
//! [`SmilesErrorKind::CgUnsupportedAnnotation`]:
//!     crate::io::smiles::SmilesErrorKind::CgUnsupportedAnnotation
//! [`SmilesErrorKind::CgRepeatOnRingMarker`]:
//!     crate::io::smiles::SmilesErrorKind::CgRepeatOnRingMarker
//! [`SmilesErrorKind::RingBondConflict`]: crate::io::smiles::SmilesErrorKind::RingBondConflict
//! [`SmilesErrorKind::CgDuplicateFragment`]:
//!     crate::io::smiles::SmilesErrorKind::CgDuplicateFragment
//! [`SmilesErrorKind::CgUndefinedFragment`]:
//!     crate::io::smiles::SmilesErrorKind::CgUndefinedFragment
//! [`SmilesErrorKind::CgLastBlockNotAtomistic`]:
//!     crate::io::smiles::SmilesErrorKind::CgLastBlockNotAtomistic
//! [`SmilesErrorKind::CgEmptyBlock`]: crate::io::smiles::SmilesErrorKind::CgEmptyBlock
//! [`SmilesErrorKind::CgSquashUnsupported`]:
//!     crate::io::smiles::SmilesErrorKind::CgSquashUnsupported

mod ast;
#[cfg(test)]
pub(super) mod fixtures;
mod instantiate;
mod parser;
mod resolve;
mod templates;
mod to_atomistic;
mod to_coarsegrain;
mod validate;

use crate::io::smiles::SmilesError;
use parser::CgParser;
use resolve::resolve;

pub use ast::{
    CgBondOrder, CgEdge, CgFragmentDef, CgGraph, CgNode, CgSmilesIr, EdgeOrigin, FragmentBody,
    PairEnd, ResolvedPair,
};

impl CgSmilesIr {
    /// Parse a `CGsmiles` string — every resolution block it writes — into its
    /// intermediate representation.
    ///
    /// `CGsmiles` is a line notation that writes a molecule at a *coarse-grained*
    /// resolution: where SMILES names atoms, `CGsmiles` names **beads**, each one
    /// standing for a whole named fragment of the molecule. A block `{…}` is one
    /// resolution level, so `{[#PEO][#PEO][#PEO]}` is a chain of three beads of
    /// the fragment called `PEO`. Inside a block, `[#NAME]` is a node, adjacency
    /// is a bond, the symbols `-`, `=`, `#` and `$` write bond multiplicities
    /// 1..=4 (the default is 1), `( … )` branches, a bare digit or `%` and a digit
    /// run opens a ring marker that its second occurrence closes, `|n` repeats the
    /// preceding unit `n` times in total, and `;`-separated annotations decorate a
    /// node — `[#A;q=-0.5]`, or positionally `[#A;-0.5]`, since annotation slot 2
    /// is `q`.
    ///
    /// Blocks are separated by `.`, and every block after the first is a
    /// **fragment table** — `{#PEO=[>]COC[<],#PE=[>]CC[<]}` — naming the bodies of
    /// the nodes written one level up. Each intermediate table is expanded here
    /// into the level it denotes: one disjoint copy of a body per node that names
    /// it, each copy's [`parent`](CgNode::parent) pointing back at that node. The
    /// bonding descriptors the copies carry are then paired — every written edge
    /// matched to a concrete pair of free compatible ports, recorded in
    /// [`CgSmilesIr::pairs`], and each pair of an intermediate level appended to
    /// the level below as a [`Derived`](EdgeOrigin::Derived) edge.
    ///
    /// The **last** block is atomistic by position — the notation marks it
    /// nowhere — so its bodies are parsed as OpenSMILES fragment bodies and kept
    /// as [`FragmentBody::Smiles`], descriptors intact, without being expanded
    /// into atoms. The limitation this buys is real and not worked around: a
    /// string whose deepest resolution is meant to stay coarse-grained cannot be
    /// written, and a coarse body in the last block is refused with
    /// [`CgLastBlockNotAtomistic`]. Nothing here builds a molecular graph or a
    /// frame.
    ///
    /// Units: a node's [`charge`](CgNode::charge) is a **partial charge in
    /// elementary charge units `e`** — the fractional force-field charge carried
    /// by a bead, not the integer formal charge of an atom. A bond order is a
    /// dimensionless multiplicity, 1..=4.
    ///
    /// # Valid notation this parser refuses
    ///
    /// Four constructs are legal `CGsmiles` that molrs does not yet model, and
    /// are refused loudly rather than dropped silently: bond order 0, written `.`
    /// (`{[#A].[#B]}`), which denotes a virtual edge between virtual particles;
    /// a mapping weight `w` at any value other than its default 1, keyword
    /// (`[#A;w=2]`) or positional (`[#A;0;0.5]`); a chirality annotation
    /// (`[#A;x=S]`), a coarse bead having no stereocentre to place; and the squash
    /// operator `[!]` (`{[#A][!]}`), the one descriptor that shares a single atom
    /// between two beads instead of marking a site for a later join. Everything
    /// else the annotation table does not reserve is kept verbatim in
    /// [`CgNode::annotations`].
    ///
    /// # Invariants and caveats
    ///
    /// `fragments[k]` is the authority for a fragment's shape and `levels[k + 1]`
    /// is the expansion it denotes. A base-only string has
    /// `fragments.is_empty() && levels.len() == 1`; otherwise
    /// `levels.len() == fragments.len()` — one level short of the number of
    /// tables, because the last table is atomistic and a level is a graph of
    /// beads. Nodes and edges are in parse order, a ring-closure
    /// edge appearing where its closing marker was read — the reference
    /// implementation orders edges by its graph library's adjacency instead, so a
    /// ring-bearing string indexes edges differently there. The copies `|n` makes
    /// are re-read from the template's bytes, so each copy **shares** the
    /// template's [`Span`](crate::io::smiles::Span): spans within a level are
    /// neither disjoint nor monotonic, and a diagnostic raised inside a copy
    /// points at the template's text. A bonding descriptor inside a repeated unit
    /// is cloned into every copy — a descriptor is a pairing *class*, consumed one
    /// occurrence at a time — whereas a ring marker is a one-shot identity and
    /// cannot be replayed at all ([`CgRepeatOnRingMarker`]).
    ///
    /// The constructor of the IR, as [`SmilesIr::parse`](crate::io::smiles::SmilesIr::parse)
    /// is SMILES's: parsing is stateless, and the parser that does the work is a
    /// private type with exactly one user-visible step.
    ///
    /// # Errors
    ///
    /// Returns a [`SmilesError`] stamped
    /// [`Notation::CgSmiles`](crate::io::smiles::Notation::CgSmiles) and carrying
    /// the whole of `text`, so the rendered message reads `CGsmiles parse error at
    /// position N: …` with a caret under column N.
    ///
    /// Twenty-one kinds name inputs only this notation has: [`CgEmptyBlock`] (`{}`,
    /// in any position), [`CgMalformedAnnotation`] (`{[#A;]}`, `{[#A;=1]}`,
    /// `{[#A;q=x]}`, or a fourth positional field), [`CgUnsupportedAnnotation`]
    /// (the weight and chirality refusals above),
    /// [`CgAnnotationOnWildcard`] (`{[#*;q=1]}`, the
    /// wildcard bead standing for *any* fragment and so having no properties of
    /// its own), [`CgInvalidBondOrder`] (`.`, or any other non-bond character in
    /// bond position), [`CgInvalidRepeatCount`] (`{[#A]|0}`, `{[#A]|x}`),
    /// [`CgDuplicateEdge`] (a ring closure re-forming a pair the graph already
    /// has, or bonding a node to itself), [`CgRepeatOnBranchedNode`] (`|` inside
    /// an open branch, or after a node carrying two branches),
    /// [`CgRepeatOnRingMarker`], [`CgDanglingBond`] (`{[#A]=}`),
    /// [`CgInvalidRingMarker`] (`%` with no digits after it, or a number past
    /// 65535), [`CgExpectedBlock`] (`{[#A]}.#A=CC`, `{[#A]}{#A=CC}`, or anything
    /// after the last `}`), [`CgMalformedFragmentDef`] (`{[#A]}.{#A}`),
    /// [`CgEmptyFragmentBody`] (`{[#A]}.{#A=}`), [`CgDuplicateFragment`]
    /// (`{[#A]}.{#A=CC,#A=CCC}`), [`CgUndefinedFragment`] (`{[#A][#B]}.{#A=CC}`),
    /// [`CgSquashUnsupported`] (the squash operator `[!]` at any level, the
    /// shortest such string being `{[#A][!]}`), [`CgLastBlockNotAtomistic`]
    /// (`{[#A]}.{#A=[#X][#X]}`, boxing the reason the body did not read as
    /// OpenSMILES), [`CgUnmatchableEdge`] (a written bond whose two fragments
    /// offer no free pair of compatible bonding descriptors,
    /// `{[#A][#B]}.{#A=[$a]C,#B=[$b]C}`), [`CgNotExpandable`] (raised by
    /// [`CgSmilesIr::to_atomistic`], not by this function: a string with no
    /// atomistic body to expand) and [`CgBuild`] (an internal invariant of the
    /// reader, unreachable from user input).
    ///
    /// Seven kinds are shared with SMILES and SMARTS, and are told apart from them
    /// by [`SmilesError::notation`] rather than by their name: [`EmptyInput`] (an
    /// empty `text`), [`UnexpectedChar`], [`UnexpectedEnd`] (the block never
    /// closes), [`UnclosedBracket`] (a `[…]` never closes), [`UnclosedBranch`],
    /// [`UnmatchedRingClosure`] (a marker still open at `}`) and
    /// [`RingBondConflict`].
    ///
    /// A body of the last block is parsed by
    /// [`SmilesIr::from_fragment`](crate::io::smiles::SmilesIr::from_fragment), and
    /// its diagnostic is **re-based** before it leaves this function: the span is
    /// shifted by the body's offset, the input is the whole `CGsmiles` string and
    /// the notation is `CGsmiles`, so the caret lands on the offending token of
    /// the string the caller passed in. The kind is kept, boxed inside
    /// [`CgLastBlockNotAtomistic`] — except
    /// [`AtomAnnotationUnsupported`](crate::io::smiles::SmilesErrorKind::AtomAnnotationUnsupported)
    /// (`[C;0.5]`, `[*;s=C,0]`), which already names an unsupported feature and
    /// propagates un-wrapped.
    ///
    /// A body that parses and then fails to **build** gets that same treatment:
    /// parsing a fragment body leaves its ring closures unchecked, so
    /// `{[#A][#B]}.{#A=[$]C1CC,#B=[$]C}` is refused when resolution converts the
    /// body, as [`CgLastBlockNotAtomistic`] boxing
    /// [`UnmatchedRingClosure`] and spanned at the marker inside the `#A=` body.
    ///
    /// A bonding descriptor written beside a node is checked by the same grammar
    /// rules the SMILES fragment dialect uses, so [`BondInsideDescriptor`]
    /// (`{[#A][$=]}`), [`InvalidDescriptorLabel`] (`{[#A][$a+]}`) and
    /// [`DanglingDescriptor`] can be raised too, stamped as `CGsmiles`. The last
    /// of those takes a graph with no node in it at all (`{[$]}`): a descriptor
    /// written *before* the first node waits for that node and binds to it, so
    /// `{[$][#A]}` is a `$` on `[#A]` and not an error.
    ///
    /// [`CgEmptyBlock`]: crate::io::smiles::SmilesErrorKind::CgEmptyBlock
    /// [`CgExpectedBlock`]: crate::io::smiles::SmilesErrorKind::CgExpectedBlock
    /// [`CgMalformedFragmentDef`]: crate::io::smiles::SmilesErrorKind::CgMalformedFragmentDef
    /// [`CgEmptyFragmentBody`]: crate::io::smiles::SmilesErrorKind::CgEmptyFragmentBody
    /// [`CgDuplicateFragment`]: crate::io::smiles::SmilesErrorKind::CgDuplicateFragment
    /// [`CgUndefinedFragment`]: crate::io::smiles::SmilesErrorKind::CgUndefinedFragment
    /// [`CgSquashUnsupported`]: crate::io::smiles::SmilesErrorKind::CgSquashUnsupported
    /// [`CgLastBlockNotAtomistic`]: crate::io::smiles::SmilesErrorKind::CgLastBlockNotAtomistic
    /// [`CgBuild`]: crate::io::smiles::SmilesErrorKind::CgBuild
    /// [`CgUnmatchableEdge`]: crate::io::smiles::SmilesErrorKind::CgUnmatchableEdge
    /// [`CgNotExpandable`]: crate::io::smiles::SmilesErrorKind::CgNotExpandable
    /// [`CgMalformedAnnotation`]: crate::io::smiles::SmilesErrorKind::CgMalformedAnnotation
    /// [`CgUnsupportedAnnotation`]: crate::io::smiles::SmilesErrorKind::CgUnsupportedAnnotation
    /// [`CgAnnotationOnWildcard`]: crate::io::smiles::SmilesErrorKind::CgAnnotationOnWildcard
    /// [`CgInvalidBondOrder`]: crate::io::smiles::SmilesErrorKind::CgInvalidBondOrder
    /// [`CgInvalidRepeatCount`]: crate::io::smiles::SmilesErrorKind::CgInvalidRepeatCount
    /// [`CgDuplicateEdge`]: crate::io::smiles::SmilesErrorKind::CgDuplicateEdge
    /// [`CgRepeatOnBranchedNode`]: crate::io::smiles::SmilesErrorKind::CgRepeatOnBranchedNode
    /// [`CgRepeatOnRingMarker`]: crate::io::smiles::SmilesErrorKind::CgRepeatOnRingMarker
    /// [`CgDanglingBond`]: crate::io::smiles::SmilesErrorKind::CgDanglingBond
    /// [`CgInvalidRingMarker`]: crate::io::smiles::SmilesErrorKind::CgInvalidRingMarker
    /// [`EmptyInput`]: crate::io::smiles::SmilesErrorKind::EmptyInput
    /// [`UnexpectedChar`]: crate::io::smiles::SmilesErrorKind::UnexpectedChar
    /// [`UnexpectedEnd`]: crate::io::smiles::SmilesErrorKind::UnexpectedEnd
    /// [`UnclosedBracket`]: crate::io::smiles::SmilesErrorKind::UnclosedBracket
    /// [`UnclosedBranch`]: crate::io::smiles::SmilesErrorKind::UnclosedBranch
    /// [`UnmatchedRingClosure`]: crate::io::smiles::SmilesErrorKind::UnmatchedRingClosure
    /// [`RingBondConflict`]: crate::io::smiles::SmilesErrorKind::RingBondConflict
    /// [`BondInsideDescriptor`]: crate::io::smiles::SmilesErrorKind::BondInsideDescriptor
    /// [`InvalidDescriptorLabel`]: crate::io::smiles::SmilesErrorKind::InvalidDescriptorLabel
    /// [`DanglingDescriptor`]: crate::io::smiles::SmilesErrorKind::DanglingDescriptor
    /// [`SmilesError::notation`]: crate::io::smiles::SmilesError::notation
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::io::cgsmiles::CgSmilesIr;
    ///
    /// let ir = CgSmilesIr::parse("{[#PEO][#PEO][#PEO]}").unwrap();
    /// assert_eq!(ir.levels.len(), 1);
    /// assert_eq!(ir.levels[0].nodes.len(), 3);
    /// assert_eq!(ir.levels[0].nodes[0].name, "PEO");
    /// assert_eq!(ir.levels[0].edges.len(), 2);
    /// assert_eq!(ir.levels[0].edges[0].order.multiplicity(), 1);
    /// ```
    pub fn parse(text: &str) -> Result<CgSmilesIr, SmilesError> {
        let mut ir = CgParser::new(text).parse()?;
        resolve(&mut ir, text)?;
        Ok(ir)
    }
}
