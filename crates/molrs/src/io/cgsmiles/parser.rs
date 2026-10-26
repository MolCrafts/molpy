//! Recursive-descent parser for a `CGsmiles` string: block structure, coarse
//! graphs and fragment tables.
//!
//! Three layers sit in this file. [`CgParser::split_blocks`] cuts the string
//! into its `{…}` blocks lexically — `.` separates blocks and means something
//! else entirely inside one. [`CgParser::parse_fragment_block`] reads a block
//! after the first as a table of `#NAME=body` entries and dispatches each body
//! **by position**: the last block's bodies are atomistic and go to
//! `parse_fragment_smiles`, every earlier one's is a coarse graph and goes to
//! [`CgParser::parse_body`]. And the token loop below reads one coarse graph,
//! whether it came from a block or from a body.
//!
//! A graph is always read by a *fresh* parser seeked to its absolute offset in
//! the whole input, never by a scanner over a substring: that is what keeps
//! `|n`'s replay, a body-level diagnostic and a re-based atomistic diagnostic
//! all pointing at bytes of the string the caller passed in.
//!
//! The grammar of one graph is small enough to read in one token loop:
//! [`CgParser::step`] reads the next token and folds it into the graph being
//! built, and every construct — a node, a bonding descriptor, a bond symbol, a
//! branch, a ring marker, a `|n` repeat — is one arm of it.
//!
//! Two decisions are worth stating before the code. The bond symbol is *not*
//! a different token for each position the notation allows it in: it is read
//! once and remembered as `pending`, and whatever comes next spends it — a
//! node as the chain bond (which is also how the branch bonds of
//! `{[#A](=[#B])[#C]}` and `{[#A]([#B])=[#C]}` arrive, since a node follows
//! the symbol in both), a ring marker as the ring bond, a descriptor bracket
//! as the order of the bond its pairing will create, and `|` as the bond that
//! chains the repeated copies. And the `|n` repeat re-reads the unit's bytes
//! by rewinding this same scanner over the full input
//! ([`Scanner::seek`](crate::line_notation::scanner::Scanner::seek)), so a
//! diagnostic raised inside a copy still points at a position in the string
//! the caller passed in.
//!
//! Comments below cite grammar rules as `R2.4`, `R2.20` and so on. Those are
//! the numbered rules of `.claude/specs/cgsmiles-01b-graph.md` and
//! `.claude/specs/cgsmiles-01c-fragments.md` § Domain basis,
//! each of which records where it came from — the `CGsmiles` documentation,
//! the reference implementation at a named commit, or an inference marked as
//! such. `F1.9` and the like name those specs' worked fixtures. `01d` is the
//! next spec in the same chain (`.claude/specs/cgsmiles-01d-resolve.md`, which
//! pairs the bonding descriptors this file only carries, and expands a level
//! into atoms); naming it marks work this file deliberately leaves undone.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::io::cgsmiles::instantiate::instantiate;
use crate::io::cgsmiles::validate::validate_ir;
use crate::io::cgsmiles::{
    CgBondOrder, CgEdge, CgFragmentDef, CgGraph, CgNode, CgSmilesIr, EdgeOrigin, FragmentBody,
};
use crate::io::smiles::{BondKind, BondingDescriptor, DescriptorKind, FragmentIr, Span};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::parser::parse_fragment_smiles;
use crate::line_notation::scanner::Scanner;
use crate::line_notation::validation::validate_descriptor;
use crate::op::F;

/// One fragment table: the names one block defines, in name order.
type FragmentTable = BTreeMap<String, CgFragmentDef>;

/// The only mapping weight the parser accepts: `w = 1`, the notation's own
/// default (R2.14). `w` is the dimensionless mapping weight the notation
/// attaches to a node; molrs models no other value yet, so writing one is
/// refused rather than silently dropped.
const DEFAULT_WEIGHT: F = 1.0;

/// The largest `|n` repeat count the parser accepts.
///
/// A **molrs** rule, not a `CGsmiles` one: the notation states no limit, and
/// `n` is the number of copies of the unit that end up in the level, so
/// `{[#A]|900000000}` asks for nine hundred million nodes and the reader dies
/// of memory exhaustion — an abort where a refusal was owed, and a process
/// kill rather than an exception for every binding that calls in. The cap is
/// `u16::MAX`, the same ceiling the notation's own ring markers carry, so one
/// number bounds both counted constructs.
const MAX_REPEAT_COUNT: usize = u16::MAX as usize;

/// A node's annotations after the dialect's reserved keys have been read out
/// of them.
struct BoundAnnotations {
    /// The partial charge written as `q`, in elementary charge units `e`.
    charge: Option<F>,
    /// Every pair the table does not reserve, verbatim and in written order.
    rest: Vec<(String, String)>,
}

/// A ring marker that has been opened and is waiting for its closing twin.
struct OpenRing {
    /// The node the opening marker was written after.
    node: usize,
    /// The bond order written before the *opening* marker, `None` when the
    /// notation wrote no symbol there.
    ///
    /// The order is a property of the bond, not of the end it was written at
    /// (OpenSMILES § 3.4 writes `C1CCCCC=1`), so a symbol at *either* end sets
    /// it and two differing explicit symbols are a conflict.
    order: Option<CgBondOrder>,
    /// Byte range of the opening marker, for the unmatched-closure error.
    span: Span,
}

/// The stretch of input `|n` would repeat: the node most recently written at
/// branch depth 0, plus everything attached to it since.
#[derive(Clone)]
struct RepeatUnit {
    /// Byte offset the unit starts at (the node's `[`).
    start: usize,
    /// Byte offset one past the unit's last token. A bond symbol never moves
    /// it, which is what keeps the symbol before `|` out of the copied text.
    end: usize,
    /// Index of the node each copy is chained through (R2.20: the anchor is
    /// part of the unit).
    anchor: usize,
    /// How many branches were opened on the anchor. More than one is
    /// [`SmilesErrorKind::CgRepeatOnBranchedNode`] (R2.23).
    branches: usize,
    /// Whether a ring marker was opened or closed inside the unit, which
    /// [`SmilesErrorKind::CgRepeatOnRingMarker`] refuses.
    ring_touched: bool,
}

/// Parser state for one `CGsmiles` block.
///
/// Private by design: it has exactly one user-visible step, so the public
/// surface is the free function
/// [`CgSmilesIr::parse`](crate::io::cgsmiles::CgSmilesIr::parse) that runs it, in the
/// shape `parse_smiles` and `parse_smarts` already use.
pub(super) struct CgParser<'a> {
    /// Cursor over the whole input, and the source of every diagnostic.
    scanner: Scanner<'a>,
    /// Nodes of the level being built, in parse order.
    nodes: Vec<CgNode>,
    /// Edges of the level being built, in parse order.
    edges: Vec<CgEdge>,
    /// Ring markers opened and not yet closed, by marker number.
    open_rings: HashMap<u16, OpenRing>,
    /// Anchor node of each open branch, innermost last.
    branches: Vec<usize>,
    /// The node the next chain element bonds to.
    current: Option<usize>,
    /// A bond symbol that has been read and not yet spent, with its offset.
    pending: Option<(CgBondOrder, usize)>,
    /// The stretch of input a `|n` written here would repeat — `None` before
    /// the first node of the block, and again after a `|n` consumes it.
    unit: Option<RepeatUnit>,
    /// Descriptors written before the first node of the graph, waiting for it
    /// (01a's R4.2 leading rule: `[>][#PEO][#PEO][<]` puts the `>` on the
    /// first `[#PEO]`). Still full when the graph ends is
    /// [`SmilesErrorKind::DanglingDescriptor`].
    leading: Vec<(BondingDescriptor, Span)>,
}

// ---- block driver ----
impl<'a> CgParser<'a> {
    /// A parser over the whole of `text`, reading it as
    /// [`Notation::CgSmiles`].
    pub(super) fn new(text: &'a str) -> Self {
        Self {
            scanner: Scanner::new(text, Notation::CgSmiles),
            nodes: Vec::new(),
            edges: Vec::new(),
            open_rings: HashMap::new(),
            branches: Vec::new(),
            current: None,
            pending: None,
            unit: None,
            leading: Vec::new(),
        }
    }

    /// Read a whole `CGsmiles` string: syntax, then per level coverage and
    /// instantiation, then validation.
    ///
    /// The three steps are one call because no caller may hold the state
    /// between them — an un-instantiated [`CgSmilesIr`] is not a value this
    /// module hands out. A graph is read by a **fresh** parser seeked to the
    /// block or body it covers, so every diagnostic keeps a span into the
    /// whole input while the per-graph state (rings, branches, the repeat
    /// unit) never leaks from one graph into the next.
    ///
    /// # Errors
    ///
    /// Every malformed input named by [`SmilesErrorKind`]'s `Cg*` variants,
    /// plus the kinds shared with the atomistic notations —
    /// [`SmilesErrorKind::EmptyInput`], [`SmilesErrorKind::UnexpectedChar`],
    /// [`SmilesErrorKind::UnexpectedEnd`], [`SmilesErrorKind::UnclosedBracket`],
    /// [`SmilesErrorKind::UnclosedBranch`],
    /// [`SmilesErrorKind::UnmatchedRingClosure`] and
    /// [`SmilesErrorKind::RingBondConflict`] — and the three descriptor kinds
    /// [`SmilesErrorKind::BondInsideDescriptor`],
    /// [`SmilesErrorKind::InvalidDescriptorLabel`] and
    /// [`SmilesErrorKind::DanglingDescriptor`] that `validate_descriptor` and
    /// `anchor_descriptor` raise.
    /// [`SmilesErrorKind::AtomAnnotationUnsupported`] is raised un-wrapped —
    /// re-based into the full input, but not boxed inside
    /// [`SmilesErrorKind::CgLastBlockNotAtomistic`] — when a `;` annotation is
    /// written inside a bracket atom of an atomistic body.
    /// [`SmilesErrorKind::InvalidDescriptorOrder`] cannot occur: a coarse bond
    /// order only ever maps to one of the four orders it allows.
    pub(super) fn parse(self) -> Result<CgSmilesIr, SmilesError> {
        let input = self.scanner.input();
        if input.is_empty() {
            return Err(self.scanner.error(SmilesErrorKind::EmptyInput));
        }
        let blocks = self.split_blocks()?;
        let base = CgParser::new(input).parse_block(blocks[0])?;
        let last = blocks.len() - 1;
        let mut fragments = Vec::with_capacity(last);
        for (index, block) in blocks.iter().enumerate().skip(1) {
            fragments.push(self.parse_fragment_block(*block, index == last)?);
        }
        let levels = Self::build_levels(base, &fragments, input)?;
        // One empty pair list per level: this parser reads syntax, and
        // pairing descriptors is `resolve`'s step, run by `CgSmilesIr::parse`
        // once the levels exist.
        let pairs = vec![Vec::new(); levels.len()];
        let ir = CgSmilesIr {
            levels,
            fragments,
            pairs,
            span: Span::new(0, input.len()),
        };
        validate_ir(&ir, input)?;
        Ok(ir)
    }

    /// Resolve every level in turn: check that table *k* covers level *k*,
    /// then — unless it is the atomistic last table — expand level *k* through
    /// it into level *k + 1*.
    ///
    /// Coverage runs **before** the expansion it feeds, so a name the string
    /// never defines reaches the user as
    /// [`SmilesErrorKind::CgUndefinedFragment`] and `instantiate`'s
    /// [`SmilesErrorKind::CgBuild`] stays unreachable from user input.
    fn build_levels(
        base: CgGraph,
        fragments: &[FragmentTable],
        input: &str,
    ) -> Result<Vec<CgGraph>, SmilesError> {
        let mut levels = vec![base];
        for (index, table) in fragments.iter().enumerate() {
            Self::check_coverage(&levels[index], table, input)?;
            if index + 1 == fragments.len() {
                break;
            }
            let defs = Self::graph_defs(table);
            let next = instantiate(&levels[index], &defs, input)?;
            levels.push(next);
        }
        Ok(levels)
    }

    /// Every name of `level` must be defined by `table` (R3.2 / R3.4).
    ///
    /// The converse is not checked: a table may define more than the level
    /// above it uses, which is how a shared library block is written.
    ///
    /// R3.2's exemption for a node all of whose edges have order 0 is not
    /// implemented — order 0 is [`SmilesErrorKind::CgInvalidBondOrder`] here,
    /// so the exempt case cannot be reached.
    fn check_coverage(
        level: &CgGraph,
        table: &FragmentTable,
        input: &str,
    ) -> Result<(), SmilesError> {
        for node in &level.nodes {
            if !table.contains_key(&node.name) {
                let kind = SmilesErrorKind::CgUndefinedFragment(node.name.clone());
                return Err(SmilesError::new(kind, node.span, input, Notation::CgSmiles));
            }
        }
        Ok(())
    }

    /// A borrowed view of the coarse-graph bodies of one table, the shape
    /// `instantiate` takes.
    ///
    /// Every body of an intermediate table is a [`FragmentBody::Graph`] by
    /// dispatch, so the filter drops nothing in practice — it is what stops an
    /// atomistic body from reaching expansion *by type* instead of by a
    /// runtime check.
    fn graph_defs(table: &FragmentTable) -> BTreeMap<&str, &CgGraph> {
        table
            .iter()
            .filter_map(|(name, def)| match &def.body {
                FragmentBody::Graph(graph) => Some((name.as_str(), graph)),
                FragmentBody::Smiles(_) => None,
            })
            .collect()
    }

    // -- block structure ----------------------------------------------------

    /// Cut the input into its `{…}` blocks, as absolute byte ranges.
    ///
    /// Lexical rather than a regular expression (R1.3): a block runs from its
    /// `{` to the first `}` after it — a `}` may not be written inside a
    /// block, so there is no nesting to track — and between two blocks the
    /// only thing that may stand is the `.` separator. A `.` *within* a block
    /// never reaches this scan, which jumps from the opening `{` straight to
    /// the closing `}`; in there the same character is something else entirely
    /// (R1.4) — the order-0 bond a coarse graph refuses, or the OpenSMILES
    /// disconnection of an atomistic body, which is why
    /// `{#OHter=[$][O-].[Na+]}` is one salt and not two blocks.
    ///
    /// # Errors
    ///
    /// [`SmilesErrorKind::CgExpectedBlock`] when a separator is not followed
    /// by `{`, when a block follows another with no separator between them, or
    /// when anything at all follows the last `}`;
    /// [`SmilesErrorKind::UnexpectedChar`] when the string does not open with
    /// `{`, and [`SmilesErrorKind::UnexpectedEnd`] when a block never closes.
    fn split_blocks(&self) -> Result<Vec<Span>, SmilesError> {
        let input = self.scanner.input();
        let mut blocks = Vec::new();
        let mut pos = 0usize;
        loop {
            match input[pos..].chars().next() {
                Some('{') => {}
                Some(c) if blocks.is_empty() => {
                    let span = Span::new(pos, pos + 1);
                    return Err(self
                        .scanner
                        .error_at(SmilesErrorKind::UnexpectedChar(c), span));
                }
                _ => return Err(self.expected_block(pos)),
            }
            let Some(offset) = input[pos..].find('}') else {
                let span = Span::new(input.len(), input.len() + 1);
                return Err(self.scanner.error_at(SmilesErrorKind::UnexpectedEnd, span));
            };
            let end = pos + offset + 1;
            blocks.push(Span::new(pos, end));
            pos = end;
            match input[pos..].chars().next() {
                None => return Ok(blocks),
                Some('.') => pos += 1,
                Some(_) => return Err(self.expected_block(pos)),
            }
        }
    }

    /// The missing-block error, pointing at the byte a `{` was expected at.
    fn expected_block(&self, at: usize) -> SmilesError {
        let span = Span::new(at, at + 1);
        self.scanner
            .error_at(SmilesErrorKind::CgExpectedBlock, span)
    }

    // -- fragment tables ----------------------------------------------------

    /// Read one `{#NAME=body,…}` block into the table it defines.
    ///
    /// `atomistic` says whether this is the **last** block, which is the only
    /// thing that decides how a body is read (R1.5): the notation marks it
    /// nowhere, so dispatch is positional.
    fn parse_fragment_block(
        &self,
        block: Span,
        atomistic: bool,
    ) -> Result<FragmentTable, SmilesError> {
        let interior = Span::new(block.start + 1, block.end - 1);
        if interior.start >= interior.end {
            return Err(self.scanner.error_at(SmilesErrorKind::CgEmptyBlock, block));
        }
        let mut table = FragmentTable::new();
        for entry in self.split_fragment_defs(interior) {
            let def = self.parse_fragment_def(entry, atomistic)?;
            if let Some(previous) = table.insert(def.name.clone(), def) {
                let kind = SmilesErrorKind::CgDuplicateFragment(previous.name);
                return Err(self.scanner.error_at(kind, entry));
            }
        }
        Ok(table)
    }

    /// Cut a fragment block's interior into its entries, as absolute byte
    /// ranges.
    ///
    /// The separator is `,` at bracket depth 0 (R3.0), so the comma inside
    /// `[*;s=C,0]` stays where it was written instead of splitting the entry
    /// that carries it.
    fn split_fragment_defs(&self, interior: Span) -> Vec<Span> {
        let interior_bytes = &self.scanner.input().as_bytes()[interior.start..interior.end];
        let mut entries = Vec::new();
        let mut depth = 0usize;
        let mut start = interior.start;
        for (offset, byte) in interior_bytes.iter().enumerate() {
            let at = interior.start + offset;
            match byte {
                b'[' => depth += 1,
                b']' => depth = depth.saturating_sub(1),
                b',' if depth == 0 => {
                    entries.push(Span::new(start, at));
                    start = at + 1;
                }
                _ => {}
            }
        }
        entries.push(Span::new(start, interior.end));
        entries
    }

    /// Read one `#NAME=body` entry into the definition it writes.
    ///
    /// The split is on the **first** `=` (R3.0): a body may carry further
    /// ones, which is how `#A=C=C` names a double bond.
    fn parse_fragment_def(
        &self,
        entry: Span,
        atomistic: bool,
    ) -> Result<CgFragmentDef, SmilesError> {
        let input = self.scanner.input();
        let text = &input[entry.start..entry.end];
        let malformed = || {
            let kind = SmilesErrorKind::CgMalformedFragmentDef;
            self.scanner.error_at(kind, entry)
        };
        let Some(rest) = text.strip_prefix('#') else {
            return Err(malformed());
        };
        let Some(equals) = rest.find('=') else {
            return Err(malformed());
        };
        let name = &rest[..equals];
        if name.is_empty() || !name.chars().all(Self::is_fragment_name_char) {
            return Err(malformed());
        }
        let body = Span::new(entry.start + 1 + equals + 1, entry.end);
        if body.start >= body.end {
            let kind = SmilesErrorKind::CgEmptyFragmentBody;
            return Err(self.scanner.error_at(kind, entry));
        }
        Ok(CgFragmentDef {
            name: name.to_owned(),
            body: self.parse_body_of(body, atomistic)?,
            span: entry,
        })
    }

    /// Read one body at whichever resolution its block sits at.
    ///
    /// The last block's bodies are atomistic and go to `parse_fragment_smiles`
    /// — not `parse_smiles`, which refuses the bonding descriptors a fragment
    /// body is written with. Every earlier block's body is a coarse graph over
    /// the next level's `[#X]` nodes and goes to [`CgParser::parse_body`].
    fn parse_body_of(&self, body: Span, atomistic: bool) -> Result<FragmentBody, SmilesError> {
        let input = self.scanner.input();
        if !atomistic {
            let graph = CgParser::new(input).parse_body(body)?;
            return Ok(FragmentBody::Graph(graph));
        }
        match parse_fragment_smiles(&input[body.start..body.end]) {
            Ok(ir) => Ok(FragmentBody::Smiles(FragmentIr::new(ir))),
            Err(err) => Err(Self::rebase(err, body.start, input)),
        }
    }

    /// Re-base an atomistic body's diagnostic into the whole `CGsmiles`
    /// string: the span is shifted by the body's start offset, the input
    /// becomes the full text and the notation becomes `CGsmiles`.
    ///
    /// The kind is *kept*, boxed inside
    /// [`SmilesErrorKind::CgLastBlockNotAtomistic`], so the message names the
    /// molrs rule and then the reason — a body of coarse nodes reports the
    /// last-block rule rather than a bare `unexpected character '#'`.
    ///
    /// One kind propagates un-wrapped:
    /// [`SmilesErrorKind::AtomAnnotationUnsupported`], which the fragment
    /// dialect raises for a `;` annotation inside a bracket atom. It already
    /// says "this notation feature is unsupported" and is no evidence that the
    /// block is the wrong resolution.
    ///
    /// Visible to the module because the body's *conversion*, run by the
    /// `resolve` step, catches what parsing a body does not check — an
    /// unmatched ring closure inside it — and that failure class has to reach
    /// the reader one way, not two.
    pub(super) fn rebase(err: SmilesError, offset: usize, input: &str) -> SmilesError {
        let span = Span::new(err.span.start + offset, err.span.end + offset);
        let kind = match err.kind {
            annotation @ SmilesErrorKind::AtomAnnotationUnsupported(_) => annotation,
            inner => SmilesErrorKind::CgLastBlockNotAtomistic(Box::new(inner)),
        };
        SmilesError::new(kind, span, input, Notation::CgSmiles)
    }
}

// ---- graph body ----
impl<'a> CgParser<'a> {
    // -- graphs -------------------------------------------------------------

    /// Read one `{…}` block as a coarse graph, `block` being its absolute byte
    /// range, braces included.
    ///
    /// The cursor is bounded at the closing `}` for the same reason
    /// [`CgParser::parse_body`] bounds it at the body's end: a token left open
    /// inside the block must be refused where it was written, not chased into
    /// whatever follows.
    fn parse_block(mut self, block: Span) -> Result<CgGraph, SmilesError> {
        self.scanner.seek(block.start);
        self.scanner.set_limit(block.end - 1);
        self.scanner.expect('{')?;
        while self.scanner.pos() < block.end - 1 {
            self.step()?;
        }
        self.check_finished()?;
        self.scanner.set_limit(self.scanner.input().len());
        self.scanner.seek(block.end); // past '}'
        if self.nodes.is_empty() {
            return Err(self.scanner.error_at(SmilesErrorKind::CgEmptyBlock, block));
        }
        Ok(self.take_graph())
    }

    /// Read a fragment body as a coarse graph, `body` being its absolute byte
    /// range — the entry point that has no braces around it.
    ///
    /// Entered at the body's offset in the **full** input rather than over a
    /// substring, so `|n`'s replay (which rewinds this scanner) and every
    /// diagnostic raised inside a body keep spans into the string the caller
    /// passed in. A `|n` written as the body's last token therefore works,
    /// which the reference implementation crashes on (R2.24).
    ///
    /// Seeking sets where the read *starts*; the cursor is bounded at
    /// `body.end` as well, so a token left open at the body's last byte — the
    /// `[` of `{[#A]}.{#A=[#B;k=1}.{#B=[$]C}` — is reported inside this body
    /// instead of scanning on into the block that follows it.
    fn parse_body(mut self, body: Span) -> Result<CgGraph, SmilesError> {
        self.scanner.seek(body.start);
        self.scanner.set_limit(body.end);
        while self.scanner.pos() < body.end {
            self.step()?;
        }
        self.check_finished()?;
        Ok(self.take_graph())
    }

    /// The checks every finished graph shares, in the order a reader meets
    /// them: an unspent bond symbol, an open branch, an unpaired ring marker,
    /// a descriptor with no node to bind to, and finally the simple-graph
    /// rule over the whole edge list.
    fn check_finished(&self) -> Result<(), SmilesError> {
        if let Some((_, at)) = self.pending {
            let span = Span::new(at, at + 1);
            return Err(self.scanner.error_at(SmilesErrorKind::CgDanglingBond, span));
        }
        if !self.branches.is_empty() {
            return Err(self.scanner.error(SmilesErrorKind::UnclosedBranch));
        }
        if let Some((&rnum, open)) = self.open_rings.iter().next() {
            let kind = SmilesErrorKind::UnmatchedRingClosure(rnum);
            return Err(self.scanner.error_at(kind, open.span));
        }
        if let Some((_, span)) = self.leading.first() {
            let kind = SmilesErrorKind::DanglingDescriptor;
            return Err(self.scanner.error_at(kind, *span));
        }
        self.check_simple_graph()
    }

    /// Hand over the nodes and edges read so far.
    fn take_graph(&mut self) -> CgGraph {
        CgGraph {
            nodes: std::mem::take(&mut self.nodes),
            edges: std::mem::take(&mut self.edges),
        }
    }

    /// Refuse a multigraph: a second edge between one pair of nodes, or an
    /// edge from a node to itself (R2.12 — "use bond order symbols instead").
    ///
    /// A pass over the finished edge list, in the shape of
    /// `chem::validation::validate_ring_closures` over a finished IR, and it
    /// has to be one: the duplicate only comes into being when the closing
    /// marker is read, while a structural error written after that closure —
    /// a `|` over a unit that closed a ring — is the earlier complaint about
    /// the same text and must be the one reported.
    fn check_simple_graph(&self) -> Result<(), SmilesError> {
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for edge in &self.edges {
            let pair = (edge.i.min(edge.j), edge.i.max(edge.j));
            if edge.i == edge.j || !seen.insert(pair) {
                let kind = SmilesErrorKind::CgDuplicateEdge {
                    i: edge.i,
                    j: edge.j,
                };
                return Err(self.scanner.error_at(kind, edge.span));
            }
        }
        Ok(())
    }

    /// Read the next token of the block and fold it into the graph.
    ///
    /// Every token but one also extends the repeatable unit to the cursor,
    /// which is why [`CgParser::mark_unit_end`] is called here rather than in
    /// each arm. The exception is the bond symbol: it returns early, so the
    /// symbol written before a `|` stays out of the copied text (R2.20).
    fn step(&mut self) -> Result<(), SmilesError> {
        match self.scanner.peek() {
            Some('[') if self.at_descriptor() => self.read_descriptor()?,
            Some('[') => self.read_node()?,
            Some('(') => self.open_branch()?,
            Some(')') => self.close_branch()?,
            Some('|') => self.read_repeat()?,
            Some('%') => self.read_ring_marker()?,
            Some(c) if c.is_ascii_digit() => self.read_ring_marker()?,
            Some(c) => return self.read_bond(c),
            None => return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd)),
        }
        self.mark_unit_end();
        Ok(())
    }

    // -- vocabulary ---------------------------------------------------------

    /// True when the cursor sits on a bonding-descriptor bracket rather than
    /// on a node: `[` followed by one of `$ < > !`. `[#` starts a node, and
    /// the two are told apart by this one character of lookahead.
    fn at_descriptor(&self) -> bool {
        self.scanner.peek() == Some('[')
            && matches!(self.scanner.peek_next(), Some('$' | '<' | '>' | '!'))
    }

    /// True for a character a fragment name may contain: ASCII alphanumeric,
    /// or the `*` of the wildcard bead. A name is read in two places — the
    /// `[#NAME]` of a node and the `#NAME=` of a table entry — and both take
    /// this one class, so a name written at one is spelled the same at the
    /// other.
    fn is_fragment_name_char(c: char) -> bool {
        c.is_ascii_alphanumeric() || c == '*'
    }

    /// The multiplicity a bond symbol writes, or `None` for a character that
    /// is not one. `.` (order 0) is deliberately not among them.
    fn bond_order(ch: char) -> Option<CgBondOrder> {
        match ch {
            '-' => Some(CgBondOrder::Single),
            '=' => Some(CgBondOrder::Double),
            '#' => Some(CgBondOrder::Triple),
            '$' => Some(CgBondOrder::Quadruple),
            _ => None,
        }
    }

    /// The atomistic bond kind a coarse multiplicity annotates a bonding
    /// descriptor with: a descriptor's order is stored as a `BondKind`,
    /// because the bond its pairing creates is an ordinary atomistic bond.
    fn bond_kind(order: CgBondOrder) -> BondKind {
        match order {
            CgBondOrder::Single => BondKind::Single,
            CgBondOrder::Double => BondKind::Double,
            CgBondOrder::Triple => BondKind::Triple,
            CgBondOrder::Quadruple => BondKind::Quadruple,
        }
    }

    /// Spend the pending bond symbol, defaulting to a single bond.
    fn take_pending(&mut self) -> CgBondOrder {
        self.pending
            .take()
            .map_or(CgBondOrder::Single, |(order, _)| order)
    }

    /// Extend the repeatable unit up to the cursor.
    fn mark_unit_end(&mut self) {
        let pos = self.scanner.pos();
        if let Some(unit) = self.unit.as_mut() {
            unit.end = pos;
        }
    }

    // -- bonds and branches -------------------------------------------------

    /// Remember a bond symbol for whatever token comes next.
    fn read_bond(&mut self, ch: char) -> Result<(), SmilesError> {
        let at = self.scanner.pos();
        let span = Span::new(at, at + 1);
        let Some(order) = Self::bond_order(ch) else {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::CgInvalidBondOrder, span));
        };
        if self.pending.is_some() {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::CgInvalidBondOrder, span));
        }
        self.scanner.advance();
        self.pending = Some((order, at));
        Ok(())
    }

    /// Open `(`: push the node written before it as the branch's anchor.
    fn open_branch(&mut self) -> Result<(), SmilesError> {
        let Some(anchor) = self.current else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar('(')));
        };
        if self.branches.is_empty()
            && let Some(unit) = self.unit.as_mut()
        {
            unit.branches += 1;
        }
        self.scanner.advance();
        self.branches.push(anchor);
        Ok(())
    }

    /// Close `)`: the chain continues from the branch's anchor, not from the
    /// last node inside the branch.
    fn close_branch(&mut self) -> Result<(), SmilesError> {
        if let Some((_, at)) = self.pending {
            let span = Span::new(at, at + 1);
            return Err(self.scanner.error_at(SmilesErrorKind::CgDanglingBond, span));
        }
        let Some(anchor) = self.branches.pop() else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(')')));
        };
        self.scanner.advance();
        self.current = Some(anchor);
        Ok(())
    }

    // -- nodes --------------------------------------------------------------

    /// Read `[#NAME;annotations]` and append it to the level.
    fn read_node(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('[')?;
        self.scanner.expect('#')?;
        let name_start = self.scanner.pos();
        while matches!(self.scanner.peek(), Some(c) if Self::is_fragment_name_char(c)) {
            self.scanner.advance();
        }
        let name = self.scanner.input()[name_start..self.scanner.pos()].to_owned();
        if name.is_empty() {
            return Err(match self.scanner.peek() {
                Some(c) => self.scanner.error(SmilesErrorKind::UnexpectedChar(c)),
                None => self.scanner.error(SmilesErrorKind::UnexpectedEnd),
            });
        }
        let fields = self.read_annotation_fields(start)?;
        self.scanner.expect(']')?;
        let span = self.scanner.span_from(start);
        let bound = self.bind_annotations(&name, &fields, span)?;
        self.push_node(
            CgNode {
                name,
                charge: bound.charge,
                annotations: bound.rest,
                descriptors: Vec::new(),
                parent: None,
                span,
            },
            start,
        )
    }

    /// Collect the raw `;`-separated annotation fields of a node, unparsed.
    fn read_annotation_fields(
        &mut self,
        node_start: usize,
    ) -> Result<Vec<(String, Span)>, SmilesError> {
        let mut fields = Vec::new();
        while self.scanner.peek() == Some(';') {
            self.scanner.advance();
            let start = self.scanner.pos();
            while !matches!(self.scanner.peek(), Some(';' | ']') | None) {
                self.scanner.advance();
            }
            if self.scanner.is_done() {
                let span = self.scanner.span_from(node_start);
                return Err(self
                    .scanner
                    .error_at(SmilesErrorKind::UnclosedBracket, span));
            }
            let text = self.scanner.input()[start..self.scanner.pos()].to_owned();
            fields.push((text, self.scanner.span_from(start)));
        }
        Ok(fields)
    }

    /// Bind the raw annotation fields to the dialect's reserved table.
    ///
    /// The table is positional (R2.14): slot 1 is the `#NAME` itself, slot 2
    /// is `q`, slot 3 is `w`. A field spelled `key=value` names its own key
    /// and takes no slot; a bare field takes the next one, and a fourth is
    /// [`SmilesErrorKind::CgMalformedAnnotation`]. Everything the table does
    /// not reserve is kept verbatim (R2.16).
    fn bind_annotations(
        &self,
        name: &str,
        fields: &[(String, Span)],
        span: Span,
    ) -> Result<BoundAnnotations, SmilesError> {
        if name == "*" && !fields.is_empty() {
            let text = fields
                .iter()
                .map(|(field, _)| field.as_str())
                .collect::<Vec<_>>()
                .join(";");
            let kind = SmilesErrorKind::CgAnnotationOnWildcard(text);
            return Err(self.scanner.error_at(kind, span));
        }
        let mut slot = 2usize;
        let mut bound = BoundAnnotations {
            charge: None,
            rest: Vec::new(),
        };
        for (text, field_span) in fields {
            let (key, value) = self.split_annotation(text, *field_span, &mut slot)?;
            match key.as_str() {
                // Slot 2 is `q`, so writing it by keyword *and* positionally
                // gives the charge two values in either written order. That is
                // malformed, not a last-one-wins override.
                "q" if bound.charge.is_some() => {
                    let kind = SmilesErrorKind::CgMalformedAnnotation(text.clone());
                    return Err(self.scanner.error_at(kind, *field_span));
                }
                "q" => bound.charge = Some(self.read_charge(&value, text, *field_span)?),
                "w" => self.check_weight(&value, text, *field_span)?,
                "x" => {
                    let kind = SmilesErrorKind::CgUnsupportedAnnotation { key, value };
                    return Err(self.scanner.error_at(kind, *field_span));
                }
                _ => bound.rest.push((key, value)),
            }
        }
        Ok(bound)
    }

    /// Split one annotation field into the key it names and the value it
    /// carries, taking a positional slot when it names no key.
    fn split_annotation(
        &self,
        text: &str,
        span: Span,
        slot: &mut usize,
    ) -> Result<(String, String), SmilesError> {
        let malformed = || {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            self.scanner.error_at(kind, span)
        };
        if let Some((key, value)) = text.split_once('=') {
            if key.is_empty() || value.is_empty() || value.contains('=') {
                return Err(malformed());
            }
            return Ok((key.to_owned(), value.to_owned()));
        }
        if text.is_empty() {
            return Err(malformed());
        }
        let key = match *slot {
            2 => "q",
            3 => "w",
            _ => return Err(malformed()),
        };
        *slot += 1;
        Ok((key.to_owned(), text.to_owned()))
    }

    /// Read `q` as a partial charge in `e`.
    fn read_charge(&self, value: &str, text: &str, span: Span) -> Result<F, SmilesError> {
        value.parse::<F>().map_err(|_| {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            self.scanner.error_at(kind, span)
        })
    }

    /// Accept `w` only at its default value.
    ///
    /// A mapping weight is meaningful notation that molrs does not model;
    /// refusing a non-default one is a stated choice, not a parse failure, so
    /// it names the key and the value rather than the character.
    fn check_weight(&self, value: &str, text: &str, span: Span) -> Result<(), SmilesError> {
        let Ok(weight) = value.parse::<F>() else {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            return Err(self.scanner.error_at(kind, span));
        };
        if weight == DEFAULT_WEIGHT {
            return Ok(());
        }
        let kind = SmilesErrorKind::CgUnsupportedAnnotation {
            key: "w".to_owned(),
            value: value.to_owned(),
        };
        Err(self.scanner.error_at(kind, span))
    }

    /// Append a node, bond it to the chain, and start a new repeatable unit
    /// when it is written at branch depth 0.
    ///
    /// Descriptors written before any node of the graph have been waiting for
    /// this one (01a's R4.2 leading rule) and are moved onto it first, so they
    /// keep their written order ahead of anything written after the node.
    fn push_node(&mut self, mut node: CgNode, start: usize) -> Result<(), SmilesError> {
        let span = node.span;
        node.descriptors
            .extend(self.leading.drain(..).map(|(descriptor, _)| descriptor));
        let index = self.nodes.len();
        match self.current {
            Some(previous) => {
                let order = self.take_pending();
                self.nodes.push(node);
                self.edges.push(CgEdge {
                    i: previous,
                    j: index,
                    order,
                    span,
                    origin: EdgeOrigin::Written,
                });
            }
            None => {
                if let Some((_, at)) = self.pending {
                    let at_span = Span::new(at, at + 1);
                    return Err(self
                        .scanner
                        .error_at(SmilesErrorKind::CgDanglingBond, at_span));
                }
                self.nodes.push(node);
            }
        }
        self.current = Some(index);
        if self.branches.is_empty() {
            self.unit = Some(RepeatUnit {
                start,
                end: self.scanner.pos(),
                anchor: index,
                branches: 0,
                ring_touched: false,
            });
        }
        Ok(())
    }

    // -- bonding descriptors ------------------------------------------------

    /// Read `[glyph label]` and anchor it on the node written before it.
    fn read_descriptor(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('[')?;
        let kind = match self.scanner.advance() {
            Some('$') => DescriptorKind::Symmetric,
            Some('<') => DescriptorKind::Left,
            Some('>') => DescriptorKind::Right,
            Some('!') => DescriptorKind::Shared,
            Some(c) => return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(c))),
            None => return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd)),
        };
        let label_start = self.scanner.pos();
        loop {
            match self.scanner.peek() {
                Some(']') => break,
                Some(c) if Self::bond_order(c).is_some() => {
                    return Err(self.scanner.error(SmilesErrorKind::BondInsideDescriptor));
                }
                Some(_) => {
                    self.scanner.advance();
                }
                None => {
                    let span = self.scanner.span_from(start);
                    return Err(self
                        .scanner
                        .error_at(SmilesErrorKind::UnclosedBracket, span));
                }
            }
        }
        let label = self.scanner.input()[label_start..self.scanner.pos()].to_owned();
        self.scanner.advance(); // ']'
        let order = self.pending.take().map(|(o, _)| Self::bond_kind(o));
        let descriptor = BondingDescriptor { kind, label, order };
        self.anchor_descriptor(descriptor, self.scanner.span_from(start))
    }

    /// Validate a descriptor and hang it on the current node.
    ///
    /// `validate_descriptor` is the shared grammar check; it is told which
    /// notation to stamp, so its error needs no re-stamping here.
    ///
    /// Written before the graph has a node, the descriptor is held until the
    /// first one arrives (01a's R4.2): `{#B1=[>][#PEO][#PEO][<]}` puts the `>`
    /// on the leading `[#PEO]`. A graph that ends with the queue still full
    /// had nothing to bind to, which is
    /// [`SmilesErrorKind::DanglingDescriptor`] — raised by
    /// [`CgParser::check_finished`], not here, because at this point a node
    /// may still be coming.
    fn anchor_descriptor(
        &mut self,
        descriptor: BondingDescriptor,
        span: Span,
    ) -> Result<(), SmilesError> {
        validate_descriptor(&descriptor, span, self.scanner.input(), Notation::CgSmiles)?;
        match self.current.and_then(|index| self.nodes.get_mut(index)) {
            Some(node) => node.descriptors.push(descriptor),
            None => self.leading.push((descriptor, span)),
        }
        Ok(())
    }

    // -- ring markers -------------------------------------------------------

    /// Read `1`..`9` or `%nn` and open or close the ring it names.
    fn read_ring_marker(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        let Some(glyph) = self.scanner.peek() else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd));
        };
        let Some(node) = self.current else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(glyph)));
        };
        let rnum = self.read_marker_number()?;
        if let Some(unit) = self.unit.as_mut() {
            unit.ring_touched = true;
        }
        match self.open_rings.remove(&rnum) {
            Some(open) => self.close_ring(rnum, open, node, start)?,
            None => {
                let order = self.pending.take().map(|(order, _)| order);
                let span = self.scanner.span_from(start);
                self.open_rings.insert(rnum, OpenRing { node, order, span });
            }
        }
        Ok(())
    }

    /// The marker number: one digit, or `%` and the whole digit run after it.
    ///
    /// `%` takes **any** number of digits (`%1` is marker 1, `%123` is marker
    /// 123), which is what the notation's own description says and what the
    /// reference implementation accepts. Zero is a marker number like any
    /// other in both spellings — OpenSMILES § 3.4 writes `C0CCCCC0` — so `0`
    /// and `%00` name marker 0.
    ///
    /// The marker key is a `u16`, so a run naming a larger number is refused
    /// rather than wrapped.
    fn read_marker_number(&mut self) -> Result<u16, SmilesError> {
        let start = self.scanner.pos();
        if self.scanner.peek() != Some('%') {
            return match self.scanner.eat_digit() {
                Some(digit) => Ok(u16::from(digit)),
                None => Err(self.invalid_marker(start)),
            };
        }
        self.scanner.advance();
        let digits = self.scanner.eat_digits();
        if digits.is_empty() {
            return Err(self.invalid_marker(start));
        }
        digits
            .parse::<u16>()
            .map_err(|_| self.invalid_marker(start))
    }

    /// The malformed-marker error, spanning the marker text read so far.
    fn invalid_marker(&self, start: usize) -> SmilesError {
        let span = self.scanner.span_from(start);
        self.scanner
            .error_at(SmilesErrorKind::CgInvalidRingMarker, span)
    }

    /// Close an open marker, emitting the ring edge at the closing marker.
    ///
    /// The order belongs to the bond, not to the end it was written at
    /// (OpenSMILES § 3.4 writes `C1CCCCC=1`), so a symbol at **either** end
    /// sets it and the same symbol at both agrees with itself. Two *differing*
    /// explicit symbols are a conflict rather than an override, because
    /// neither end is the authority. Whether the closure duplicates an edge
    /// the graph already has is decided by
    /// [`CgParser::check_simple_graph`] once the graph is read, not here.
    fn close_ring(
        &mut self,
        rnum: u16,
        open: OpenRing,
        node: usize,
        start: usize,
    ) -> Result<(), SmilesError> {
        let span = self.scanner.span_from(start);
        let order = match (open.order, self.pending.take()) {
            (Some(opening), Some((closing, at))) if opening != closing => {
                let kind = SmilesErrorKind::RingBondConflict { rnum };
                return Err(self.scanner.error_at(kind, Span::new(at, at + 1)));
            }
            (Some(opening), _) => opening,
            (None, Some((closing, _))) => closing,
            (None, None) => CgBondOrder::Single,
        };
        self.edges.push(CgEdge {
            i: open.node,
            j: node,
            order,
            span,
            origin: EdgeOrigin::Written,
        });
        Ok(())
    }

    // -- the `|n` repeat operator -------------------------------------------

    /// Repeat the preceding unit `n` times in total, chaining the copies.
    ///
    /// The repeat *consumes* the unit: `self.unit` is cleared, so a second `|`
    /// written straight after finds nothing to repeat and is refused as an
    /// unexpected character rather than replaying the same bytes again. The
    /// next node written at branch depth 0 starts a fresh unit.
    fn read_repeat(&mut self) -> Result<(), SmilesError> {
        let at = self.scanner.pos();
        self.scanner.advance(); // '|'
        let count = self.read_repeat_count()?;
        let unit = self.repeatable_unit(at)?;
        let order = self.take_pending();
        let resume = self.scanner.pos();
        let mut previous = unit.anchor;
        for _ in 1..count {
            previous = self.replay(&unit, previous, order, at)?;
        }
        self.scanner.seek(resume);
        self.current = Some(previous);
        self.unit = None;
        Ok(())
    }

    /// The count after `|`: a positive decimal integer, `1` being the
    /// identity (one copy, no chaining edge), at most
    /// [`MAX_REPEAT_COUNT`].
    ///
    /// The upper bound is molrs's own — see [`MAX_REPEAT_COUNT`] — and is
    /// reported exactly like `0` and `x` are, as
    /// [`SmilesErrorKind::CgInvalidRepeatCount`] carrying the digits that were
    /// written: one rule about what a count may be, stated at both ends of the
    /// range.
    fn read_repeat_count(&mut self) -> Result<usize, SmilesError> {
        let start = self.scanner.pos();
        let digits = self.scanner.eat_digits();
        if digits.is_empty() {
            let text = self.scanner.peek().map(String::from).unwrap_or_default();
            return Err(self.invalid_count(text, start));
        }
        match digits.parse::<usize>() {
            Ok(count) if (1..=MAX_REPEAT_COUNT).contains(&count) => Ok(count),
            _ => Err(self.invalid_count(digits.to_owned(), start)),
        }
    }

    /// The bad-repeat-count error, carrying the text read after `|`.
    fn invalid_count(&self, text: String, start: usize) -> SmilesError {
        let span = self.scanner.span_from(start);
        self.scanner
            .error_at(SmilesErrorKind::CgInvalidRepeatCount(text), span)
    }

    /// The unit `|` may repeat, with the shapes that cannot be replayed
    /// refused.
    ///
    /// A ring marker inside the unit is a one-shot identity and replaying it
    /// would either overwrite the open marker or close it twice; a bonding
    /// descriptor inside the unit, by contrast, is cloned into every copy,
    /// because a descriptor label is a pairing *class* consumed one occurrence
    /// at a time.
    fn repeatable_unit(&self, at: usize) -> Result<RepeatUnit, SmilesError> {
        let span = Span::new(at, at + 1);
        if !self.branches.is_empty() {
            let kind = SmilesErrorKind::CgRepeatOnBranchedNode;
            return Err(self.scanner.error_at(kind, span));
        }
        let Some(unit) = self.unit.as_ref() else {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::UnexpectedChar('|'), span));
        };
        if unit.ring_touched {
            let kind = SmilesErrorKind::CgRepeatOnRingMarker;
            return Err(self.scanner.error_at(kind, span));
        }
        if unit.branches > 1 {
            let kind = SmilesErrorKind::CgRepeatOnBranchedNode;
            return Err(self.scanner.error_at(kind, span));
        }
        Ok(unit.clone())
    }

    /// Re-read the unit's bytes as one more copy and chain it to `previous`.
    ///
    /// The scanner rewinds over the *full* input, so a node parsed here keeps
    /// the template's span and an error raised here points at the template's
    /// text — the copies share spans, which is why spans in a level are
    /// neither disjoint nor monotonic.
    fn replay(
        &mut self,
        unit: &RepeatUnit,
        previous: usize,
        order: CgBondOrder,
        at: usize,
    ) -> Result<usize, SmilesError> {
        self.scanner.seek(unit.start);
        self.current = None;
        let anchor = self.nodes.len();
        while self.scanner.pos() < unit.end {
            self.step()?;
        }
        self.edges.push(CgEdge {
            i: previous,
            j: anchor,
            order,
            span: Span::new(at, at + 1),
            origin: EdgeOrigin::Written,
        });
        Ok(anchor)
    }
}

#[cfg(test)]
mod tests {
    use crate::io::cgsmiles::{CgBondOrder, CgGraph, CgNode, CgSmilesIr, EdgeOrigin, FragmentBody};
    use crate::io::smiles::{DescriptorKind, Notation, SmilesError, SmilesErrorKind, SmilesIr};
    use crate::line_notation::fixtures::{atom_nodes, descriptors};

    // -- helpers ------------------------------------------------------------

    /// The single resolution level of a `CGsmiles` string that must parse.
    fn level0(text: &str) -> CgGraph {
        let ir = CgSmilesIr::parse(text)
            .unwrap_or_else(|e| panic!("CgSmilesIr::parse({text:?}) failed: {e}"));
        ir.levels
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("CgSmilesIr::parse({text:?}) produced no level"))
    }

    /// `(i, j, order)` of every edge, in parse order.
    fn edges(graph: &CgGraph) -> Vec<(usize, usize, CgBondOrder)> {
        graph.edges.iter().map(|e| (e.i, e.j, e.order)).collect()
    }

    /// `(i, j)` of every edge, in parse order — for fixtures whose expectation
    /// is the connectivity alone.
    fn pairs(graph: &CgGraph) -> Vec<(usize, usize)> {
        graph.edges.iter().map(|e| (e.i, e.j)).collect()
    }

    /// The error kind `text` must be refused with.
    fn kind_of(text: &str) -> SmilesErrorKind {
        CgSmilesIr::parse(text)
            .err()
            .unwrap_or_else(|| panic!("CgSmilesIr::parse({text:?}) was accepted"))
            .kind
    }

    // -- F1.1 ---------------------------------------------------------------

    #[test]
    fn test_single_node_block_has_one_named_node_and_no_edge() {
        let graph = level0("{[#PEO]}");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].name, "PEO");
        assert!(graph.edges.is_empty(), "edges were {:?}", graph.edges);
    }

    // -- F1.2, F1.3: adjacency and bond symbols -----------------------------

    #[test]
    fn test_adjacent_nodes_are_joined_by_single_edges() {
        let graph = level0("{[#PEO][#PEO][#PEO]}");
        assert_eq!(
            edges(&graph),
            vec![(0, 1, CgBondOrder::Single), (1, 2, CgBondOrder::Single),]
        );
    }

    #[test]
    fn test_bond_symbols_set_the_edge_order() {
        let graph = level0("{[#A]=[#B]#[#C]$[#D]}");
        assert_eq!(
            edges(&graph),
            vec![
                (0, 1, CgBondOrder::Double),
                (1, 2, CgBondOrder::Triple),
                (2, 3, CgBondOrder::Quadruple),
            ]
        );
    }

    // -- F1.5: branches -----------------------------------------------------

    #[test]
    fn test_branch_attaches_to_the_node_written_before_it() {
        let graph = level0("{[#A]([#B])[#C]}");
        assert_eq!(pairs(&graph), vec![(0, 1), (0, 2)]);
    }

    // -- F1.6, F1.14, F1.15: ring markers -----------------------------------

    #[test]
    fn test_ring_closure_edge_is_emitted_last() {
        let graph = level0("{[#A]1[#B][#C]1}");
        assert_eq!(pairs(&graph), vec![(0, 1), (1, 2), (0, 2)]);
    }

    #[test]
    fn test_two_digit_ring_marker_closes_the_ring() {
        let graph = level0("{[#A]%12[#B][#C]%12}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn test_ring_marker_number_is_reusable_after_it_closes() {
        let graph = level0("{[#A]1[#B][#A]1[#C]}");
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 4);
    }

    #[test]
    fn test_ring_bond_order_comes_from_the_opening_marker() {
        let graph = level0("{[#A]=1[#B][#C]1}");
        let ring = graph.edges.last().expect("ring closure edge is missing");
        assert_eq!((ring.i, ring.j, ring.order), (0, 2, CgBondOrder::Double));
    }

    /// A symbol written before the *closing* marker only sets the closure's
    /// order just as well (OpenSMILES § 3.4 writes `C1CCCCC=1`); the order is
    /// a property of the bond, not of the end it was written at.
    #[test]
    fn test_ring_bond_order_may_be_written_on_the_closing_marker() {
        let graph = level0("{[#A]1[#B][#C]=1}");
        let ring = graph.edges.last().expect("ring closure edge is missing");
        assert_eq!((ring.i, ring.j, ring.order), (0, 2, CgBondOrder::Double));
    }

    /// The same symbol at both ends agrees with itself and is not a conflict.
    #[test]
    fn test_ring_bond_order_written_at_both_ends_agrees() {
        let graph = level0("{[#A]=1[#B][#C]=1}");
        let ring = graph.edges.last().expect("ring closure edge is missing");
        assert_eq!((ring.i, ring.j, ring.order), (0, 2, CgBondOrder::Double));
    }

    // -- F1.7, F1.11, F1.12: annotations ------------------------------------

    #[test]
    fn test_keyword_charge_is_read_as_a_partial_charge_in_e() {
        let graph = level0("{[#A;q=-0.5;kind=ether]}");
        let q = graph.nodes[0].charge.expect("q annotation was dropped");
        assert!((q - (-0.5)).abs() < 1e-10, "charge was {q}");
    }

    #[test]
    fn test_unknown_annotation_is_retained_verbatim_without_q_or_w() {
        let graph = level0("{[#A;q=-0.5;kind=ether]}");
        assert_eq!(
            graph.nodes[0].annotations,
            vec![("kind".to_owned(), "ether".to_owned())]
        );
    }

    #[test]
    fn test_second_positional_slot_is_the_charge() {
        let graph = level0("{[#A;0.5]}");
        let q = graph.nodes[0].charge.expect("positional q was dropped");
        assert!((q - 0.5).abs() < 1e-10, "charge was {q}");
    }

    #[test]
    fn test_third_positional_slot_is_the_weight_and_a_non_default_one_is_refused() {
        let kind = kind_of("{[#A;0;0.5]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "w" && value == "0.5"
            ),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_default_positional_weight_parses_with_a_zero_charge() {
        let graph = level0("{[#A;0;1]}");
        let q = graph.nodes[0].charge.expect("positional q was dropped");
        assert!(q.abs() < 1e-10, "charge was {q}");
        assert!(
            graph.nodes[0].annotations.is_empty(),
            "reserved slots must not be kept verbatim, annotations were {:?}",
            graph.nodes[0].annotations
        );
    }

    /// Slot 2 is `q`: writing it by keyword and then positionally gives the
    /// charge two values, which is malformed rather than a last-one-wins
    /// override.
    #[test]
    fn test_charge_bound_by_keyword_then_positionally_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;q=1;0.5]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    /// The same collision in the other written order.
    #[test]
    fn test_charge_bound_positionally_then_by_keyword_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;0;q=1]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    // -- F1.8: bonding descriptors ------------------------------------------

    #[test]
    fn test_descriptor_binds_to_the_node_written_before_it() {
        let graph = level0("{[#A][$][#B][$]}");
        let kinds: Vec<Vec<DescriptorKind>> = graph
            .nodes
            .iter()
            .map(|n| n.descriptors.iter().map(|d| d.kind).collect())
            .collect();
        assert_eq!(
            kinds,
            vec![
                vec![DescriptorKind::Symmetric],
                vec![DescriptorKind::Symmetric],
            ]
        );
    }

    #[test]
    fn test_descriptor_is_not_a_node_of_the_chain() {
        let graph = level0("{[#A][$][#B][$]}");
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    // -- F1.4, F1.9, F1.13: the `|n` repeat operator ------------------------

    #[test]
    fn test_repeat_chains_one_node_into_a_run() {
        let graph = level0("{[#PEO]|4}");
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(
            edges(&graph),
            vec![
                (0, 1, CgBondOrder::Single),
                (1, 2, CgBondOrder::Single),
                (2, 3, CgBondOrder::Single),
            ]
        );
    }

    #[test]
    fn test_repeat_count_one_is_the_identity() {
        let graph = level0("{[#A]|1}");
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.edges.is_empty(), "edges were {:?}", graph.edges);
    }

    /// F1.9 (R2.20): the repeated unit includes its anchor, and successive
    /// anchors are chained — copy 1 is nodes 0,1,2 with edges (0,1) and (1,2),
    /// copy 2 is nodes 3,4,5 with edges (3,4) and (4,5), joined by (0,3). The
    /// expectation is the *parse-order* edge vector, unsorted: the chaining
    /// edge (0,3) is pushed after the copy's own internal edges, so it lands
    /// last rather than second.
    #[test]
    fn test_repeated_branch_unit_is_copied_with_its_anchor() {
        let graph = level0("{[#A]([#B][#C])|2}");
        assert_eq!(graph.nodes.len(), 6);
        assert_eq!(graph.edges.len(), 5);
        assert_eq!(pairs(&graph), vec![(0, 1), (1, 2), (3, 4), (4, 5), (0, 3)]);
    }

    #[test]
    fn test_repeat_of_a_two_node_branch_expands_to_nine_nodes() {
        let graph = level0("{[#A]([#B][#B])|3}");
        assert_eq!(graph.nodes.len(), 9);
        assert_eq!(graph.edges.len(), 8);
    }

    #[test]
    fn test_repeat_with_a_late_anchor_expands_to_seven_nodes() {
        let graph = level0("{[#A][#B]([#C])|3}");
        assert_eq!(graph.nodes.len(), 7);
        assert_eq!(graph.edges.len(), 6);
    }

    #[test]
    fn test_descriptor_inside_a_repeated_unit_is_cloned_into_every_copy() {
        let graph = level0("{[#A][$1]|3}");
        let labels: Vec<Vec<(DescriptorKind, String)>> = graph
            .nodes
            .iter()
            .map(|n| {
                n.descriptors
                    .iter()
                    .map(|d| (d.kind, d.label.clone()))
                    .collect()
            })
            .collect();
        assert_eq!(
            labels,
            vec![
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
            ]
        );
    }

    /// R2.22: a bond symbol written before `|` is promoted to the order of the
    /// edge that joins each copy to the previous one.
    #[test]
    fn test_bond_symbol_before_the_repeat_sets_the_chaining_order() {
        let graph = level0("{[#A]=|3}");
        assert_eq!(
            edges(&graph),
            vec![(0, 1, CgBondOrder::Double), (1, 2, CgBondOrder::Double),]
        );
    }

    /// R2.22 in its branch form: the symbol before `|` orders the edge that
    /// chains each copy of the *whole branch unit* to the previous anchor.
    /// `{[#A]([#B])=|3}` is A0(B1), A2(B3), A4(B5) — five edges, of which the
    /// two chaining ones are double. Compared as a sorted set: the edge order
    /// within a copy is not the behaviour under test here.
    #[test]
    fn test_bond_symbol_before_a_branch_repeat_orders_the_chaining_edges() {
        let graph = level0("{[#A]([#B])=|3}");
        assert_eq!(graph.nodes.len(), 6);
        let mut got = edges(&graph);
        got.sort_by_key(|&(i, j, _)| (i, j));
        assert_eq!(
            got,
            vec![
                (0, 1, CgBondOrder::Single),
                (0, 2, CgBondOrder::Double),
                (2, 3, CgBondOrder::Single),
                (2, 4, CgBondOrder::Double),
                (4, 5, CgBondOrder::Single),
            ]
        );
    }

    /// Every construct is validated on the first pass over the input, so there
    /// is no replay-only error path: the non-default weight below is refused
    /// while the template is read, before any copy of it exists. The error a
    /// repeated unit produces therefore carries the full input and points
    /// inside the bracket that was actually written.
    #[test]
    fn test_error_inside_a_repeated_unit_carries_the_full_input() {
        let text = "{[#A]([#B;w=2])|2}";
        let err = CgSmilesIr::parse(text).expect_err("a non-default weight must be refused");
        assert_eq!(err.input, text);
        assert!(
            err.span.start < text.len(),
            "span was {:?} over {} bytes",
            err.span,
            text.len()
        );
        let open = text.find("[#B").expect("fixture writes a [#B bracket");
        let close = open + "[#B;w=2]".len();
        assert!(
            (open..close).contains(&err.span.start),
            "span {:?} is outside the [#B;w=2] bracket at {open}..{close}",
            err.span
        );
    }

    // -- rejections ---------------------------------------------------------

    #[test]
    fn test_empty_block_is_refused() {
        assert!(matches!(kind_of("{}"), SmilesErrorKind::CgEmptyBlock));
    }

    #[test]
    fn test_empty_annotation_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_annotation_without_a_key_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;=1]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_non_numeric_charge_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;q=x]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_fourth_positional_slot_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;0;1;x;y]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_keyword_weight_is_unsupported() {
        let kind = kind_of("{[#A;w=2]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "w" && value == "2"
            ),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_chirality_annotation_is_unsupported() {
        let kind = kind_of("{[#A;x=S]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "x" && value == "S"
            ),
            "kind was {kind:?}"
        );
    }

    /// With no fragment table to resolve it against, a wildcard bead is just
    /// a node name: it parses and bonds like any other.
    #[test]
    fn test_wildcard_bead_in_a_base_only_string_parses() {
        let graph = level0("{[#*][#A]}");
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.nodes[0].name, "*");
        assert_eq!(graph.edges.len(), 1);
    }

    #[test]
    fn test_annotation_on_a_wildcard_node_is_refused() {
        assert!(matches!(
            kind_of("{[#*;q=1]}"),
            SmilesErrorKind::CgAnnotationOnWildcard(_)
        ));
    }

    #[test]
    fn test_zero_bond_order_is_refused() {
        assert!(matches!(
            kind_of("{[#A].[#B]}"),
            SmilesErrorKind::CgInvalidBondOrder
        ));
    }

    #[test]
    fn test_bond_symbol_before_the_block_close_is_dangling() {
        assert!(matches!(
            kind_of("{[#A]=}"),
            SmilesErrorKind::CgDanglingBond
        ));
    }

    /// `%` with no digit run after it names no marker.
    #[test]
    fn test_percent_without_digits_is_an_invalid_ring_marker() {
        assert!(matches!(
            kind_of("{[#A]%}"),
            SmilesErrorKind::CgInvalidRingMarker
        ));
    }

    /// The digit run is unbounded in the notation but the marker key is a
    /// `u16`, so a number past that range is refused rather than wrapped.
    #[test]
    fn test_ring_marker_number_past_the_key_range_is_refused() {
        assert!(matches!(
            kind_of("{[#A]%99999[#B]%99999}"),
            SmilesErrorKind::CgInvalidRingMarker
        ));
    }

    #[test]
    fn test_ring_closure_duplicating_an_edge_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1[#B]1[#A]}"),
            SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 }
        ));
    }

    /// A marker opened and closed on one node would bond that node to itself,
    /// which the simple-graph check refuses under the duplicate-edge kind.
    #[test]
    fn test_ring_marker_opened_and_closed_on_one_node_is_a_self_bond() {
        assert!(matches!(
            kind_of("{[#A]11}"),
            SmilesErrorKind::CgDuplicateEdge { i: 0, j: 0 }
        ));
    }

    /// The self-bond reads as one: the rendered message says the node was
    /// bonded to itself, not that two nodes are "already joined".
    #[test]
    fn test_self_bond_message_says_the_node_is_bonded_to_itself() {
        let rendered = err_of("{[#A]11}").to_string();
        assert!(
            rendered.contains("itself"),
            "rendered error was {rendered:?}"
        );
    }

    /// R2.5: the order belongs to the bond rather than to the end it was
    /// written at, so neither marker is the authority and two *differing*
    /// explicit symbols are a conflict, not an override by the later one.
    #[test]
    fn test_conflicting_ring_bond_orders_are_refused() {
        assert!(matches!(
            kind_of("{[#A]=1[#B][#C]-1}"),
            SmilesErrorKind::RingBondConflict { .. }
        ));
    }

    #[test]
    fn test_unclosed_ring_marker_is_unmatched() {
        assert!(matches!(
            kind_of("{[#A]1[#B]}"),
            SmilesErrorKind::UnmatchedRingClosure(1)
        ));
    }

    /// Two markers are left open, and which of them is reported is
    /// unspecified — the open markers live in a `HashMap` — so the assertion
    /// is on the kind only, never on the marker number.
    #[test]
    fn test_two_unclosed_ring_markers_report_an_unmatched_closure() {
        assert!(matches!(
            kind_of("{[#A]1[#B]2}"),
            SmilesErrorKind::UnmatchedRingClosure(_)
        ));
    }

    /// A descriptor binds to a node, and one written before the first node
    /// waits for it — so the refusal needs a block with no node at all, not
    /// merely one whose descriptor comes first (`{[$][#A]}` parses).
    #[test]
    fn test_descriptor_without_a_node_to_anchor_it_is_refused() {
        assert!(matches!(
            kind_of("{[$]}"),
            SmilesErrorKind::DanglingDescriptor
        ));
    }

    /// `CGsmiles` writes a descriptor's bond order outside the bracket, so the
    /// `BigSMILES` in-bracket spelling is refused rather than translated.
    #[test]
    fn test_bond_symbol_inside_a_descriptor_bracket_is_refused() {
        assert!(matches!(
            kind_of("{[#A][$=]}"),
            SmilesErrorKind::BondInsideDescriptor
        ));
    }

    /// The annotation run swallows the block's `}` and hits end of input, which
    /// is a missing `]`, not a missing `}`.
    /// A body is bounded by its own block: an unclosed `[` inside one must be
    /// reported *within* that block, never by scanning on into the next one.
    /// The assertion is on the bound rather than on the kind, because which
    /// rule the truncated body breaks is the parser's to choose — running past
    /// the `}` is what is forbidden.
    #[test]
    fn test_unclosed_bracket_in_a_body_does_not_scan_into_the_next_block() {
        let text = "{[#A]}.{#A=[#B;k=1}.{#B=[$]C}";
        let second_block_end = text
            .find("}.{#B")
            .expect("the second block closes before the third opens")
            + 1;
        let err = err_of(text);
        assert!(
            err.span.end <= second_block_end,
            "span {:?} of {:?} runs past the second block, which ends at {second_block_end}",
            err.span,
            err.kind
        );
    }

    #[test]
    fn test_node_bracket_left_open_after_an_annotation_is_refused() {
        assert!(matches!(
            kind_of("{[#A;k=v}"),
            SmilesErrorKind::UnclosedBracket
        ));
    }

    #[test]
    fn test_zero_repeat_count_is_refused() {
        assert!(matches!(
            kind_of("{[#A]|0}"),
            SmilesErrorKind::CgInvalidRepeatCount(_)
        ));
    }

    /// The cap is a notation rule, not an allocation detail: `|n` replays the
    /// unit `n` times, so an uncapped count is a memory bomb written in four
    /// characters. `65535` is the last count that parses.
    #[test]
    fn test_repeat_count_at_the_cap_is_accepted() {
        let graph = level0("{[#A]|65535}");
        assert_eq!(graph.nodes.len(), 65535);
    }

    /// One past the cap is a bad count, reported like `0` and `x` — the same
    /// rule, stated at the other end of the range.
    #[test]
    fn test_repeat_count_past_the_cap_is_refused() {
        assert!(matches!(
            kind_of("{[#A]|65536}"),
            SmilesErrorKind::CgInvalidRepeatCount(_)
        ));
    }

    #[test]
    fn test_non_numeric_repeat_count_is_refused() {
        assert!(matches!(
            kind_of("{[#A]|x}"),
            SmilesErrorKind::CgInvalidRepeatCount(_)
        ));
    }

    /// A repeat consumes the unit it copies, so a second `|` written straight
    /// after the first has nothing left to repeat and is an unexpected
    /// character — not a silent second replay of the same bytes.
    #[test]
    fn test_stacked_repeat_has_no_unit_left_to_repeat() {
        assert!(matches!(
            kind_of("{[#A]|2|3}"),
            SmilesErrorKind::UnexpectedChar('|')
        ));
    }

    #[test]
    fn test_ring_marker_opened_inside_a_repeated_unit_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1|2[#B]1}"),
            SmilesErrorKind::CgRepeatOnRingMarker
        ));
    }

    #[test]
    fn test_ring_marker_closed_inside_a_repeated_unit_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1[#B]1|2}"),
            SmilesErrorKind::CgRepeatOnRingMarker
        ));
    }

    #[test]
    fn test_repeat_inside_an_open_branch_is_refused() {
        assert!(matches!(
            kind_of("{[#A](|2[#B])}"),
            SmilesErrorKind::CgRepeatOnBranchedNode
        ));
    }

    #[test]
    fn test_repeat_on_a_node_with_two_branches_is_refused() {
        assert!(matches!(
            kind_of("{[#A]([#B])([#C])|3}"),
            SmilesErrorKind::CgRepeatOnBranchedNode
        ));
    }

    #[test]
    fn test_every_cgsmiles_error_is_stamped_with_the_cgsmiles_notation() {
        for text in [
            "{}",
            "{[#A;]}",
            "{[#A].[#B]}",
            "{[#A]|0}",
            "{[#PEO][#PEO]}[#X]",
        ] {
            let err = CgSmilesIr::parse(text).expect_err("input must be refused");
            assert_eq!(err.notation, Notation::CgSmiles, "input was {text}");
        }
    }

    // =======================================================================
    // 01c: multi-block fragment tables
    // =======================================================================
    //
    // Fixtures F2, F8, F9 and the reviewer-added strings of
    // `.claude/specs/cgsmiles-01c-fragments.md` § Domain basis. Every count,
    // parent vector, edge order and descriptor placement below is hand-derived
    // from the rules cited there (R1.1–R1.5, R2.24, R3.0–R3.4, R4.10, R4.20,
    // R5.1, R5.2, R5.5); no external program produced any expected value.

    /// F2 — an OH-capped PEO trimer over one atomistic fragment table.
    const F2: &str = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}";

    /// F8 — three resolutions: beads of blocks, blocks of beads, beads of
    /// atoms.
    const F8: &str = "{[#B1][#B2][#B1]}.{#B1=[>][#PEO][#PEO][<],#B2=[>][#PE][#PE][<]}.\
                      {#PEO=[>]COC[<],#PE=[>]CC[<]}";

    // -- helpers ------------------------------------------------------------

    /// The whole IR of a `CGsmiles` string that must parse.
    fn ir_of(text: &str) -> CgSmilesIr {
        CgSmilesIr::parse(text)
            .unwrap_or_else(|e| panic!("CgSmilesIr::parse({text:?}) failed: {e}"))
    }

    /// The whole error a `CGsmiles` string must be refused with.
    fn err_of(text: &str) -> SmilesError {
        CgSmilesIr::parse(text)
            .err()
            .unwrap_or_else(|| panic!("CgSmilesIr::parse({text:?}) was accepted"))
    }

    /// The coarse-graph body stored for `name` in fragment table `table`.
    fn graph_body<'a>(ir: &'a CgSmilesIr, table: usize, name: &str) -> &'a CgGraph {
        match &ir.fragments[table][name].body {
            FragmentBody::Graph(graph) => graph,
            FragmentBody::Smiles(_) => panic!("fragment {name:?} holds an atomistic body"),
        }
    }

    /// The atomistic body stored for `name` in fragment table `table`.
    fn smiles_body<'a>(ir: &'a CgSmilesIr, table: usize, name: &str) -> &'a SmilesIr {
        match &ir.fragments[table][name].body {
            FragmentBody::Smiles(body) => body.ir(),
            FragmentBody::Graph(_) => panic!("fragment {name:?} holds a coarse-graph body"),
        }
    }

    /// The descriptor kinds one coarse node carries, in written order.
    fn node_kinds(node: &CgNode) -> Vec<DescriptorKind> {
        node.descriptors.iter().map(|d| d.kind).collect()
    }

    /// The descriptor kinds every node of a level carries, in index order.
    fn level_kinds(graph: &CgGraph) -> Vec<Vec<DescriptorKind>> {
        graph.nodes.iter().map(node_kinds).collect()
    }

    /// The descriptor kinds an atomistic body carries, in atom-visit order.
    fn body_kinds(body: &SmilesIr) -> Vec<DescriptorKind> {
        descriptors(body).iter().map(|d| d.kind).collect()
    }

    /// Node names of a level, in index order.
    fn names(graph: &CgGraph) -> Vec<&str> {
        graph.nodes.iter().map(|n| n.name.as_str()).collect()
    }

    /// `parent` of every node of a level, in index order.
    fn parents(graph: &CgGraph) -> Vec<Option<usize>> {
        graph.nodes.iter().map(|n| n.parent).collect()
    }

    // -- F2: base level and its atomistic table -----------------------------

    #[test]
    fn test_f2_base_level_has_five_nodes_and_four_edges() {
        let ir = ir_of(F2);
        assert_eq!(ir.levels[0].nodes.len(), 5);
        assert_eq!(ir.levels[0].edges.len(), 4);
    }

    /// `parent` is `None` in `levels[0]`: nothing above it instantiated it.
    #[test]
    fn test_f2_base_level_nodes_have_no_parent() {
        let ir = ir_of(F2);
        assert_eq!(parents(&ir.levels[0]), vec![None; 5]);
    }

    #[test]
    fn test_f2_fragment_table_is_keyed_by_the_two_written_names() {
        let ir = ir_of(F2);
        assert_eq!(
            ir.fragments[0].keys().collect::<Vec<&String>>(),
            vec!["OH", "PEO"]
        );
    }

    #[test]
    fn test_f2_last_block_bodies_are_atomistic() {
        let ir = ir_of(F2);
        assert!(matches!(
            ir.fragments[0]["OH"].body,
            FragmentBody::Smiles(_)
        ));
        assert!(matches!(
            ir.fragments[0]["PEO"].body,
            FragmentBody::Smiles(_)
        ));
    }

    /// That `[$]COC[$]` parses at all is the proof the atomistic path runs
    /// through `parse_fragment_smiles`: `parse_smiles` refuses a descriptor
    /// with `DescriptorInPlainSmiles`.
    #[test]
    fn test_f2_peo_body_has_three_atoms_and_two_symmetric_descriptors() {
        let ir = ir_of(F2);
        let body = smiles_body(&ir, 0, "PEO");
        assert_eq!(atom_nodes(body).len(), 3);
        assert_eq!(
            body_kinds(body),
            vec![DescriptorKind::Symmetric, DescriptorKind::Symmetric]
        );
    }

    #[test]
    fn test_f2_oh_body_has_one_atom_and_one_symmetric_descriptor() {
        let ir = ir_of(F2);
        let body = smiles_body(&ir, 0, "OH");
        assert_eq!(atom_nodes(body).len(), 1);
        assert_eq!(body_kinds(body), vec![DescriptorKind::Symmetric]);
    }

    // -- F8: intermediate table (coarse bodies) -----------------------------

    #[test]
    fn test_f8_intermediate_body_is_a_graph_of_two_nodes_and_one_edge() {
        let ir = ir_of(F8);
        let b1 = graph_body(&ir, 0, "B1");
        assert_eq!(b1.nodes.len(), 2);
        assert_eq!(pairs(b1), vec![(0, 1)]);
    }

    /// R4.2: a descriptor written before the first node of a body binds to
    /// that node — F8's leading `[>]` lands on B1's first `[#PEO]`.
    #[test]
    fn test_f8_leading_descriptor_binds_to_the_first_node_of_the_body() {
        let ir = ir_of(F8);
        let b1 = graph_body(&ir, 0, "B1");
        assert_eq!(node_kinds(&b1.nodes[0]), vec![DescriptorKind::Right]);
    }

    #[test]
    fn test_f8_trailing_descriptor_binds_to_the_last_node_of_the_body() {
        let ir = ir_of(F8);
        let b1 = graph_body(&ir, 0, "B1");
        assert_eq!(node_kinds(&b1.nodes[1]), vec![DescriptorKind::Left]);
    }

    // -- F8: last table (atomistic bodies) ----------------------------------

    #[test]
    fn test_f8_peo_body_has_three_atoms_with_a_right_and_a_left_descriptor() {
        let ir = ir_of(F8);
        let body = smiles_body(&ir, 1, "PEO");
        assert_eq!(atom_nodes(body).len(), 3);
        assert_eq!(
            body_kinds(body),
            vec![DescriptorKind::Right, DescriptorKind::Left]
        );
    }

    #[test]
    fn test_f8_pe_body_has_two_atoms_with_a_right_and_a_left_descriptor() {
        let ir = ir_of(F8);
        let body = smiles_body(&ir, 1, "PE");
        assert_eq!(atom_nodes(body).len(), 2);
        assert_eq!(
            body_kinds(body),
            vec![DescriptorKind::Right, DescriptorKind::Left]
        );
    }

    // -- F8: the level the intermediate table denotes (R4.10 / R4.20) -------

    #[test]
    fn test_f8_level_one_holds_one_copy_of_each_body_per_referencing_node() {
        let ir = ir_of(F8);
        assert_eq!(
            names(&ir.levels[1]),
            vec!["PEO", "PEO", "PE", "PE", "PEO", "PEO"]
        );
    }

    #[test]
    fn test_f8_level_one_parents_index_the_level_zero_nodes() {
        let ir = ir_of(F8);
        assert_eq!(
            parents(&ir.levels[1]),
            vec![Some(0), Some(0), Some(1), Some(1), Some(2), Some(2)]
        );
    }

    /// Expansion writes one edge inside each copy and nothing else; the
    /// edges 01d derives from the level above are pinned by
    /// `resolve.rs::test_f8_appends_one_derived_edge_per_level_zero_pair`.
    #[test]
    fn test_f8_level_one_has_three_single_written_intra_fragment_edges() {
        let ir = ir_of(F8);
        let written: Vec<(usize, usize, CgBondOrder)> = ir.levels[1]
            .edges
            .iter()
            .filter(|e| e.origin == EdgeOrigin::Written)
            .map(|e| (e.i, e.j, e.order))
            .collect();
        assert_eq!(
            written,
            vec![
                (0, 1, CgBondOrder::Single),
                (2, 3, CgBondOrder::Single),
                (4, 5, CgBondOrder::Single),
            ]
        );
    }

    /// Pairing descriptors across copies is 01d's job: every edge the
    /// notation *wrote* at this level stays inside one copy, and the ones
    /// that cross are the derived edges 01d appends.
    #[test]
    fn test_f8_level_one_has_no_written_edge_between_two_copies() {
        let ir = ir_of(F8);
        let level = &ir.levels[1];
        let crossing: Vec<(usize, usize)> = level
            .edges
            .iter()
            .filter(|e| e.origin == EdgeOrigin::Written)
            .filter(|e| level.nodes[e.i].parent != level.nodes[e.j].parent)
            .map(|e| (e.i, e.j))
            .collect();
        assert!(
            crossing.is_empty(),
            "inter-copy written edges were {crossing:?}"
        );
    }

    #[test]
    fn test_f8_level_one_copies_carry_the_descriptors_of_their_body() {
        let ir = ir_of(F8);
        assert_eq!(
            level_kinds(&ir.levels[1]),
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

    // -- alignment invariant, one test per shape ----------------------------

    #[test]
    fn test_base_only_string_has_no_fragment_table_and_one_level() {
        let ir = ir_of("{[#A][#B]}");
        assert!(ir.fragments.is_empty(), "tables were {:?}", ir.fragments);
        assert_eq!(ir.levels.len(), 1);
    }

    /// The last table is atomistic and builds no level, so F2's one table
    /// leaves one level.
    #[test]
    fn test_f2_has_one_fragment_table_and_one_level() {
        let ir = ir_of(F2);
        assert_eq!(ir.fragments.len(), 1);
        assert_eq!(ir.levels.len(), 1);
    }

    #[test]
    fn test_f8_has_two_fragment_tables_and_two_levels() {
        let ir = ir_of(F8);
        assert_eq!(ir.fragments.len(), 2);
        assert_eq!(ir.levels.len(), 2);
    }

    // -- edge cases ---------------------------------------------------------

    #[test]
    fn test_repeated_base_block_alone_has_no_fragment_table() {
        let ir = ir_of("{[#EO]|5}");
        assert!(ir.fragments.is_empty(), "tables were {:?}", ir.fragments);
    }

    /// R2.24: `|n` as the last token of a body is legal notation (the
    /// reference implementation crashes there).
    #[test]
    fn test_repeat_as_the_last_token_of_a_body_is_replayed() {
        let ir = ir_of("{[#A]}.{#A=[#B]|3}.{#B=[>]CC[<]}");
        assert_eq!(ir.levels[1].nodes.len(), 3);
    }

    /// The replay re-scans the full input, so a diagnostic raised inside a
    /// body points at the body's own bytes: the `|` of `[#B]1[#C]1|2` is byte
    /// 21 of `{[#A]}.{#A=[#B]1[#C]1|2}.{#B=CC,#C=CC}`, counted by hand —
    /// `{[#A]}.` is 7 bytes, `{#A=` four more, `[#B]1[#C]1` ten more.
    #[test]
    fn test_body_level_repeat_over_a_ring_marker_spans_the_full_input() {
        let text = "{[#A]}.{#A=[#B]1[#C]1|2}.{#B=CC,#C=CC}";
        let err = err_of(text);
        assert!(
            matches!(err.kind, SmilesErrorKind::CgRepeatOnRingMarker),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.span.start, 21, "span was {:?} in {text:?}", err.span);
    }

    /// The converse of coverage is not an error: a library block may define
    /// more than the level above it uses.
    #[test]
    fn test_definition_that_is_never_referenced_stays_in_the_table() {
        let ir = ir_of("{[#A]}.{#A=CC,#B=CCC}");
        assert_eq!(
            ir.fragments[0].keys().collect::<Vec<&String>>(),
            vec!["A", "B"]
        );
    }

    /// R1.4: inside an atomistic body `.` is the OpenSMILES disconnection —
    /// `{#OHter=[$][O-].[Na+]}` is a salt, not two blocks.
    #[test]
    fn test_disconnection_inside_an_atomistic_body_is_not_a_block_separator() {
        let ir = ir_of("{[#OHter]}.{#OHter=[$][O-].[Na+]}");
        assert_eq!(ir.fragments.len(), 1);
        assert_eq!(
            ir.fragments[0].keys().collect::<Vec<&String>>(),
            vec!["OHter"]
        );
    }

    /// R3.0: an entry splits on its **first** `=`; the rest belongs to the
    /// body, where `C=C` is a double bond.
    #[test]
    fn test_entry_splits_on_the_first_equals_sign_only() {
        let ir = ir_of("{[#A]}.{#A=C=C}");
        assert_eq!(atom_nodes(smiles_body(&ir, 0, "A")).len(), 2);
    }

    /// R3.3 / 01a's branch rule: a descriptor-only branch binds to its parent
    /// anchor — the `[#B]` node, not a node of its own.
    #[test]
    fn test_descriptor_only_branch_binds_to_the_anchor_of_a_graph_body() {
        let ir = ir_of("{[#A]}.{#A=[#B]([>])[#C]}.{#B=[>]CC[<],#C=[>]CC[<]}");
        let body = graph_body(&ir, 0, "A");
        assert_eq!(node_kinds(&body.nodes[0]), vec![DescriptorKind::Right]);
    }

    #[test]
    fn test_descriptor_only_branch_binds_to_the_anchor_atom_of_an_atomistic_body() {
        let ir = ir_of("{[#A]}.{#A=N([>])C}");
        let body = smiles_body(&ir, 0, "A");
        let nitrogen = atom_nodes(body)[0];
        assert_eq!(
            nitrogen
                .descriptors
                .iter()
                .map(|d| d.kind)
                .collect::<Vec<DescriptorKind>>(),
            vec![DescriptorKind::Right]
        );
    }

    /// `parent` is overwritten, never inherited: one body used by two nodes
    /// yields copies with different parents.
    #[test]
    fn test_body_used_twice_yields_copies_with_different_parents() {
        let ir = ir_of("{[#A][#A]}.{#A=[>][#B][#B][<]}.{#B=[>]CC[<]}");
        assert_eq!(
            parents(&ir.levels[1]),
            vec![Some(0), Some(0), Some(1), Some(1)]
        );
    }

    // -- 01b follow-ups (behaviour 01b owns, tests it owed) -----------------

    /// Marker number 0 is a marker like any other (OpenSMILES § 3.4 writes
    /// `C0CCCCC0`), in the bare spelling …
    #[test]
    fn test_bare_ring_marker_zero_closes_the_ring() {
        let graph = level0("{[#A]0[#B][#C]0}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    /// … and in the `%` spelling.
    #[test]
    fn test_percent_ring_marker_zero_closes_the_ring() {
        let graph = level0("{[#A]%00[#B][#C]%00}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    /// `%` takes the whole digit run that follows it, of any length (CGsmiles
    /// paper, JCIM 2025, § 2.1.4), so `%123` is one marker, not `%12` then
    /// `3`.
    #[test]
    fn test_percent_ring_marker_takes_the_whole_digit_run() {
        let graph = level0("{[#A]%123[#B][#C]%123}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    /// One digit after `%` is the same rule with a run of length one.
    #[test]
    fn test_one_digit_percent_ring_marker_closes_the_ring() {
        let graph = level0("{[#A]%1[#B][#C]%1}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    /// A `|n` consumes its unit; the node written after it starts a fresh one,
    /// which a second `|n` repeats.
    #[test]
    fn test_repeat_after_a_consumed_unit_repeats_the_fresh_unit() {
        let graph = level0("{[#A]|2[#B]|2}");
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 3);
    }

    // -- must-raise: squash (R5.2 / R5.5) -----------------------------------

    /// F9's squash fixture: `[!]` in an atomistic body.
    #[test]
    fn test_squash_descriptor_in_an_atomistic_body_is_refused() {
        assert!(matches!(
            kind_of("{[#SC4]1[#TC5][#TC5]1}.{#SC4=Cc(c[!])c[!],#TC5=[!]ccc[!]}"),
            SmilesErrorKind::CgSquashUnsupported
        ));
    }

    /// One structural rule covers every level, base graph included.
    #[test]
    fn test_squash_descriptor_in_the_base_graph_is_refused() {
        assert!(matches!(
            kind_of("{[#A][!]}"),
            SmilesErrorKind::CgSquashUnsupported
        ));
    }

    /// The third position the one structural rule covers: a `[!]` written in
    /// an **intermediate** (coarse-graph) body, which is neither the base
    /// graph nor the atomistic last block. The refusal carries the whole
    /// string and the `CGsmiles` notation, and its caret lands inside it.
    #[test]
    fn test_squash_descriptor_in_a_graph_body_is_refused() {
        let text = "{[#A]}.{#A=[!][#X]}.{#X=[$]C[$]}";
        let err = err_of(text);
        assert!(
            matches!(err.kind, SmilesErrorKind::CgSquashUnsupported),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert!(
            err.span.start < text.len(),
            "span was {:?} in {text:?}",
            err.span
        );
    }

    // -- must-raise: atom annotations propagate un-wrapped (R5.5) -----------

    /// A positional atom weight. The diagnostic is 01a's, un-wrapped, re-based
    /// into the full string: the bracket atom `[C;0.5]` spans bytes 15..22.
    #[test]
    fn test_atom_weight_annotation_is_unsupported() {
        let text = "{[#A][#B]}.{#A=[C;0.5]C,#B=CC}";
        let err = err_of(text);
        assert!(
            matches!(err.kind, SmilesErrorKind::AtomAnnotationUnsupported(_)),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert!(
            (15..22).contains(&err.span.start),
            "span was {:?} in {text:?}",
            err.span
        );
    }

    /// Wildcard overloading. That this reports the annotation rather than a
    /// malformed entry also proves the entry splitter respects `[]` depth:
    /// the `,` inside `[*;s=C,0]` did not split the entry.
    #[test]
    fn test_wildcard_overloading_annotation_is_unsupported() {
        let text = "{[#T]}.{#T=C1=CCCC[*;s=C,0][*;s=C,0]}";
        let err = err_of(text);
        assert!(
            matches!(err.kind, SmilesErrorKind::AtomAnnotationUnsupported(_)),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert!(
            (18..27).contains(&err.span.start),
            "span was {:?} in {text:?}",
            err.span
        );
    }

    // -- must-raise: the fragment-table grammar -----------------------------

    /// R3.1: the reference keeps the first definition silently; molrs names it.
    #[test]
    fn test_duplicate_fragment_name_is_refused() {
        let kind = kind_of("{[#A]}.{#A=CC,#A=CCC}");
        assert!(
            matches!(&kind, SmilesErrorKind::CgDuplicateFragment(name) if name == "A"),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_name_in_the_base_graph_with_no_definition_is_refused() {
        let kind = kind_of("{[#A][#B]}.{#A=CC}");
        assert!(
            matches!(&kind, SmilesErrorKind::CgUndefinedFragment(name) if name == "B"),
            "kind was {kind:?}"
        );
    }

    /// A wildcard bead is a name like any other once a fragment table is
    /// written: `*` is looked up in the table and reported undefined when it
    /// is not there. Only a base-only string may leave a bead undefined.
    #[test]
    fn test_wildcard_bead_with_a_fragment_table_is_undefined() {
        let kind = kind_of("{[#*]}.{#A=CC}");
        assert!(
            matches!(&kind, SmilesErrorKind::CgUndefinedFragment(name) if name == "*"),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_name_in_an_intermediate_body_with_no_definition_is_refused() {
        let kind = kind_of("{[#A]}.{#A=[#X]}.{#Y=CC}");
        assert!(
            matches!(&kind, SmilesErrorKind::CgUndefinedFragment(name) if name == "X"),
            "kind was {kind:?}"
        );
    }

    /// Coverage of level *k* runs before level *k+1* is built, so the missing
    /// base-level name is reported rather than an internal build failure.
    #[test]
    fn test_coverage_of_a_level_runs_before_the_next_level_is_built() {
        let kind = kind_of("{[#A][#Q]}.{#A=[#X]}.{#X=CC}");
        assert!(
            matches!(&kind, SmilesErrorKind::CgUndefinedFragment(name) if name == "Q"),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_text_after_the_last_block_is_refused() {
        assert!(matches!(
            kind_of("{[#A]}.{#A=CC}}"),
            SmilesErrorKind::CgExpectedBlock
        ));
    }

    #[test]
    fn test_separator_not_followed_by_a_block_is_refused() {
        assert!(matches!(
            kind_of("{[#A]}.#A=[$]C[$]"),
            SmilesErrorKind::CgExpectedBlock
        ));
    }

    /// F1.10, whose kind 01c changes: a second block needs a `.` before it.
    #[test]
    fn test_second_block_without_a_separator_is_refused() {
        assert!(matches!(
            kind_of("{[#PEO][#PEO]}[#X]"),
            SmilesErrorKind::CgExpectedBlock
        ));
    }

    /// 01b's `CgEmptyBlock` is reused rather than doubled: `{}` is refused in
    /// every position.
    #[test]
    fn test_empty_fragment_block_is_refused() {
        assert!(matches!(
            kind_of("{[#A]}.{}"),
            SmilesErrorKind::CgEmptyBlock
        ));
    }

    #[test]
    fn test_empty_fragment_body_is_refused() {
        assert!(matches!(
            kind_of("{[#A]}.{#A=}"),
            SmilesErrorKind::CgEmptyFragmentBody
        ));
    }

    #[test]
    fn test_fragment_entry_without_an_equals_sign_is_refused() {
        assert!(matches!(
            kind_of("{[#A]}.{#A}"),
            SmilesErrorKind::CgMalformedFragmentDef
        ));
    }

    // -- must-raise: the last block must be atomistic -----------------------

    /// A coarse node in the last block: the inner `UnexpectedChar('#')` is
    /// kept, boxed, and its span re-based — byte 12 is the `#` of the body's
    /// first `[#X]`.
    #[test]
    fn test_coarse_node_in_the_last_block_is_not_atomistic() {
        let text = "{[#A]}.{#A=[#X][#X]}";
        let err = err_of(text);
        assert!(
            matches!(
                &err.kind,
                SmilesErrorKind::CgLastBlockNotAtomistic(inner)
                    if **inner == SmilesErrorKind::UnexpectedChar('#')
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span.start, 12);
        assert_eq!(text.as_bytes()[12], b'#');
    }

    /// The rendered message names the notation and the rule, instead of
    /// leaking a bare `unexpected character '#'`.
    #[test]
    fn test_last_block_message_names_the_notation_and_the_rule() {
        let rendered = err_of("{[#A]}.{#A=[#X][#X]}").to_string();
        assert!(
            rendered.starts_with("CGsmiles parse error"),
            "rendered error was {rendered:?}"
        );
        let lower = rendered.to_lowercase();
        assert!(
            lower.contains("last block"),
            "rendered error was {rendered:?}"
        );
        assert!(
            lower.contains("atomistic"),
            "rendered error was {rendered:?}"
        );
    }

    /// A body that is atomistic but malformed: byte 13 is its `(`.
    #[test]
    fn test_unclosed_branch_in_a_last_block_body_is_boxed_and_rebased() {
        let text = "{[#A]}.{#A=CC(}";
        let err = err_of(text);
        assert!(
            matches!(
                &err.kind,
                SmilesErrorKind::CgLastBlockNotAtomistic(inner)
                    if **inner == SmilesErrorKind::UnclosedBranch
            ),
            "kind was {:?}",
            err.kind
        );
        assert_eq!(err.input, text);
        assert_eq!(err.notation, Notation::CgSmiles);
        assert_eq!(err.span.start, 13);
        assert_eq!(text.as_bytes()[13], b'(');
    }
}
