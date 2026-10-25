//! Error types shared by the three line notations this module family reads
//! and writes: SMILES, SMARTS and `CGsmiles`.
//!
//! There is one error struct, [`SmilesError`], one list of reasons,
//! [`SmilesErrorKind`], and one field, [`Notation`], saying which language the
//! offending text was meant to be — because many reasons (`UnclosedBracket`
//! and friends) are raised by all three, and the entry point is the only place
//! that knows which was being parsed.

use std::fmt;

use crate::line_notation::ast::{BondKind, Span};
use molrs::core::MolRsError;

/// Which line notation was being read or written when an error was raised.
///
/// One error type serves all three notations, so the notation is a *fact owned
/// by the entry point* rather than something a reader could infer from the
/// variant name: `UnclosedBracket` is raised by the SMILES, SMARTS and
/// CGsmiles parsers alike, and only the entry point knows which language
/// the offending text was meant to be. Every construction site therefore
/// states it — see [`SmilesError::new`].
///
/// [`Display`](fmt::Display) renders the conventional spelling of each
/// notation (`SMILES`, `SMARTS`, `CGsmiles`), which is the prefix of every
/// rendered [`SmilesError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notation {
    /// SMILES, including the SMILES fragment body (the fragment dialect is
    /// SMILES widened with bonding descriptors, not a notation of its own).
    Smiles,
    /// SMARTS, the SMILES query language.
    Smarts,
    /// `CGsmiles`, the coarse-grained resolution notation.
    CgSmiles,
}

impl fmt::Display for Notation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Notation::Smiles => "SMILES",
            Notation::Smarts => "SMARTS",
            Notation::CgSmiles => "CGsmiles",
        };
        f.write_str(name)
    }
}

/// Error produced by any stage of the line-notation pipeline: the parser, the
/// validator, the IR → graph walker, or the writer.
#[derive(Debug, Clone, PartialEq)]
pub struct SmilesError {
    /// Which rule was broken; the payload, if any, is the offending text.
    pub kind: SmilesErrorKind,
    /// Byte range of the offending text within `input`. Its start is the
    /// position reported by [`Display`](fmt::Display) and the column the caret
    /// points at.
    pub span: Span,
    /// The original input string (kept for diagnostic display). Empty when the
    /// error was raised away from the parser — by the IR → graph walker or the
    /// writer, which are handed an IR and never see the text it came from.
    pub input: String,
    /// Which notation was being parsed or written, as the rendered message
    /// names it. Stamped by the entry point, never derived from [`kind`]:
    /// several kinds are shared by all three notations.
    ///
    /// [`kind`]: SmilesError::kind
    pub notation: Notation,
}

/// Why a string was refused: the rule that was broken, with the offending
/// text as payload where naming it helps.
///
/// A variant says *what* went wrong, never *which notation* was being read —
/// most of the first group below is raised by all three — so a caller that
/// wants the language reads [`SmilesError::notation`] instead. The variants
/// fall into three groups: the shared grammar rules, the bonding-descriptor
/// rules of the SMILES fragment dialect (`DescriptorInPlainSmiles` onwards),
/// and the rules of `CGsmiles` (`Cg*`) — first those of one coarse graph,
/// then those of the fragment blocks that resolve it into a finer one.
#[derive(Debug, Clone, PartialEq)]
pub enum SmilesErrorKind {
    /// A character was encountered that is not valid in the current context.
    UnexpectedChar(char),
    /// Reached end-of-input while more tokens were expected.
    UnexpectedEnd,
    /// `[` was opened but never closed.
    UnclosedBracket,
    /// `(` was opened but never closed.
    UnclosedBranch,
    /// A ring-closure digit was opened but never paired.
    UnmatchedRingClosure(u16),
    /// A `%` in a plain SMILES or SMARTS string is not followed by exactly
    /// two digits: `C%1C` stops after one, and a trailing `C%` has none.
    ///
    /// OpenSMILES §3.4 spells a two-digit ring marker `%nn` and fixes its
    /// width at two, so `%1` is not marker 1 written short — it is a marker
    /// whose second digit is missing. `CGsmiles` reads `%` differently (there
    /// it takes the whole digit run that follows, `%123` being marker 123) and
    /// so has its own [`SmilesErrorKind::CgInvalidRingMarker`]; the two rules
    /// are different rules, and each notation reports its own.
    InvalidRingMarker,
    /// The element symbol is not recognised.
    InvalidElement(String),
    /// A charge specification could not be parsed.
    InvalidCharge,
    /// An empty string was passed to the parser.
    EmptyInput,
    /// Characters remain after the molecule was fully parsed. Raised by the
    /// atomistic parsers only: a `CGsmiles` string with text after its last
    /// block is [`SmilesErrorKind::CgExpectedBlock`], which says what was
    /// expected there instead of only that something was left over.
    TrailingCharacters,
    /// A SMARTS query primitive is not recognised.
    InvalidQueryPrimitive(String),
    /// `$(` was opened but the matching `)` was not found.
    UnclosedRecursive,
    /// Recursive SMARTS nesting exceeded the depth limit.
    RecursionLimit,
    /// Ring closure bond types are inconsistent between open and close.
    RingBondConflict { rnum: u16 },
    /// Two directional substituents of one double-bond atom are marked on the
    /// same side (`C(/F)(/Cl)=C/F`). Opposite marks agree: they are one
    /// configuration. A ring digit that disagrees with itself is
    /// [`Self::RingBondConflict`], not this.
    BondStereoConflict,
    /// Graph → IR / IR → string emit failure (message is the reason).
    Emit(String),
    /// An IR → graph construction step failed — an internal invariant of the
    /// SMILES builder was violated, not a rule the input broke. The payload is
    /// the reason, usually the message of the underlying
    /// [`MolRsError`].
    ///
    /// This is the plain-SMILES twin of [`SmilesErrorKind::CgBuild`]: the same
    /// "no input can reach this state, so report it by value rather than
    /// panicking" contract, raised by the atomistic builder instead of by the
    /// `CGsmiles` reader.
    Build(String),
    /// A bonding descriptor (`[$]`, `[<]`, `[>]`, `[!]`) met a stage that
    /// speaks plain SMILES, which has no such notation: `SmilesIr::parse` on a
    /// descriptor bracket, `SmilesIr::validate` on a node carrying one, or
    /// `smiles_text` / `smarts_text` asked to emit one. This is a routing
    /// error, and the rendered message says where to go instead — it names
    /// `SmilesIr::from_fragment`, the entry point of the dialect that does
    /// accept descriptors.
    ///
    /// SMARTS *parsing* never raises it: there `[$(C)]` is recursive SMARTS
    /// and `[!C]` is a negated atom, so the bracket belongs to the query
    /// parser and is left to it.
    DescriptorInPlainSmiles,
    /// An IR carrying bonding descriptors was handed to
    /// `SmilesIr::to_atomistic`, the plain IR → graph conversion, which has
    /// nowhere to put them and would silently drop them. The rendered message
    /// names the fragment route: `SmilesIr::from_fragment`, then
    /// `FragmentIr::to_atomistic` (which returns them alongside the graph) or
    /// `FragmentIr::to_template`.
    DescriptorsUnconvertible,
    /// A bond symbol was written *inside* a descriptor bracket — the
    /// `BigSMILES` spelling `[<=1]`, or `[$-]`. `CGsmiles` writes the order
    /// outside the bracket (`CC=[$]`), and accepting both spellings would give
    /// one meaning two notations, so the in-bracket form is refused rather
    /// than translated.
    BondInsideDescriptor,
    /// A descriptor label is not ASCII alphanumeric, as in `[$a+]`; the
    /// payload is the rejected label text (here `a+`).
    InvalidDescriptorLabel(String),
    /// A bond order was written next to a descriptor that no formed bond could
    /// take — aromatic (`c:[$]`), directional (`/`, `\`), wildcard (`~`) or
    /// ring (`@`); the payload is the offending kind. Raised by the parser at
    /// the descriptor's construction site, and by `fragment_smiles_text` for
    /// a hand-built IR that no parser could have produced.
    InvalidDescriptorOrder(BondKind),
    /// A descriptor was written where no atom can anchor it: `[$]` alone, or
    /// `C.[$]`, whose second `.`-separated component contains no atom for the
    /// descriptor to bind to.
    DanglingDescriptor,
    /// A `CGsmiles` atom-level annotation — a weight `[C;0.5]`, a chirality
    /// `[C;1;S]`, or wildcard overloading `[*;s=C,0]` — was written. The
    /// fragment dialect does not support these, and this error names the
    /// feature rather than claiming a missing `]`. The payload is the
    /// annotation text between `;` and the closing `]`, or the end of input if
    /// the bracket is never closed. Raised in the fragment dialect
    /// only; `SmilesIr::parse("[C;0.5]")` still reports
    /// [`SmilesErrorKind::UnclosedBracket`].
    AtomAnnotationUnsupported(String),

    // -- CGsmiles coarse-graph kinds ----------------------------------------
    /// A `CGsmiles` block contains nothing: `{}`. The reference implementation
    /// skips such a block silently; molrs names it, because a resolution level
    /// with no node is never what the writer meant.
    CgEmptyBlock,
    /// A node annotation could not be read as a `key=value` pair or as a
    /// positional field: `[#A;]`, `[#A;=1]`, `[#A;q=x]`, or a fourth
    /// positional slot (the table has three: the name, `q` and `w`). The
    /// payload is the offending annotation text.
    CgMalformedAnnotation(String),
    /// A node annotation was read but molrs does not model it yet: a
    /// non-default weight (`[#A;w=2]`, `[#A;0;0.5]`) or a chirality
    /// (`[#A;x=S]`). The payload names the key and the value that was written.
    CgUnsupportedAnnotation {
        /// The annotation key, after positional binding (`w`, `x`, …).
        key: String,
        /// The value as written.
        value: String,
    },
    /// An annotation was written on the wildcard node `[#*;…]`, which stands
    /// for "any fragment" and therefore has no properties of its own. The
    /// payload is the annotation text.
    CgAnnotationOnWildcard(String),
    /// A symbol in bond position is not one of `-`, `=`, `#`, `$`, or a second
    /// bond symbol was written before the first had anything to bond
    /// (`{[#A]=-[#B]}`). The refused symbols include `.`, the zero-order
    /// (virtual) edge, which the notation permits and molrs declines to model.
    CgInvalidBondOrder,
    /// The count after `|` is not a positive decimal integer: `{[#A]|0}`,
    /// `{[#A]|x}`. The payload is the text that was read as the count.
    CgInvalidRepeatCount(String),
    /// A ring closure would repeat an edge the graph already has, or would
    /// bond a node to itself. The coarse graph is a *simple* graph — at most
    /// one edge per pair of nodes, and no self-loops — so a second bond
    /// between the same two nodes is written with a bond-order symbol, not
    /// with a second ring marker. Checked once the block has been read, so
    /// `i` and `j` are final node indices.
    CgDuplicateEdge {
        /// Index of the node that opened the marker.
        i: usize,
        /// Index of the node that closed it.
        j: usize,
    },
    /// `|` was written where the unit to repeat is not a single well-formed
    /// subgraph: inside an open branch, or after a node carrying more than one
    /// branch.
    CgRepeatOnBranchedNode,
    /// A ring marker was opened or closed inside a repeated unit. A marker is
    /// a one-shot identity — [`SmilesErrorKind::CgDuplicateEdge`] forbids two
    /// closures of one marker — so replaying it is either a silent overwrite
    /// or a self-bond.
    CgRepeatOnRingMarker,
    /// A bond symbol has nothing on one side to bond to: written immediately
    /// before the closing `}` of a block (`{[#A]=}`), before the `)` that ends
    /// a branch (`{[#A]([#B]=)}`), or before the first node of the block
    /// (`{=[#A]}`). The input is not exhausted in any of these, so
    /// `UnexpectedEnd` would be a false report.
    CgDanglingBond,
    /// A ring marker is malformed: `%` with no digit run after it, as in
    /// `{[#A]%}`, or a marker number past `u16::MAX`. `%` takes the *whole*
    /// digit run that follows it — `%1` is marker 1 and `%123` is marker 123 —
    /// and zero is a marker like any other in both spellings (`0`, `%00`).
    CgInvalidRingMarker,

    // -- CGsmiles fragment-block kinds --------------------------------------
    /// A `{…}` block was expected and not found: a `.` separator not followed
    /// by `{` (`{[#A]}.#A=[$]C[$]`), a second block written with no separator
    /// before it (`{[#PEO][#PEO]}[#X]`), or text left over after the last `}`
    /// (`{[#A]}.{#A=CC}}`). A `CGsmiles` string is a `.`-separated sequence of
    /// blocks and nothing else.
    CgExpectedBlock,
    /// An entry of a fragment block does not read as `#NAME=body`: it does not
    /// open with `#` (`{[#A]}.{A=CC}`), carries no `=` (`{[#A]}.{#A}`), names
    /// nothing between the two (`{[#A]}.{#=CC}`), or spells the name with a
    /// character a node bracket could not spell (`{[#A]}.{#A-1=CC}`; a
    /// fragment name is ASCII alphanumeric, or the single wildcard `*`, the
    /// same alphabet `[#NAME]` accepts).
    CgMalformedFragmentDef,
    /// A fragment entry defines an empty body: `{[#A]}.{#A=}`. The name is
    /// written, the `=` is written, and nothing follows it before the `,` or
    /// the `}`.
    CgEmptyFragmentBody,
    /// One fragment block defines the same name twice:
    /// `{[#A]}.{#A=CC,#A=CCC}`. The payload is the repeated name. The
    /// reference implementation keeps the first definition and drops the rest
    /// silently; molrs names the collision, because which of two bodies was
    /// meant is not something a reader may guess.
    CgDuplicateFragment(String),
    /// A node name used at one resolution has no definition in the fragment
    /// block that resolves it: `{[#A][#B]}.{#A=CC}` never defines `B`. The
    /// payload is the missing name, and the span is the node that referenced
    /// it.
    CgUndefinedFragment(String),
    /// The squash operator `[!]` was written. It is the only `CGsmiles`
    /// syntax that places one atom in two beads, and molrs does not model that
    /// yet — so it is refused wherever it appears (base graph, coarse fragment
    /// body or atomistic fragment body) by one structural rule, rather than
    /// parsed into a graph that quietly loses the sharing.
    CgSquashUnsupported,
    /// A body of the **last** block failed to parse as an atomistic
    /// (OpenSMILES) fragment body. The payload is the inner reason, kept
    /// rather than discarded, so `{[#A]}.{#A=[#X][#X]}` names the molrs rule
    /// and then the `UnexpectedChar('#')` that proved it broken.
    ///
    /// The last block is atomistic **by position**: the notation carries no
    /// flag for it (the reference implementation passes one to its reader), so
    /// a string whose deepest resolution is meant to stay coarse-grained
    /// cannot be written.
    CgLastBlockNotAtomistic(Box<SmilesErrorKind>),
    /// An internal invariant of the reader was violated — a state no input
    /// can reach, reported by value rather than by panicking. The payload
    /// says which invariant.
    CgBuild(String),
    /// A written coarse edge could not be matched to a free pair of compatible
    /// bonding descriptors: `{[#A][#B]}.{#A=[$a]C,#B=[$b]C}` writes a bond
    /// between two fragments whose only ports carry different labels. `level`
    /// is the resolution level being resolved and `edge` indexes
    /// `levels[level].edges`.
    ///
    /// The reference implementation drops such an edge silently; molrs refuses
    /// it, because a bond the user wrote and the reader cannot form is not a
    /// fact either of them may discard.
    ///
    /// The span is the edge's own. For a **derived** edge — one resolution
    /// created from a pair of the level above
    /// ([`EdgeOrigin::Derived`](crate::io::cgsmiles::EdgeOrigin::Derived)) —
    /// that span is a copy of the coarse edge's, so the caret lands on the
    /// bond that created it rather than on text that does not exist.
    CgUnmatchableEdge {
        /// The resolution level being resolved.
        level: usize,
        /// Index into that level's edge list of the edge that cannot be
        /// formed.
        edge: usize,
    },
    /// A `CGsmiles` string has no atomistic body to expand into atoms: either
    /// the lowest level's bodies are coarse graphs, or the string wrote no
    /// fragment table at all. The payload is what could not be expanded — a
    /// fragment name, or the phrase `base-only string (no fragment table)`
    /// when there is no fragment to name — and the message reads
    /// `<payload>: no atomistic body to expand` for both.
    CgNotExpandable(String),
}

impl SmilesError {
    /// Build an error at `span` within `input`, reported as an error of
    /// `notation`.
    ///
    /// `notation` has no default on purpose: several kinds are shared by all
    /// three notations, so a defaulted [`Notation::Smiles`] would silently
    /// mislabel every SMARTS and `CGsmiles` diagnostic that reuses one.
    pub fn new(kind: SmilesErrorKind, span: Span, input: &str, notation: Notation) -> Self {
        Self {
            kind,
            span,
            input: input.to_owned(),
            notation,
        }
    }
}

impl SmilesErrorKind {
    /// The one-line reason this kind reports, without the notation prefix or
    /// the caret context that `Display for SmilesError` wraps it in.
    fn message(&self) -> String {
        match self {
            SmilesErrorKind::UnexpectedChar(c) => format!("unexpected character '{c}'"),
            SmilesErrorKind::UnexpectedEnd => "unexpected end of input".to_owned(),
            SmilesErrorKind::UnclosedBracket => "unclosed bracket '['".to_owned(),
            SmilesErrorKind::UnclosedBranch => "unclosed branch '('".to_owned(),
            SmilesErrorKind::UnmatchedRingClosure(n) => {
                format!("unmatched ring closure {n}")
            }
            SmilesErrorKind::InvalidRingMarker => {
                "invalid ring marker — '%' takes exactly two digits, as in '%12'".to_owned()
            }
            SmilesErrorKind::InvalidElement(s) => format!("invalid element '{s}'"),
            SmilesErrorKind::InvalidCharge => "invalid charge specification".to_owned(),
            SmilesErrorKind::EmptyInput => "empty input".to_owned(),
            SmilesErrorKind::TrailingCharacters => "trailing characters after molecule".to_owned(),
            SmilesErrorKind::InvalidQueryPrimitive(s) => {
                format!("invalid SMARTS query primitive '{s}'")
            }
            SmilesErrorKind::UnclosedRecursive => "unclosed recursive SMARTS '$('".to_owned(),
            SmilesErrorKind::RecursionLimit => "recursive SMARTS nesting limit exceeded".to_owned(),
            SmilesErrorKind::RingBondConflict { rnum } => {
                format!("conflicting bond types on ring closure {rnum}")
            }
            SmilesErrorKind::BondStereoConflict => {
                "two substituents of one double-bond atom are marked on the same side".to_owned()
            }
            SmilesErrorKind::Emit(s) => format!("emit error: {s}"),
            SmilesErrorKind::Build(s) => format!("graph construction failed: {s}"),
            SmilesErrorKind::DescriptorInPlainSmiles => {
                "bonding descriptor is not plain SMILES notation \
                 — parse a fragment body with SmilesIr::from_fragment"
                    .to_owned()
            }
            SmilesErrorKind::DescriptorsUnconvertible => "this atom carries bonding descriptors \
                 — parse a fragment body with SmilesIr::from_fragment and convert it with \
                 FragmentIr::to_atomistic or FragmentIr::to_template"
                .to_owned(),
            SmilesErrorKind::BondInsideDescriptor => {
                "bond symbol inside a bonding descriptor bracket \
                 — the bond order is written outside it, as in CC=[$]"
                    .to_owned()
            }
            SmilesErrorKind::InvalidDescriptorLabel(s) => {
                format!("bonding descriptor label '{s}' is not alphanumeric")
            }
            SmilesErrorKind::InvalidDescriptorOrder(k) => {
                format!("bond order {k:?} cannot annotate a bonding descriptor")
            }
            SmilesErrorKind::DanglingDescriptor => {
                "bonding descriptor has no atom to bind to".to_owned()
            }
            SmilesErrorKind::AtomAnnotationUnsupported(s) => {
                format!("atom annotation ';{s}' is unsupported")
            }
            SmilesErrorKind::CgEmptyBlock
            | SmilesErrorKind::CgMalformedAnnotation(_)
            | SmilesErrorKind::CgUnsupportedAnnotation { .. }
            | SmilesErrorKind::CgAnnotationOnWildcard(_)
            | SmilesErrorKind::CgInvalidBondOrder
            | SmilesErrorKind::CgInvalidRepeatCount(_)
            | SmilesErrorKind::CgDuplicateEdge { .. }
            | SmilesErrorKind::CgRepeatOnBranchedNode
            | SmilesErrorKind::CgRepeatOnRingMarker
            | SmilesErrorKind::CgDanglingBond
            | SmilesErrorKind::CgInvalidRingMarker
            | SmilesErrorKind::CgExpectedBlock
            | SmilesErrorKind::CgMalformedFragmentDef
            | SmilesErrorKind::CgEmptyFragmentBody
            | SmilesErrorKind::CgDuplicateFragment(_)
            | SmilesErrorKind::CgUndefinedFragment(_)
            | SmilesErrorKind::CgSquashUnsupported
            | SmilesErrorKind::CgLastBlockNotAtomistic(_)
            | SmilesErrorKind::CgBuild(_)
            | SmilesErrorKind::CgUnmatchableEdge { .. }
            | SmilesErrorKind::CgNotExpandable(_) => self.cg_message(),
        }
    }

    /// The reason a **structural** `Cg*` kind reports: the `CGsmiles` half of
    /// [`SmilesErrorKind::message`], split off so neither function has to be
    /// read past its own notation.
    ///
    /// Reached only through `message`, whose match routes exactly the `Cg*`
    /// kinds here and answers every other kind itself. The kinds the
    /// resolution stage raises are answered by `cg_resolution_message`, which
    /// this function delegates to.
    fn cg_message(&self) -> String {
        match self {
            SmilesErrorKind::CgEmptyBlock => {
                "empty coarse-graph block '{}' — a resolution level needs at least one node"
                    .to_owned()
            }
            SmilesErrorKind::CgMalformedAnnotation(s) => {
                format!("malformed node annotation '{s}'")
            }
            SmilesErrorKind::CgUnsupportedAnnotation { key, value } => {
                format!("node annotation '{key}={value}' is not supported")
            }
            SmilesErrorKind::CgAnnotationOnWildcard(s) => {
                format!("annotation '{s}' written on the wildcard node '[#*]'")
            }
            SmilesErrorKind::CgInvalidBondOrder => {
                "invalid coarse bond order — write one of '-', '=', '#', '$'".to_owned()
            }
            SmilesErrorKind::CgInvalidRepeatCount(s) => {
                format!(
                    "invalid repeat count '{s}' — '|' takes a positive integer up to \
                     65535, a molrs bound the notation does not state"
                )
            }
            SmilesErrorKind::CgDuplicateEdge { i, j } if i == j => {
                format!("node {i} is bonded to itself — a coarse graph has no self-loops")
            }
            SmilesErrorKind::CgDuplicateEdge { i, j } => format!(
                "nodes {i} and {j} are already joined — use bond order symbols instead of a \
                 second edge"
            ),
            SmilesErrorKind::CgRepeatOnBranchedNode => {
                "'|' cannot repeat a unit with an open branch or with more than one branch"
                    .to_owned()
            }
            SmilesErrorKind::CgRepeatOnRingMarker => {
                "'|' cannot repeat a unit that opens or closes a ring marker".to_owned()
            }
            SmilesErrorKind::CgDanglingBond => {
                "bond symbol with nothing to bond to before the end of the block".to_owned()
            }
            SmilesErrorKind::CgInvalidRingMarker => {
                "invalid ring marker — '%' must be followed by digits; marker numbers up to 65535"
                    .to_owned()
            }
            SmilesErrorKind::CgExpectedBlock => {
                "expected a '{…}' resolution block — blocks are separated by '.', and nothing \
                 may follow the last one"
                    .to_owned()
            }
            SmilesErrorKind::CgMalformedFragmentDef => {
                "malformed fragment entry — write '#NAME=body', entries separated by ','".to_owned()
            }
            SmilesErrorKind::CgEmptyFragmentBody => {
                "empty fragment body — '#NAME=' defines nothing".to_owned()
            }
            SmilesErrorKind::CgDuplicateFragment(name) => {
                format!("fragment '{name}' is defined twice — each name must be unique")
            }
            SmilesErrorKind::CgUndefinedFragment(name) => {
                format!("fragment '{name}' is used but never defined")
            }
            SmilesErrorKind::CgSquashUnsupported => {
                "the squash operator '[!]' is not supported — molrs does not model an atom \
                 shared by two beads"
                    .to_owned()
            }
            SmilesErrorKind::CgLastBlockNotAtomistic(inner) => format!(
                "the last block must be an atomistic (OpenSMILES) body: {}",
                inner.message()
            ),
            _ => self.cg_resolution_message(),
        }
    }

    /// The reason a `Cg*` kind of the **resolution** stage reports: descriptor
    /// pairing, atomistic expansion, and the internal-invariant kind they
    /// share with the reader's earlier stages.
    ///
    /// Split from `cg_message` along the line the reader itself draws — the
    /// kinds above name notation a string got wrong, the kinds here name
    /// something the reader could not do with a string it accepted — so
    /// neither function has to be read past its own stage.
    ///
    /// Reached only through `cg_message`, which answers every structural kind
    /// itself; the remaining kinds are unreachable rather than handled.
    fn cg_resolution_message(&self) -> String {
        match self {
            SmilesErrorKind::CgUnmatchableEdge { level, edge } => format!(
                "edge {edge} of resolution level {level} has no free pair of compatible \
                 bonding descriptors to form it"
            ),
            SmilesErrorKind::CgNotExpandable(what) => {
                format!("{what}: no atomistic body to expand")
            }
            SmilesErrorKind::CgBuild(reason) => {
                format!("an internal invariant of the reader was violated: {reason}")
            }
            kind => unreachable!("{kind:?} is not a CGsmiles resolution error kind"),
        }
    }
}

/// Renders as one message line, `<notation> parse error at position <n>:
/// <reason>`, optionally followed by two context lines: the input, and a caret
/// under the byte the span starts at. The prefix is [`SmilesError::notation`]
/// as the entry point stamped it — `SMILES`, `SMARTS` or `CGsmiles`.
///
/// The context lines are skipped when there is no input text to point into —
/// errors raised by the IR → graph walker and by the writer carry none — and
/// when the input is longer than 120 bytes, where a caret under a wrapped line
/// helps nobody.
impl fmt::Display for SmilesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = self.span.start;
        let msg = self.kind.message();
        write!(f, "{} parse error at position {pos}: {msg}", self.notation)?;

        // Caret-style context line, when there is an input to point into and
        // it is short enough to be useful. Errors raised away from the scanner
        // (the IR → graph walker, the writer) carry no input text, and a caret
        // under an empty line points at nothing.
        if !self.input.is_empty() && self.input.len() <= 120 {
            write!(f, "\n  {}\n  ", self.input)?;
            for _ in 0..pos.min(self.input.len()) {
                write!(f, " ")?;
            }
            write!(f, "^")?;
        }
        Ok(())
    }
}

impl std::error::Error for SmilesError {}

impl From<SmilesError> for MolRsError {
    fn from(e: SmilesError) -> Self {
        MolRsError::Parse {
            line: None,
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_with_caret() {
        let err = SmilesError::new(
            SmilesErrorKind::UnexpectedChar('X'),
            Span::new(3, 4),
            "CC(X)O",
            Notation::Smiles,
        );
        let s = err.to_string();
        assert!(s.contains("position 3"));
        assert!(s.contains("unexpected character 'X'"));
        assert!(s.contains("CC(X)O"));
        assert!(s.contains("   ^"));
    }

    #[test]
    fn test_into_molrs_error() {
        let err = SmilesError::new(
            SmilesErrorKind::EmptyInput,
            Span::new(0, 0),
            "",
            Notation::Smiles,
        );
        let molrs: MolRsError = err.into();
        let msg = format!("{molrs}");
        assert!(msg.contains("empty input"));
    }

    // -- bonding-descriptor variants ----------------------------------------

    use crate::line_notation::ast::BondKind;

    /// The message body of a rendered error: the first line after the
    /// `"… position N: "` prefix that every variant shares.
    fn message(kind: SmilesErrorKind) -> String {
        let rendered = SmilesError::new(kind, Span::new(0, 3), "[$]", Notation::Smiles).to_string();
        let first = rendered.lines().next().expect("rendered error is empty");
        match first.split_once(": ") {
            Some((_, body)) => body.to_owned(),
            None => first.to_owned(),
        }
    }

    #[test]
    fn test_display_descriptor_in_plain_smiles_names_fragment_parser() {
        let msg = message(SmilesErrorKind::DescriptorInPlainSmiles);
        assert!(
            msg.contains("SmilesIr::from_fragment"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_descriptors_unconvertible_names_fragment_converter() {
        let msg = message(SmilesErrorKind::DescriptorsUnconvertible);
        assert!(
            msg.contains("FragmentIr::to_atomistic"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_bond_inside_descriptor_is_non_empty() {
        let msg = message(SmilesErrorKind::BondInsideDescriptor);
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_invalid_descriptor_label_shows_the_label() {
        let msg = message(SmilesErrorKind::InvalidDescriptorLabel("a+".to_owned()));
        assert!(msg.contains("a+"), "message was {msg:?}");
    }

    #[test]
    fn test_display_invalid_descriptor_order_is_non_empty() {
        let msg = message(SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic));
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_dangling_descriptor_is_non_empty() {
        let msg = message(SmilesErrorKind::DanglingDescriptor);
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_atom_annotation_unsupported_shows_the_annotation() {
        let msg = message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned()));
        assert!(msg.contains("0.5"), "message was {msg:?}");
        assert!(
            msg.to_lowercase().contains("unsupported"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_atom_annotation_unsupported_does_not_claim_unclosed_bracket() {
        let msg = message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned()));
        assert!(
            !msg.to_lowercase().contains("unclosed"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_descriptor_messages_are_pairwise_distinct() {
        let messages = [
            message(SmilesErrorKind::DescriptorInPlainSmiles),
            message(SmilesErrorKind::DescriptorsUnconvertible),
            message(SmilesErrorKind::BondInsideDescriptor),
            message(SmilesErrorKind::InvalidDescriptorLabel("a+".to_owned())),
            message(SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)),
            message(SmilesErrorKind::DanglingDescriptor),
            message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned())),
        ];
        let mut sorted = messages.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), messages.len(), "messages were {messages:?}");
    }

    // -- notation prefix ----------------------------------------------------

    /// The full text of a kind shared with the atomistic notations, stamped
    /// `CGsmiles` and spanned at byte 14 of `{[#PEO][#PEO]}[#X]` — the `[` that
    /// follows the only block. The value is built by hand to exercise
    /// `Display`'s notation prefix on a non-`Cg*` kind; `CgSmilesIr::parse` itself
    /// reports [`SmilesErrorKind::CgExpectedBlock`] for that string.
    fn cgsmiles_trailing_rendered() -> String {
        SmilesError::new(
            SmilesErrorKind::TrailingCharacters,
            Span::new(14, 15),
            "{[#PEO][#PEO]}[#X]",
            Notation::CgSmiles,
        )
        .to_string()
    }

    #[test]
    fn test_display_cgsmiles_reused_kind_names_the_cgsmiles_notation() {
        let rendered = cgsmiles_trailing_rendered();
        assert!(
            rendered.starts_with("CGsmiles parse error at position 14"),
            "rendered error was {rendered:?}"
        );
    }

    #[test]
    fn test_display_cgsmiles_caret_points_at_the_column_of_the_full_input() {
        let rendered = cgsmiles_trailing_rendered();
        let caret = rendered.lines().nth(2).expect("caret line is missing");
        assert_eq!(caret, format!("  {}^", " ".repeat(14)));
    }

    #[test]
    fn test_display_smiles_notation_renders_the_smiles_prefix() {
        let rendered = SmilesError::new(
            SmilesErrorKind::UnclosedBranch,
            Span::new(2, 3),
            "CC(",
            Notation::Smiles,
        )
        .to_string();
        assert!(
            rendered.starts_with("SMILES parse error"),
            "rendered error was {rendered:?}"
        );
    }

    #[test]
    fn test_display_smarts_notation_renders_the_smarts_prefix() {
        let rendered = SmilesError::new(
            SmilesErrorKind::InvalidQueryPrimitive("Q".to_owned()),
            Span::new(1, 2),
            "[Q]",
            Notation::Smarts,
        )
        .to_string();
        assert!(
            rendered.starts_with("SMARTS parse error"),
            "rendered error was {rendered:?}"
        );
    }

    // -- CGsmiles coarse-graph variants -------------------------------------

    /// The message body of a `CGsmiles` error, the counterpart of
    /// [`message`] for the coarse-graph kinds.
    fn cg_message(kind: SmilesErrorKind) -> String {
        let rendered =
            SmilesError::new(kind, Span::new(0, 6), "{[#A]}", Notation::CgSmiles).to_string();
        let first = rendered.lines().next().expect("rendered error is empty");
        match first.split_once(": ") {
            Some((_, body)) => body.to_owned(),
            None => first.to_owned(),
        }
    }

    #[test]
    fn test_display_cg_empty_block_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgEmptyBlock);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("empty"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_malformed_annotation_shows_the_annotation() {
        let msg = cg_message(SmilesErrorKind::CgMalformedAnnotation("q=x".to_owned()));
        assert!(msg.contains("q=x"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_unsupported_annotation_shows_key_and_value() {
        let msg = cg_message(SmilesErrorKind::CgUnsupportedAnnotation {
            key: "w".to_owned(),
            value: "0.5".to_owned(),
        });
        assert!(msg.contains('w'), "message was {msg:?}");
        assert!(msg.contains("0.5"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_annotation_on_wildcard_shows_the_annotation() {
        let msg = cg_message(SmilesErrorKind::CgAnnotationOnWildcard("q=1".to_owned()));
        assert!(msg.contains("q=1"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_bond_order_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgInvalidBondOrder);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("bond"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_repeat_count_shows_the_count() {
        let msg = cg_message(SmilesErrorKind::CgInvalidRepeatCount("0".to_owned()));
        assert!(msg.contains('0'), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_duplicate_edge_names_both_nodes() {
        let msg = cg_message(SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 });
        assert!(msg.contains('0'), "message was {msg:?}");
        assert!(msg.contains('1'), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_repeat_on_branched_node_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgRepeatOnBranchedNode);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("branch"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_repeat_on_ring_marker_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgRepeatOnRingMarker);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("ring"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_dangling_bond_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgDanglingBond);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("bond"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_ring_marker_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgInvalidRingMarker);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("ring"), "message was {msg:?}");
    }

    // -- 01c fragment-block variants ----------------------------------------
    //
    // The eight kinds `.claude/specs/cgsmiles-01c-fragments.md` § Design adds
    // for the multi-block grammar. Each message is hand-written from the rule
    // it reports; nothing here is captured from another program.

    #[test]
    fn test_display_cg_expected_block_names_the_missing_block() {
        let msg = cg_message(SmilesErrorKind::CgExpectedBlock);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("block"), "message was {msg:?}");
    }

    /// The caret of a block-structure error points at the offending byte of
    /// the whole `CGsmiles` string — byte 14, where the second block of
    /// `{[#PEO][#PEO]}[#X]` should have started with a separator.
    #[test]
    fn test_display_cg_expected_block_caret_points_at_the_offending_byte() {
        let rendered = SmilesError::new(
            SmilesErrorKind::CgExpectedBlock,
            Span::new(14, 15),
            "{[#PEO][#PEO]}[#X]",
            Notation::CgSmiles,
        )
        .to_string();
        let caret = rendered.lines().nth(2).expect("caret line is missing");
        assert_eq!(caret, format!("  {}^", " ".repeat(14)));
    }

    #[test]
    fn test_display_cg_malformed_fragment_def_names_the_definition_grammar() {
        let msg = cg_message(SmilesErrorKind::CgMalformedFragmentDef);
        assert!(!msg.is_empty());
        assert!(
            msg.to_lowercase().contains("fragment"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_cg_empty_fragment_body_names_the_empty_body() {
        let msg = cg_message(SmilesErrorKind::CgEmptyFragmentBody);
        assert!(msg.to_lowercase().contains("empty"), "message was {msg:?}");
        assert!(msg.to_lowercase().contains("body"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_duplicate_fragment_shows_the_name() {
        let msg = cg_message(SmilesErrorKind::CgDuplicateFragment("PEO".to_owned()));
        assert!(msg.contains("PEO"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_undefined_fragment_shows_the_name() {
        let msg = cg_message(SmilesErrorKind::CgUndefinedFragment("PEO".to_owned()));
        assert!(msg.contains("PEO"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_squash_unsupported_names_the_squash_operator() {
        let msg = cg_message(SmilesErrorKind::CgSquashUnsupported);
        assert!(!msg.is_empty());
        assert!(msg.contains("[!]"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_build_shows_the_reason() {
        let msg = cg_message(SmilesErrorKind::CgBuild(
            "no definition for 'B1'".to_owned(),
        ));
        assert!(
            msg.contains("no definition for 'B1'"),
            "message was {msg:?}"
        );
    }

    /// The boxed inner kind is reported, not discarded: the message states the
    /// molrs rule ("the last block must be an atomistic (OpenSMILES) body")
    /// and then the inner reason, so `{[#A]}.{#A=CC(}` names both.
    #[test]
    fn test_display_cg_last_block_not_atomistic_states_the_rule() {
        let msg = cg_message(SmilesErrorKind::CgLastBlockNotAtomistic(Box::new(
            SmilesErrorKind::UnclosedBranch,
        )));
        let lower = msg.to_lowercase();
        assert!(lower.contains("last block"), "message was {msg:?}");
        assert!(lower.contains("atomistic"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_last_block_not_atomistic_appends_the_inner_message() {
        let inner = cg_message(SmilesErrorKind::UnclosedBranch);
        let msg = cg_message(SmilesErrorKind::CgLastBlockNotAtomistic(Box::new(
            SmilesErrorKind::UnclosedBranch,
        )));
        assert!(
            msg.ends_with(&inner),
            "message was {msg:?}, inner message was {inner:?}"
        );
    }

    /// `CgNotExpandable`'s one arm serves two payload shapes; this is the
    /// fragment-name one, `{0}: no atomistic body to expand`.
    #[test]
    fn test_display_cg_not_expandable_names_the_fragment() {
        let msg = cg_message(SmilesErrorKind::CgNotExpandable("PEO".to_owned()));
        assert!(
            msg.contains("PEO: no atomistic body to expand"),
            "message was {msg:?}"
        );
    }

    /// The second shape, the phrase `templates` (link 02b) reuses for a
    /// string that names no fragment at all. The same arm has to read
    /// correctly for it, which is why the payload is a whole phrase rather
    /// than a name the arm decorates.
    #[test]
    fn test_display_cg_not_expandable_reads_correctly_for_the_base_only_phrase() {
        let msg = cg_message(SmilesErrorKind::CgNotExpandable(
            "base-only string (no fragment table)".to_owned(),
        ));
        assert!(
            msg.contains("base-only string (no fragment table): no atomistic body to expand"),
            "message was {msg:?}"
        );
    }

    /// An unmatchable edge is reported by its two indices: which level was
    /// being resolved, and which of that level's edges could not be formed.
    #[test]
    fn test_display_cg_unmatchable_edge_names_the_level_and_the_edge() {
        let msg = cg_message(SmilesErrorKind::CgUnmatchableEdge { level: 1, edge: 4 });
        assert!(
            msg.contains(
                "edge 4 of resolution level 1 has no free pair of compatible bonding \
                 descriptors to form it"
            ),
            "message was {msg:?}"
        );
    }

    // -- graph construction and ring markers --------------------------------

    /// `Build` carries the reason construction failed, and the reason is the
    /// whole point of the payload: a message that dropped it would say only
    /// that something went wrong.
    #[test]
    fn test_display_build_shows_the_reason() {
        let msg = message(SmilesErrorKind::Build(
            "bond 0-1 could not be added".to_owned(),
        ));
        assert!(
            msg.contains("bond 0-1 could not be added"),
            "message was {msg:?}"
        );
    }

    /// The SMILES rule is `%nn`: `%` followed by exactly two digits. The
    /// message states it, so a reader of `C%1C` learns what to write instead.
    #[test]
    fn test_display_invalid_ring_marker_states_the_percent_rule() {
        let msg = message(SmilesErrorKind::InvalidRingMarker);
        let lower = msg.to_lowercase();
        assert!(msg.contains('%'), "message was {msg:?}");
        assert!(lower.contains("ring marker"), "message was {msg:?}");
        assert!(lower.contains("two digits"), "message was {msg:?}");
    }

    /// Each of the two new kinds has a `CGsmiles` twin whose rule is a
    /// different one — `%` takes the whole digit run there, and `CgBuild`
    /// reports a resolution failure — so the two messages must not read alike.
    #[test]
    fn test_display_new_kinds_do_not_read_like_their_cgsmiles_twins() {
        assert_ne!(
            message(SmilesErrorKind::InvalidRingMarker),
            cg_message(SmilesErrorKind::CgInvalidRingMarker)
        );
        assert_ne!(
            message(SmilesErrorKind::Build("no definition for 'B1'".to_owned())),
            cg_message(SmilesErrorKind::CgBuild(
                "no definition for 'B1'".to_owned()
            ))
        );
    }

    #[test]
    fn test_display_cg_messages_are_pairwise_distinct() {
        let messages = [
            cg_message(SmilesErrorKind::CgEmptyBlock),
            cg_message(SmilesErrorKind::CgMalformedAnnotation("q=x".to_owned())),
            cg_message(SmilesErrorKind::CgUnsupportedAnnotation {
                key: "w".to_owned(),
                value: "0.5".to_owned(),
            }),
            cg_message(SmilesErrorKind::CgAnnotationOnWildcard("q=1".to_owned())),
            cg_message(SmilesErrorKind::CgInvalidBondOrder),
            cg_message(SmilesErrorKind::CgInvalidRepeatCount("0".to_owned())),
            cg_message(SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 }),
            cg_message(SmilesErrorKind::CgRepeatOnBranchedNode),
            cg_message(SmilesErrorKind::CgRepeatOnRingMarker),
            cg_message(SmilesErrorKind::CgDanglingBond),
            cg_message(SmilesErrorKind::CgInvalidRingMarker),
            cg_message(SmilesErrorKind::CgExpectedBlock),
            cg_message(SmilesErrorKind::CgMalformedFragmentDef),
            cg_message(SmilesErrorKind::CgEmptyFragmentBody),
            cg_message(SmilesErrorKind::CgDuplicateFragment("PEO".to_owned())),
            cg_message(SmilesErrorKind::CgUndefinedFragment("PEO".to_owned())),
            cg_message(SmilesErrorKind::CgSquashUnsupported),
            cg_message(SmilesErrorKind::CgBuild(
                "no definition for 'B1'".to_owned(),
            )),
            cg_message(SmilesErrorKind::CgLastBlockNotAtomistic(Box::new(
                SmilesErrorKind::UnclosedBranch,
            ))),
            cg_message(SmilesErrorKind::CgUnmatchableEdge { level: 1, edge: 4 }),
            cg_message(SmilesErrorKind::CgNotExpandable("PEO".to_owned())),
        ];
        let mut sorted = messages.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), messages.len(), "messages were {messages:?}");
    }
}
