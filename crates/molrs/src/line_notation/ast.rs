//! Intermediate representation (IR) types for SMILES and SMARTS notation.
//!
//! [`SmilesIr`] is a pure syntax tree that captures the notation faithfully
//! without committing to atomistic or coarse-grained semantics.
//! SMARTS is modelled as a superset: [`AtomSpec::Query`] and [`BondQuery`]
//! extend the SMILES-only variants without breaking existing consumers. The
//! fragment dialect is a second extension: [`AtomNode::descriptors`] carries
//! the `CGsmiles` / `BigSMILES` bonding descriptors and is empty for the two
//! plain dialects.
//!
//! This module lives under `chem/` because the AST is the shared vocabulary of
//! all three dialects. Language-specific processing — parsing entry points,
//! validation and graph conversion — lives in the sibling `smiles/` module.
//! SMARTS *matching* ([`crate::perceive::smarts`]) compiles its queries from
//! this AST: there is one SMARTS parser, the one in this module.

use molrs::core::{BondNumber, BondOrder};

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// Byte-offset range within the input string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

/// Intermediate representation produced by the SMILES / SMARTS parser.
///
/// This is a pure syntax tree — it captures the notation faithfully without
/// committing to atomistic or coarse-grained semantics. Convert to
/// [`Atomistic`](crate::core::Atomistic) for domain use; a
/// coarse-grained bead graph comes from `CGsmiles` instead, via
/// [`CgSmilesIr::to_coarsegrain`](crate::io::cgsmiles::CgSmilesIr::to_coarsegrain).
///
/// Multiple disconnected components are separated by `.` in the input.
#[derive(Debug, Clone, PartialEq)]
pub struct SmilesIr {
    /// Connected components (separated by `.` in the input).
    pub components: Vec<Chain>,
    /// Span covering the entire input.
    pub span: Span,
}

/// A linear chain of atoms with branches and ring closures.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    /// The first atom in the chain.
    pub head: AtomNode,
    /// Subsequent elements: bonded atoms, branches, or ring closures.
    pub tail: Vec<ChainElement>,
}

/// An element following the head atom in a chain.
///
/// Bond slots are [`BondQuery`] rather than plain [`BondKind`] so the AST
/// can faithfully represent SMARTS bond operators (`!`, `,`, `&`) alongside
/// simple SMILES bond kinds. For SMILES inputs the parser always emits
/// `Some(BondQuery::Kind(_))` or `None`.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainElement {
    /// An atom bonded to the previous atom.
    BondedAtom {
        bond: Option<BondQuery>,
        atom: AtomNode,
    },
    /// A parenthesised branch: `(` bond? chain `)`.
    Branch {
        bond: Option<BondQuery>,
        chain: Chain,
        span: Span,
    },
    /// A ring-closure digit or `%nn`.
    RingClosure {
        bond: Option<BondQuery>,
        rnum: u16,
        span: Span,
    },
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/// An atom node carrying its specification and source span.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomNode {
    /// What the notation wrote at this position: an organic-subset symbol, a
    /// bracket atom, `*`, or a SMARTS query expression.
    pub spec: AtomSpec,
    /// Byte range of the atom's own text in the parsed input, used for error
    /// carets. Descriptor brackets written next to it are *not* covered.
    pub span: Span,
    /// Bonding descriptors anchored on this atom, in written order.
    ///
    /// A descriptor binds to the node written immediately before it — or, when
    /// it is written before any atom of its chain, to that chain's head atom —
    /// so this vector holds the descriptors the notation attached to *this*
    /// atom, in the order they appear in the text: `[>][$1]C` gives the carbon
    /// a `>` first and a `$` labelled `1` second. Empty for plain SMILES and
    /// SMARTS, which have no such notation.
    pub descriptors: Vec<BondingDescriptor>,
}

/// Atom specification — the extensibility point for SMARTS.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomSpec {
    /// Organic-subset shorthand (no brackets): `C`, `N`, `c`, `n`, etc.
    Organic { symbol: String, aromatic: bool },
    /// Bracket atom: `[isotope? symbol chirality? hcount? charge? class?]`.
    Bracket {
        isotope: Option<u16>,
        symbol: BracketSymbol,
        chirality: Option<Chirality>,
        hcount: Option<u8>,
        /// Formal charge as written, `[NH4+]` giving `Some(1)`: a whole number
        /// of elementary charges assigned by the valence bookkeeping of the
        /// notation. It is **not** a partial charge — the fractional
        /// force-field charge of a coarse-grained bead is
        /// [`CgNode::charge`](crate::io::cgsmiles::CgNode::charge), an `f64` in
        /// elementary charge units `e`.
        charge: Option<i8>,
        atom_class: Option<u16>,
    },
    /// Wildcard `*`.
    Wildcard,
    /// SMARTS query expression (logical combination of primitives).
    Query(AtomQuery),
}

impl AtomSpec {
    /// Whether the **notation wrote this atom aromatic** — a lowercase organic
    /// symbol (`c`, `n`) or a lowercase bracket element (`[nH]`).
    ///
    /// This is a question about the text, not about the molecule: it reports
    /// the aromaticity the string *declares*, before any perception step has
    /// looked at rings. A Kekulé-spelled benzene (`C1=CC=CC=C1`) writes no
    /// aromatic atom and answers `false` for every one of its carbons, however
    /// aromatic [`crate::perceive::assign_aromaticity`] would later find the ring.
    ///
    /// The SMARTS primitives `a` ([`BracketSymbol::Aromatic`]) and
    /// [`AtomSpec::Query`] answer `false`: a query *asks* whether an atom is
    /// aromatic, and asking is not declaring. [`AtomSpec::Wildcard`] declares
    /// no element at all, so it declares no aromaticity either.
    ///
    /// The one implementation of the predicate, and one evaluator of it: the
    /// SMILES builder asks it here and records the answer as the atom property
    /// `is_aromatic`. The `CGsmiles` resolver — which promotes a bond written
    /// with no symbol between two written-aromatic atoms to an aromatic one —
    /// never holds an `AtomSpec` of its own, and reads that stamp instead, off
    /// a fragment body just converted and never perceived, where the stamp can
    /// only say what the notation wrote.
    ///
    /// Reference: Daylight Theory Manual, *SMILES*, § 3 (aromatic atoms are
    /// written in lower case).
    pub(crate) fn written_aromatic(&self) -> bool {
        match self {
            AtomSpec::Organic { aromatic, .. } => *aromatic,
            AtomSpec::Bracket {
                symbol: BracketSymbol::Element { aromatic, .. },
                ..
            } => *aromatic,
            AtomSpec::Bracket { .. } | AtomSpec::Wildcard | AtomSpec::Query(_) => false,
        }
    }
}

/// Symbol inside a bracket atom.
#[derive(Debug, Clone, PartialEq)]
pub enum BracketSymbol {
    /// A concrete element, possibly aromatic.
    Element { symbol: String, aromatic: bool },
    /// `*` inside brackets.
    Any,
    /// `A` — any aliphatic atom (SMARTS).
    Aliphatic,
    /// `a` — any aromatic atom (SMARTS).
    Aromatic,
}

/// Tetrahedral chirality marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chirality {
    /// `@` — counter-clockwise (S).
    CounterClockwise,
    /// `@@` — clockwise (R).
    Clockwise,
}

// ---------------------------------------------------------------------------
// Bonds
// ---------------------------------------------------------------------------

/// Bond kind (covers both SMILES and SMARTS bond types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondKind {
    /// `-` explicit single bond.
    Single,
    /// `=` double bond.
    Double,
    /// `#` triple bond.
    Triple,
    /// `$` quadruple bond.
    ///
    /// The `$` glyph is overloaded with [`DescriptorKind::Symmetric`] and is
    /// disambiguated only by bracket position: outside a bracket it is this
    /// bond kind (`C$C` is two carbons joined by a quadruple bond), inside one
    /// it is a bonding descriptor (`C[$]C`). A bond symbol written immediately
    /// *before* a descriptor bracket annotates that descriptor instead of
    /// joining two atoms, so in the fragment dialect `C$[$]C` is two carbons
    /// with an unannotated bond between them and a `$` descriptor of order
    /// `Quadruple` on the first — the quadruple bond is the one that
    /// descriptor's future pairing will create, not one present in the string.
    Quadruple,
    /// `:` aromatic bond.
    Aromatic,
    /// `/` directional up (cis/trans).
    Up,
    /// `\` directional down (cis/trans).
    Down,
    /// `~` any bond (SMARTS wildcard).
    Any,
    /// `@` ring bond (SMARTS).
    Ring,
}

impl BondKind {
    /// The bond's **chemical class**, as
    /// [`Atomistic::set_bond_class`](crate::core::Atomistic::set_bond_class)
    /// records it.
    ///
    /// `Aromatic` is a class of its own rather than a number: the notation
    /// declares the ring delocalized and says nothing about which Kekulé
    /// structure to pick, so [`BondKind::bond_number`] leaves that `Unknown`.
    /// The directional kinds `/` and `\` and the SMARTS wildcards `~` and `@`
    /// are structurally single bonds — the direction is stereochemistry
    /// recorded elsewhere, and a wildcard states no order at all.
    ///
    /// # Approximation
    ///
    /// [`BondKind::Quadruple`] maps to [`BondOrder::Double`], because
    /// [`BondOrder`](crate::core::BondOrder) has no quadruple variant
    /// (`core/bond_order.rs`). The number is exact —
    /// [`BondKind::bond_number`] answers [`BondNumber::Quadruple`] — so no
    /// count is lost, only the class is coarsened. Widening `BondOrder` is a
    /// change to the core bond vocabulary and is not made here.
    ///
    /// Reference: Daylight Theory Manual, *SMILES*, § 3 (bond symbols).
    pub(crate) fn bond_type(self) -> BondOrder {
        match self {
            BondKind::Single | BondKind::Up | BondKind::Down => BondOrder::Single,
            BondKind::Double => BondOrder::Double,
            BondKind::Triple => BondOrder::Triple,
            // A quadruple bond has no aromatic character; it is a plain class whose
            // number the notation states outright.
            BondKind::Quadruple => BondOrder::Double,
            BondKind::Aromatic => BondOrder::Aromatic,
            BondKind::Any | BondKind::Ring => BondOrder::Single,
        }
    }

    /// The **localized (Kekulé) number** this kind states, when it states one.
    ///
    /// [`BondKind::Aromatic`] states none — it declares delocalization, not a
    /// Kekulé phase — and answers [`BondNumber::Unknown`], which kekulization
    /// later replaces. Every other kind, including the quadruple bond the
    /// class approximates, states its own count.
    ///
    /// Reference: Daylight Theory Manual, *SMILES*, § 3 (bond symbols).
    pub(crate) fn bond_number(self) -> BondNumber {
        match self {
            BondKind::Single | BondKind::Up | BondKind::Down => BondNumber::Single,
            BondKind::Double => BondNumber::Double,
            BondKind::Triple => BondNumber::Triple,
            BondKind::Quadruple => BondNumber::Quadruple,
            // The notation declares delocalization, not a Kekulé phase.
            BondKind::Aromatic => BondNumber::Unknown,
            BondKind::Any | BondKind::Ring => BondNumber::Single,
        }
    }
}

/// SMARTS bond query with logical operators.
///
/// A SMILES bond is one concrete kind; a SMARTS bond may be a logical
/// combination of kinds, which is why every bond slot in the AST holds this
/// rather than a bare [`BondKind`]. SMILES and fragment inputs only ever
/// produce [`BondQuery::Kind`].
#[derive(Debug, Clone, PartialEq)]
pub enum BondQuery {
    /// One concrete bond kind — the only variant a non-SMARTS parse produces.
    Kind(BondKind),
    /// `!expr` — matches any bond the inner query does not.
    Not(Box<BondQuery>),
    /// `expr & expr` — matches a bond satisfying every listed query.
    And(Vec<BondQuery>),
    /// `expr , expr` — matches a bond satisfying at least one listed query.
    Or(Vec<BondQuery>),
}

// ---------------------------------------------------------------------------
// Bonding descriptors (CGsmiles / BigSMILES)
// ---------------------------------------------------------------------------

/// A bonding descriptor anchored on an atom: `[$]`, `[<]`, `[>]`, `[!]`.
///
/// A descriptor marks a site where the fragment can be joined to another
/// fragment: `[$]COC[$]` is a C-O-C ether unit offering one joining site on
/// each of its carbons. It is notation of the `CGsmiles` / `BigSMILES` fragment dialect,
/// not of plain SMILES, so plain-SMILES nodes carry none. The descriptor is
/// only *data* here — matching two descriptors up and creating the bond
/// between their atoms is a later step, outside this module.
///
/// References: cgsmiles.readthedocs.io (fragments page); Lin, T.-S. et al.,
/// *BigSMILES: A Structurally-Based Line Notation for Describing
/// Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019).
/// DOI: 10.1021/acscentsci.9b00476
#[derive(Debug, Clone, PartialEq)]
pub struct BondingDescriptor {
    /// Which operator was written, and hence which descriptors it may pair with.
    pub kind: DescriptorKind,
    /// Label distinguishing descriptor classes of the same kind, `""` when the
    /// descriptor is unnamed (`[$]` vs `[$a]`).
    ///
    /// `BigSMILES` restricts labels to positive integers; `CGsmiles` widens them
    /// to alphanumerics, which is what this field holds.
    pub label: String,
    /// Bond order written next to the bracket, outside it (`CC=[$]`).
    ///
    /// `None` means no bond symbol was written next to the bracket; `Some(k)`
    /// means `k` was written. The effective order of the bond formed when two
    /// descriptors pair is [`BondKind::Single`] when this is `None`.
    ///
    /// The distinction between `None` and `Some(BondKind::Single)` is not
    /// cosmetic, which is why the field is an `Option` rather than a plain
    /// [`BondKind`]. SMILES promotes a bond between two aromatic atoms to
    /// [`BondKind::Aromatic`] when — and only when — the notation wrote no
    /// bond symbol at all (the Daylight rule: `c1ccccc1` is a ring of aromatic
    /// bonds, while the explicit `-` of biphenyl's `c1ccccc1-c1ccccc1` stays
    /// single). A descriptor that will one day form a bond between two
    /// aromatic atoms therefore has to keep "nothing was written" and "`-` was
    /// written" apart, because they promote differently.
    pub order: Option<BondKind>,
}

/// Which bonding-descriptor operator an atom carries.
///
/// Variants are named for the pairing role, not for the glyph, matching the
/// in-file precedent of [`BondKind`] and [`Chirality`]. Pairing itself is not
/// performed here; these are the rules the pairing step enforces.
///
/// References: cgsmiles.readthedocs.io (fragments page); Lin, T.-S. et al.,
/// *BigSMILES: A Structurally-Based Line Notation for Describing
/// Macromolecules*, ACS Cent. Sci. **5**, 1523–1531 (2019).
/// DOI: 10.1021/acscentsci.9b00476
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorKind {
    /// `$` — the AA-type (self-complementary) operator: a `$` bonds only to
    /// another `$` carrying the same label, so one descriptor class describes
    /// both ends of the junction (`[$]COC[$]` polymerises with itself).
    Symmetric,
    /// `<` — the left half of the AB-type (two-role) operator pair: a `<`
    /// bonds only to a [`DescriptorKind::Right`] of the same label, never to
    /// another `<`.
    Left,
    /// `>` — the right half of the AB-type (two-role) operator pair: a `>`
    /// bonds only to a [`DescriptorKind::Left`] of the same label, never to
    /// another `>`.
    Right,
    /// `!` — the `CGsmiles` squash operator: instead of forming a bond it
    /// merges (squashes) the two nodes it joins into one. Only the notation is
    /// modelled here; no stage of this module performs the merge.
    Shared,
}

impl DescriptorKind {
    /// The notation glyph this operator is written as: `$`, `<`, `>` or `!`.
    ///
    /// This is the same string
    /// [`core::PortKind::as_str`](crate::core::PortKind::as_str)
    /// returns for the port role the descriptor is stored as, so a user reads
    /// and writes one spelling per role whether the value came from the
    /// notation side or from the stored side. The two enums stay distinct by
    /// design — this one names what was written, `PortKind` names what is
    /// stored — and this method is what keeps their single user-facing
    /// spelling in step.
    pub fn as_str(self) -> &'static str {
        match self {
            DescriptorKind::Symmetric => "$",
            DescriptorKind::Left => "<",
            DescriptorKind::Right => ">",
            DescriptorKind::Shared => "!",
        }
    }
}

// ---------------------------------------------------------------------------
// SMARTS query algebra
// ---------------------------------------------------------------------------

/// SMARTS atom query primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomPrimitive {
    /// Concrete element (possibly aromatic).
    Element { symbol: String, aromatic: bool },
    /// `#<n>` — atomic number, aromatic or aliphatic alike (`[#6]` matches
    /// both `C` and `c`, unlike the element symbol `C`).
    AtomicNumber(u8),
    /// `*` — any atom.
    Wildcard,
    /// `A` — any aliphatic atom.
    Aliphatic,
    /// `a` — any aromatic atom.
    Aromatic,
    /// `D<n>` — explicit degree (number of explicit bonds).
    Degree(u8),
    /// `X<n>` — total connections (explicit + implicit H).
    TotalConnections(u8),
    /// `H<n>` — total hydrogen count.
    HCount(u8),
    /// `h<n>` — implicit hydrogen count.
    ImplicitH(u8),
    /// `R<n>` — ring membership count (0 = any ring).
    RingMembership(Option<u8>),
    /// `r<n>` — smallest ring size.
    RingSize(u8),
    /// `r{lo-hi}` / `r{lo-}` / `r{-hi}` — smallest ring size in a range (the
    /// RDKit extension). `lo == 0` is no lower bound, `hi == None` no upper.
    RingSizeRange { lo: u8, hi: Option<u8> },
    /// `x<n>` — number of incident ring bonds (ring connectivity).
    RingBondCount(u8),
    /// `v<n>` — total valence.
    Valence(u8),
    /// Formal charge.
    Charge(i8),
    /// Isotope (mass number).
    Isotope(u16),
    /// `:<n>` — atom class / map number.
    AtomClass(u16),
    /// Chirality specification.
    Chirality(Chirality),
    /// `$(...)` — recursive SMARTS (environment match).
    Recursive(Box<SmilesIr>),
    /// `%LABEL` — a molrs extension, not standard SMARTS: the atom carries
    /// exactly this label in a caller-supplied label map (the iterative
    /// typifiers' "already assigned type", e.g. `%opls_154`). The label starts
    /// with a letter or `_`, so it never reads as a `%nn` ring closure, which
    /// only appears outside brackets anyway.
    ContextLabel(String),
}

/// SMARTS atom query expression with logical operators.
///
/// Operator precedence (highest to lowest):
/// 1. `!` — NOT (unary)
/// 2. `&` — AND (high precedence, also implicit between adjacent primitives)
/// 3. `,` — OR
/// 4. `;` — AND (low precedence)
#[derive(Debug, Clone, PartialEq)]
pub enum AtomQuery {
    /// A single primitive.
    Primitive(AtomPrimitive),
    /// `!expr` — logical NOT.
    Not(Box<AtomQuery>),
    /// `expr & expr` or implicit adjacency — high-precedence AND.
    And(Vec<AtomQuery>),
    /// `expr , expr` — OR.
    Or(Vec<AtomQuery>),
    /// `expr ; expr` — low-precedence AND.
    LowAnd(Vec<AtomQuery>),
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Every expectation below is hand-written from the notation itself: the
    // Daylight Theory Manual § SMILES (lowercase symbols are the *written*
    // aromatic declaration; a SMARTS primitive is a query, not a declaration)
    // and the `BondKind` → `(BondOrder, BondNumber)` table already in use at
    // `io/smiles/smiles/to_atomistic.rs`. No external program produced any
    // value here.

    /// A bracket atom carrying nothing but `symbol`.
    fn bracket(symbol: BracketSymbol) -> AtomSpec {
        AtomSpec::Bracket {
            isotope: None,
            symbol,
            chirality: None,
            hcount: None,
            charge: None,
            atom_class: None,
        }
    }

    // -- AtomSpec::written_aromatic -----------------------------------------

    #[test]
    fn test_written_aromatic_is_true_for_a_lowercase_organic_atom() {
        let spec = AtomSpec::Organic {
            symbol: "c".to_owned(),
            aromatic: true,
        };
        assert!(spec.written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_false_for_an_uppercase_organic_atom() {
        let spec = AtomSpec::Organic {
            symbol: "C".to_owned(),
            aromatic: false,
        };
        assert!(!spec.written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_true_for_an_aromatic_bracket_element() {
        let spec = bracket(BracketSymbol::Element {
            symbol: "n".to_owned(),
            aromatic: true,
        });
        assert!(spec.written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_false_for_an_aliphatic_bracket_element() {
        let spec = bracket(BracketSymbol::Element {
            symbol: "N".to_owned(),
            aromatic: false,
        });
        assert!(!spec.written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_false_for_the_any_bracket_symbol() {
        assert!(!bracket(BracketSymbol::Any).written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_false_for_the_aliphatic_bracket_symbol() {
        assert!(!bracket(BracketSymbol::Aliphatic).written_aromatic());
    }

    /// The SMARTS primitive `a` asks a *question* about an atom; it does not
    /// declare one aromatic, so it is not a written aromatic atom.
    #[test]
    fn test_written_aromatic_is_false_for_the_aromatic_bracket_symbol() {
        assert!(!bracket(BracketSymbol::Aromatic).written_aromatic());
    }

    #[test]
    fn test_written_aromatic_is_false_for_the_wildcard() {
        assert!(!AtomSpec::Wildcard.written_aromatic());
    }

    /// Same reason as [`BracketSymbol::Aromatic`]: a query is not a
    /// declaration, whatever primitive it holds.
    #[test]
    fn test_written_aromatic_is_false_for_a_query_atom() {
        let spec = AtomSpec::Query(AtomQuery::Primitive(AtomPrimitive::Aromatic));
        assert!(!spec.written_aromatic());
    }

    // -- DescriptorKind::as_str ---------------------------------------------

    /// The whole glyph table, every variant pinned. The glyphs are the four
    /// operators of the `CGsmiles` / `BigSMILES` grammar, so this table is the
    /// notation-side mirror of `core::PortKind::as_str` and the two must keep
    /// spelling the same role the same way.
    #[test]
    fn test_descriptor_kind_as_str_is_the_notation_glyph() {
        assert_eq!(DescriptorKind::Symmetric.as_str(), "$");
        assert_eq!(DescriptorKind::Left.as_str(), "<");
        assert_eq!(DescriptorKind::Right.as_str(), ">");
        assert_eq!(DescriptorKind::Shared.as_str(), "!");
    }

    // -- BondKind::bond_type / bond_number ----------------------------------

    /// The whole class table, every variant pinned: the directional and
    /// wildcard kinds are structurally single bonds, and `Aromatic` is its own
    /// class rather than a number.
    #[test]
    fn test_bond_type_maps_every_kind_to_its_class() {
        assert_eq!(BondKind::Single.bond_type(), BondOrder::Single);
        assert_eq!(BondKind::Double.bond_type(), BondOrder::Double);
        assert_eq!(BondKind::Triple.bond_type(), BondOrder::Triple);
        assert_eq!(BondKind::Quadruple.bond_type(), BondOrder::Double);
        assert_eq!(BondKind::Aromatic.bond_type(), BondOrder::Aromatic);
        assert_eq!(BondKind::Up.bond_type(), BondOrder::Single);
        assert_eq!(BondKind::Down.bond_type(), BondOrder::Single);
        assert_eq!(BondKind::Any.bond_type(), BondOrder::Single);
        assert_eq!(BondKind::Ring.bond_type(), BondOrder::Single);
    }

    /// The whole number table, every variant pinned.
    #[test]
    fn test_bond_number_maps_every_kind_to_its_number() {
        assert_eq!(BondKind::Single.bond_number(), BondNumber::Single);
        assert_eq!(BondKind::Double.bond_number(), BondNumber::Double);
        assert_eq!(BondKind::Triple.bond_number(), BondNumber::Triple);
        assert_eq!(BondKind::Quadruple.bond_number(), BondNumber::Quadruple);
        assert_eq!(BondKind::Aromatic.bond_number(), BondNumber::Unknown);
        assert_eq!(BondKind::Up.bond_number(), BondNumber::Single);
        assert_eq!(BondKind::Down.bond_number(), BondNumber::Single);
        assert_eq!(BondKind::Any.bond_number(), BondNumber::Single);
        assert_eq!(BondKind::Ring.bond_number(), BondNumber::Single);
    }

    /// `BondOrder` has no quadruple variant, so the class is the documented
    /// approximation `Double` while the number states the quadruple outright.
    /// Pinned as a pair so the approximation cannot be widened silently.
    #[test]
    fn test_quadruple_is_the_documented_double_class_with_a_quadruple_number() {
        assert_eq!(
            (
                BondKind::Quadruple.bond_type(),
                BondKind::Quadruple.bond_number()
            ),
            (BondOrder::Double, BondNumber::Quadruple)
        );
    }

    /// The notation declares delocalization, not a Kekulé phase: an aromatic
    /// bond has no localized number until kekulization picks one.
    #[test]
    fn test_aromatic_is_an_aromatic_class_with_an_unknown_number() {
        assert_eq!(
            (
                BondKind::Aromatic.bond_type(),
                BondKind::Aromatic.bond_number()
            ),
            (BondOrder::Aromatic, BondNumber::Unknown)
        );
    }
}
