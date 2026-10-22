//! The compiled form of a SMARTS pattern: per-atom and per-bond *tests* the
//! matcher evaluates against an [`Atomistic`].
//!
//! This is not a second syntax tree. The pattern's syntax is the one
//! line-notation AST (shared with SMILES); [`super::compile`] resolves it into
//! these tests — element symbols to atomic numbers, `$(...)` subpatterns to
//! indices of compiled graphs, the unwritten bond to "single or aromatic" —
//! so evaluation never re-reads the text.
//!
//! Ported (semantics only) from RDKit's query primitives under BSD-3:
//! - `Code/GraphMol/QueryAtom.h`, `Code/GraphMol/QueryBond.h`
//! - `Code/Query/` (query trees)
//!
//! The tests mirror RDKit's atom/bond query trees: leaf *predicates* (element,
//! charge, H-count, ...) combined by AND / OR / NOT nodes. Evaluation is a
//! pure read against a [`Atomistic`] plus a precomputed [`SmartsTarget`] that
//! carries ring info and the project-convention aromaticity perception.

use std::collections::HashMap;

use crate::core::Atomistic;
use crate::core::Element;
use crate::core::NodeId;
use crate::line_notation::atom_keys;
use crate::perceive::{RingSet, perceive_rings};

/// The molecule a pattern is matched against, with the facts every
/// primitive evaluation reads, precomputed once and read-only.
///
/// Holds ring info, each atom's atomic number and the per-atom aromatic
/// marker. An atom is aromatic when it carries the `is_aromatic` marker —
/// written by the SMILES reader, by aromaticity perception, or by a loader
/// that already knows the truth (e.g. transplanted from RDKit). The species
/// is read once, from [`keys::ATOMIC_NUMBER`](crate::core::keys::ATOMIC_NUMBER),
/// so an element primitive compares atomic numbers and never re-reads a
/// symbol.
///
/// It optionally carries an **external label map** (`atom → label`) consumed by
/// the [`AtomPredicate::HasContextLabel`] predicate. This is a general,
/// domain-neutral mechanism: a caller (e.g. an iterative typifier) supplies a
/// "current assignment" map and a `%LABEL` token matches an atom iff that map
/// assigns it the exact label. [`SmartsTarget::new`] leaves the map empty, so the
/// default label-free [`SmartsPattern::find`](super::SmartsPattern::find) path
/// behaves exactly as before; [`SmartsTarget::with_labels`] supplies one.
pub struct SmartsTarget<'m> {
    pub mol: &'m Atomistic,
    pub rings: RingSet,
    /// atom → atomic number (read once, up front; `0` is the wildcard atom).
    atomic_number: HashMap<NodeId, u8>,
    /// atom → is-aromatic (perceived once, up front).
    aromatic_atom: HashMap<NodeId, bool>,
    /// atom → total H count (explicit H neighbours; see module note).
    h_count: HashMap<NodeId, u32>,
    /// atom → explicit degree (number of bonded neighbours).
    degree: HashMap<NodeId, u32>,
    /// atom → number of incident ring bonds (RDKit `x<n>`).
    ring_bond_count: HashMap<NodeId, u32>,
    /// External "current label" map for the `%LABEL` context predicate.
    /// Borrowed from the caller; empty for the label-free match path.
    labels: &'m HashMap<NodeId, String>,
}

/// Shared empty label map for the label-free match path, so
/// [`SmartsTarget::new`] can borrow a `&'static HashMap` without allocating.
static EMPTY_LABELS: std::sync::LazyLock<HashMap<NodeId, String>> =
    std::sync::LazyLock::new(HashMap::new);

impl<'m> SmartsTarget<'m> {
    /// Build the match target for `mol` (runs ring perception once), with **no**
    /// external label map — `%LABEL` predicates never match until
    /// [`with_labels`](Self::with_labels) supplies one. This is the default
    /// label-free path used by [`SmartsPattern::find`](super::SmartsPattern::find).
    pub fn new(mol: &'m Atomistic) -> Self {
        let rings = perceive_rings(mol);
        let mut atomic_number = HashMap::new();
        let mut aromatic_atom = HashMap::new();
        let mut h_count = HashMap::new();
        let mut degree = HashMap::new();

        for (id, atom) in mol.atoms() {
            // Every atom of an `Atomistic` carries a valid atomic number (its
            // invariant), so the map is total.
            if let Some(z) = atom.atomic_number() {
                atomic_number.insert(id, z);
            }

            // §11.1: an aromatic atom expression reads `is_aromatic` and
            // nothing else. There is no numeric fallback — inferring atom
            // aromaticity from a bond value is how `1.5` became a second,
            // silent source of truth.
            aromatic_atom.insert(id, atom_keys::is_aromatic(&atom));

            degree.insert(id, mol.neighbor_bonds(id).count() as u32);
        }

        // Total H count = number of bonded neighbour atoms that are hydrogen,
        // Z = 1 (bonds only: a port is not a bond).
        for (id, _) in mol.atoms() {
            let h = mol
                .neighbor_bonds(id)
                .filter(|(nb, _)| atomic_number.get(nb) == Some(&Element::H.z()))
                .count() as u32;
            h_count.insert(id, h);
        }

        // Ring-bond connectivity (`x<n>`): for each ring bond, both endpoints
        // gain one incident ring bond. Counted from the bond side so it uses
        // the same `RingSet::is_bond_in_ring` truth as the `@` bond primitive.
        let mut ring_bond_count: HashMap<NodeId, u32> =
            mol.atoms().map(|(id, _)| (id, 0)).collect();
        for (bid, bond) in mol.bonds() {
            if rings.is_bond_in_ring(bid) {
                for &end in &bond.nodes {
                    *ring_bond_count.entry(end).or_insert(0) += 1;
                }
            }
        }

        Self {
            mol,
            rings,
            atomic_number,
            aromatic_atom,
            h_count,
            degree,
            ring_bond_count,
            labels: &EMPTY_LABELS,
        }
    }

    /// Use `labels` as the external label map for `%LABEL` context
    /// predicates: `labels[atom] == "L"` makes `[...;%L]` match `atom`.
    pub fn with_labels(mut self, labels: &'m HashMap<NodeId, String>) -> Self {
        self.labels = labels;
        self
    }

    /// Whether atom `id` has atomic number `z`.
    fn has_atomic_number(&self, id: NodeId, z: u8) -> bool {
        self.atomic_number.get(&id) == Some(&z)
    }

    fn is_aromatic(&self, id: NodeId) -> bool {
        self.aromatic_atom.get(&id).copied().unwrap_or(false)
    }

    /// Whether the external label map assigns `id` exactly `label`.
    fn has_label(&self, id: NodeId, label: &str) -> bool {
        self.labels.get(&id).map(String::as_str) == Some(label)
    }

    fn h_count(&self, id: NodeId) -> u32 {
        self.h_count.get(&id).copied().unwrap_or(0)
    }

    fn degree(&self, id: NodeId) -> u32 {
        self.degree.get(&id).copied().unwrap_or(0)
    }

    fn ring_bond_count(&self, id: NodeId) -> u32 {
        self.ring_bond_count.get(&id).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Atom query
// ---------------------------------------------------------------------------

/// A leaf atom primitive.
#[derive(Debug, Clone)]
pub enum AtomPredicate {
    /// `*` — matches any atom.
    Any,
    /// `a` — any aromatic atom.
    AnyAromatic,
    /// `A` — any aliphatic atom.
    AnyAliphatic,
    /// Aliphatic element by atomic number (`C`, `N`, `Cl`, ...).
    AliphaticElement(u8),
    /// Aromatic element by atomic number (`c`, `n`, `o`, ...).
    AromaticElement(u8),
    /// `#<n>` — atomic number, either aromatic or aliphatic.
    AtomicNum(u8),
    /// `H<n>` — total hydrogen count.
    TotalH(u32),
    /// `X<n>` — total connections (degree incl. implicit H; here = degree).
    TotalConnections(u32),
    /// `D<n>` — explicit degree.
    Degree(u32),
    /// `R` — in any ring (`R0` = in no ring).
    RingMembership(Option<u32>),
    /// `r<n>` — in a ring of size exactly `n` (`r` alone = in any ring).
    RingSize(Option<u32>),
    /// `r{lo-hi}` / `r{lo-}` / `r{-hi}` — the atom's *smallest* ring size lies
    /// in `[lo, hi]`. `lo == 0` means no lower bound (RDKit `r{-hi}`, which
    /// also matches acyclic atoms since their smallest ring size is 0);
    /// `hi == None` means no upper bound (RDKit `r{lo-}`).
    RingSizeRange { lo: u32, hi: Option<u32> },
    /// `x<n>` — ring-bond connectivity: the atom is incident to exactly `n`
    /// ring bonds.
    RingBondCount(u32),
    /// `+`/`-`/`+n`/`-n` — formal charge.
    Charge(i32),
    /// `%LABEL` — context-label predicate (a general molrs extension, not part
    /// of standard SMARTS). Matches an atom iff the [`SmartsTarget`]'s external
    /// label map assigns it exactly this label. Used by iterative typifiers
    /// (e.g. OPLS layered typing supplies the per-atom assigned-type map);
    /// the engine is otherwise domain-neutral about what the label means.
    HasContextLabel(String),
}

/// An atom query tree (logical combination of primitives).
#[derive(Debug, Clone)]
pub enum AtomTest {
    Prim(AtomPredicate),
    /// `$(...)` — recursive SMARTS rooted at the candidate atom.
    /// The boxed pattern is stored separately to break the type cycle.
    Recursive(usize),
    Not(Box<AtomTest>),
    And(Vec<AtomTest>),
    Or(Vec<AtomTest>),
}

impl AtomPredicate {
    fn eval(&self, ctx: &SmartsTarget, id: NodeId) -> bool {
        let mol = ctx.mol;
        match self {
            AtomPredicate::Any => true,
            AtomPredicate::AnyAromatic => ctx.is_aromatic(id),
            AtomPredicate::AnyAliphatic => !ctx.is_aromatic(id),
            AtomPredicate::AliphaticElement(z) => {
                !ctx.is_aromatic(id) && ctx.has_atomic_number(id, *z)
            }
            AtomPredicate::AromaticElement(z) => {
                ctx.is_aromatic(id) && ctx.has_atomic_number(id, *z)
            }
            AtomPredicate::AtomicNum(z) => ctx.has_atomic_number(id, *z),
            AtomPredicate::TotalH(n) => ctx.h_count(id) == *n,
            AtomPredicate::TotalConnections(n) => ctx.degree(id) == *n,
            AtomPredicate::Degree(n) => ctx.degree(id) == *n,
            AtomPredicate::RingMembership(None) => ctx.rings.is_atom_in_ring(id),
            AtomPredicate::RingMembership(Some(n)) => ctx.rings.n_atom_rings(id) as u32 == *n,
            AtomPredicate::RingSize(None) => ctx.rings.is_atom_in_ring(id),
            // RDKit's `r<n>` matches when the atom's *smallest* ring has size
            // exactly `n` (not "is in any ring of size n"). E.g. a fused-ring
            // atom shared by a 5- and a 6-ring matches `r5`, never `r6`.
            AtomPredicate::RingSize(Some(n)) => {
                ctx.rings.smallest_ring_containing_atom(id) == Some(*n as usize)
            }
            // RDKit's `r{lo-hi}` range compares the atom's *smallest* ring size
            // against the range (same data func as `r<n>`). An acyclic atom has
            // smallest ring size 0, so it satisfies only an open-low `r{-hi}`
            // bound (`lo == 0`), matching RDKit (`r{-8}` selects acyclic atoms).
            AtomPredicate::RingSizeRange { lo, hi } => {
                let size = ctx
                    .rings
                    .smallest_ring_containing_atom(id)
                    .map_or(0, |s| s as u32);
                size >= *lo && hi.is_none_or(|h| size <= h)
            }
            AtomPredicate::RingBondCount(n) => ctx.ring_bond_count(id) == *n,
            AtomPredicate::Charge(c) => mol.get_atom(id).map_or(0, |a| a.formal_charge()) == *c,
            AtomPredicate::HasContextLabel(label) => ctx.has_label(id, label),
        }
    }
}

// ---------------------------------------------------------------------------
// Bond query
// ---------------------------------------------------------------------------

/// A leaf bond primitive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondPredicate {
    /// `-` single.
    Single,
    /// `=` double.
    Double,
    /// `#` triple.
    Triple,
    /// `:` aromatic.
    Aromatic,
    /// `~` any bond.
    Any,
    /// `@` ring bond.
    InRing,
    /// Default bond (no symbol) — single OR aromatic.
    SingleOrAromatic,
}

/// A bond query tree.
#[derive(Debug, Clone)]
pub enum BondTest {
    Prim(BondPredicate),
    Not(Box<BondTest>),
    And(Vec<BondTest>),
    Or(Vec<BondTest>),
}

/// Resolved facts about a concrete molecule bond, for primitive evaluation.
pub struct BondFacts {
    pub order: f64,
    pub aromatic: bool,
    pub in_ring: bool,
}

impl BondPredicate {
    fn eval(&self, f: &BondFacts) -> bool {
        let is_single = !f.aromatic && (f.order - 1.0).abs() < 1e-6;
        let is_double = !f.aromatic && (f.order - 2.0).abs() < 1e-6;
        let is_triple = !f.aromatic && (f.order - 3.0).abs() < 1e-6;
        match self {
            BondPredicate::Single => is_single,
            BondPredicate::Double => is_double,
            BondPredicate::Triple => is_triple,
            BondPredicate::Aromatic => f.aromatic,
            BondPredicate::Any => true,
            BondPredicate::InRing => f.in_ring,
            BondPredicate::SingleOrAromatic => is_single || f.aromatic,
        }
    }
}

impl BondTest {
    /// Evaluate the bond query against resolved facts.
    pub fn eval(&self, f: &BondFacts) -> bool {
        match self {
            BondTest::Prim(p) => p.eval(f),
            BondTest::Not(inner) => !inner.eval(f),
            BondTest::And(items) => items.iter().all(|q| q.eval(f)),
            BondTest::Or(items) => items.iter().any(|q| q.eval(f)),
        }
    }
}

// ---------------------------------------------------------------------------
// Atom-query evaluation with recursive-SMARTS support
// ---------------------------------------------------------------------------

/// Callback used to evaluate a recursive `$(...)` subpattern rooted at `id`.
///
/// Implemented by the matcher (which owns the compiled subpatterns) to avoid
/// a hard type cycle between `ast` and `matcher`.
pub trait RecursiveEval {
    fn eval_recursive(&self, sub_index: usize, ctx: &SmartsTarget, id: NodeId) -> bool;
}

impl AtomTest {
    /// Evaluate this atom query against atom `id`.
    pub fn eval(&self, ctx: &SmartsTarget, id: NodeId, rec: &dyn RecursiveEval) -> bool {
        match self {
            AtomTest::Prim(p) => p.eval(ctx, id),
            AtomTest::Recursive(idx) => rec.eval_recursive(*idx, ctx, id),
            AtomTest::Not(inner) => !inner.eval(ctx, id, rec),
            AtomTest::And(items) => items.iter().all(|q| q.eval(ctx, id, rec)),
            AtomTest::Or(items) => items.iter().any(|q| q.eval(ctx, id, rec)),
        }
    }

    /// Collect every [`AtomPredicate::HasContextLabel`] label appearing in this
    /// query tree (does not descend into recursive `$(...)` subpatterns — those
    /// are handled by the owning [`QueryGraph`]).
    pub fn collect_context_labels(&self, out: &mut Vec<String>) {
        match self {
            AtomTest::Prim(AtomPredicate::HasContextLabel(l)) => out.push(l.clone()),
            AtomTest::Prim(_) | AtomTest::Recursive(_) => {}
            AtomTest::Not(inner) => inner.collect_context_labels(out),
            AtomTest::And(items) | AtomTest::Or(items) => {
                for q in items {
                    q.collect_context_labels(out);
                }
            }
        }
    }

    /// Collect ring-related leaf primitives (syntax only). Recursive `$(...)`
    /// indices are ignored here; the owning [`QueryGraph`] walks subpatterns.
    pub fn collect_ring_primitives(&self, out: &mut Vec<super::RingPrimitive>) {
        use super::RingPrimitive;
        match self {
            AtomTest::Prim(AtomPredicate::RingMembership(None)) => {
                out.push(RingPrimitive::Membership);
            }
            AtomTest::Prim(AtomPredicate::RingMembership(Some(n))) => {
                // RDKit `R0` = in no ring → still Membership-class syntax.
                if *n == 0 {
                    out.push(RingPrimitive::Membership);
                } else {
                    out.push(RingPrimitive::RingCount(*n));
                }
            }
            AtomTest::Prim(AtomPredicate::RingSize(None)) => {
                out.push(RingPrimitive::Membership);
            }
            AtomTest::Prim(AtomPredicate::RingSize(Some(n))) => {
                out.push(RingPrimitive::Sized(*n));
            }
            AtomTest::Prim(AtomPredicate::RingSizeRange { lo, hi }) => {
                // Report the finite size endpoints that appear in the range.
                if *lo > 0 {
                    out.push(RingPrimitive::Sized(*lo));
                }
                if let Some(h) = *hi {
                    out.push(RingPrimitive::Sized(h));
                }
                // Open-ended range without a finite size still mentions ring-ness.
                if *lo == 0 && hi.is_none() {
                    out.push(RingPrimitive::Membership);
                }
            }
            AtomTest::Prim(AtomPredicate::RingBondCount(n)) => {
                out.push(RingPrimitive::RingBondCount(*n));
            }
            AtomTest::Prim(_) | AtomTest::Recursive(_) => {}
            AtomTest::Not(inner) => inner.collect_ring_primitives(out),
            AtomTest::And(items) | AtomTest::Or(items) => {
                for q in items {
                    q.collect_ring_primitives(out);
                }
            }
        }
    }
}
