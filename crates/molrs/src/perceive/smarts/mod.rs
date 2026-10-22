//! SMARTS substructure-matching engine.
//!
//! SMARTS, wholly: the pattern is parsed by the crate's one line-notation
//! grammar (shared with SMILES), compiled into a query graph, matched by a
//! backtracking subgraph-isomorphism matcher covering the SMARTS
//! feature subset used by RDKit's ETKDGv3 experimental-torsion preference
//! tables (`torsionPreferences_v2 / _smallrings / _macrocycles`), including
//! recursive SMARTS `[$(...)]`.
//!
//! Match semantics follow RDKit `GetSubstructMatches(uniquify=False)`: every
//! distinct query-atom → mol-atom embedding is reported, ordered by query-atom
//! index. Ported (semantics only) from RDKit under the BSD-3 licence:
//! - `Code/GraphMol/Substruct/SubstructMatch.cpp` (matching + recursive eval)
//! - `Code/GraphMol/QueryAtom.h` / `QueryBond.h` (query primitives)
//!
//! # Aromaticity convention
//!
//! `Atomistic` carries no aromatic model. An atom is aromatic when it carries
//! the `is_aromatic` marker (written by the SMILES reader and by aromaticity
//! perception) and a bond when its `bond_type` is aromatic. A caller can
//! transplant a reference perception (e.g. RDKit's) by writing those markers,
//! so that aromatic queries (`a`, `c`, `:` bonds) agree exactly.
//!
//! # Species
//!
//! Element primitives (`C`, `c`, `#6`) compare the atom's
//! [`atomic_number`](crate::core::keys::ATOMIC_NUMBER); an `element` symbol
//! is never read.
//!
//! # Supported features
//!
//! - Atom primitives: aliphatic/aromatic elements, `*`, `a`, `A`, `#<n>`,
//!   `H<n>`, `X<n>`, `D<n>`, `R`/`R<n>`, `r<n>`, `r{lo-hi}`, `x<n>`,
//!   `+`/`++`/`+<n>`/`-`/`-<n>`, atom-map `:<n>`, and the molrs context
//!   label `%LABEL` (see [`MatchOptions::labels`]).
//! - Atom logic: implicit/`&` high AND, `;` low AND, `,` OR, `!` NOT.
//! - Recursive SMARTS `[$(...)]` (nestable), rooted at the candidate atom.
//! - Bond primitives: `-` `=` `#` `:` `~` `@`, `!`, logical combos
//!   (`!@;-`, `-,:`); default bond = single-or-aromatic.
//! - Branches `( )`, ring closures incl. `%nn`.
//!
//! Out of scope, refused with an error: chirality `@`/`@@`, isotopes, `h<n>`,
//! `v<n>`, directional and quadruple bonds, and `.`-separated component
//! SMARTS (a [`Reaction`] splits its reactants itself).
//!
//! # Example
//!
//! ```
//! use molrs::perceive::smarts::SmartsPattern;
//! use molrs::core::{Atomistic, BondNumber, BondOrder, Element, keys};
//! use molrs::props;
//!
//! // Acetamide skeleton C-C(=O)-N (no Hs needed for this query).
//! let mut g = Atomistic::new();
//! let c0 = g.add_atom(props! { keys::ATOMIC_NUMBER => Element::C }).unwrap();
//! let c1 = g.add_atom(props! { keys::ATOMIC_NUMBER => Element::C }).unwrap();
//! let o = g.add_atom(props! { keys::ATOMIC_NUMBER => Element::O }).unwrap();
//! let n = g.add_atom(props! { keys::ATOMIC_NUMBER => Element::N }).unwrap();
//! g.add_bond(c0, c1, props! {}).unwrap();
//! g.add_bond(
//!     c1,
//!     o,
//!     props! { keys::BOND_TYPE => BondOrder::Double, keys::BOND_NUMBER => BondNumber::Double },
//! )
//! .unwrap();
//! g.add_bond(c1, n, props! {}).unwrap();
//!
//! let pat = SmartsPattern::parse("[$([CX3]=[OX1]):1]~[*:2]").unwrap();
//! assert!(pat.has_match(&g, molrs::perceive::smarts::MatchOptions::default()));
//! assert_eq!(pat.map_label(0), Some(1));
//! ```

mod compile;
mod environment;
mod matcher;
mod predicate;
mod reaction;

use std::collections::HashMap;

use crate::core::Atomistic;
use crate::core::MolRsError;
use crate::core::NodeId;

use compile::QueryGraph;

pub use environment::{EnvironmentOptions, NeighborStyle};
pub use reaction::{Reaction, ReactionBatch};

/// Ring-related SMARTS atom primitives found in a compiled pattern.
///
/// This is **syntax only**: the caller decides what a kind means for it (for
/// example, whether an untyped `[R]` leaves a pattern set without a bounded
/// ring size). The engine never returns a boolean "is_bounded".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RingPrimitive {
    /// Sized ring membership, e.g. `[r6]` or a finite endpoint of `r{lo-hi}`.
    Sized(u32),
    /// Untyped ring membership: `[R]`, `[!R]`, `[R0]`, bare `r`.
    Membership,
    /// Ring-count token `[Rn]` with `n >= 1` (`RingMembership(Some(n))`).
    RingCount(u32),
    /// Ring-bond connectivity `[xn]`.
    RingBondCount(u32),
}

/// Matching controls for [`SmartsPattern::find`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchOptions<'a> {
    /// Optional `%LABEL` context (`atom -> current label`).
    pub labels: Option<&'a HashMap<NodeId, String>>,
    /// Optional root pin for query atom 0.
    pub root: Option<NodeId>,
    /// Optional maximum number of matches to return.
    pub limit: Option<usize>,
}

/// One SMARTS match, indexed by query atom order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartsMatch {
    pub atoms: Vec<NodeId>,
}

impl SmartsMatch {
    pub fn atoms(&self) -> &[NodeId] {
        &self.atoms
    }
}

/// A compiled SMARTS query.
///
/// Its [`Display`](std::fmt::Display) is the SMARTS text: what
/// [`parse`](Self::parse) read, or what
/// [`from_environment`](Self::from_environment) generated.
#[derive(Debug, Clone)]
pub struct SmartsPattern {
    graph: QueryGraph,
    smarts: String,
}

impl std::fmt::Display for SmartsPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.smarts)
    }
}

impl SmartsPattern {
    /// Parse a SMARTS string and compile it for matching. Returns `Err` on any
    /// syntax error, or on a construct the matcher does not evaluate (never
    /// panics).
    pub fn parse(smarts: &str) -> Result<SmartsPattern, MolRsError> {
        let graph = compile::compile(smarts)?;
        Ok(SmartsPattern {
            graph,
            smarts: smarts.to_owned(),
        })
    }

    /// The pattern that states the local environment of `center` in `mol` —
    /// its element, and as many of degree, hydrogen count, charge,
    /// aromaticity, ring membership and neighbours out to `options.reach`
    /// bonds as `options` asks for.
    ///
    /// The pattern matches `mol` at `center` (as atom 0). With
    /// [`NeighborStyle::Chain`] its
    /// [`max_bond_depth`](Self::max_bond_depth) is at most `options.reach`.
    ///
    /// # Errors
    ///
    /// `options.reach` is 0, `center` is not an atom of `mol`, or an atom
    /// carries no valid `atomic_number` (the species is read from Z alone).
    /// The wildcard atom (`Z = 0`) is written `*` — or `#0` under
    /// [`EnvironmentOptions::atomic_number`].
    pub fn from_environment(
        mol: &Atomistic,
        center: NodeId,
        options: &EnvironmentOptions,
    ) -> Result<SmartsPattern, MolRsError> {
        let ir = environment::environment_ir(mol, center, options)
            .map_err(|e| MolRsError::parse(e.to_string()))?;
        let smarts = crate::line_notation::writer::smarts_text(&ir)
            .map_err(|e| MolRsError::parse(e.to_string()))?;
        Self::parse(&smarts)
    }

    /// All matches (non-uniquified), controlled by [`MatchOptions`].
    pub fn find(&self, mol: &Atomistic, options: MatchOptions<'_>) -> Vec<SmartsMatch> {
        matcher::find(&self.graph, mol, options)
    }

    pub(crate) fn find_in_target(
        &self,
        target: &predicate::SmartsTarget<'_>,
        root: Option<NodeId>,
    ) -> Vec<SmartsMatch> {
        matcher::find_in_target(&self.graph, target, root, None)
    }

    /// Whether at least one match exists.
    pub fn has_match(&self, mol: &Atomistic, options: MatchOptions<'_>) -> bool {
        matcher::has_match(&self.graph, mol, options)
    }

    /// Project a match into `{atom_map_label -> molecule atom}`.
    pub fn mapped(&self, m: &SmartsMatch) -> HashMap<u32, NodeId> {
        m.atoms
            .iter()
            .enumerate()
            .filter_map(|(i, &atom)| self.map_label(i).map(|label| (label, atom)))
            .collect()
    }

    /// The SMARTS atom-map label (`:1` etc.) of query atom `query_atom`, or
    /// `None` if unlabelled / out of range.
    pub fn map_label(&self, query_atom: usize) -> Option<u32> {
        self.graph.atoms.get(query_atom).and_then(|a| a.map_label)
    }

    /// Number of query atoms.
    pub fn n_query_atoms(&self) -> usize {
        self.graph.atoms.len()
    }

    /// Longest shortest-path length (in bonds) on the **query atom graph**.
    ///
    /// Isolated single-atom queries return `0`. Recursive `$(...)` subpatterns
    /// do not contribute edges — they are atom predicates, not topological
    /// neighbours of the main query skeleton.
    pub fn max_bond_depth(&self) -> usize {
        self.graph.max_bond_depth()
    }

    /// Ring primitives used anywhere in this pattern (including recursive
    /// subpatterns). Order is traversal order; duplicates are not collapsed
    /// (callers may dedup). Does not judge boundedness.
    pub fn ring_primitives(&self) -> Vec<RingPrimitive> {
        self.graph.ring_primitives()
    }

    /// Every `%LABEL` context-label referenced by this pattern (including those
    /// inside recursive `$(...)` subpatterns), in traversal order with
    /// duplicates kept.
    ///
    /// Iterative typifiers use this to discover a def's dependencies on
    /// previously-assigned labels (e.g. OPLS `%opls_NNN` references).
    pub fn context_labels(&self) -> Vec<String> {
        self.graph.context_labels()
    }
}

#[cfg(test)]
mod pattern_syntax_tests {
    use super::{RingPrimitive, SmartsPattern};

    #[test]
    fn max_bond_depth_linear() {
        assert_eq!(SmartsPattern::parse("C").unwrap().max_bond_depth(), 0);
        assert_eq!(SmartsPattern::parse("CC").unwrap().max_bond_depth(), 1);
        assert_eq!(SmartsPattern::parse("CCO").unwrap().max_bond_depth(), 2);
    }

    #[test]
    fn max_bond_depth_branch() {
        // C(C)C — propane skeleton: leaf–centre–leaf longest shortest path = 2
        let p = SmartsPattern::parse("C(C)C").unwrap();
        assert_eq!(p.max_bond_depth(), 2);
        // Explicit centre with two single-bond neighbours as separate patterns:
        // CC(C)C is isobutane-like depth 2 (any methyl to methyl via tertiary C).
        assert_eq!(SmartsPattern::parse("CC(C)C").unwrap().max_bond_depth(), 2);
    }

    #[test]
    fn ring_primitives_membership_and_sized() {
        let empty = SmartsPattern::parse("[C]").unwrap().ring_primitives();
        assert!(empty.is_empty());

        let r = SmartsPattern::parse("[R]").unwrap().ring_primitives();
        assert!(r.contains(&RingPrimitive::Membership));

        let r2 = SmartsPattern::parse("[R2]").unwrap().ring_primitives();
        assert!(r2.contains(&RingPrimitive::RingCount(2)));

        let r6 = SmartsPattern::parse("[r6]").unwrap().ring_primitives();
        assert!(r6.contains(&RingPrimitive::Sized(6)));

        let x2 = SmartsPattern::parse("[x2]").unwrap().ring_primitives();
        assert!(x2.contains(&RingPrimitive::RingBondCount(2)));
    }
}
