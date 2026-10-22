//! SMARTS syntax → matcher query graph.
//!
//! The SMARTS text is parsed by the crate's one line-notation grammar; this
//! module compiles its [`SmilesIr`] into the [`QueryGraph`] the matcher walks:
//! element symbols resolve to atomic numbers, `$(...)` subpatterns compile to
//! their own graphs, ring closures become bonds and an unwritten bond becomes
//! "single or aromatic".
//!
//! The matcher evaluates every primitive the parser reads except chirality,
//! isotopes, implicit-H counts and valence, and it matches one connected
//! pattern: those, the quadruple and directional bonds, and a `.`-separated
//! pattern are refused here with an error naming them.

use std::collections::HashMap;

use crate::core::MolRsError;
use crate::line_notation::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BracketSymbol, Chain,
    ChainElement, SmilesIr,
};
use crate::line_notation::parser::parse_smarts;

use super::predicate::{AtomPredicate, AtomTest, BondPredicate, BondTest};

/// A compiled query atom: its query tree + optional atom-map label (`:n`).
#[derive(Debug, Clone)]
pub struct QueryAtom {
    pub query: AtomTest,
    pub map_label: Option<u32>,
}

/// A compiled query bond between two query-atom indices.
#[derive(Debug, Clone)]
pub struct QueryBond {
    pub a: usize,
    pub b: usize,
    pub query: BondTest,
}

/// The whole compiled query graph. Atoms are numbered in the order the
/// pattern writes them, so atom 0 is the root and every later atom is bonded
/// to an earlier one.
#[derive(Debug, Clone, Default)]
pub struct QueryGraph {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryBond>,
    /// Compiled recursive subpatterns, addressed by `AtomTest::Recursive(i)`.
    pub recursives: Vec<QueryGraph>,
}

impl QueryGraph {
    /// Collect every `%LABEL` context-label appearing anywhere in this query
    /// graph, including inside recursive `$(...)` subpatterns. Order is the
    /// traversal order; duplicates are kept (callers dedup as needed).
    pub fn context_labels(&self) -> Vec<String> {
        let mut out = Vec::new();
        for atom in &self.atoms {
            atom.query.collect_context_labels(&mut out);
        }
        for sub in &self.recursives {
            out.extend(sub.context_labels());
        }
        out
    }

    /// See [`super::SmartsPattern::ring_primitives`].
    pub fn ring_primitives(&self) -> Vec<super::RingPrimitive> {
        let mut out = Vec::new();
        for atom in &self.atoms {
            atom.query.collect_ring_primitives(&mut out);
        }
        for sub in &self.recursives {
            out.extend(sub.ring_primitives());
        }
        out
    }

    /// See [`super::SmartsPattern::max_bond_depth`].
    pub fn max_bond_depth(&self) -> usize {
        let n = self.atoms.len();
        if n <= 1 {
            return 0;
        }
        let mut adj = vec![Vec::new(); n];
        for b in &self.bonds {
            if b.a < n && b.b < n {
                adj[b.a].push(b.b);
                adj[b.b].push(b.a);
            }
        }
        let mut best = 0usize;
        for start in 0..n {
            let mut dist = vec![None; n];
            let mut q = std::collections::VecDeque::new();
            dist[start] = Some(0usize);
            q.push_back(start);
            while let Some(u) = q.pop_front() {
                let du = dist[u].expect("visited");
                for &v in &adj[u] {
                    if dist[v].is_none() {
                        dist[v] = Some(du + 1);
                        q.push_back(v);
                    }
                }
            }
            for d in dist.into_iter().flatten() {
                best = best.max(d);
            }
        }
        best
    }
}

/// Parse `smarts` with the one SMARTS parser and compile it into a
/// [`QueryGraph`].
///
/// # Errors
///
/// [`MolRsError::parse`] carrying the parser's message for a syntax error, or
/// naming the construct for a pattern the matcher cannot evaluate (see the
/// module docs).
pub fn compile(smarts: &str) -> Result<QueryGraph, MolRsError> {
    let ir = parse_smarts(smarts).map_err(|e| MolRsError::parse(e.to_string()))?;
    Compiler { src: smarts }.graph(&ir)
}

struct Compiler<'s> {
    src: &'s str,
}

/// An open ring closure: the atom that opened it and the bond written there.
type OpenRing<'ir> = (usize, Option<&'ir BondQuery>);

impl Compiler<'_> {
    fn err(&self, msg: impl std::fmt::Display) -> MolRsError {
        MolRsError::parse(format!("{msg} in SMARTS '{}'", self.src))
    }

    fn graph(&self, ir: &SmilesIr) -> Result<QueryGraph, MolRsError> {
        let [component] = ir.components.as_slice() else {
            return Err(self
                .err("a '.'-separated pattern is not one connected query; match each component"));
        };
        let mut g = QueryGraph::default();
        let mut rings: HashMap<u16, OpenRing<'_>> = HashMap::new();
        self.chain(&mut g, component, None, &mut rings)?;
        if !rings.is_empty() {
            return Err(self.err("unclosed ring bond"));
        }
        Ok(g)
    }

    /// Compile `chain`; its head bonds to `parent` (an atom and the bond
    /// written before the branch) when there is one.
    fn chain<'ir>(
        &self,
        g: &mut QueryGraph,
        chain: &'ir Chain,
        parent: Option<(usize, Option<&'ir BondQuery>)>,
        rings: &mut HashMap<u16, OpenRing<'ir>>,
    ) -> Result<(), MolRsError> {
        let head = self.atom(g, &chain.head)?;
        if let Some((p, bond)) = parent {
            self.bond(g, p, head, bond)?;
        }
        let mut cur = head;
        for element in &chain.tail {
            match element {
                ChainElement::BondedAtom { bond, atom } => {
                    let next = self.atom(g, atom)?;
                    self.bond(g, cur, next, bond.as_ref())?;
                    cur = next;
                }
                ChainElement::Branch { bond, chain, .. } => {
                    self.chain(g, chain, Some((cur, bond.as_ref())), rings)?;
                }
                ChainElement::RingClosure { bond, rnum, .. } => match rings.remove(rnum) {
                    Some((other, open_bond)) => {
                        if other == cur {
                            return Err(self.err("ring bond to self"));
                        }
                        // Either end may spell the bond; the closing one wins.
                        self.bond(g, other, cur, bond.as_ref().or(open_bond))?;
                    }
                    None => {
                        rings.insert(*rnum, (cur, bond.as_ref()));
                    }
                },
            }
        }
        Ok(())
    }

    fn bond(
        &self,
        g: &mut QueryGraph,
        a: usize,
        b: usize,
        bond: Option<&BondQuery>,
    ) -> Result<(), MolRsError> {
        let query = match bond {
            None => BondTest::Prim(BondPredicate::SingleOrAromatic),
            Some(q) => self.bond_query(q)?,
        };
        g.bonds.push(QueryBond { a, b, query });
        Ok(())
    }

    fn bond_query(&self, q: &BondQuery) -> Result<BondTest, MolRsError> {
        Ok(match q {
            BondQuery::Kind(kind) => BondTest::Prim(match kind {
                BondKind::Single => BondPredicate::Single,
                BondKind::Double => BondPredicate::Double,
                BondKind::Triple => BondPredicate::Triple,
                BondKind::Aromatic => BondPredicate::Aromatic,
                BondKind::Any => BondPredicate::Any,
                BondKind::Ring => BondPredicate::InRing,
                BondKind::Quadruple | BondKind::Up | BondKind::Down => {
                    return Err(
                        self.err(format!("the {kind:?} bond is not supported by the matcher"))
                    );
                }
            }),
            BondQuery::Not(inner) => BondTest::Not(Box::new(self.bond_query(inner)?)),
            BondQuery::And(parts) => BondTest::And(self.bond_queries(parts)?),
            BondQuery::Or(parts) => BondTest::Or(self.bond_queries(parts)?),
        })
    }

    fn bond_queries(&self, parts: &[BondQuery]) -> Result<Vec<BondTest>, MolRsError> {
        parts.iter().map(|p| self.bond_query(p)).collect()
    }

    /// Append the query atom `node` compiles to; returns its index.
    fn atom(&self, g: &mut QueryGraph, node: &AtomNode) -> Result<usize, MolRsError> {
        let mut map_label = None;
        let query = match &node.spec {
            AtomSpec::Organic { symbol, aromatic } => self.element(symbol, *aromatic)?,
            AtomSpec::Wildcard => AtomTest::Prim(AtomPredicate::Any),
            AtomSpec::Bracket {
                isotope: None,
                symbol,
                chirality: None,
                hcount: None,
                charge: None,
                atom_class: None,
            } => match symbol {
                BracketSymbol::Element { symbol, aromatic } => self.element(symbol, *aromatic)?,
                BracketSymbol::Any => AtomTest::Prim(AtomPredicate::Any),
                BracketSymbol::Aliphatic => AtomTest::Prim(AtomPredicate::AnyAliphatic),
                BracketSymbol::Aromatic => AtomTest::Prim(AtomPredicate::AnyAromatic),
            },
            AtomSpec::Bracket { .. } => {
                return Err(self.err("a SMILES bracket atom is not a SMARTS query"));
            }
            AtomSpec::Query(q) => self.atom_query(g, q, &mut map_label)?,
        };
        g.atoms.push(QueryAtom { query, map_label });
        Ok(g.atoms.len() - 1)
    }

    fn element(&self, symbol: &str, aromatic: bool) -> Result<AtomTest, MolRsError> {
        let z = molrs::core::Element::by_symbol(symbol)
            .ok_or_else(|| self.err(format!("unknown element '{symbol}'")))?
            .z();
        Ok(AtomTest::Prim(if aromatic {
            AtomPredicate::AromaticElement(z)
        } else {
            AtomPredicate::AliphaticElement(z)
        }))
    }

    fn atom_query(
        &self,
        g: &mut QueryGraph,
        q: &AtomQuery,
        map_label: &mut Option<u32>,
    ) -> Result<AtomTest, MolRsError> {
        Ok(match q {
            AtomQuery::Primitive(p) => self.primitive(g, p, map_label)?,
            AtomQuery::Not(inner) => AtomTest::Not(Box::new(self.atom_query(g, inner, map_label)?)),
            AtomQuery::And(parts) | AtomQuery::LowAnd(parts) => {
                AtomTest::And(self.atom_queries(g, parts, map_label)?)
            }
            AtomQuery::Or(parts) => AtomTest::Or(self.atom_queries(g, parts, map_label)?),
        })
    }

    fn atom_queries(
        &self,
        g: &mut QueryGraph,
        parts: &[AtomQuery],
        map_label: &mut Option<u32>,
    ) -> Result<Vec<AtomTest>, MolRsError> {
        parts
            .iter()
            .map(|p| self.atom_query(g, p, map_label))
            .collect()
    }

    fn primitive(
        &self,
        g: &mut QueryGraph,
        p: &AtomPrimitive,
        map_label: &mut Option<u32>,
    ) -> Result<AtomTest, MolRsError> {
        let prim = match p {
            AtomPrimitive::Element { symbol, aromatic } => return self.element(symbol, *aromatic),
            AtomPrimitive::AtomicNumber(z) => AtomPredicate::AtomicNum(*z),
            AtomPrimitive::Wildcard => AtomPredicate::Any,
            AtomPrimitive::Aliphatic => AtomPredicate::AnyAliphatic,
            AtomPrimitive::Aromatic => AtomPredicate::AnyAromatic,
            AtomPrimitive::Degree(n) => AtomPredicate::Degree(u32::from(*n)),
            AtomPrimitive::TotalConnections(n) => AtomPredicate::TotalConnections(u32::from(*n)),
            AtomPrimitive::HCount(n) => AtomPredicate::TotalH(u32::from(*n)),
            AtomPrimitive::RingMembership(n) => AtomPredicate::RingMembership(n.map(u32::from)),
            AtomPrimitive::RingSize(n) => AtomPredicate::RingSize(Some(u32::from(*n))),
            AtomPrimitive::RingSizeRange { lo, hi } => AtomPredicate::RingSizeRange {
                lo: u32::from(*lo),
                hi: hi.map(u32::from),
            },
            AtomPrimitive::RingBondCount(n) => AtomPredicate::RingBondCount(u32::from(*n)),
            AtomPrimitive::Charge(c) => AtomPredicate::Charge(i32::from(*c)),
            AtomPrimitive::ContextLabel(label) => AtomPredicate::HasContextLabel(label.clone()),
            AtomPrimitive::AtomClass(n) => {
                // A map label constrains nothing: it labels the atom, and
                // stands in the tree as "any atom" so it composes under AND.
                *map_label = Some(u32::from(*n));
                AtomPredicate::Any
            }
            AtomPrimitive::Recursive(ir) => {
                let sub = self.graph(ir)?;
                g.recursives.push(sub);
                return Ok(AtomTest::Recursive(g.recursives.len() - 1));
            }
            AtomPrimitive::ImplicitH(_)
            | AtomPrimitive::Valence(_)
            | AtomPrimitive::Isotope(_)
            | AtomPrimitive::Chirality(_) => {
                return Err(self.err(format!(
                    "the primitive {p:?} is not supported by the matcher"
                )));
            }
        };
        Ok(AtomTest::Prim(prim))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_chain_has_one_bond_per_link_and_ring_closures_add_bonds() {
        let chain = compile("CCC").unwrap();
        assert_eq!((chain.atoms.len(), chain.bonds.len()), (3, 2));
        let ring = compile("C1CC1").unwrap();
        assert_eq!((ring.atoms.len(), ring.bonds.len()), (3, 3));
        let branched = compile("C(C)(C)C").unwrap();
        assert_eq!((branched.atoms.len(), branched.bonds.len()), (4, 3));
    }

    #[test]
    fn recursive_subpatterns_are_compiled_separately() {
        let q = compile("[$(C=O)]N").unwrap();
        assert_eq!(q.atoms.len(), 2);
        assert_eq!(q.recursives.len(), 1);
        assert_eq!(q.recursives[0].atoms.len(), 2);
    }

    #[test]
    fn malformed_input_is_an_error() {
        assert!(compile("").is_err());
        assert!(compile("C(C").is_err());
        assert!(compile("[C").is_err());
        assert!(compile("C1CC").is_err(), "an unclosed ring closure");
    }

    #[test]
    fn what_the_matcher_cannot_evaluate_is_refused_by_name() {
        for smarts in [
            "C.C",
            "[C@H](F)Cl",
            "[13C]",
            "C$C",
            "C/C=C/C",
            "[Cv4]",
            "[Ch1]",
        ] {
            assert!(compile(smarts).is_err(), "{smarts}");
        }
    }

    #[test]
    fn max_bond_depth_counts_bonds_from_the_first_atom() {
        assert_eq!(compile("C").unwrap().max_bond_depth(), 0);
        assert_eq!(compile("CCCC").unwrap().max_bond_depth(), 3);
        // The depth is the graph diameter: branch tip to branch tip is two bonds.
        assert_eq!(compile("C(C)(C)C").unwrap().max_bond_depth(), 2);
    }

    #[test]
    fn map_labels_land_on_their_atoms() {
        let q = compile("[C:1]-[N;H2:2]").unwrap();
        assert_eq!(q.atoms[0].map_label, Some(1));
        assert_eq!(q.atoms[1].map_label, Some(2));
    }
}
