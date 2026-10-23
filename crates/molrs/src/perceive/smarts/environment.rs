//! An atom's local environment as a SMARTS pattern —
//! [`SmartsPattern::from_environment`](super::SmartsPattern::from_environment).

use std::collections::{HashMap, VecDeque};

use crate::core::NeighborKinds;
use crate::line_notation::ast::*;
use crate::line_notation::atom_keys;
use crate::line_notation::error::{Notation, SmilesError, SmilesErrorKind};
use crate::perceive::perceive_rings;
use molrs::core::Atomistic;
use molrs::core::BondOrder;
use molrs::core::Element;
use molrs::core::NodeId;
use molrs::core::Props;

/// Build a query [`SmilesIr`] for `center` with the given options.
///
/// Bound on pattern bond depth: with default flags and [`NeighborStyle::Chain`],
/// `SmartsPattern::max_bond_depth() <= reach` (k = 0). Recursive style may add
/// nested depth equal to the recursive fragment depth (still ≤ `reach`).
pub(crate) fn environment_ir(
    mol: &Atomistic,
    center: NodeId,
    opts: &EnvironmentOptions,
) -> Result<SmilesIr, SmilesError> {
    if opts.reach < 1 {
        return Err(emit_err("reach must be >= 1"));
    }
    if mol.get_atom(center).is_err() {
        return Err(emit_err("center atom not in molecule"));
    }

    let rings = perceive_rings(mol);

    // BFS ball of radius reach
    let mut depth: HashMap<NodeId, u32> = HashMap::new();
    let mut q = VecDeque::new();
    depth.insert(center, 0);
    q.push_back(center);
    while let Some(cur) = q.pop_front() {
        let d = depth[&cur];
        if d >= opts.reach {
            continue;
        }
        for (nb, _) in mol.neighbors(cur, NeighborKinds::BONDS) {
            if !opts.include_explicit_h_atoms && is_h(mol, nb) {
                continue;
            }
            if depth.contains_key(&nb) {
                continue;
            }
            depth.insert(nb, d + 1);
            q.push_back(nb);
        }
    }

    let chain = match opts.neighbor_style {
        NeighborStyle::Chain => build_chain_env(mol, center, &depth, opts, &rings)?,
        NeighborStyle::Recursive => build_recursive_env(mol, center, &depth, opts, &rings)?,
    };

    Ok(SmilesIr {
        components: vec![chain],
        span: Span::new(0, 0),
    })
}

fn emit_err(msg: impl Into<String>) -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::Emit(msg.into()),
        Span::new(0, 0),
        "",
        Notation::Smarts,
    )
}

/// Whether atom `id` is a hydrogen (`Z = 1`).
fn is_h(mol: &Atomistic, id: NodeId) -> bool {
    mol.get_atom(id)
        .is_ok_and(|a| a.atomic_number() == Some(Element::H.z()))
}

/// The element primitive of atom `id`: `[#Z]` when `as_atomic_number`,
/// otherwise its element symbol (aromatic when `aromatic`), or `*` for the
/// wildcard atom (`Z = 0`), which names no element.
///
/// # Errors
///
/// An emit error when the atom carries no valid
/// [`keys::ATOMIC_NUMBER`](molrs::core::keys::ATOMIC_NUMBER): the species is
/// read from Z alone, never from an `element` symbol.
fn species_query(
    atom: &Props,
    id: NodeId,
    as_atomic_number: bool,
    aromatic: bool,
) -> Result<AtomQuery, SmilesError> {
    let z = atom.atomic_number().ok_or_else(|| {
        emit_err(format!(
            "atom {id:?} has no valid '{}'; a symbol-only graph is converted with \
             Element::fill_atomic_numbers first",
            molrs::core::keys::ATOMIC_NUMBER
        ))
    })?;
    let primitive = if as_atomic_number {
        AtomPrimitive::AtomicNumber(z)
    } else {
        match Element::by_number(z) {
            Some(element) => AtomPrimitive::Element {
                symbol: element.symbol().to_owned(),
                aromatic,
            },
            None => AtomPrimitive::Wildcard,
        }
    };
    Ok(AtomQuery::Primitive(primitive))
}

fn center_query(
    mol: &Atomistic,
    id: NodeId,
    opts: &EnvironmentOptions,
    rings: &crate::perceive::RingSet,
) -> Result<AtomQuery, SmilesError> {
    let mut prims: Vec<AtomQuery> = Vec::new();

    let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
    let aromatic = opts.include_aromatic && atom_keys::is_aromatic(&atom);
    prims.push(species_query(&atom, id, opts.atomic_number, aromatic)?);

    if opts.include_aromatic && !opts.atomic_number {
        // already folded into element aromatic flag
    } else if aromatic {
        prims.push(AtomQuery::Primitive(AtomPrimitive::Aromatic));
    }

    if opts.include_degree {
        let d = mol.neighbors(id, NeighborKinds::BONDS).count() as u8;
        prims.push(AtomQuery::Primitive(AtomPrimitive::Degree(d)));
    }

    if opts.include_h_count {
        let h = count_h(mol, id)?;
        prims.push(AtomQuery::Primitive(AtomPrimitive::HCount(h)));
    }

    if opts.include_charge
        && let Ok(c) = i8::try_from(atom.formal_charge())
        && c != 0
    {
        prims.push(AtomQuery::Primitive(AtomPrimitive::Charge(c)));
    }

    if opts.include_ring_membership {
        if rings.is_atom_in_ring(id) {
            let n = rings.n_atom_rings(id) as u8;
            prims.push(AtomQuery::Primitive(AtomPrimitive::RingMembership(Some(
                n.max(1),
            ))));
        } else {
            prims.push(AtomQuery::Primitive(AtomPrimitive::RingMembership(Some(0))));
        }
    }

    if opts.include_ring_size
        && let Some(sz) = rings.smallest_ring_containing_atom(id)
    {
        prims.push(AtomQuery::Primitive(AtomPrimitive::RingSize(sz as u8)));
    }

    Ok(if prims.len() == 1 {
        prims.pop().unwrap()
    } else {
        AtomQuery::LowAnd(prims)
    })
}

/// The hydrogen count of atom `id`: drawn and implicit
/// ([`n_hydrogens`](crate::perceive::n_hydrogens)), the count the SMARTS
/// `H<n>` primitive matches.
///
/// # Errors
///
/// An emit error when the count does not fit the `0..=255` a bracket atom
/// can write.
fn count_h(mol: &Atomistic, id: NodeId) -> Result<u8, SmilesError> {
    let h = crate::perceive::n_hydrogens(mol, id);
    u8::try_from(h)
        .map_err(|_| emit_err(format!("atom {id:?}: {h} hydrogens is not a bracket count")))
}

fn bond_query(
    mol: &Atomistic,
    a: NodeId,
    b: NodeId,
    opts: &EnvironmentOptions,
) -> Option<BondQuery> {
    if !opts.include_bond_orders {
        return None;
    }
    let bid = mol
        .neighbors(a, NeighborKinds::BONDS)
        .find(|(nb, _)| *nb == b)?
        .1;
    match mol.bond_type(bid).expect("neighbors yields live bonds") {
        BondOrder::Double => Some(BondQuery::Kind(BondKind::Double)),
        BondOrder::Triple => Some(BondQuery::Kind(BondKind::Triple)),
        BondOrder::Aromatic => Some(BondQuery::Kind(BondKind::Aromatic)),
        _ => None, // default single omitted
    }
}

fn ordered_neighbors(
    mol: &Atomistic,
    id: NodeId,
    depth: &HashMap<NodeId, u32>,
    opts: &EnvironmentOptions,
    parent: Option<NodeId>,
) -> Vec<NodeId> {
    let d0 = depth[&id];
    let mut nbs: Vec<NodeId> = mol
        .neighbors(id, NeighborKinds::BONDS)
        .map(|(nb, _)| nb)
        .filter(|nb| Some(*nb) != parent)
        .filter(|nb| depth.get(nb).is_some_and(|d| *d == d0 + 1))
        .filter(|nb| opts.include_explicit_h_atoms || !is_h(mol, *nb))
        .collect();

    if opts.canonical_neighbor_order {
        let order = mol.canonical_order();
        let rank: HashMap<NodeId, usize> =
            order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        nbs.sort_by_key(|id| rank.get(id).copied().unwrap_or(usize::MAX));
    } else {
        nbs.sort_by_key(|id| u64::from(*id));
    }
    nbs
}

fn leaf_atom_query(
    mol: &Atomistic,
    id: NodeId,
    opts: &EnvironmentOptions,
) -> Result<AtomQuery, SmilesError> {
    let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
    let aromatic = opts.include_aromatic && atom_keys::is_aromatic(&atom);
    species_query(&atom, id, opts.atomic_number, aromatic)
}

fn build_chain_env(
    mol: &Atomistic,
    center: NodeId,
    depth: &HashMap<NodeId, u32>,
    opts: &EnvironmentOptions,
    rings: &crate::perceive::RingSet,
) -> Result<Chain, SmilesError> {
    fn rec(
        mol: &Atomistic,
        id: NodeId,
        parent: Option<NodeId>,
        depth: &HashMap<NodeId, u32>,
        opts: &EnvironmentOptions,
        rings: &crate::perceive::RingSet,
        is_center: bool,
    ) -> Result<Chain, SmilesError> {
        let q = if is_center {
            center_query(mol, id, opts, rings)?
        } else {
            leaf_atom_query(mol, id, opts)?
        };
        let head = AtomNode {
            spec: AtomSpec::Query(q),
            span: Span::new(0, 0),
            descriptors: Vec::new(),
        };
        let nbs = ordered_neighbors(mol, id, depth, opts, parent);
        if nbs.is_empty() {
            return Ok(Chain { head, tail: vec![] });
        }
        let mut tail = Vec::new();
        let (branches, last) = nbs.split_at(nbs.len().saturating_sub(1));
        for &ch in branches {
            let bond = bond_query(mol, id, ch, opts);
            let sub = rec(mol, ch, Some(id), depth, opts, rings, false)?;
            tail.push(ChainElement::Branch {
                bond,
                chain: sub,
                span: Span::new(0, 0),
            });
        }
        if let Some(&ch) = last.first() {
            let bond = bond_query(mol, id, ch, opts);
            let sub = rec(mol, ch, Some(id), depth, opts, rings, false)?;
            tail.push(ChainElement::BondedAtom {
                bond,
                atom: sub.head,
            });
            tail.extend(sub.tail);
        }
        Ok(Chain { head, tail })
    }

    rec(mol, center, None, depth, opts, rings, true)
}

fn build_recursive_env(
    mol: &Atomistic,
    center: NodeId,
    depth: &HashMap<NodeId, u32>,
    opts: &EnvironmentOptions,
    rings: &crate::perceive::RingSet,
) -> Result<Chain, SmilesError> {
    // Centre with LowAnd of primitives + Recursive fragments for each neighbour path
    let mut prims: Vec<AtomQuery> = match center_query(mol, center, opts, rings)? {
        AtomQuery::LowAnd(ps) => ps,
        other => vec![other],
    };

    for nb in ordered_neighbors(mol, center, depth, opts, None) {
        // Build a chain starting with bond? Recursive SMARTS is a molecule:
        // neighbour atom as head of a chain of remaining depth.
        let mut sub_opts = opts.clone();
        // recursive fragment uses remaining depth
        if sub_opts.reach > 1 {
            sub_opts.reach -= 1;
        }
        let sub_chain = {
            // re-depth relative to nb with remaining reach
            let mut dmap = HashMap::new();
            let mut q = VecDeque::new();
            dmap.insert(nb, 0u32);
            q.push_back(nb);
            let max_d = opts.reach.saturating_sub(1);
            while let Some(cur) = q.pop_front() {
                let d = dmap[&cur];
                if d >= max_d {
                    continue;
                }
                for (n2, _) in mol.neighbors(cur, NeighborKinds::BONDS) {
                    if n2 == center {
                        continue;
                    }
                    if !opts.include_explicit_h_atoms && is_h(mol, n2) {
                        continue;
                    }
                    if dmap.contains_key(&n2) {
                        continue;
                    }
                    dmap.insert(n2, d + 1);
                    q.push_back(n2);
                }
            }
            // small chain from nb
            let leaf = leaf_atom_query(mol, nb, opts)?;
            Chain {
                head: AtomNode {
                    spec: AtomSpec::Query(leaf),
                    span: Span::new(0, 0),
                    descriptors: Vec::new(),
                },
                tail: vec![],
            }
        };
        let rec_ir = SmilesIr {
            components: vec![sub_chain],
            span: Span::new(0, 0),
        };
        prims.push(AtomQuery::Primitive(AtomPrimitive::Recursive(Box::new(
            rec_ir,
        ))));
    }

    let q = if prims.len() == 1 {
        prims.pop().unwrap()
    } else {
        AtomQuery::LowAnd(prims)
    };

    Ok(Chain {
        head: AtomNode {
            spec: AtomSpec::Query(q),
            span: Span::new(0, 0),
            descriptors: Vec::new(),
        },
        tail: vec![],
    })
}

/// How [`SmartsPattern::from_environment`](super::SmartsPattern::from_environment) encodes the neighbours of the
/// centre atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborStyle {
    /// Spanning-tree branches from the centre (chain form).
    Chain,
    /// Centre atom with recursive `$(...)` environments.
    Recursive,
}

/// Options for [`SmartsPattern::from_environment`](super::SmartsPattern::from_environment): how much of an atom's
/// environment the generated pattern states.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvironmentOptions {
    /// Bond depth from the centre atom. Must be >= 1.
    pub reach: u32,
    /// Encode centre as `[#Z]` rather than element symbol.
    pub atomic_number: bool,
    pub include_degree: bool,
    pub include_h_count: bool,
    pub include_charge: bool,
    pub include_aromatic: bool,
    /// Emit ring membership, `[R]` / `[Rn]`, on each atom.
    pub include_ring_membership: bool,
    pub include_ring_size: bool,
    pub include_explicit_h_atoms: bool,
    pub include_bond_orders: bool,
    pub neighbor_style: NeighborStyle,
    pub canonical_neighbor_order: bool,
}

impl Default for EnvironmentOptions {
    fn default() -> Self {
        Self {
            reach: 1,
            atomic_number: true,
            include_degree: true,
            include_h_count: true,
            include_charge: true,
            include_aromatic: true,
            include_ring_membership: false,
            include_ring_size: false,
            include_explicit_h_atoms: false,
            include_bond_orders: true,
            neighbor_style: NeighborStyle::Chain,
            canonical_neighbor_order: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perceive::smarts::SmartsPattern;

    fn heavy_atoms(mol: &Atomistic) -> Vec<NodeId> {
        mol.atoms()
            .filter(|(id, _)| !is_h(mol, *id))
            .map(|(id, _)| id)
            .collect()
    }

    #[test]
    fn local_smarts_matches_center() {
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let opts = EnvironmentOptions::default();
        for center in heavy_atoms(&mol) {
            for reach in [1u32, 2] {
                let o = EnvironmentOptions {
                    reach,
                    ..opts.clone()
                };
                let s = SmartsPattern::from_environment(&mol, center, &o)
                    .unwrap()
                    .to_string();
                let pat = SmartsPattern::parse(&s).expect(&s);
                assert!(
                    pat.has_match(&mol, crate::perceive::smarts::MatchOptions::default()),
                    "pattern {s} should match CCO"
                );
            }
        }
    }

    #[test]
    fn flags_change_string() {
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let center = heavy_atoms(&mol)[0];
        let a = EnvironmentOptions {
            atomic_number: true,
            ..Default::default()
        };
        let b = EnvironmentOptions {
            atomic_number: false,
            ..Default::default()
        };
        let sa = SmartsPattern::from_environment(&mol, center, &a)
            .unwrap()
            .to_string();
        let sb = SmartsPattern::from_environment(&mol, center, &b)
            .unwrap()
            .to_string();
        assert!(sa.contains('#'), "{sa}");
        assert_ne!(sa, sb);

        let c = EnvironmentOptions {
            include_degree: false,
            ..Default::default()
        };
        let sc = SmartsPattern::from_environment(&mol, center, &c)
            .unwrap()
            .to_string();
        assert!(!sc.contains('D'), "{sc}");
    }

    #[test]
    fn max_bond_depth_bounded_by_reach() {
        let mol = (SmilesIr::parse("CCCC").unwrap()).to_atomistic().unwrap();
        let center = heavy_atoms(&mol)[0];
        for reach in [1u32, 2, 3] {
            let o = EnvironmentOptions {
                reach,
                neighbor_style: NeighborStyle::Chain,
                ..Default::default()
            };
            let s = SmartsPattern::from_environment(&mol, center, &o)
                .unwrap()
                .to_string();
            let pat = SmartsPattern::parse(&s).expect(&s);
            assert!(
                pat.max_bond_depth() as u32 <= reach,
                "reach={reach} depth={} s={s}",
                pat.max_bond_depth()
            );
        }
    }

    #[test]
    fn recursive_style_parses() {
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let center = heavy_atoms(&mol)[1]; // middle C
        let o = EnvironmentOptions {
            neighbor_style: NeighborStyle::Recursive,
            reach: 1,
            ..Default::default()
        };
        let s = SmartsPattern::from_environment(&mol, center, &o)
            .unwrap()
            .to_string();
        assert!(s.contains("$(") || s.contains('#'), "{s}");
        SmartsPattern::parse(&s).expect(&s);
    }
}
