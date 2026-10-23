//! Build a concrete [`SmilesIr`] from an [`Atomistic`] graph.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::NeighborKinds;
use crate::io::smiles::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::ast::*;
use crate::line_notation::atom_keys::{self, H_COUNT, ISOTOPE, STEREO, WILDCARD_Z};
use molrs::core::Atomistic;
use molrs::core::Element;
use molrs::core::NodeId;
use molrs::core::Props;
use molrs::core::{BondNumber, BondOrder};

impl SmilesIr {
    /// The concrete SMILES IR of a molecular graph, written as `opts` says
    /// (canonical order, root, aromatic and hydrogen policy, components).
    pub fn from_atomistic(
        mol: &Atomistic,
        opts: &SmilesEmitOptions,
    ) -> Result<SmilesIr, SmilesError> {
        ir_from_atomistic(mol, opts)
    }
}

fn ir_from_atomistic(mol: &Atomistic, opts: &SmilesEmitOptions) -> Result<SmilesIr, SmilesError> {
    if mol.n_atoms() == 0 {
        return Err(emit_err("empty molecule"));
    }

    let skip_h = matches!(
        opts.hydrogens,
        HydrogenEmit::OrganicSubset | HydrogenEmit::AsStored
    ) && opts.hydrogens != HydrogenEmit::ExplicitAll;

    let include_h_atoms = matches!(opts.hydrogens, HydrogenEmit::ExplicitAll);

    let mut components = connected_components(mol, include_h_atoms);
    if components.is_empty() {
        return Err(emit_err("no atoms to emit"));
    }

    // Sort components for stability when canonical: by size, then by the
    // sorted symbols of their atoms.
    if opts.canonical {
        let mut keyed = components
            .into_iter()
            .map(|comp| {
                let mut labels = comp
                    .iter()
                    .map(|&id| {
                        let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
                        Ok(smiles_symbol(species(&atom, id)?))
                    })
                    .collect::<Result<Vec<_>, SmilesError>>()?;
                labels.sort_unstable();
                Ok(((comp.len(), labels), comp))
            })
            .collect::<Result<Vec<_>, SmilesError>>()?;
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        components = keyed.into_iter().map(|(_, comp)| comp).collect();
    }

    match opts.multi_component {
        MultiComponentEmit::ErrorIfMultiple if components.len() > 1 => {
            return Err(emit_err(format!(
                "molecule has {} components; set multi_component=JoinDot or FirstOnly",
                components.len()
            )));
        }
        MultiComponentEmit::FirstOnly => {
            if let Some(root) = opts.root {
                components.retain(|c| c.contains(&root));
                if components.is_empty() {
                    return Err(emit_err("root atom not found in any component"));
                }
            } else {
                components.truncate(1);
            }
        }
        MultiComponentEmit::JoinDot | MultiComponentEmit::ErrorIfMultiple => {}
    }

    let mut chains = Vec::with_capacity(components.len());
    for comp in &components {
        let root = choose_root(mol, comp, opts)?;
        let chain = emit_component(mol, comp, root, opts, include_h_atoms, skip_h)?;
        chains.push(chain);
    }

    Ok(SmilesIr {
        components: chains,
        span: Span::new(0, 0),
    })
}

fn emit_err(msg: impl Into<String>) -> SmilesError {
    SmilesError::new(
        SmilesErrorKind::Emit(msg.into()),
        Span::new(0, 0),
        "",
        Notation::Smiles,
    )
}

/// The atomic number of `atom` (`id` names it in the error).
///
/// # Errors
///
/// [`SmilesErrorKind::Emit`] when the atom carries no valid
/// [`keys::ATOMIC_NUMBER`](molrs::core::keys::ATOMIC_NUMBER) — the writer reads
/// the species from Z alone, never from an `element` symbol.
fn species(atom: &Props, id: NodeId) -> Result<u8, SmilesError> {
    atom.atomic_number().ok_or_else(|| {
        emit_err(format!(
            "atom {id:?} has no valid '{}'; a symbol-only graph is converted with \
             Element::fill_atomic_numbers before writing SMILES",
            molrs::core::keys::ATOMIC_NUMBER
        ))
    })
}

/// The element symbol SMILES writes for atomic number `z`: `*` for the
/// wildcard, `Z = 0`.
///
/// `z` comes from [`Props::atomic_number`], which guarantees `0..=118`, so
/// every non-wildcard value names an element.
fn smiles_symbol(z: u8) -> &'static str {
    match Element::by_number(z) {
        Some(element) => element.symbol(),
        None => "*",
    }
}

/// Whether atom `id` is a hydrogen (`Z = 1`).
fn is_hydrogen(mol: &Atomistic, id: NodeId) -> bool {
    mol.get_atom(id)
        .is_ok_and(|a| a.atomic_number() == Some(Element::H.z()))
}

fn connected_components(mol: &Atomistic, include_h: bool) -> Vec<Vec<NodeId>> {
    let mut seen = HashSet::new();
    let mut comps = Vec::new();
    for (id, _) in mol.atoms() {
        if !include_h && is_hydrogen(mol, id) {
            continue;
        }
        if !seen.insert(id) {
            continue;
        }
        let mut q = VecDeque::new();
        let mut comp = Vec::new();
        q.push_back(id);
        while let Some(cur) = q.pop_front() {
            comp.push(cur);
            for (nb, _) in mol.neighbors(cur, NeighborKinds::BONDS) {
                if !include_h && is_hydrogen(mol, nb) {
                    continue;
                }
                if seen.insert(nb) {
                    q.push_back(nb);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

fn choose_root(
    mol: &Atomistic,
    comp: &[NodeId],
    opts: &SmilesEmitOptions,
) -> Result<NodeId, SmilesError> {
    if let Some(r) = opts.root {
        if comp.contains(&r) {
            return Ok(r);
        }
        return Err(emit_err("root atom not in component"));
    }
    if opts.canonical {
        let order = mol.canonical_order();
        for id in order {
            if comp.contains(&id) && !is_hydrogen(mol, id) {
                return Ok(id);
            }
        }
    }
    // Prefer non-H, lowest handle order in component.
    let mut ids = comp.to_vec();
    ids.sort_by_key(|id| u64::from(*id));
    ids.into_iter()
        .find(|&id| !is_hydrogen(mol, id))
        .or_else(|| comp.first().copied())
        .ok_or_else(|| emit_err("empty component"))
}

#[derive(Default)]
struct Tree {
    /// parent[child] = Some(parent); root has None
    parent: HashMap<NodeId, Option<NodeId>>,
    children: HashMap<NodeId, Vec<NodeId>>,
    /// ring closures: atom → list of (rnum, optional bond kind for write)
    rings: HashMap<NodeId, Vec<(u16, Option<BondKind>)>>,
}

fn emit_component(
    mol: &Atomistic,
    comp: &[NodeId],
    root: NodeId,
    opts: &SmilesEmitOptions,
    include_h: bool,
    _skip_h_implicit: bool,
) -> Result<Chain, SmilesError> {
    let comp_set: HashSet<NodeId> = comp.iter().copied().collect();
    let tree = build_tree(mol, &comp_set, root, opts, include_h)?;
    build_chain_from(mol, root, &tree, opts)
}

fn build_tree(
    mol: &Atomistic,
    comp: &HashSet<NodeId>,
    root: NodeId,
    opts: &SmilesEmitOptions,
    include_h: bool,
) -> Result<Tree, SmilesError> {
    let mut tree = Tree::default();
    let mut parent_edge: HashMap<NodeId, NodeId> = HashMap::new();
    let mut stack = vec![root];
    tree.parent.insert(root, None);
    tree.children.insert(root, Vec::new());

    // DFS to build spanning tree
    let mut visited = HashSet::new();
    visited.insert(root);

    while let Some(cur) = stack.pop() {
        let mut nbs: Vec<NodeId> = mol
            .neighbors(cur, NeighborKinds::BONDS)
            .map(|(nb, _)| nb)
            .filter(|nb| comp.contains(nb))
            .filter(|nb| include_h || !is_hydrogen(mol, *nb))
            .collect();

        if opts.canonical {
            let order = mol.canonical_order();
            let rank: HashMap<NodeId, usize> =
                order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
            nbs.sort_by_key(|id| rank.get(id).copied().unwrap_or(usize::MAX));
        } else {
            nbs.sort_by_key(|id| u64::from(*id));
        }

        for nb in nbs {
            if visited.insert(nb) {
                tree.parent.insert(nb, Some(cur));
                tree.children.entry(cur).or_default().push(nb);
                tree.children.entry(nb).or_default();
                parent_edge.insert(nb, cur);
                stack.push(nb);
            }
        }
    }

    // Non-tree edges → ring digits
    let mut next_rnum: u16 = 1;
    let mut assigned_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
    for &a in comp {
        if !include_h && is_hydrogen(mol, a) {
            continue;
        }
        for (b, bid) in mol.neighbors(a, NeighborKinds::BONDS) {
            if !comp.contains(&b) {
                continue;
            }
            if !include_h && is_hydrogen(mol, b) {
                continue;
            }
            let edge = if u64::from(a) < u64::from(b) {
                (a, b)
            } else {
                (b, a)
            };
            if !assigned_edges.insert(edge) {
                continue;
            }
            // tree edge?
            let is_tree = parent_edge.get(&b) == Some(&a) || parent_edge.get(&a) == Some(&b);
            if is_tree {
                continue;
            }
            let rnum = next_rnum;
            next_rnum += 1;
            if next_rnum > 99 {
                return Err(emit_err("too many ring closures (>99)"));
            }
            let bk = bond_kind_for(mol, bid, opts)?;
            tree.rings.entry(a).or_default().push((rnum, bk));
            tree.rings.entry(b).or_default().push((rnum, bk));
        }
    }

    Ok(tree)
}

fn bond_kind_for(
    mol: &Atomistic,
    bid: molrs::core::RelationId,
    opts: &SmilesEmitOptions,
) -> Result<Option<BondKind>, SmilesError> {
    let bt = mol.bond_type(bid).map_err(|e| emit_err(e.to_string()))?;
    let number = mol.bond_number(bid).map_err(|e| emit_err(e.to_string()))?;
    match opts.aromatic {
        AromaticEmit::AsMarked if bt.is_aromatic() => Ok(Some(BondKind::Aromatic)),
        AromaticEmit::KekuleOnly if bt.is_aromatic() => {
            // Prefer bond_number if set
            match number {
                BondNumber::Single => Ok(Some(BondKind::Single)),
                BondNumber::Double => Ok(Some(BondKind::Double)),
                BondNumber::Triple => Ok(Some(BondKind::Triple)),
                BondNumber::Quadruple => Ok(Some(BondKind::Quadruple)),
                BondNumber::Unknown => Err(emit_err(
                    "KekuleOnly requires integer bond_number on aromatic bonds",
                )),
            }
        }
        _ => match bt {
            BondOrder::Single => Ok(None), // default single omitted
            BondOrder::Double => Ok(Some(BondKind::Double)),
            BondOrder::Triple => Ok(Some(BondKind::Triple)),
            BondOrder::Aromatic => Ok(Some(BondKind::Aromatic)),
            BondOrder::Unknown => match number {
                BondNumber::Double => Ok(Some(BondKind::Double)),
                BondNumber::Triple => Ok(Some(BondKind::Triple)),
                BondNumber::Quadruple => Ok(Some(BondKind::Quadruple)),
                _ => Ok(None),
            },
        },
    }
}

fn find_bond(mol: &Atomistic, a: NodeId, b: NodeId) -> Option<molrs::core::RelationId> {
    mol.neighbors(a, NeighborKinds::BONDS)
        .find(|(nb, _)| *nb == b)
        .map(|(_, bid)| bid)
}

fn build_chain_from(
    mol: &Atomistic,
    atom: NodeId,
    tree: &Tree,
    opts: &SmilesEmitOptions,
) -> Result<Chain, SmilesError> {
    let head = atom_node(mol, atom, opts)?;
    let mut tail = Vec::new();

    // Ring closures on this atom first (Daylight: after atom, before branches)
    if let Some(rings) = tree.rings.get(&atom) {
        for (rnum, bk) in rings {
            tail.push(ChainElement::RingClosure {
                bond: bk.map(BondQuery::Kind),
                rnum: *rnum,
                span: Span::new(0, 0),
            });
        }
    }

    let children = tree.children.get(&atom).cloned().unwrap_or_default();
    if children.is_empty() {
        return Ok(Chain { head, tail });
    }

    // All but last child are branches; last continues the main chain
    let (branches, last) = children.split_at(children.len().saturating_sub(1));
    for &ch in branches {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts)?;
        tail.push(ChainElement::Branch {
            bond: bond.map(BondQuery::Kind),
            chain: sub,
            span: Span::new(0, 0),
        });
    }
    if let Some(&ch) = last.first() {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts)?;
        // Flatten: bond + head of sub + sub.tail into main chain
        tail.push(ChainElement::BondedAtom {
            bond: bond.map(BondQuery::Kind),
            atom: sub.head,
        });
        tail.extend(sub.tail);
    }

    Ok(Chain { head, tail })
}

/// The integer prop `key` of `atom`, if it states one.
///
/// # Errors
///
/// [`SmilesErrorKind::Emit`] when the prop is present but is not an integer
/// that fits `T` — a hydrogen count of `-1` or `2.5` is not written as
/// something else.
fn int_prop<T: TryFrom<i32>>(
    atom: &Props,
    id: NodeId,
    key: &str,
) -> Result<Option<T>, SmilesError> {
    let Some(value) = atom.get(key) else {
        return Ok(None);
    };
    atom.get_int(key)
        .and_then(|v| T::try_from(v).ok())
        .map(Some)
        .ok_or_else(|| {
            emit_err(format!(
                "atom {id:?}: '{key}' = {value:?} is not a writable integer"
            ))
        })
}

fn organic_subset_ok(sym: &str) -> bool {
    matches!(
        sym,
        "B" | "C"
            | "N"
            | "O"
            | "P"
            | "S"
            | "F"
            | "Cl"
            | "Br"
            | "I"
            | "b"
            | "c"
            | "n"
            | "o"
            | "p"
            | "s"
    )
}

fn atom_node(
    mol: &Atomistic,
    id: NodeId,
    opts: &SmilesEmitOptions,
) -> Result<AtomNode, SmilesError> {
    let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
    let z = species(&atom, id)?;
    let sym = smiles_symbol(z).to_owned();

    let aromatic = matches!(opts.aromatic, AromaticEmit::AsMarked) && atom_keys::is_aromatic(&atom);
    let charge = i8::try_from(atom.formal_charge()).map_err(|_| {
        emit_err(format!(
            "atom {id:?}: formal charge {} does not fit a SMILES charge",
            atom.formal_charge()
        ))
    })?;
    let charge = (charge != 0).then_some(charge);
    let isotope = int_prop::<u16>(&atom, id, ISOTOPE)?;
    let hcount = match opts.hydrogens {
        HydrogenEmit::OrganicSubset => None,
        // An atom that states no count has no hydrogen *atoms* to fold in
        // either: `ExplicitAll` writes every H as its own atom.
        HydrogenEmit::ExplicitAll => Some(int_prop::<u8>(&atom, id, H_COUNT)?.unwrap_or(0)),
        HydrogenEmit::AsStored => int_prop::<u8>(&atom, id, H_COUNT)?,
    };
    let stereo = if opts.include_stereo {
        atom.get_str(STEREO).and_then(|s| match s {
            "CW" | "@@" => Some(Chirality::Clockwise),
            "CCW" | "@" => Some(Chirality::CounterClockwise),
            _ => None,
        })
    } else {
        None
    };

    let need_bracket = !opts.organic_subset
        || !organic_subset_ok(&sym)
        || charge.is_some()
        || isotope.is_some()
        || hcount.is_some()
        || stereo.is_some()
        || z == WILDCARD_Z;

    let spec = if need_bracket {
        AtomSpec::Bracket {
            isotope,
            symbol: BracketSymbol::Element {
                symbol: sym,
                aromatic,
            },
            chirality: stereo,
            hcount,
            charge,
            atom_class: None,
        }
    } else {
        AtomSpec::Organic {
            symbol: sym,
            aromatic,
        }
    };

    Ok(AtomNode {
        spec,
        span: Span::new(0, 0),
        descriptors: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::line_notation::fixtures::atom_nodes;

    /// The molecule as SMILES text: its IR, written.
    fn smiles(mol: &Atomistic, opts: &SmilesEmitOptions) -> Result<String, SmilesError> {
        SmilesIr::from_atomistic(mol, opts).map(|ir| ir.to_string())
    }

    #[test]
    fn ethanol_round_trip_stable() {
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let opts = SmilesEmitOptions::default();
        let s1 = smiles(&mol, &opts).unwrap();
        let mol2 = (SmilesIr::parse(&s1).unwrap()).to_atomistic().unwrap();
        let s2 = smiles(&mol2, &opts).unwrap();
        assert_eq!(s1, s2);
    }

    #[test]
    fn a_partial_charge_is_not_written_as_a_formal_charge() {
        // `charge` is the partial charge; only `formal_charge` is written.
        let mut mol = (SmilesIr::parse("[Na]").unwrap()).to_atomistic().unwrap();
        let id = mol.atoms().next().unwrap().0;
        mol.set_atom(id, molrs::core::keys::CHARGE, 1.0).unwrap();
        let opts = SmilesEmitOptions::default();
        assert_eq!(smiles(&mol, &opts).unwrap(), "[Na]");
        mol.set_atom(id, molrs::core::keys::FORMAL_CHARGE, 1)
            .unwrap();
        assert_eq!(smiles(&mol, &opts).unwrap(), "[Na+]");
    }

    #[test]
    fn acetic_and_benzene() {
        let opts = SmilesEmitOptions::default();
        for src in ["CC(=O)O", "c1ccccc1"] {
            let mol = (SmilesIr::parse(src).unwrap()).to_atomistic().unwrap();
            let s = smiles(&mol, &opts).unwrap();
            assert!(SmilesIr::parse(&s).is_ok(), "src={src} wrote={s}");
            let mol2 = (SmilesIr::parse(&s).unwrap()).to_atomistic().unwrap();
            let s2 = smiles(&mol2, &opts).unwrap();
            assert_eq!(s, s2);
        }
    }

    #[test]
    fn multi_component_error_and_join() {
        let mol = (SmilesIr::parse("CCO.O").unwrap()).to_atomistic().unwrap();
        let err_opts = SmilesEmitOptions::default();
        assert!(smiles(&mol, &err_opts).is_err());

        let join = SmilesEmitOptions {
            multi_component: MultiComponentEmit::JoinDot,
            ..Default::default()
        };
        let s = smiles(&mol, &join).unwrap();
        assert!(s.contains('.'), "got {s}");
    }

    #[test]
    fn hydrogens_flag_changes_output() {
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let a = SmilesEmitOptions {
            hydrogens: HydrogenEmit::OrganicSubset,
            ..Default::default()
        };
        let b = SmilesEmitOptions {
            hydrogens: HydrogenEmit::ExplicitAll,
            ..Default::default()
        };
        // ExplicitAll still has no H atoms in graph from to_atomistic, but
        // may force brackets with H0 — at least options path runs.
        let sa = smiles(&mol, &a).unwrap();
        let sb = smiles(&mol, &b).unwrap();
        assert!(!sa.is_empty());
        assert!(!sb.is_empty());
        // Bracket form under ExplicitAll for every atom
        assert!(sb.contains('['), "explicit expected brackets: {sb}");
    }

    #[test]
    fn no_stereo_when_flag_false() {
        // Bracket stereo marker if present on graph; default flag must not invent @.
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let opts = SmilesEmitOptions::default(); // include_stereo = false
        let s = smiles(&mol, &opts).unwrap();
        assert!(!s.contains('@'), "got {s}");
    }

    #[test]
    fn emitted_nodes_carry_no_descriptors() {
        // An `Atomistic` never stored bonding descriptors, so the IR built from
        // one has none to emit — the graph → IR direction stays plain SMILES.
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let ir = SmilesIr::from_atomistic(&mol, &SmilesEmitOptions::default()).unwrap();
        for node in atom_nodes(&ir) {
            assert!(node.descriptors.is_empty(), "node {:?}", node.spec);
        }
    }

    #[test]
    fn ethanol_writes_the_unchanged_smiles_string() {
        // The descriptor field must not perturb what the plain writer emits.
        // Written in input order (not canonical), so the expected string is
        // the walk from the first atom and does not hang on the canonical
        // ranking's hash values.
        let mol = (SmilesIr::parse("CCO").unwrap()).to_atomistic().unwrap();
        let opts = SmilesEmitOptions {
            canonical: false,
            ..SmilesEmitOptions::default()
        };
        let ir = SmilesIr::from_atomistic(&mol, &opts).unwrap();
        assert_eq!(
            crate::line_notation::writer::smiles_text(&ir).unwrap(),
            "CCO"
        );
    }
}
