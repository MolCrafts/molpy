//! Build a concrete [`SmilesIr`] from an [`Atomistic`] graph.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::core::NeighborKinds;
use crate::io::smiles::bond_stereo;
use crate::io::smiles::chirality::{
    Substituent, chirality_in, emission_order, opensmiles_order, reference_order,
};
use crate::io::smiles::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};
use crate::io::smiles::{Notation, SmilesError, SmilesErrorKind};
use crate::line_notation::ast::*;
use crate::line_notation::atom_keys::{self, H_COUNT, ISOTOPE, STEREO, WILDCARD_Z};
use crate::perceive::priority_substituent;
use molrs::core::Atomistic;
use molrs::core::Element;
use molrs::core::NodeId;
use molrs::core::Props;
use molrs::core::RelationId;
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

/// A protium atom that a bracket `H` can name.
///
/// Isotope, charge, or a non-zero atom map cannot be written as that count,
/// so those hydrogens stay atoms. An absent isotope or map is the same as zero.
fn is_bare_protium(mol: &Atomistic, id: NodeId) -> bool {
    let Ok(atom) = mol.get_atom(id) else {
        return false;
    };
    atom.atomic_number() == Some(Element::H.z())
        && atom.formal_charge() == 0
        && atom.get_int(ISOTOPE).is_none_or(|n| n == 0)
        && atom
            .get_int(molrs::core::keys::ATOM_MAP)
            .is_none_or(|n| n == 0)
}

/// Bare protium neighbors `id` will not draw as atoms.
fn omitted_protium_count(mol: &Atomistic, id: NodeId, opts: &SmilesEmitOptions) -> u8 {
    if matches!(opts.hydrogens, HydrogenEmit::ExplicitAll) {
        return 0;
    }
    let count = mol
        .neighbors(id, NeighborKinds::BONDS)
        .filter(|(neighbor, _)| is_bare_protium(mol, *neighbor))
        .count();
    u8::try_from(count).unwrap_or(u8::MAX)
}

fn connected_components(mol: &Atomistic, include_h: bool) -> Vec<Vec<NodeId>> {
    let mut seen = HashSet::new();
    let mut comps = Vec::new();
    for (id, _) in mol.atoms() {
        if !include_h && is_bare_protium(mol, id) {
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
                if !include_h && is_bare_protium(mol, nb) {
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
    let plan = slash_forward(mol, opts);
    let tree = build_tree(mol, &comp_set, root, opts, include_h, &plan)?;
    build_chain_from(mol, root, &tree, opts, &plan)
}

fn build_tree(
    mol: &Atomistic,
    comp: &HashSet<NodeId>,
    root: NodeId,
    opts: &SmilesEmitOptions,
    include_h: bool,
    plan: &HashMap<RelationId, bool>,
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
            .filter(|nb| include_h || !is_bare_protium(mol, *nb))
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
            // Each digit carries the character in that atom's outgoing direction,
            // so the two ends of a directional ring bond are opposite characters.
            let forward = bond_kind_for(mol, bid, opts, a, b, plan)?;
            let backward = bond_kind_for(mol, bid, opts, b, a, plan)?;
            tree.rings.entry(a).or_default().push((rnum, forward));
            tree.rings.entry(b).or_default().push((rnum, backward));
        }
    }

    Ok(tree)
}

fn bond_kind_for(
    mol: &Atomistic,
    bid: RelationId,
    opts: &SmilesEmitOptions,
    from: NodeId,
    to: NodeId,
    plan: &HashMap<RelationId, bool>,
) -> Result<Option<BondKind>, SmilesError> {
    let bt = mol.bond_type(bid).map_err(|e| emit_err(e.to_string()))?;
    let number = mol.bond_number(bid).map_err(|e| emit_err(e.to_string()))?;
    let chemical = match opts.aromatic {
        AromaticEmit::AsMarked if bt.is_aromatic() => Some(BondKind::Aromatic),
        AromaticEmit::KekuleOnly if bt.is_aromatic() => {
            // Prefer bond_number if set
            match number {
                BondNumber::Single => Some(BondKind::Single),
                BondNumber::Double => Some(BondKind::Double),
                BondNumber::Triple => Some(BondKind::Triple),
                BondNumber::Quadruple => Some(BondKind::Quadruple),
                BondNumber::Unknown => {
                    return Err(emit_err(
                        "KekuleOnly requires integer bond_number on aromatic bonds",
                    ));
                }
            }
        }
        _ => match bt {
            BondOrder::Single => None, // default single omitted
            BondOrder::Double => Some(BondKind::Double),
            BondOrder::Triple => Some(BondKind::Triple),
            BondOrder::Aromatic => Some(BondKind::Aromatic),
            BondOrder::Unknown => match number {
                BondNumber::Double => Some(BondKind::Double),
                BondNumber::Triple => Some(BondKind::Triple),
                BondNumber::Quadruple => Some(BondKind::Quadruple),
                _ => None,
            },
        },
    };
    // A double, triple or aromatic bond is the chemical class. `/` and `\`
    // are a direction on a single bond. The plan carries a stored `"up"` /
    // `"down"` and, where the double bond stores `E` / `Z`, the slash derived
    // for its priority substituent.
    if chemical.is_some() || !opts.include_stereo {
        return Ok(chemical);
    }
    let Some(&forward_up) = plan.get(&bid) else {
        return Ok(chemical);
    };
    let bond = mol.get_bond(bid).map_err(|e| emit_err(e.to_string()))?;
    let stored = if forward_up { "up" } else { "down" };
    Ok(bond_stereo::kind_along(
        stored,
        from,
        to,
        [bond.nodes[0], bond.nodes[1]],
    ))
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
    plan: &HashMap<RelationId, bool>,
) -> Result<Chain, SmilesError> {
    let parent = tree.parent.get(&atom).copied().flatten();
    let partners = ring_partners(tree, atom);
    let children = tree.children.get(&atom).cloned().unwrap_or_default();
    let (stereo, fold_hydrogen) = emitted_stereo(mol, atom, parent, &partners, &children, opts)?;
    let head = atom_node(mol, atom, opts, stereo, fold_hydrogen)?;
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

    if children.is_empty() {
        return Ok(Chain { head, tail });
    }

    // All but last child are branches; last continues the main chain
    let (branches, last) = children.split_at(children.len().saturating_sub(1));
    for &ch in branches {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts, atom, ch, plan))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts, plan)?;
        tail.push(ChainElement::Branch {
            bond: bond.map(BondQuery::Kind),
            chain: sub,
            span: Span::new(0, 0),
        });
    }
    if let Some(&ch) = last.first() {
        let bond = find_bond(mol, atom, ch)
            .map(|bid| bond_kind_for(mol, bid, opts, atom, ch, plan))
            .transpose()?
            .flatten();
        let sub = build_chain_from(mol, ch, tree, opts, plan)?;
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

/// `/` while traversing endpoint 0 toward endpoint 1, for every single bond
/// this string should draw as a slash.
///
/// A stored `"up"` / `"down"` is that bit. A simple alkene's `E` / `Z` assigns
/// the bit of each priority substituent: the earlier double-bond end is `/`
/// outward when its bond is still free, and the other end matches for `Z` and
/// disagrees for `E`. A substituent bond already claimed by an earlier alkene
/// keeps its bit, and the free end follows. Cumulene ends stay on the stored
/// slash and are not reassigned.
fn slash_forward(mol: &Atomistic, opts: &SmilesEmitOptions) -> HashMap<RelationId, bool> {
    let mut forward = HashMap::new();
    if !opts.include_stereo {
        return forward;
    }
    for (bid, rel) in mol.bonds() {
        match rel.props.get_str(STEREO) {
            Some("up") => {
                forward.insert(bid, true);
            }
            Some("down") => {
                forward.insert(bid, false);
            }
            _ => {}
        }
    }
    let mut alkenes = Vec::new();
    for (bid, rel) in mol.bonds() {
        if mol.bond_type(bid).ok() != Some(BondOrder::Double) {
            continue;
        }
        let same_side = match rel.props.get_str(STEREO) {
            Some("Z") => true,
            Some("E") => false,
            _ => continue,
        };
        let (a, b) = (rel.nodes[0], rel.nodes[1]);
        if other_double(mol, a, b) || other_double(mol, b, a) {
            continue;
        }
        let rank = atom_index(mol, a).min(atom_index(mol, b));
        alkenes.push((rank, a, b, same_side));
    }
    alkenes.sort_by_key(|row| row.0);
    for (_, a, b, same_side) in alkenes {
        assign_alkene_slashes(mol, &mut forward, a, b, same_side);
    }
    forward
}

fn assign_alkene_slashes(
    mol: &Atomistic,
    forward: &mut HashMap<RelationId, bool>,
    a: NodeId,
    b: NodeId,
    same_side: bool,
) {
    let (earlier, later) = if atom_index(mol, a) <= atom_index(mol, b) {
        (a, b)
    } else {
        (b, a)
    };
    let Some(rep_earlier) = priority_substituent(mol, &other_neighbors(mol, earlier, later)) else {
        return;
    };
    let Some(rep_later) = priority_substituent(mol, &other_neighbors(mol, later, earlier)) else {
        return;
    };
    let Some(bond_earlier) = find_bond(mol, earlier, rep_earlier) else {
        return;
    };
    let Some(bond_later) = find_bond(mol, later, rep_later) else {
        return;
    };
    if bond_earlier == bond_later {
        return;
    }
    let up_earlier = up_from_plan(mol, forward, earlier, rep_earlier, bond_earlier);
    let up_later = up_from_plan(mol, forward, later, rep_later, bond_later);
    match (up_earlier, up_later) {
        (None, None) => {
            set_up_from(mol, forward, earlier, rep_earlier, bond_earlier, true);
            set_up_from(mol, forward, later, rep_later, bond_later, same_side);
        }
        (Some(up), None) => {
            set_up_from(
                mol,
                forward,
                later,
                rep_later,
                bond_later,
                if same_side { up } else { !up },
            );
        }
        (None, Some(up)) => {
            set_up_from(
                mol,
                forward,
                earlier,
                rep_earlier,
                bond_earlier,
                if same_side { up } else { !up },
            );
        }
        (Some(_), Some(_)) => {}
    }
}

fn up_from_plan(
    mol: &Atomistic,
    forward: &HashMap<RelationId, bool>,
    carbon: NodeId,
    substituent: NodeId,
    bid: RelationId,
) -> Option<bool> {
    let forward_up = *forward.get(&bid)?;
    let rel = mol.get_bond(bid).ok()?;
    if rel.nodes[0] == carbon && rel.nodes[1] == substituent {
        Some(forward_up)
    } else if rel.nodes[1] == carbon && rel.nodes[0] == substituent {
        Some(!forward_up)
    } else {
        None
    }
}

fn set_up_from(
    mol: &Atomistic,
    forward: &mut HashMap<RelationId, bool>,
    carbon: NodeId,
    substituent: NodeId,
    bid: RelationId,
    up: bool,
) {
    let Ok(rel) = mol.get_bond(bid) else {
        return;
    };
    let forward_up = if rel.nodes[0] == carbon && rel.nodes[1] == substituent {
        up
    } else {
        !up
    };
    forward.insert(bid, forward_up);
}

fn other_double(mol: &Atomistic, atom: NodeId, partner: NodeId) -> bool {
    mol.neighbors(atom, NeighborKinds::BONDS)
        .any(|(neighbor, bid)| {
            neighbor != partner && mol.bond_type(bid).ok() == Some(BondOrder::Double)
        })
}

fn other_neighbors(mol: &Atomistic, atom: NodeId, partner: NodeId) -> Vec<NodeId> {
    mol.neighbors(atom, NeighborKinds::BONDS)
        .map(|(neighbor, _)| neighbor)
        .filter(|neighbor| *neighbor != partner)
        .collect()
}

fn atom_index(mol: &Atomistic, id: NodeId) -> usize {
    mol.atoms()
        .position(|(atom, _)| atom == id)
        .unwrap_or(usize::MAX)
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

/// Ring-closure partners of `atom` in the order its digits are written.
fn ring_partners(tree: &Tree, atom: NodeId) -> Vec<NodeId> {
    let Some(rings) = tree.rings.get(&atom) else {
        return Vec::new();
    };
    rings
        .iter()
        .filter_map(|(rnum, _)| {
            tree.rings.iter().find_map(|(other, listed)| {
                (*other != atom && listed.iter().any(|(n, _)| n == rnum)).then_some(*other)
            })
        })
        .collect()
}

/// `@` / `@@` for the order this string writes, or `None` when stereo is off
/// or the written substituents are not the reference four.
///
/// The graph stores the clock sense of the reference order. The glyph is that
/// sense in the OpenSMILES order of this traversal. A lone bare protium this
/// policy skips is written as the bracket `H` and stands in the implicit
/// hydrogen's slot. Any other set of substituents that is not the reference
/// four omits the marker: copying it would name a clock those substituents
/// do not have.
///
/// The bool reports that hydrogen fold, so the bracket states `H` rather than
/// the stored count of zero.
fn emitted_stereo(
    mol: &Atomistic,
    id: NodeId,
    parent: Option<NodeId>,
    ring_partners: &[NodeId],
    children: &[NodeId],
    opts: &SmilesEmitOptions,
) -> Result<(Option<Chirality>, bool), SmilesError> {
    if !opts.include_stereo {
        return Ok((None, false));
    }
    let stored = {
        let atom = mol.get_atom(id).map_err(|e| emit_err(e.to_string()))?;
        atom.get_str(STEREO).and_then(stored_chirality)
    };
    let Some(stored) = stored else {
        return Ok((None, false));
    };
    if let Some(glyph) =
        folded_lone_hydrogen(mol, id, parent, ring_partners, children, stored, opts)
    {
        return Ok((Some(glyph), true));
    }
    let Some(reference) = reference_order(mol, id) else {
        return Ok((None, false));
    };
    let Some(emission) = emission_order(mol, id, parent, ring_partners, children) else {
        return Ok((None, false));
    };
    Ok((chirality_in(stored, &reference, &emission), false))
}

/// The clock sense after folding one skipped explicit hydrogen into the
/// bracket `H`, when that fold names the same four substituents.
///
/// The hydrogen is explicit (`h_count` 0, four bond neighbors) and this
/// policy does not write it as an atom. It is bonded only to `id`. Two
/// omitted hydrogens, or a hydrogen that also bonds elsewhere, are not one
/// implicit hydrogen: there is no order to invent.
fn folded_lone_hydrogen(
    mol: &Atomistic,
    id: NodeId,
    parent: Option<NodeId>,
    ring_partners: &[NodeId],
    children: &[NodeId],
    stored: Chirality,
    opts: &SmilesEmitOptions,
) -> Option<Chirality> {
    if matches!(opts.hydrogens, HydrogenEmit::ExplicitAll) {
        return None;
    }
    let neighbors: Vec<NodeId> = mol
        .neighbors(id, NeighborKinds::BONDS)
        .map(|(nb, _)| nb)
        .collect();
    if neighbors.len() != 4 {
        return None;
    }
    let hydrogens: Vec<NodeId> = neighbors
        .into_iter()
        .filter(|nb| is_bare_protium(mol, *nb))
        .collect();
    if hydrogens.len() != 1 {
        return None;
    }
    let hydrogen = hydrogens[0];
    let mut degree = 0;
    for (other, _) in mol.neighbors(hydrogen, NeighborKinds::BONDS) {
        degree += 1;
        if other != id {
            return None;
        }
    }
    if degree != 1 {
        return None;
    }
    let mut reference = reference_order(mol, id)?;
    let mut found = false;
    for slot in &mut reference {
        if *slot == Substituent::Atom(hydrogen) {
            *slot = Substituent::Hydrogen;
            found = true;
            break;
        }
    }
    if !found {
        return None;
    }
    let mut heavy = Vec::with_capacity(3);
    if let Some(parent) = parent {
        heavy.push(parent);
    }
    heavy.extend_from_slice(ring_partners);
    heavy.extend_from_slice(children);
    let emission = opensmiles_order(parent.is_some(), &heavy, 1)?;
    chirality_in(stored, &reference, &emission)
}

/// `"CW"` / `"CCW"` as a clock sense. Anything else, including the SMILES
/// glyphs `"@"` and `"@@"`, is not a stored tetrahedral label.
fn stored_chirality(label: &str) -> Option<Chirality> {
    match label {
        "CW" => Some(Chirality::Clockwise),
        "CCW" => Some(Chirality::CounterClockwise),
        _ => None,
    }
}

fn atom_node(
    mol: &Atomistic,
    id: NodeId,
    opts: &SmilesEmitOptions,
    stereo: Option<Chirality>,
    fold_hydrogen: bool,
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
    // The reader fills no count for a bare atom and reads an omitted bracket
    // count as zero, so a stored count means the atom was stated in brackets
    // and goes back into them. `OrganicSubset` omits the count only on an
    // atom that states none, and spells a stored zero as the bare bracket.
    let stored_h = int_prop::<u8>(&atom, id, H_COUNT)?;
    let omitted = omitted_protium_count(mol, id, opts);
    let (hcount, states_h) = match opts.hydrogens {
        // An atom that states no count has no hydrogen *atoms* to fold in
        // either: `ExplicitAll` writes every H as its own atom.
        HydrogenEmit::ExplicitAll => (Some(stored_h.unwrap_or(0)), true),
        HydrogenEmit::OrganicSubset | HydrogenEmit::AsStored => {
            let total = stored_h.unwrap_or(0).saturating_add(omitted);
            let stated = stored_h.is_some() || omitted > 0;
            if !stated {
                (None, false)
            } else if total == 0 {
                // A stored zero with nothing omitted is the bare bracket, not `H0`.
                (None, true)
            } else {
                (Some(total), true)
            }
        }
    };
    // The skipped lone explicit hydrogen is the bracket H of the implicit
    // order. The stored count is zero, because the hydrogen was an atom.
    let (hcount, states_h) = if fold_hydrogen {
        (Some(1), true)
    } else {
        (hcount, states_h)
    };

    let need_bracket = !opts.organic_subset
        || !organic_subset_ok(&sym)
        || charge.is_some()
        || isotope.is_some()
        || states_h
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

    /// A bracket atom states its hydrogen count exactly and the reader fills
    /// none for a bare atom, so a stored count keeps its bracket under the
    /// default policy: the Daylight omission is for atoms that state none.
    #[test]
    fn a_stored_hydrogen_count_keeps_its_bracket() {
        let opts = SmilesEmitOptions::default();
        for (src, want) in [("[NH4+]", "[NH4+]"), ("[C]", "[C]")] {
            let mol = SmilesIr::parse(src).unwrap().to_atomistic().unwrap();
            assert_eq!(smiles(&mol, &opts).unwrap(), want, "src={src}");
        }
        let methyl = SmilesIr::parse("[CH3]C").unwrap().to_atomistic().unwrap();
        let s = smiles(&methyl, &opts).unwrap();
        assert!(s.contains("[CH3]"), "the stated count was dropped: {s}");
        let pyrrole = SmilesIr::parse("c1cc[nH]c1")
            .unwrap()
            .to_atomistic()
            .unwrap();
        let s = smiles(&pyrrole, &opts).unwrap();
        assert!(s.contains("[nH]"), "pyrrole lost its N-H: {s}");
        let chiral = SmilesIr::parse("[C@@H](F)(Cl)Br")
            .unwrap()
            .to_atomistic()
            .unwrap();
        let stereo = SmilesEmitOptions {
            include_stereo: true,
            ..SmilesEmitOptions::default()
        };
        let s = smiles(&chiral, &stereo).unwrap();
        assert!(s.contains("H]"), "the stereocentre lost its H: {s}");
    }

    #[test]
    fn drawn_hydrogens_stay_in_the_bracket_count() {
        let mol = SmilesIr::parse("[CH3]C").unwrap().to_atomistic().unwrap();
        let mol = crate::perceive::add_hydrogens(&mol).unwrap();
        let text = smiles(&mol, &SmilesEmitOptions::default()).unwrap();
        assert!(
            !text.contains("[C]"),
            "drawn hydrogens collapsed to a bare bracket: {text}"
        );
        let back = SmilesIr::parse(&text).unwrap().to_atomistic().unwrap();
        for (_, atom) in back.atoms() {
            if atom.atomic_number() != Some(6) {
                continue;
            }
            assert_ne!(
                atom.get_int(H_COUNT),
                Some(0),
                "{text} read a carbon back as a bare bracket"
            );
        }
    }

    #[test]
    fn a_deuterium_is_not_folded_into_a_bracket_hydrogen() {
        let lone = SmilesIr::parse("[2H]").unwrap().to_atomistic().unwrap();
        assert_eq!(
            smiles(&lone, &SmilesEmitOptions::default()).unwrap(),
            "[2H]"
        );

        let mol = SmilesIr::parse("N[C@]([2H])(O)C")
            .unwrap()
            .to_atomistic()
            .unwrap();
        let text = smiles(
            &mol,
            &SmilesEmitOptions {
                include_stereo: true,
                canonical: false,
                ..SmilesEmitOptions::default()
            },
        )
        .unwrap();
        assert!(text.contains("2H"), "the deuterium was folded away: {text}");
        let back = SmilesIr::parse(&text).unwrap().to_atomistic().unwrap();
        assert_eq!(back.n_atoms(), mol.n_atoms(), "{text}");
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

    fn parse(smiles: &str) -> Atomistic {
        SmilesIr::parse(smiles)
            .unwrap_or_else(|err| panic!("{smiles}: {err}"))
            .to_atomistic()
            .unwrap_or_else(|err| panic!("{smiles}: {err}"))
    }

    fn rooted(root: NodeId) -> SmilesEmitOptions {
        SmilesEmitOptions {
            canonical: false,
            include_stereo: true,
            root: Some(root),
            ..SmilesEmitOptions::default()
        }
    }

    fn write_at(mol: &Atomistic, root: NodeId) -> String {
        smiles(mol, &rooted(root)).unwrap()
    }

    /// Trans when the one double bond is `E`, or when a cumulene's two
    /// terminal slashes have opposite up-ness from their carbons.
    fn is_trans(mol: &Atomistic) -> bool {
        let mut geometry = Vec::new();
        for (bid, rel) in mol.bonds() {
            if mol.bond_type(bid).ok() != Some(BondOrder::Double) {
                continue;
            }
            match rel.props.get_str(STEREO) {
                Some("E") => geometry.push(true),
                Some("Z") => geometry.push(false),
                _ => {}
            }
        }
        if geometry.len() == 1 {
            return geometry[0];
        }
        let mut upness = Vec::new();
        for (_, rel) in mol.bonds() {
            let forward_up = match rel.props.get_str(STEREO) {
                Some("up") => true,
                Some("down") => false,
                _ => continue,
            };
            let (a, b) = (rel.nodes[0], rel.nodes[1]);
            let double = |id: NodeId| {
                mol.neighbors(id, NeighborKinds::BONDS)
                    .any(|(_, other)| mol.bond_type(other).ok() == Some(BondOrder::Double))
            };
            let (carbon, _) = match (double(a), double(b)) {
                (true, false) => (a, b),
                (false, true) => (b, a),
                _ => continue,
            };
            let up_from_carbon = if carbon == a { forward_up } else { !forward_up };
            upness.push(up_from_carbon);
        }
        assert_eq!(upness.len(), 2, "expected one pair of directional bonds");
        upness[0] != upness[1]
    }

    /// Cis/trans of every double bond. `true` is trans: stored `E`, or opposite
    /// up-ness of the marked neighbors when the bond still carries slashes.
    fn alkene_bits(mol: &Atomistic) -> Vec<bool> {
        let mut bits = Vec::new();
        for (bid, rel) in mol.bonds() {
            if mol.bond_type(bid).unwrap() != BondOrder::Double {
                continue;
            }
            match rel.props.get_str(STEREO) {
                Some("E") => {
                    bits.push(true);
                    continue;
                }
                Some("Z") => {
                    bits.push(false);
                    continue;
                }
                _ => {}
            }
            let (c1, c2) = (rel.nodes[0], rel.nodes[1]);
            let Some(u1) = marked_upness(mol, c1, c2) else {
                continue;
            };
            let Some(u2) = marked_upness(mol, c2, c1) else {
                continue;
            };
            bits.push(u1 != u2);
        }
        bits.sort_unstable();
        bits
    }

    /// Up-ness of the one marked neighbor of `carbon` other than `partner`.
    ///
    /// `"up"` is `/` from the bond's first endpoint toward its second.
    fn marked_upness(mol: &Atomistic, carbon: NodeId, partner: NodeId) -> Option<bool> {
        let mut found = None;
        for (nb, bid) in mol.neighbors(carbon, NeighborKinds::BONDS) {
            if nb == partner {
                continue;
            }
            let bond = mol.get_bond(bid).unwrap();
            let forward_up = match bond.props.get_str(STEREO) {
                Some("up") => true,
                Some("down") => false,
                _ => continue,
            };
            let up = if bond.nodes[0] == carbon {
                forward_up
            } else {
                !forward_up
            };
            assert!(found.is_none(), "two marked neighbors on {carbon:?}");
            found = Some(up);
        }
        found
    }

    #[test]
    fn directional_bonds_round_trip_the_cis_trans_configuration() {
        let trans = ["F/C=C/F", "F\\C=C\\F", "C(\\F)=C/F"];
        let cis = ["F/C=C\\F", "C(/F)=C/F"];
        for src in trans {
            assert!(is_trans(&parse(src)), "{src} parsed as cis");
        }
        for src in cis {
            assert!(!is_trans(&parse(src)), "{src} parsed as trans");
        }
        assert!(is_trans(&parse("F/C=C=C=C/F")));
        assert!(!is_trans(&parse("F/C=C=C=C\\F")));

        let branched = parse("C(\\F)=C/F");
        let ids: Vec<NodeId> = branched.atoms().map(|(id, _)| id).collect();
        for root in [ids[0], ids[3]] {
            let text = write_at(&branched, root);
            assert!(is_trans(&parse(&text)), "root {root:?} wrote {text}");
        }

        let cumulene = parse("F/C=C=C=C/F");
        for (id, _) in cumulene.atoms() {
            let text = write_at(&cumulene, id);
            assert!(is_trans(&parse(&text)), "root {id:?} wrote {text}");
        }

        let silent = smiles(&branched, &SmilesEmitOptions::default()).unwrap();
        assert!(!silent.contains('/'), "{silent}");
        assert!(!silent.contains('\\'), "{silent}");
    }

    #[test]
    fn a_directional_ring_keeps_each_double_bond() {
        let src = "C/1=C/C=C\\C=C/C=C\\1";
        let mol = parse(src);
        let want = alkene_bits(&mol);
        assert_eq!(want.len(), 4, "{src} marked {want:?}");
        for (id, _) in mol.atoms() {
            let text = write_at(&mol, id);
            let again = parse(&text);
            assert_eq!(alkene_bits(&again), want, "root {id:?} wrote {text}");
        }

        // The spanning tree closes the ring on a plain bond. The directional
        // bond stays `/` from its first endpoint toward its second.
        let pentagon = parse("C/1CCCC\\1");
        let first = pentagon.atoms().next().unwrap().0;
        assert_eq!(write_at(&pentagon, first), "C(C1)/CCC1");
        let conflict = SmilesIr::parse("C/1CCCC/1").unwrap().to_atomistic();
        assert!(conflict.is_err());
    }
}
