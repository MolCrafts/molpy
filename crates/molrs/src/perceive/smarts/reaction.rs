//! Daylight reaction-SMARTS (SMIRKS) transform engine.

use std::collections::{HashMap, HashSet};

use crate::core::Atomistic;
use crate::core::Element;
use crate::core::MolRsError;
use crate::core::{BondNumber, BondOrder};
use crate::core::{NodeId, Props, keys};
use crate::props;

use super::SmartsPattern;
use super::compile::QueryGraph;
use super::predicate::SmartsTarget;
use super::predicate::{AtomPredicate, AtomTest, BondPredicate, BondTest};
use crate::perceive::aromaticity::rewrite_aromatic_flags;

/// What [`Reaction::apply_many`] did, one entry per binding, in `bindings`
/// order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReactionBatch {
    /// Per binding: the surviving touched atoms, deduplicated and ordered as
    /// [`Reaction::apply`] returns them.
    pub touched: Vec<Vec<NodeId>>,
    /// Per binding: the atoms the product added, in product creation order.
    /// This is the stable correspondence a compile-first caller uses to replay
    /// annotations onto RHS-added atoms; sorting opaque graph handles cannot
    /// provide it once deleted slots have been reused.
    pub created: Vec<Vec<NodeId>>,
}

// ---------------------------------------------------------------------------
// Query-tree extraction helpers (read-only reads of the chain-01 AST)
// ---------------------------------------------------------------------------

/// The concrete element pinned by an atom query (`C`, `[N:1]`, `[#8]`,
/// aromatic `c`, ...), or `None` for a wildcard / purely-logical query —
/// `[#0]` included, since the wildcard atom names no element.
fn query_element(q: &AtomTest) -> Option<Element> {
    match q {
        AtomTest::Prim(AtomPredicate::AliphaticElement(z))
        | AtomTest::Prim(AtomPredicate::AromaticElement(z))
        | AtomTest::Prim(AtomPredicate::AtomicNum(z)) => Element::by_number(*z),
        AtomTest::And(items) | AtomTest::Or(items) => items.iter().find_map(query_element),
        _ => None,
    }
}

/// An explicit formal charge pinned by an atom query, or `None`.
fn query_charge(q: &AtomTest) -> Option<i32> {
    match q {
        AtomTest::Prim(AtomPredicate::Charge(c)) => Some(*c),
        AtomTest::And(items) | AtomTest::Or(items) => items.iter().find_map(query_charge),
        _ => None,
    }
}

/// The bond a product query builds, as its two facts: `-`→single, `=`→double,
/// `#`→triple, `:`→aromatic-with-no-phase-yet; a default / `~` / ring bond is a
/// single bond. For a logical bond query the first sub-term decides.
///
/// `:` yields `(Aromatic, Unknown)` on purpose — the query declares the product
/// bond delocalized and says nothing about which Kekulé phase it takes.
/// Perception on the product is what decides that.
fn query_bond_class(q: &BondTest) -> (BondOrder, BondNumber) {
    match q {
        BondTest::Prim(BondPredicate::Double) => (BondOrder::Double, BondNumber::Double),
        BondTest::Prim(BondPredicate::Triple) => (BondOrder::Triple, BondNumber::Triple),
        BondTest::Prim(BondPredicate::Aromatic) => (BondOrder::Aromatic, BondNumber::Unknown),
        BondTest::Prim(_) => (BondOrder::Single, BondNumber::Single),
        BondTest::And(items) | BondTest::Or(items) => items
            .first()
            .map(query_bond_class)
            .unwrap_or((BondOrder::Single, BondNumber::Single)),
        BondTest::Not(_) => (BondOrder::Single, BondNumber::Single),
    }
}

/// Per-atom facts read out of a [`QueryGraph`] for the map diff.
struct QueryAtomFacts {
    label: Option<u32>,
    element: Option<Element>,
    charge: Option<i32>,
    /// `(mapped-neighbour label, bond order)` for each bond to a mapped atom.
    attach: Vec<(u32, (BondOrder, BondNumber))>,
}

/// Read the per-atom facts (label, element, charge, mapped-neighbour bonds).
fn analyze_graph(g: &QueryGraph) -> Vec<QueryAtomFacts> {
    let mut facts: Vec<QueryAtomFacts> = g
        .atoms
        .iter()
        .map(|a| QueryAtomFacts {
            label: a.map_label,
            element: query_element(&a.query),
            charge: query_charge(&a.query),
            attach: Vec::new(),
        })
        .collect();
    for b in &g.bonds {
        let order = query_bond_class(&b.query);
        let (la, lb) = (g.atoms[b.a].map_label, g.atoms[b.b].map_label);
        if let Some(l) = la
            && lb.is_none()
        {
            facts[b.b].attach.push((l, order));
        }
        if let Some(l) = lb
            && la.is_none()
        {
            facts[b.a].attach.push((l, order));
        }
    }
    facts
}

/// Canonical `(min, max)` key for an unordered mapped-atom bond.
fn ordered(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

/// Collect the `{(map_a, map_b): (type, number)}` bonds whose *both*
/// endpoints are mapped.
fn collect_mapped_bonds(g: &QueryGraph, out: &mut HashMap<(u32, u32), (BondOrder, BondNumber)>) {
    for b in &g.bonds {
        if let (Some(la), Some(lb)) = (g.atoms[b.a].map_label, g.atoms[b.b].map_label) {
            out.insert(ordered(la, lb), query_bond_class(&b.query));
        }
    }
}

/// Whether two unmapped atoms are the *same* atom across the arrow: identical
/// concrete element and a shared `(mapped-neighbour, bond class)` attachment.
fn atoms_pair(r: &QueryAtomFacts, p: &QueryAtomFacts) -> bool {
    match (&r.element, &p.element) {
        (Some(re), Some(pe)) if re == pe => {}
        _ => return false,
    }
    r.attach
        .iter()
        .any(|&(rl, ro)| p.attach.iter().any(|&(pl, po)| rl == pl && ro == po))
}

// ---------------------------------------------------------------------------
// Transform (compiled map diff)
// ---------------------------------------------------------------------------

/// An endpoint of a bond the transform must form: either a preserved (mapped)
/// atom, resolved through the apply binding, or a freshly added product atom,
/// resolved through the add-atom id map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NodeRef {
    Mapped(u32),
    Added(usize),
}

/// A product atom to create (unmapped, unpaired): its element/charge template
/// and its product-graph index (identity used by [`NodeRef::Added`]).
///
/// The species is an [`Element`], not a bare atomic number: an added atom
/// needs a definite element, and the wildcard (`Z = 0`) is not one.
#[derive(Debug, Clone)]
struct AddAtomSpec {
    product_idx: usize,
    element: Element,
    charge: i32,
}

/// Leaving atoms of one reactant component: the component's mapped pins
/// (`query_idx → label`) anchor a re-match to the apply binding, from which the
/// `delete_idxs` query atoms resolve to molecule atom ids.
#[derive(Debug, Clone)]
struct DeleteSpec {
    component: usize,
    pins: Vec<(usize, u32)>,
    delete_idxs: Vec<usize>,
}

/// A property change on a preserved (mapped) atom.
#[derive(Debug, Clone)]
struct PropDelta {
    label: u32,
    element: Option<Element>,
    charge: Option<i32>,
}

/// The compiled reaction transform: a pure description of the graph edit,
/// independent of any concrete molecule.
#[derive(Debug, Clone)]
pub struct Transform {
    delete: Vec<DeleteSpec>,
    add_atoms: Vec<AddAtomSpec>,
    form_bonds: Vec<(NodeRef, NodeRef, (BondOrder, BondNumber))>,
    break_bonds: Vec<(u32, u32)>,
    set_order: Vec<(u32, u32, (BondOrder, BondNumber))>,
    set_props: Vec<PropDelta>,
}

/// Classification of a product atom.
#[derive(PartialEq, Eq)]
enum Cls {
    Mapped,
    Paired,
    Added,
}

impl Transform {
    /// Compile the LHS↔RHS map diff into a transform (Daylight SMIRKS semantics).
    fn compile(
        reactants: &[SmartsPattern],
        product: &SmartsPattern,
    ) -> Result<Transform, MolRsError> {
        let r_facts: Vec<Vec<QueryAtomFacts>> =
            reactants.iter().map(|p| analyze_graph(&p.graph)).collect();
        let p_facts = analyze_graph(&product.graph);

        let (r_labels, p_labels) = Self::index_maps(&r_facts, &p_facts)?;

        // Pair unmapped atoms; classify the rest as delete / add.
        let mut p_claimed = vec![false; p_facts.len()];
        let mut delete_atoms: Vec<(usize, usize)> = Vec::new();
        for (ci, facts) in r_facts.iter().enumerate() {
            for (ai, r_atom) in facts.iter().enumerate() {
                if r_atom.label.is_some() {
                    continue; // preserved
                }
                let pair = p_facts.iter().enumerate().find(|&(pi, p_atom)| {
                    !p_claimed[pi] && p_atom.label.is_none() && atoms_pair(r_atom, p_atom)
                });
                match pair {
                    Some((pi, _)) => p_claimed[pi] = true,
                    None => delete_atoms.push((ci, ai)),
                }
            }
        }

        let mut add_atoms: Vec<AddAtomSpec> = Vec::new();
        for (pi, p_atom) in p_facts.iter().enumerate() {
            if p_atom.label.is_none() && !p_claimed[pi] {
                let element = p_atom.element.ok_or_else(|| {
                    MolRsError::validation(format!(
                        "product atom {pi} is added but declares no concrete element"
                    ))
                })?;
                add_atoms.push(AddAtomSpec {
                    product_idx: pi,
                    element,
                    charge: p_atom.charge.unwrap_or(0),
                });
            }
        }

        let delete = Self::group_deletes(&r_facts, &delete_atoms);
        let classify = |idx: usize| -> Cls {
            if p_facts[idx].label.is_some() {
                Cls::Mapped
            } else if p_claimed[idx] {
                Cls::Paired
            } else {
                Cls::Added
            }
        };

        // Mapped-mapped bond diff.
        let mut r_mm = HashMap::new();
        for p in reactants {
            collect_mapped_bonds(&p.graph, &mut r_mm);
        }
        let mut p_mm = HashMap::new();
        collect_mapped_bonds(&product.graph, &mut p_mm);

        let mut form_bonds: Vec<(NodeRef, NodeRef, (BondOrder, BondNumber))> = Vec::new();
        let mut set_order: Vec<(u32, u32, (BondOrder, BondNumber))> = Vec::new();
        for (&(a, b), &po) in &p_mm {
            match r_mm.get(&(a, b)) {
                None => form_bonds.push((NodeRef::Mapped(a), NodeRef::Mapped(b), po)),
                Some(&ro) if ro != po => set_order.push((a, b, po)),
                Some(_) => {}
            }
        }
        let break_bonds: Vec<(u32, u32)> = r_mm
            .keys()
            .filter(|k| !p_mm.contains_key(k))
            .copied()
            .collect();

        // Bonds touching an added atom (from the product template).
        for bnd in &product.graph.bonds {
            let (ca, cb) = (classify(bnd.a), classify(bnd.b));
            if ca == Cls::Paired || cb == Cls::Paired {
                continue; // pre-existing (preserved) bond, unchanged
            }
            if ca != Cls::Added && cb != Cls::Added {
                continue; // handled by the mapped-mapped diff above
            }
            let na = Self::node_ref(bnd.a, &p_facts)?;
            let nb = Self::node_ref(bnd.b, &p_facts)?;
            form_bonds.push((na, nb, query_bond_class(&bnd.query)));
        }

        // Property changes on preserved atoms.
        let mut set_props: Vec<PropDelta> = Vec::new();
        for (&l, &pi) in &p_labels {
            let (rc, ra) = r_labels[&l];
            let r_atom = &r_facts[rc][ra];
            let p_atom = &p_facts[pi];
            let element = match (r_atom.element, p_atom.element) {
                (Some(re), Some(pe)) if re != pe => Some(pe),
                _ => None,
            };
            let charge = match p_atom.charge {
                Some(pc) if r_atom.charge != Some(pc) => Some(pc),
                _ => None,
            };
            if element.is_some() || charge.is_some() {
                set_props.push(PropDelta {
                    label: l,
                    element,
                    charge,
                });
            }
        }

        Ok(Transform {
            delete,
            add_atoms,
            form_bonds,
            break_bonds,
            set_order,
            set_props,
        })
    }

    /// Build the label→position indices for both sides and enforce the Daylight
    /// pairwise-map rule (each label ≤ once per side, present on both sides).
    #[allow(clippy::type_complexity)]
    fn index_maps(
        r_facts: &[Vec<QueryAtomFacts>],
        p_facts: &[QueryAtomFacts],
    ) -> Result<(HashMap<u32, (usize, usize)>, HashMap<u32, usize>), MolRsError> {
        let mut r_labels: HashMap<u32, (usize, usize)> = HashMap::new();
        for (ci, facts) in r_facts.iter().enumerate() {
            for (ai, r_atom) in facts.iter().enumerate() {
                if let Some(l) = r_atom.label
                    && r_labels.insert(l, (ci, ai)).is_some()
                {
                    return Err(MolRsError::validation(format!(
                        "atom map :{l} appears more than once on the reactant side"
                    )));
                }
            }
        }
        let mut p_labels: HashMap<u32, usize> = HashMap::new();
        for (ai, p_atom) in p_facts.iter().enumerate() {
            if let Some(l) = p_atom.label
                && p_labels.insert(l, ai).is_some()
            {
                return Err(MolRsError::validation(format!(
                    "atom map :{l} appears more than once on the product side"
                )));
            }
        }
        for l in r_labels.keys() {
            if !p_labels.contains_key(l) {
                return Err(MolRsError::validation(format!(
                    "atom map :{l} appears only on the reactant side (Daylight requires pairwise maps)"
                )));
            }
        }
        for l in p_labels.keys() {
            if !r_labels.contains_key(l) {
                return Err(MolRsError::validation(format!(
                    "atom map :{l} appears only on the product side (Daylight requires pairwise maps)"
                )));
            }
        }
        Ok((r_labels, p_labels))
    }

    /// Group per-atom deletes into per-component [`DeleteSpec`]s with pins.
    fn group_deletes(
        r_facts: &[Vec<QueryAtomFacts>],
        delete_atoms: &[(usize, usize)],
    ) -> Vec<DeleteSpec> {
        let mut comps: Vec<usize> = delete_atoms.iter().map(|&(c, _)| c).collect();
        comps.sort_unstable();
        comps.dedup();
        comps
            .into_iter()
            .map(|ci| {
                let pins: Vec<(usize, u32)> = r_facts[ci]
                    .iter()
                    .enumerate()
                    .filter_map(|(ai, r_atom)| r_atom.label.map(|l| (ai, l)))
                    .collect();
                let delete_idxs: Vec<usize> = delete_atoms
                    .iter()
                    .filter(|&&(c, _)| c == ci)
                    .map(|&(_, a)| a)
                    .collect();
                DeleteSpec {
                    component: ci,
                    pins,
                    delete_idxs,
                }
            })
            .collect()
    }

    /// The [`NodeRef`] for a product atom that participates in an added-atom bond.
    fn node_ref(idx: usize, p_facts: &[QueryAtomFacts]) -> Result<NodeRef, MolRsError> {
        match p_facts[idx].label {
            Some(l) => Ok(NodeRef::Mapped(l)),
            None => Ok(NodeRef::Added(idx)),
        }
    }

    /// Resolve every unmapped LHS atom while the reactant world is still intact.
    fn resolve_leaving(
        &self,
        target: &SmartsTarget<'_>,
        binding: &HashMap<u32, NodeId>,
        reactants: &[SmartsPattern],
    ) -> Result<HashSet<NodeId>, MolRsError> {
        let mut leaving: HashSet<NodeId> = HashSet::new();
        for spec in &self.delete {
            // Re-anchor the reactant to the binding to resolve its leaving atoms.
            // When the root query atom (index 0) is pinned — the usual case, e.g.
            // `[C;%cx:1][H]` — seed the match at its already-bound image so the
            // search grows locally from that anchor instead of scanning the whole
            // (possibly huge) graph: O(local), not O(N) per apply. The `%LABEL`
            // target is threaded so a labelled reactant still resolves; an empty
            // map behaves like plain matching.
            let root = spec
                .pins
                .iter()
                .find(|&&(qi, _)| qi == 0)
                .and_then(|&(_, l)| binding.get(&l).copied());
            let matches = reactants[spec.component].find_in_target(target, root);
            let chosen = matches
                .iter()
                .find(|m| {
                    spec.pins.iter().all(|&(qi, l)| {
                        binding
                            .get(&l)
                            .is_some_and(|&aid| m.atoms.get(qi) == Some(&aid))
                    })
                })
                .ok_or_else(|| {
                    MolRsError::validation(format!(
                        "reaction apply: could not re-anchor reactant component {} to the binding \
                         to resolve its leaving atoms",
                        spec.component
                    ))
                })?;
            for &qi in &spec.delete_idxs {
                if let Some(&aid) = chosen.atoms.get(qi) {
                    leaving.insert(aid);
                }
            }
        }
        Ok(leaving)
    }

    /// Apply everything except LHS deletion, which a batch performs once.
    fn apply_after_delete(
        &self,
        mol: &mut Atomistic,
        binding: &HashMap<u32, NodeId>,
        touched: &mut Vec<NodeId>,
        created: &mut Vec<NodeId>,
    ) -> Result<(), MolRsError> {
        // 2. Add unmapped product atoms (no coordinates).
        let mut added: HashMap<usize, NodeId> = HashMap::new();
        for spec in &self.add_atoms {
            let mut props = Props::new().with(keys::ATOMIC_NUMBER, spec.element);
            if spec.charge != 0 {
                props.set(keys::FORMAL_CHARGE, spec.charge);
            }
            let id = mol.add_atom(props)?;
            added.insert(spec.product_idx, id);
            touched.push(id);
            created.push(id);
        }

        let resolve = |n: &NodeRef| -> Result<NodeId, MolRsError> {
            match n {
                NodeRef::Mapped(l) => binding.get(l).copied().ok_or_else(|| {
                    MolRsError::validation(format!("reaction apply: binding missing atom map :{l}"))
                }),
                NodeRef::Added(pi) => added.get(pi).copied().ok_or_else(|| {
                    MolRsError::validation(format!(
                        "reaction apply: added atom {pi} was not created"
                    ))
                }),
            }
        };
        let mapped = |l: &u32| -> Result<NodeId, MolRsError> {
            binding.get(l).copied().ok_or_else(|| {
                MolRsError::validation(format!("reaction apply: binding missing atom map :{l}"))
            })
        };

        // 3. Break mapped-mapped bonds.
        for &(a, b) in &self.break_bonds {
            let (ida, idb) = (mapped(&a)?, mapped(&b)?);
            if let Some(bid) = mol.bond_between(ida, idb) {
                mol.del_bond(bid)?;
                touched.push(ida);
                touched.push(idb);
            }
        }

        // 4. Form bonds (mapped-mapped and added-atom).
        for (na, nb, (bond_type, bond_number)) in &self.form_bonds {
            let (ida, idb) = (resolve(na)?, resolve(nb)?);
            mol.add_bond(
                ida,
                idb,
                props! { keys::BOND_TYPE => *bond_type, keys::BOND_NUMBER => *bond_number },
            )?;
            touched.push(ida);
            touched.push(idb);
        }

        // 5. Change order of preserved bonds.
        for &(a, b, (bond_type, bond_number)) in &self.set_order {
            let (ida, idb) = (mapped(&a)?, mapped(&b)?);
            let bid = mol.bond_between(ida, idb).ok_or_else(|| {
                MolRsError::validation(format!(
                    "reaction apply: expected an existing bond between :{a} and :{b} to set its order"
                ))
            })?;
            mol.set_bond_class(bid, bond_type, bond_number)?;
            touched.push(ida);
            touched.push(idb);
        }

        // 6. Property changes on preserved atoms.
        for pd in &self.set_props {
            let id = mapped(&pd.label)?;
            if let Some(element) = pd.element {
                mol.set_atom(id, keys::ATOMIC_NUMBER, element)?;
            }
            if let Some(c) = pd.charge {
                mol.set_atom(id, keys::FORMAL_CHARGE, c)?;
            }
            touched.push(id);
        }

        Ok(())
    }

    /// Apply the transform to one pinned occurrence.
    fn apply(
        &self,
        mol: &mut Atomistic,
        binding: &HashMap<u32, NodeId>,
        reactants: &[SmartsPattern],
        labels: &HashMap<NodeId, String>,
        refresh: bool,
    ) -> Result<Vec<NodeId>, MolRsError> {
        let leaving = {
            let target = SmartsTarget::new(mol).with_labels(labels);
            self.resolve_leaving(&target, binding, reactants)?
        };
        let mut touched = Vec::new();
        for &aid in &leaving {
            for (other, _) in mol.neighbor_bonds(aid) {
                if !leaving.contains(&other) {
                    touched.push(other);
                }
            }
        }
        let mut leaving_ordered: Vec<_> = leaving.into_iter().collect();
        leaving_ordered.sort_unstable();
        mol.del_atoms(&leaving_ordered)?;
        let mut created = Vec::new();
        self.apply_after_delete(mol, binding, &mut touched, &mut created)?;

        if refresh {
            mol.generate_topology(true, true, false, false)?;
            rewrite_aromatic_flags(mol)?;
        }

        // Dedup with a deterministic order (sort by the stable atom handle).
        touched.sort_unstable();
        touched.dedup();
        Ok(touched)
    }

    /// Compile all leaving groups first, then execute the disjoint edits in one
    /// batch. Relation tables are scanned once for the union of deleted atoms.
    fn apply_many(
        &self,
        mol: &mut Atomistic,
        bindings: &[HashMap<u32, NodeId>],
        reactants: &[SmartsPattern],
        labels: &HashMap<NodeId, String>,
        refresh: bool,
    ) -> Result<ReactionBatch, MolRsError> {
        let leaving_per_edit = {
            let target = SmartsTarget::new(mol).with_labels(labels);
            bindings
                .iter()
                .map(|binding| self.resolve_leaving(&target, binding, reactants))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut all_leaving = HashSet::new();
        for leaving in &leaving_per_edit {
            for &atom in leaving {
                if !all_leaving.insert(atom) {
                    return Err(MolRsError::validation(format!(
                        "reaction batch resolves leaving atom {atom:?} in more than one edit"
                    )));
                }
            }
        }

        let mut touched_per_edit = Vec::with_capacity(bindings.len());
        for leaving in &leaving_per_edit {
            let mut touched = Vec::new();
            for &atom in leaving {
                for (other, _) in mol.neighbor_bonds(atom) {
                    if !all_leaving.contains(&other) {
                        touched.push(other);
                    }
                }
            }
            touched_per_edit.push(touched);
        }

        let mut leaving_ordered: Vec<_> = all_leaving.into_iter().collect();
        leaving_ordered.sort_unstable();
        mol.del_atoms(&leaving_ordered)?;
        let mut created_per_edit = Vec::with_capacity(bindings.len());
        for (binding, touched) in bindings.iter().zip(&mut touched_per_edit) {
            let mut created = Vec::new();
            self.apply_after_delete(mol, binding, touched, &mut created)?;
            touched.sort_unstable();
            touched.dedup();
            created_per_edit.push(created);
        }
        if refresh {
            mol.generate_topology(true, true, false, false)?;
            rewrite_aromatic_flags(mol)?;
        }
        Ok(ReactionBatch {
            touched: touched_per_edit,
            created: created_per_edit,
        })
    }
}

// ---------------------------------------------------------------------------
// Reaction (parsed reaction SMARTS)
// ---------------------------------------------------------------------------

/// A parsed Daylight reaction SMARTS and its compiled graph edit (the private
/// `Transform`: the atom/bond additions, deletions and property changes the
/// reaction applies to a match).
///
/// Parses a reaction SMARTS `reactants >> products` (tolerating, and ignoring,
/// an agent field `reactants > agents > products`), derives the graph edit from
/// the **atom-map diff** (Daylight SMIRKS transform semantics), and applies it to
/// a single matched occurrence by editing an [`Atomistic`] in place.
///
/// # Transform semantics (Daylight SMIRKS)
///
/// Atoms are keyed by their `:n` atom-map label:
///
/// - a label on **both** sides is a *preserved* atom (its molecule atom is kept);
/// - a label on **exactly one** side is an error (Daylight's pairwise-map rule);
/// - an **unmapped** atom present identically on both sides (same element and the
///   same bond to a shared mapped neighbour) is *paired* and left untouched — this
///   keeps, e.g., a carbonyl `=O` in `[C:2](=O)OC >> [C:2]=O` from being deleted
///   and re-added;
/// - an unmapped reactant atom with no pair is *deleted* (a leaving group);
/// - an unmapped product atom with no pair is *added* (element/charge from the
///   product template, **no coordinates**).
///
/// Bonds between mapped atoms are diffed: present in product-not-reactant is
/// *formed*, reactant-not-product is *broken*, and an order change is a
/// *set-order*. Bonds touching an added atom are formed from the product template.
///
/// # Reaction SMARTS, not strict SMIRKS
///
/// SMARTS queries are permitted on reacting atoms (RDKit style) so functional
/// groups can be matched (`[N;H2:1]`); only concrete product atoms
/// (`[N:1]`, `O`, `[S:3]`) can be *added*, since an added atom needs a definite
/// element. The transform mechanics follow SMIRKS; the SMILES-only restriction on
/// reacting atoms is not enforced.
#[derive(Debug, Clone)]
pub struct Reaction {
    source: String,
    reactants: Vec<SmartsPattern>,
    reactant_src: Vec<String>,
    product: SmartsPattern,
    transform: Transform,
}

impl Reaction {
    /// Parse a reaction SMARTS `reactants >> products`.
    ///
    /// The three-part `reactants > agents > products` form is accepted and the
    /// agent field ignored. The reactant side is split into components on
    /// top-level `.`; each component and the product are parsed by the chain-01
    /// SMARTS parser. Returns `Err` on any syntax or map-consistency error.
    pub fn parse(reaction_smarts: &str) -> Result<Reaction, MolRsError> {
        let segments = split_top_level(reaction_smarts, '>');
        if segments.len() != 3 {
            return Err(MolRsError::parse(format!(
                "reaction SMARTS must contain one '>>' (or '>agent>'); found {} top-level '>' segment(s)",
                segments.len()
            )));
        }
        let lhs = segments[0].trim();
        // segments[1] is the agent field — parsed-tolerated, ignored.
        let rhs = segments[2].trim();
        if lhs.is_empty() || rhs.is_empty() {
            return Err(MolRsError::parse(
                "reaction SMARTS needs non-empty reactants and products",
            ));
        }

        let reactant_src: Vec<String> = split_top_level(lhs, '.')
            .into_iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if reactant_src.is_empty() {
            return Err(MolRsError::parse(
                "reaction SMARTS has no reactant components",
            ));
        }

        let reactants = reactant_src
            .iter()
            .map(|s| SmartsPattern::parse(s))
            .collect::<Result<Vec<_>, _>>()?;
        let product = SmartsPattern::parse(rhs)?;
        let transform = Transform::compile(&reactants, &product)?;

        Ok(Reaction {
            source: reaction_smarts.to_string(),
            reactants,
            reactant_src,
            product,
            transform,
        })
    }

    /// Original reaction SMARTS used to compile this transform.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The reactant components (LHS), one [`SmartsPattern`] per top-level `.`
    /// component, for the caller to match / pair independently.
    pub fn reactants(&self) -> &[SmartsPattern] {
        &self.reactants
    }

    /// The reactant component source strings, in order.
    pub fn reactant_smarts(&self) -> Vec<String> {
        self.reactant_src.clone()
    }

    /// The product pattern (RHS).
    pub fn product(&self) -> &SmartsPattern {
        &self.product
    }

    /// The map-label pairs of newly *formed* bonds between preserved atoms — the
    /// distance criterion a caller uses to pick a reacting occurrence. Bonds that
    /// merely change order, and bonds to added atoms, are excluded.
    pub fn forming_bonds(&self) -> Vec<(u32, u32)> {
        self.transform
            .form_bonds
            .iter()
            .filter_map(|&(a, b, _)| match (a, b) {
                (NodeRef::Mapped(x), NodeRef::Mapped(y)) => Some((x, y)),
                _ => None,
            })
            .collect()
    }

    /// Apply the transform to `mol` in place at the occurrence pinned by
    /// `binding` (map label → molecule atom id). Order: delete unmapped-LHS atoms
    /// → add unmapped-RHS atoms → break/form bonds + set order → set preserved
    /// props → regenerate topology → re-perceive aromaticity. Added atoms get no
    /// coordinates. Every edit goes through the core `Atomistic` primitives.
    ///
    /// Returns the deduplicated, deterministically-ordered set of *surviving*
    /// touched atom ids — the atoms whose local environment changed and so need
    /// re-typification: endpoints of every formed / broken / order-changed bond,
    /// newly added atoms, the surviving neighbours of deleted atoms, and atoms
    /// whose props were set. Deleted atoms' own ids are never included (they no
    /// longer exist). The caller (molpy) expands this seed set into a
    /// retype-safe radius-N ball.
    pub fn apply(
        &self,
        mol: &mut Atomistic,
        binding: &HashMap<u32, NodeId>,
        labels: &HashMap<NodeId, String>,
        refresh: bool,
    ) -> Result<Vec<NodeId>, MolRsError> {
        self.transform
            .apply(mol, binding, &self.reactants, labels, refresh)
    }

    /// Compile every occurrence against the intact reactant world, then apply
    /// all disjoint transforms with one batched leaving-group deletion.
    ///
    /// Returns, per binding, the touched atoms (as [`apply`](Self::apply)
    /// returns them) and the atoms the product created — see
    /// [`ReactionBatch`].
    ///
    /// # Errors
    ///
    /// [`MolRsError::Validation`] when two bindings resolve the same leaving
    /// atom, a binding misses an atom map, or an edit is refused.
    pub fn apply_many(
        &self,
        mol: &mut Atomistic,
        bindings: &[HashMap<u32, NodeId>],
        labels: &HashMap<NodeId, String>,
        refresh: bool,
    ) -> Result<ReactionBatch, MolRsError> {
        self.transform
            .apply_many(mol, bindings, &self.reactants, labels, refresh)
    }
}

/// Split `s` on `sep` at bracket/paren depth 0 only (so `.` inside `[...]` or
/// `(...)` is preserved). Used for the `>>` arrow and the `.` component split.
fn split_top_level(s: &str, sep: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut brace = 0i32;
    let mut paren = 0i32;
    for ch in s.chars() {
        match ch {
            '[' => {
                brace += 1;
                cur.push(ch);
            }
            ']' => {
                brace -= 1;
                cur.push(ch);
            }
            '(' => {
                paren += 1;
                cur.push(ch);
            }
            ')' => {
                paren -= 1;
                cur.push(ch);
            }
            c if c == sep && brace == 0 && paren == 0 => out.push(std::mem::take(&mut cur)),
            _ => cur.push(ch),
        }
    }
    out.push(cur);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Add an atom of `element` at `xyz`.
    fn atom(mol: &mut Atomistic, element: Element, [x, y, z]: [f64; 3]) -> NodeId {
        mol.add_atom(props! {
            keys::ATOMIC_NUMBER => element,
            keys::X => x,
            keys::Y => y,
            keys::Z => z,
        })
        .unwrap()
    }

    /// A double bond `a`=`b`.
    fn double() -> Props {
        props! { keys::BOND_TYPE => BondOrder::Double, keys::BOND_NUMBER => BondNumber::Double }
    }

    #[test]
    fn parse_splits_arrow_and_components() {
        let rxn = Reaction::parse("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O").unwrap();
        assert_eq!(rxn.reactants().len(), 2);
        assert_eq!(rxn.reactant_smarts(), vec!["[N;H2:1]", "[C:2](=O)OC"]);
        assert_eq!(rxn.product().n_query_atoms(), 3);
    }

    #[test]
    fn parse_tolerates_agent_field() {
        let rxn = Reaction::parse("[C:1]=[C:2].[S;H1:3] > [Pt] > [C:1][C:2][S:3]").unwrap();
        assert_eq!(rxn.reactants().len(), 2);
    }

    #[test]
    fn parse_rejects_missing_arrow() {
        assert!(Reaction::parse("[C:1][O:2]").is_err());
    }

    #[test]
    fn compile_amide_forms_bond_and_deletes_leaving_group() {
        let rxn = Reaction::parse("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O").unwrap();
        let t = &rxn.transform;
        // forms the N-C bond
        assert_eq!(rxn.forming_bonds(), vec![(1, 2)]);
        // ester O + alkyl C leave; carbonyl =O is paired (not deleted, not added)
        assert_eq!(t.add_atoms.len(), 0);
        assert_eq!(t.delete.len(), 1);
        let del = &t.delete[0];
        assert_eq!(del.component, 1);
        let mut idxs = del.delete_idxs.clone();
        idxs.sort_unstable();
        assert_eq!(idxs, vec![2, 3]); // ester O (2) + alkyl C (3)
        assert!(t.set_order.is_empty());
        assert!(t.break_bonds.is_empty());
    }

    #[test]
    fn compile_thiol_ene_sets_order_and_forms_bond() {
        let rxn = Reaction::parse("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]").unwrap();
        let t = &rxn.transform;
        assert_eq!(rxn.forming_bonds(), vec![(2, 3)]);
        assert_eq!(t.set_order.len(), 1);
        let (a, b, class) = t.set_order[0];
        assert_eq!(ordered(a, b), (1, 2));
        assert_eq!(class, (BondOrder::Single, BondNumber::Single));
    }

    #[test]
    fn compile_rejects_one_sided_map() {
        assert!(Reaction::parse("[C:1][O:2] >> [C:1]").is_err());
    }

    #[test]
    fn compile_adds_unmapped_product_atom() {
        let rxn = Reaction::parse("[C:1]Br >> [C:1]O").unwrap();
        let t = &rxn.transform;
        assert_eq!(t.add_atoms.len(), 1);
        assert_eq!(t.add_atoms[0].element, Element::O);
        assert_eq!(t.delete.len(), 1); // Br leaves
        // the new C-O bond is a form touching an added atom (not a mapped pair)
        assert!(rxn.forming_bonds().is_empty());
        assert_eq!(t.form_bonds.len(), 1);
    }

    #[test]
    fn apply_amide_edits_graph_in_place() {
        let rxn = Reaction::parse("[N;H2:1].[C:2](=O)OC >> [N:1][C:2]=O").unwrap();

        let mut mol = Atomistic::new();
        // amine
        let n0 = atom(&mut mol, Element::N, [0.0, 0.0, 0.0]);
        let h1 = atom(&mut mol, Element::H, [0.6, 0.8, 0.0]);
        let h2 = atom(&mut mol, Element::H, [-0.6, 0.8, 0.0]);
        mol.add_bond(n0, h1, props! {}).unwrap();
        mol.add_bond(n0, h2, props! {}).unwrap();
        // ester
        let c0 = atom(&mut mol, Element::C, [5.0, 0.0, 0.0]);
        let o1 = atom(&mut mol, Element::O, [6.0, 0.8, 0.0]);
        let o2 = atom(&mut mol, Element::O, [5.0, -1.3, 0.0]);
        let c3 = atom(&mut mol, Element::C, [5.0, -2.6, 0.0]);
        mol.add_bond(c0, o1, double()).unwrap();
        mol.add_bond(c0, o2, props! {}).unwrap();
        mol.add_bond(o2, c3, props! {}).unwrap();

        let n_before = mol.n_atoms();
        let binding = HashMap::from([(1u32, n0), (2u32, c0)]);
        let touched = rxn
            .apply(&mut mol, &binding, &HashMap::new(), true)
            .unwrap();

        // two leaving atoms removed
        assert_eq!(mol.n_atoms(), n_before - 2);
        // new N-C bond present, carbonyl O preserved
        assert!(mol.bond_between(n0, c0).is_some());
        assert!(mol.bond_between(c0, o1).is_some());
        // topology regenerated around the new bond
        assert!(mol.n_angles() > 0);

        // touched = formed-bond endpoints (N, C) + the ester O's surviving
        // neighbour (C again); the deleted leaving atoms (o2, c3) are absent.
        assert!(touched.contains(&n0));
        assert!(touched.contains(&c0));
        assert!(!touched.contains(&o2));
        assert!(!touched.contains(&c3));
        // deduped + sorted
        assert!(touched.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn apply_adds_atom_without_coordinates() {
        let rxn = Reaction::parse("[C:1]Br >> [C:1]O").unwrap();
        let mut mol = Atomistic::new();
        let c0 = atom(&mut mol, Element::C, [0.0, 0.0, 0.0]);
        let br = atom(&mut mol, Element::Br, [1.9, 0.0, 0.0]);
        mol.add_bond(c0, br, props! {}).unwrap();

        let binding = HashMap::from([(1u32, c0)]);
        let touched = rxn
            .apply(&mut mol, &binding, &HashMap::new(), true)
            .unwrap();

        assert_eq!(mol.n_atoms(), 2);
        // the surviving neighbour of C is the new O
        let o = mol.bond_between(c0, br); // Br was removed; that bond is gone
        assert!(o.is_none());
        let species: Vec<Option<u8>> = mol.atoms().map(|(_, a)| a.atomic_number()).collect();
        assert!(species.contains(&Some(Element::O.z())));
        assert!(!species.contains(&Some(Element::Br.z())));

        // touched = surviving carbon (leaving Br's neighbour) + the added O;
        // the deleted Br handle never appears.
        assert!(touched.contains(&c0));
        assert!(!touched.contains(&br));
        // exactly one touched id is the freshly added atom (neither c0 nor br)
        assert_eq!(touched.iter().filter(|&&t| t != c0 && t != br).count(), 1);
    }

    #[test]
    fn apply_thiol_ene_touched_is_two_carbons_and_sulfur() {
        let rxn = Reaction::parse("[C:1]=[C:2].[S;H1:3] >> [C:1][C:2][S:3]").unwrap();
        let mut mol = Atomistic::new();
        let c1 = atom(&mut mol, Element::C, [0.0, 0.0, 0.0]);
        let c2 = atom(&mut mol, Element::C, [1.3, 0.0, 0.0]);
        let s3 = atom(&mut mol, Element::S, [3.0, 0.0, 0.0]);
        let hs = atom(&mut mol, Element::H, [3.0, 1.0, 0.0]);
        mol.add_bond(c1, c2, double()).unwrap();
        mol.add_bond(s3, hs, props! {}).unwrap();

        // No leaving group; the binding pins the three reacting atoms directly.
        let binding = HashMap::from([(1u32, c1), (2u32, c2), (3u32, s3)]);
        let touched = rxn
            .apply(&mut mol, &binding, &HashMap::new(), true)
            .unwrap();

        // C1-C2 order change (2->1) touches both carbons; C2-S3 form touches
        // C2 and S3; the spectator H is untouched.
        let mut expected = [c1, c2, s3];
        expected.sort_unstable();
        assert_eq!(touched, expected.to_vec());
    }
}
