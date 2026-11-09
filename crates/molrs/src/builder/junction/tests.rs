use std::collections::BTreeSet;

use super::*;
use crate::core::{BondNumber, Element, NodeId, PortKind};
use crate::props;

fn carbon(world: &mut Atomistic) -> NodeId {
    world
        .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
        .unwrap()
}

fn hydrogen_on(world: &mut Atomistic, on: NodeId) -> NodeId {
    let h = world
        .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
        .unwrap();
    world.add_bond(on, h, props! {}).unwrap();
    h
}

fn bond(world: &mut Atomistic, a: NodeId, b: NodeId) {
    world.add_bond(a, b, props! {}).unwrap();
}

/// Generate every graph-derived term and stamp each angle and pair: the
/// state a typed template carries.
fn typed(world: &mut Atomistic) {
    world
        .generate_topology(true, true, false, true, true)
        .unwrap();
    let angles: Vec<RelationId> = world.angles().map(|(id, _)| id).collect();
    for id in angles {
        world.set_angle(id, keys::TYPE, "stamped").unwrap();
    }
    let pairs: Vec<RelationId> = world.term_ids(PAIRS).unwrap().collect();
    for id in pairs {
        world.set_term(PAIRS, id, "epsilon", 0.5).unwrap();
    }
}

fn canonical(nodes: &[NodeId]) -> Vec<NodeId> {
    let reversed: Vec<NodeId> = nodes.iter().rev().copied().collect();
    nodes.to_vec().min(reversed)
}

fn angle_set(world: &Atomistic) -> BTreeSet<Vec<NodeId>> {
    world.angles().map(|(_, r)| canonical(&r.nodes)).collect()
}

fn dihedral_set(world: &Atomistic) -> BTreeSet<Vec<NodeId>> {
    world
        .dihedrals()
        .map(|(_, r)| canonical(&r.nodes))
        .collect()
}

/// The chain A–B–C–D with a `<` port on A and a `>` port on D, typed.
struct Chain {
    world: Atomistic,
    atoms: [NodeId; 4],
    left: RelationId,
    right: RelationId,
}

fn chain() -> Chain {
    let mut world = Atomistic::new();
    let atoms = [(); 4].map(|()| carbon(&mut world));
    for pair in atoms.windows(2) {
        bond(&mut world, pair[0], pair[1]);
    }
    let ha = hydrogen_on(&mut world, atoms[0]);
    let hd = hydrogen_on(&mut world, atoms[3]);
    let left = world
        .add_port(atoms[0], ha, PortKind::Left, "", BondNumber::Single)
        .unwrap();
    let right = world
        .add_port(atoms[3], hd, PortKind::Right, "", BondNumber::Single)
        .unwrap();
    typed(&mut world);
    Chain {
        world,
        atoms,
        left,
        right,
    }
}

fn junction_of(world: &mut Atomistic, a: RelationId, b: RelationId) -> Junction {
    world.link(a, b).unwrap()
}

#[test]
fn keep_adds_the_terms_through_the_new_bond_and_keeps_the_rest() {
    let first = chain();
    let second = chain();
    let mut world = first.world.clone();
    let map = world.merge(second.world.clone()).unwrap();
    // `merge` renames the second chain's handles: find its `<` port by anchor.
    let second_left = world
        .port_ids()
        .find(|&id| world.get_port(id).unwrap().anchor == map[&second.atoms[0]])
        .unwrap();
    let junction = junction_of(&mut world, first.right, second_left);
    let before = angle_set(&world);

    let report = update_junctions(&mut world, &[junction], &Retype::Keep).unwrap();

    let mut fresh = world.clone();
    fresh
        .generate_topology(true, true, false, true, true)
        .unwrap();
    assert_eq!(angle_set(&world), angle_set(&fresh));
    assert_eq!(dihedral_set(&world), dihedral_set(&fresh));
    assert_eq!(world.n_terms(PAIRS).unwrap(), fresh.n_terms(PAIRS).unwrap());
    assert_eq!(report.added_angles, angle_set(&world).len() - before.len());
    assert!(
        report.reclassified_pairs.is_empty(),
        "a chain shortens no path"
    );
    // The template's angles keep their stamp; the junction's are bare.
    let d = first.atoms[3];
    let a2 = map[&second.atoms[0]];
    for (_, angle) in world.angles() {
        let through = angle.nodes.contains(&d) && angle.nodes.contains(&a2);
        assert_eq!(
            angle.props.get_str(keys::TYPE).is_none(),
            through,
            "{:?}",
            angle.nodes
        );
    }
}

#[test]
fn a_ring_closure_reclassifies_the_former_one_four_pair() {
    let Chain {
        mut world,
        atoms: [a, b, c, d],
        left,
        right,
    } = chain();
    let pair_of = |world: &Atomistic, x: NodeId, y: NodeId| {
        world
            .term_ids(PAIRS)
            .unwrap()
            .find(|&id| {
                let nodes = world.get_term(PAIRS, id).unwrap().nodes;
                (nodes[0] == x && nodes[1] == y) || (nodes[0] == y && nodes[1] == x)
            })
            .unwrap()
    };
    let ad = pair_of(&world, a, d);
    assert_eq!(
        world
            .get_term(PAIRS, ad)
            .unwrap()
            .props
            .get_int(keys::N_BONDS),
        Some(3)
    );
    let junction = junction_of(&mut world, right, left);

    let report = update_junctions(&mut world, &[junction], &Retype::Keep).unwrap();

    assert_eq!(report.reclassified_pairs, vec![ad]);
    let row = world.get_term(PAIRS, ad).unwrap();
    assert_eq!(row.props.get_int(keys::N_BONDS), Some(1));
    assert_eq!(
        row.props.get_f64("epsilon"),
        None,
        "its 1-4 override is gone"
    );
    let bd = pair_of(&world, b, d);
    assert_eq!(
        world.get_term(PAIRS, bd).unwrap().props.get_f64("epsilon"),
        Some(0.5),
        "a pair whose distance did not change keeps its cells"
    );
    // The four-ring: two angles and three dihedrals through A–D, no new pair.
    assert_eq!(
        (
            report.added_angles,
            report.added_dihedrals,
            report.added_pairs
        ),
        (2, 3, 0)
    );
    assert!(dihedral_set(&world).contains(&canonical(&[c, d, a, b])));
}

#[test]
fn a_second_call_changes_nothing() {
    let Chain {
        mut world,
        left,
        right,
        ..
    } = chain();
    let junction = junction_of(&mut world, right, left);
    update_junctions(&mut world, &[junction], &Retype::Keep).unwrap();
    let (angles, dihedrals) = (angle_set(&world), dihedral_set(&world));

    let again = update_junctions(&mut world, &[junction], &Retype::Keep).unwrap();

    assert_eq!(
        (again.added_angles, again.added_dihedrals, again.added_pairs),
        (0, 0, 0)
    );
    assert!(again.reclassified_pairs.is_empty());
    assert_eq!(
        (angle_set(&world), dihedral_set(&world)),
        (angles, dihedrals)
    );
}

#[test]
fn a_graph_without_terms_gets_none() {
    let mut world = Atomistic::new();
    let a = carbon(&mut world);
    let b = carbon(&mut world);
    let (ha, hb) = (hydrogen_on(&mut world, a), hydrogen_on(&mut world, b));
    let pa = world
        .add_port(a, ha, PortKind::Right, "", BondNumber::Single)
        .unwrap();
    let pb = world
        .add_port(b, hb, PortKind::Left, "", BondNumber::Single)
        .unwrap();
    let junction = junction_of(&mut world, pa, pb);
    let report = update_junctions(&mut world, &[junction], &Retype::Keep).unwrap();
    assert_eq!((world.n_angles(), world.n_dihedrals()), (0, 0));
    assert_eq!(world.n_terms(PAIRS).unwrap(), 0);
    assert_eq!(report.junctions, vec![junction]);
}

#[test]
fn a_junction_that_is_not_a_bond_is_refused_before_any_write() {
    let Chain {
        mut world,
        atoms: [a, _, c, _],
        left,
        right,
    } = chain();
    let mut junction = junction_of(&mut world, right, left);
    junction.anchors = [a, c];
    let before = angle_set(&world);
    let err = update_junctions(&mut world, &[junction], &Retype::Keep).unwrap_err();
    assert!(
        matches!(err, JunctionError::NotABond { junction: 0 }),
        "{err}"
    );
    assert_eq!(angle_set(&world), before);
}

// ---- Retype::Patch ----

/// One copy `frag` of template "U": C0 (row 0, the `<` anchor), C1 (row 1,
/// the `>` anchor), their handles H (rows 2, 3) and an O on C0 (row 4), each
/// typed and stamped with its provenance.
struct Unit {
    atoms: [NodeId; 5],
    left: RelationId,
    right: RelationId,
}

fn unit(world: &mut Atomistic, frag: i32) -> Unit {
    let spec = [
        (Element::C, "c"),
        (Element::C, "c"),
        (Element::H, "h"),
        (Element::H, "h"),
        (Element::O, "o"),
    ];
    let atoms: [NodeId; 5] = std::array::from_fn(|row| {
        let (z, ty) = spec[row];
        world
            .add_atom(props! {
                keys::ATOMIC_NUMBER => z,
                keys::TYPE => ty,
                keys::FRAG_ID => frag,
                keys::TEMPLATE => "U",
                keys::TEMPLATE_ATOM => row as i32,
            })
            .unwrap()
    });
    for (a, b) in [(0, 1), (0, 2), (1, 3), (0, 4)] {
        bond(world, atoms[a], atoms[b]);
    }
    let left = world
        .add_port(atoms[0], atoms[2], PortKind::Left, "", BondNumber::Single)
        .unwrap();
    let right = world
        .add_port(atoms[1], atoms[3], PortKind::Right, "", BondNumber::Single)
        .unwrap();
    Unit { atoms, left, right }
}

/// Three copies linked head to tail: copy 0's `<` stays free, copy 2's `>`.
fn linked_chain() -> (Atomistic, Vec<Unit>, Vec<Junction>) {
    let mut world = Atomistic::new();
    let units: Vec<Unit> = (0..3).map(|frag| unit(&mut world, frag)).collect();
    let junctions = vec![
        world.link(units[0].right, units[1].left).unwrap(),
        world.link(units[1].right, units[2].left).unwrap(),
    ];
    (world, units, junctions)
}

fn patch(entries: &[(usize, &[(usize, &str)])]) -> Retype {
    let mut patch = Patch::new();
    for &(anchor, types) in entries {
        patch.insert(
            "U",
            anchor,
            types.iter().map(|&(row, ty)| (row, ty.to_owned())),
        );
    }
    Retype::Patch(patch)
}

fn type_of(world: &Atomistic, atom: NodeId) -> String {
    world
        .get_atom(atom)
        .unwrap()
        .get_str(keys::TYPE)
        .unwrap()
        .to_owned()
}

fn types(world: &Atomistic) -> Vec<String> {
    world.atom_ids().map(|id| type_of(world, id)).collect()
}

#[test]
fn a_patch_retypes_only_the_copies_whose_anchor_was_linked() {
    let (mut world, units, junctions) = linked_chain();
    let retype = patch(&[(0, &[(0, "c-linked"), (4, "o-linked")])]);

    let report = update_junctions(&mut world, &junctions, &retype).unwrap();

    // Row 0 is linked in copies 1 and 2; copy 0's `<` is a chain end.
    assert_eq!(type_of(&world, units[0].atoms[0]), "c");
    assert_eq!(type_of(&world, units[0].atoms[4]), "o");
    for unit in &units[1..] {
        assert_eq!(type_of(&world, unit.atoms[0]), "c-linked");
        assert_eq!(type_of(&world, unit.atoms[4]), "o-linked");
        assert_eq!(type_of(&world, unit.atoms[1]), "c", "row 1 has no entry");
    }
    assert_eq!(report.retyped.len(), 4);
    assert!(report.retyped.iter().all(|r| r.from.is_some()));
}

#[test]
fn a_second_patch_changes_nothing() {
    let (mut world, _, junctions) = linked_chain();
    let retype = patch(&[(0, &[(4, "o-linked")])]);
    update_junctions(&mut world, &junctions, &retype).unwrap();
    let again = update_junctions(&mut world, &junctions, &retype).unwrap();
    assert!(again.retyped.is_empty());
}

#[test]
fn a_conflicting_patch_is_refused_and_writes_nothing() {
    let (mut world, units, junctions) = linked_chain();
    world
        .generate_topology(true, false, false, false, false)
        .unwrap();
    let before = (types(&world), world.n_angles());
    // Copy 1 has both anchors linked: row 4 would be both "a" and "b".
    let retype = patch(&[(0, &[(4, "a")]), (1, &[(4, "b")])]);

    let err = update_junctions(&mut world, &junctions, &retype).unwrap_err();

    match err {
        JunctionError::PatchConflict { atom, types } => {
            assert_eq!(atom, units[1].atoms[4]);
            assert_eq!(types, ["a".to_owned(), "b".to_owned()]);
        }
        other => panic!("expected a conflict, got {other}"),
    }
    assert_eq!((types(&world), world.n_angles()), before);
}

#[test]
fn a_patch_naming_a_departed_atom_is_refused() {
    let (mut world, _, junctions) = linked_chain();
    // Row 2 is the `<` handle: it left with the leaving group.
    let err = update_junctions(&mut world, &junctions, &patch(&[(0, &[(2, "x")])])).unwrap_err();
    assert!(
        matches!(err, JunctionError::PatchedAtomMissing { row: 2, .. }),
        "{err}"
    );
}

#[test]
fn a_patch_needs_the_anchors_provenance() {
    let Chain {
        mut world,
        left,
        right,
        ..
    } = chain();
    let junction = world.link(right, left).unwrap();
    let err = update_junctions(&mut world, &[junction], &patch(&[(0, &[(4, "x")])])).unwrap_err();
    assert!(
        matches!(err, JunctionError::NoProvenance { junction: 0, .. }),
        "{err}"
    );
}

#[test]
fn a_copy_claimed_twice_is_ambiguous() {
    let (mut world, _, junctions) = linked_chain();
    // A second copy 1, as a merge of two assemblies leaves it.
    unit(&mut world, 1);
    let err = update_junctions(&mut world, &junctions, &patch(&[(0, &[(4, "x")])])).unwrap_err();
    assert!(
        matches!(
            err,
            JunctionError::AmbiguousCopy {
                frag_id: 1,
                row: 4,
                count: 2,
                ..
            }
        ),
        "{err}"
    );
}
