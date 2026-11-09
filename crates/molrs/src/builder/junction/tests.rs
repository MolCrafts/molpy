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
    let bond = world.link(a, b).unwrap();
    let ends = world.get_bond(bond).unwrap().nodes;
    Junction {
        bond,
        anchors: [ends[0], ends[1]],
        folds: [None, None],
    }
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
