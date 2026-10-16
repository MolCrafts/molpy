"""Contract tests for the graph bindings.

`MolGraph` is the generic ECS *world*: nodes are stable opaque `int` handles,
fields live in aligned columns, and topology is kind-tagged relations.
`Atomistic` / `CoarseGrain` are peers, not `MolGraph` subclasses: each holds its
own typed core graph and speaks only its typed vocabulary, with live views as
the one door to an entity's fields. Rigid-body moves (translate, rotate,
scale) are methods of the leaves; chemical perception has owners
(`molrs.perceive.assign_*` / `perceive_rings`, `molrs.ff.charge.*`,
`molrs.io.smiles.SmilesIr`) and is reached through them, never through a
method on the graph classes.
"""

import copy

import molrs
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Stable handles                                                              #
# --------------------------------------------------------------------------- #


def test_handles_are_stable_opaque_ints():
    g = molrs.core.MolGraph()
    handles = [g.add_node() for _ in range(3)]
    assert all(isinstance(h, int) for h in handles)
    assert len(set(handles)) == 3  # distinct


def test_del_node_middle_keeps_others_valid_no_reindex():
    g = molrs.core.MolGraph()
    e0, e1, e2 = g.add_node(), g.add_node(), g.add_node()
    g.set_node(e0, "x", 0.0)
    g.set_node(e2, "x", 2.0)

    g.del_node(e1)

    # The surviving handles still resolve to *their own* data — no positional
    # reindexing shifted e2 into e1's slot.
    assert sorted(g.node_ids()) == sorted([e0, e2])
    assert g.get_node(e0)["x"] == 0.0
    assert g.get_node(e2)["x"] == 2.0


def test_del_nodes_is_atomic_and_distinct_from_del_node():
    g = molrs.core.MolGraph()
    e0, e1, e2 = g.add_node(), g.add_node(), g.add_node()
    g.del_node(e1)
    with pytest.raises(ValueError):
        g.del_nodes([e0, e1])  # e1 is stale: nothing is deleted
    assert sorted(g.node_ids()) == sorted([e0, e2])
    with pytest.raises(ValueError):
        g.del_nodes([e0, e0])  # listed twice
    g.del_nodes([e0, e2])
    assert g.node_ids() == []


def test_add_node_carries_its_props_and_refuses_a_non_prop_value():
    g = molrs.core.MolGraph()
    e = g.add_node(x=1.0, label="a", n=2, flag=True)
    assert g.get_node(e) == {"x": 1.0, "label": "a", "n": 2, "flag": True}
    with pytest.raises(TypeError):
        g.add_node(x=None)
    with pytest.raises(TypeError):
        g.add_node(x=[1.0])
    assert g.n_nodes == 1  # a refused add leaves the graph unchanged


def test_clear_node_unsets_one_field():
    g = molrs.core.MolGraph()
    e = g.add_node(x=1.0, y=2.0)
    g.clear_node(e, "x")
    assert g.get_node(e) == {"y": 2.0}
    g.clear_node(e, "x")  # no-op when absent
    assert g.get_node(e) == {"y": 2.0}


def test_leaf_node_writes_keep_the_leaf_invariant():
    # A view is the one door onto a leaf's node fields, and every write goes
    # through the leaf's own invariant: an atom has a valid `atomic_number`,
    # a bead a string `bead_type`.
    a = molrs.core.Atomistic()
    with pytest.raises(ValueError, match="atomic_number"):
        a.add_atom(x=0.0)
    with pytest.raises(ValueError, match="atomic_number"):
        a.add_atom(atomic_number=119)
    assert a.n_atoms == 0
    atom = a.add_atom(atomic_number=6)
    with pytest.raises(ValueError):
        atom["atomic_number"] = 200
    with pytest.raises(ValueError):
        del atom["atomic_number"]
    assert atom["atomic_number"] == 6

    cg = molrs.core.CoarseGrain()
    with pytest.raises(ValueError, match="bead_type"):
        cg.add_bead(x=0.0)
    bead = cg.add_bead(bead_type="W")
    with pytest.raises(ValueError):
        bead["bead_type"] = 3
    with pytest.raises(ValueError):
        del bead["bead_type"]
    assert bead["bead_type"] == "W"


def test_leaves_are_not_molgraphs():
    # The leaves are peers of MolGraph: nothing of its generic API reaches
    # their graph.
    assert not issubclass(molrs.core.Atomistic, molrs.core.MolGraph)
    assert not issubclass(molrs.core.CoarseGrain, molrs.core.MolGraph)
    for name in ("add_node", "set_node", "add_relation", "register_kind", "column"):
        assert not hasattr(molrs.core.Atomistic, name)
        assert not hasattr(molrs.core.CoarseGrain, name)


def test_get_relation_returns_endpoints_and_fields():
    g = molrs.core.MolGraph()
    g.register_kind("link", 2)
    a, b = g.add_node(), g.add_node()
    r = g.add_relation("link", [a, b], order=2, label="ab")
    assert g.get_relation("link", r) == ([a, b], {"order": 2, "label": "ab"})
    g.set_relation_prop("link", r, "order", 3)
    g.clear_relation_prop("link", r, "label")
    assert g.get_relation("link", r) == ([a, b], {"order": 3})


def test_del_relation_and_del_relations():
    g = molrs.core.MolGraph()
    g.register_kind("link", 2)
    a, b, c = g.add_node(), g.add_node(), g.add_node()
    r0 = g.add_relation("link", [a, b])
    r1 = g.add_relation("link", [b, c])
    r2 = g.add_relation("link", [a, c])
    g.del_relation("link", r0)
    assert set(g.relation_ids("link")) == {r1, r2}
    with pytest.raises(ValueError):
        g.del_relations("link", [r1, r0])  # r0 is stale: atomic refusal
    assert g.n_relations("link") == 2
    g.del_relations("link", [r1, r2])
    assert g.n_relations("link") == 0


def test_stale_handle_raises():
    g = molrs.core.MolGraph()
    e = g.add_node()
    g.del_node(e)
    with pytest.raises(ValueError):
        g.set_node(e, "x", 1.0)
    with pytest.raises(ValueError):
        g.get_node(e)


def test_relation_endpoints_survive_del_node_no_reindex():
    g = molrs.core.MolGraph()
    g.register_kind("link", 2)
    a, b, c = g.add_node(), g.add_node(), g.add_node()
    r = g.add_relation("link", [a, c])

    g.del_node(b)  # delete a non-endpoint node

    # The relation's endpoints still resolve to a and c — handles weren't
    # shifted by the swap-remove.
    assert g.get_relation("link", r)[0] == [a, c]
    assert g.get_node(a) == {}  # a still a valid (unset) node
    assert {a, c} == set(g.node_ids())


# --------------------------------------------------------------------------- #
# Component columns                                                           #
# --------------------------------------------------------------------------- #


def test_column_is_zero_copy_view_write_through():
    g = molrs.core.MolGraph()
    h0 = g.add_node(x=1.0)
    g.add_node(x=4.0)

    col = g.column(molrs.core.keys.X)
    assert isinstance(col, np.ndarray)
    assert col.tolist() == [1.0, 4.0]

    # Mutating the view writes through to the world.
    col[0] = 9.0
    assert g.get_node(h0)["x"] == 9.0


def test_validity_mask_reflects_set_components():
    g = molrs.core.MolGraph()
    h0 = g.add_node(x=0.0)
    g.add_node(x=0.0)
    g.set_node(h0, molrs.core.keys.CHARGE, -0.5)

    v = g.validity(molrs.core.keys.CHARGE)
    assert v.dtype == np.bool_
    assert v.tolist() == [True, False]


def test_column_with_a_hole_is_a_key_error_not_a_zero_fill():
    g = molrs.core.MolGraph()
    h0 = g.add_node(x=0.0)
    g.add_node(x=0.0)
    g.set_node(h0, molrs.core.keys.CHARGE, -0.5)

    with pytest.raises(KeyError, match="1 of 2"):
        g.column(molrs.core.keys.CHARGE)
    with pytest.raises(KeyError):
        g.column("never_set")


def test_column_comes_back_in_the_component_type():
    g = molrs.core.MolGraph()
    h0 = g.add_node(n=3, flag=True, label="C1")
    g.add_node(n=-1, flag=False, label="O1")

    assert g.column("label").tolist() == ["C1", "O1"]
    ints = g.column("n")
    assert ints.dtype == np.int32 and ints.tolist() == [3, -1]
    flags = g.column("flag")
    assert flags.dtype == np.bool_ and flags.tolist() == [True, False]
    # Non-f64 columns are copies: writing them does not reach the world.
    ints[0] = 99
    assert g.get_node(h0)["n"] == 3


def test_leaf_columns_come_back_in_the_component_type():
    # A leaf's bulk read is its Refs column: a copy, in the field's type.
    a = molrs.core.Atomistic()
    a.add_atom(atomic_number=6, n=3, flag=True, label="C1", x=0.5)
    a.add_atom(atomic_number=8, n=-1, flag=False, label="O1", x=1.5)
    species = a.atoms["atomic_number"]
    assert species.dtype == np.int32 and species.tolist() == [6, 8]
    assert a.atoms["n"].dtype == np.int32
    assert a.atoms["flag"].dtype == np.bool_
    assert a.atoms["label"].tolist() == ["C1", "O1"]
    x = a.atoms["x"]
    x[0] = 9.0  # a copy: the graph is untouched
    assert a.atoms[0]["x"] == 0.5


def test_columns_lists_every_registered_component():
    g = molrs.core.MolGraph()
    h0 = g.add_node(x=0.0, atomic_number=6)
    g.set_node(h0, molrs.core.keys.CHARGE, -0.5)

    cols = g.columns()
    assert isinstance(cols, list)
    assert {
        molrs.core.keys.X.key,
        molrs.core.keys.ATOMIC_NUMBER.key,
        molrs.core.keys.CHARGE.key,
    } <= set(cols)
    assert molrs.core.MolGraph().columns() == []


def test_missing_component_is_absent_and_type_conflict_raises():
    g = molrs.core.MolGraph()
    e = g.add_node()
    assert molrs.core.keys.X.key not in g.get_node(e)  # absent
    g.set_node(e, molrs.core.keys.CHARGE, 1.0)
    with pytest.raises(ValueError):
        g.set_node(e, molrs.core.keys.CHARGE, "not-a-number")  # type conflict


# --------------------------------------------------------------------------- #
# Rigid-body moves are leaf methods; perception is owned by a type           #
# --------------------------------------------------------------------------- #


def test_translate_rotate_and_scale_are_methods_of_the_two_leaves():
    for cls in (molrs.core.Atomistic, molrs.core.CoarseGrain):
        assert callable(cls.translate)
        assert callable(cls.rotate)
        assert callable(cls.scale)
        assert callable(cls.center)


LEAVES = [molrs.core.Atomistic, molrs.core.CoarseGrain]

# The field each leaf's invariant requires of every node.
SPECIES = {
    molrs.core.Atomistic: {"atomic_number": 6},
    molrs.core.CoarseGrain: {"bead_type": "W"},
}


def _add(mol, **props):
    """Add one node to either leaf through its own door."""
    if isinstance(mol, molrs.core.CoarseGrain):
        return mol.add_bead(**SPECIES[molrs.core.CoarseGrain], **props)
    return mol.add_atom(**SPECIES[molrs.core.Atomistic], **props)


def _one_node(cls):
    mol = cls()
    node = _add(mol, x=1.0, y=0.0, z=0.0)
    return mol, node


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_translate_moves_the_leaf_in_place(cls):
    mol, node = _one_node(cls)
    assert mol.translate([1.0, 0.0, 0.0]) is None
    assert node["x"] == 2.0


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rotate_moves_the_leaf_in_place(cls):
    mol, node = _one_node(cls)
    assert mol.rotate([0.0, 0.0, 1.0], np.pi / 2) is None
    assert node["y"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_scale_moves_the_leaf_in_place(cls):
    mol, node = _one_node(cls)
    assert mol.scale([2.0, 2.0, 2.0]) is None
    assert node["x"] == 2.0


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rigid_body_moves_compose(cls):
    mol, node = _one_node(cls)
    mol.translate([1.0, 0.0, 0.0])
    mol.rotate([0.0, 0.0, 1.0], np.pi)
    mol.scale([0.5, 0.5, 0.5])
    assert node["x"] == pytest.approx(-1.0, abs=1e-12)


@pytest.mark.parametrize(
    "axis",
    [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
    ids=["zero", "nan", "inf"],
)
@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_rotate_about_a_degenerate_axis_is_a_value_error(cls, axis):
    mol, node = _one_node(cls)
    with pytest.raises(ValueError):
        mol.rotate(axis, 1.0)
    assert node["x", "y", "z"] == [1.0, 0.0, 0.0]


def test_find_rings_system():
    bz = molrs.perceive.add_hydrogens(
        molrs.io.smiles.SmilesIr.parse("C1=CC=CC=C1").to_atomistic()
    )
    rings = molrs.perceive.perceive_rings(bz).rings()
    assert len(rings) == 1
    assert len(rings[0]) == 6  # six-membered ring


def test_gasteiger_charges_system():
    eth = molrs.perceive.add_hydrogens(
        molrs.io.smiles.SmilesIr.parse("CO").to_atomistic()
    )
    charges = np.asarray(molrs.ff.charge.GasteigerModel().assign(eth))
    # One charge per atom, hydrogens included, and neutral methanol sums to ~0.
    assert charges.shape == (eth.n_atoms,)
    assert charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_perceive_aromaticity_pipeline():
    # Aromaticity perception needs explicit hydrogens (pi-electron counting).
    bz = molrs.perceive.add_hydrogens(molrs.io.smiles.SmilesIr.parse("C1=CC=CC=C1").to_atomistic())
    molrs.perceive.assign_aromaticity(bz)
    aromatic = [atom for atom in bz.atoms if atom.get("is_aromatic")]
    assert len(aromatic) == 6


# --------------------------------------------------------------------------- #
# Leaves hold a core leaf; frames, copies                                     #
# --------------------------------------------------------------------------- #


def test_leaf_frame_round_trip():
    a = molrs.core.Atomistic()
    h1 = a.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    h2 = a.add_atom(atomic_number=8, x=1.2, y=0.0, z=0.0)
    a.add_bond(h1, h2)

    frame = a.to_frame()
    a2 = molrs.core.Atomistic.from_frame(frame)
    assert a2.n_atoms == 2
    assert a2.n_bonds == 1
    assert len(a2.bonds) == 1


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_copy_is_independent_and_keeps_handles(cls, clone):
    mol = cls(name="m")
    node = _add(mol, x=1.0, y=0.0, z=0.0)
    dup = clone(mol)
    assert type(dup) is cls
    assert dup.props == {"name": "m"} and dup.props is not mol.props
    first = dup.beads[0] if cls is molrs.core.CoarseGrain else dup.atoms[0]
    assert first.handle == node.handle
    first["x"] = 5.0
    assert node["x"] == 1.0


def test_find_rotatable_unknown_bond_policy():
    # Butane skeleton; the middle bond's class is cleared to "unknown".
    mol = molrs.core.Atomistic()
    c = [mol.add_atom(atomic_number=6, x=float(i), y=0.0, z=0.0) for i in range(4)]
    bonds = [mol.add_bond(c[i], c[i + 1]) for i in range(3)]
    mol.set_bond_class(bonds[1], molrs.core.BondOrder.Unknown, molrs.core.BondNumber.Unknown)
    policy = molrs.perceive.UnknownBondPolicy

    molrs.perceive.assign_rotatable_bonds(mol)
    assert bonds[1]["is_rotatable"] == 0
    molrs.perceive.assign_rotatable_bonds(mol, unknown_bond=policy.AsSingle)
    assert bonds[1]["is_rotatable"] == 1
    with pytest.raises(TypeError):
        molrs.perceive.assign_rotatable_bonds(mol, unknown_bond="guess")


# --------------------------------------------------------------------------- #
# adopt — zero-copy move                                                      #
# --------------------------------------------------------------------------- #


def test_adopt_moves_storage_and_empties_source():
    src = molrs.core.MolGraph()
    s0 = src.add_node(x=7.0)

    dst = molrs.core.MolGraph()
    dst.adopt(src)

    assert dst.node_ids() == [s0]
    assert dst.get_node(s0)["x"] == 7.0
    assert len(src.node_ids()) == 0  # source emptied


# --------------------------------------------------------------------------- #
# keys convention                                                            #
# --------------------------------------------------------------------------- #


def test_keys_convention_exposed():
    assert isinstance(molrs.core.keys.X, molrs.core.keys.Key)
    assert molrs.core.keys.X.key == "x"
    assert molrs.core.keys.ELEMENT.key == "element"
    assert molrs.core.keys.CHARGE.key == "charge"
    # Key equals its string form for convenient comparisons.
    assert molrs.core.keys.X == "x"
    assert [k.key for k in molrs.core.keys.COORDS] == ["x", "y", "z"]
    by_str = molrs.core.schema.column("atomic_number")
    by_key = molrs.core.schema.column(molrs.core.keys.ATOMIC_NUMBER)
    assert by_str is not None and by_key is not None
    assert by_key.key == by_str.key
    assert by_key.dtype == by_str.dtype


def test_column_spec_exposes_dimension_and_the_unit_derived_from_it():
    # SchemaDocument: `x` has dimension "length", whose real-preset unit is
    # "angstrom" (molrs/src/core/schema/document.rs).
    x = molrs.core.schema.column("x")
    assert x is not None
    assert x.dimension == "length"
    assert x.unit == "angstrom"


# --------------------------------------------------------------------------- #
# relation enumeration + geometry                                             #
# --------------------------------------------------------------------------- #


def test_relation_ids_enumerates_handles():
    """Authoritative enumeration — replaces probing opaque handle ranges."""
    g = molrs.core.MolGraph()
    a, b, c = (g.add_node() for _ in range(3))
    g.register_kind("bond", 2)
    rh1 = g.add_relation("bond", [a, b])
    rh2 = g.add_relation("bond", [b, c])
    assert set(g.relation_ids("bond")) == {rh1, rh2}
    assert g.n_relations("bond") == 2


def test_relation_ids_empty_for_registered_kind():
    g = molrs.core.MolGraph()
    g.register_kind("angle", 3)
    assert g.relation_ids("angle") == []


def test_relation_ids_unregistered_kind_raises():
    g = molrs.core.MolGraph()
    with pytest.raises(ValueError):
        g.relation_ids("nope")


def test_scale_about_center():
    mol = molrs.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=6, x=float(i), y=0.0, z=0.0) for i in range(3)]
    mol.scale([2.0, 2.0, 2.0], [1.0, 0.0, 0.0])
    assert [atom["x"] for atom in atoms] == [-1.0, 1.0, 3.0]


def test_scale_uniform_about_origin():
    mol = molrs.core.Atomistic()
    atom = mol.add_atom(atomic_number=6, x=1.0, y=2.0, z=3.0)
    mol.scale([0.5, 0.5, 0.5])
    assert atom["x", "y", "z"] == [0.5, 1.0, 1.5]


# --------------------------------------------------------------------------- #
# center — a query on each leaf, same signature on both                       #
# --------------------------------------------------------------------------- #
#
# Seam only: the mass-weighted centre R = sum(m_i r_i) / sum(m_i) and every
# refusal are proven by molrs/src/op/geometry.rs. These tests check the call
# shape per leaf, the ndarray that crosses back, and the error mapping (every
# CenterError is a ValueError naming the int handle).


def _one_weighted_node(cls, position, mass):
    """A leaf holding one node at ``position`` (Å) with ``mass`` (g/mol)."""
    mol = cls()
    x, y, z = position
    node = _add(mol, x=x, y=y, z=z, mass=mass)
    return mol, node


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_returns_a_float64_triple_of_the_given_nodes(cls):
    mol, node = _one_weighted_node(cls, (1.0, 2.0, 3.0), 1.0)

    center = mol.center([node])

    assert isinstance(center, np.ndarray)
    assert center.dtype == np.float64
    assert center.shape == (3,)
    np.testing.assert_allclose(center, [1.0, 2.0, 3.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_takes_a_refs_of_the_leaf(cls):
    mol, _ = _one_weighted_node(cls, (1.0, 2.0, 3.0), 1.0)
    _add(mol, x=3.0, y=2.0, z=3.0, mass=1.0)
    nodes = mol.beads if cls is molrs.core.CoarseGrain else mol.atoms
    np.testing.assert_allclose(mol.center(nodes), [2.0, 2.0, 3.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_of_a_node_of_another_graph_is_a_value_error(cls):
    mol, _ = _one_weighted_node(cls, (0.0, 0.0, 0.0), 1.0)
    _, foreign = _one_weighted_node(cls, (0.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="another graph"):
        mol.center([foreign])


def test_center_of_a_deleted_bead_is_a_value_error_naming_the_handle():
    cg, live = _one_weighted_node(molrs.core.CoarseGrain, (0.0, 0.0, 0.0), 1.0)
    stale = cg.add_bead(bead_type="W")
    cg.del_bead(stale)

    with pytest.raises(ValueError) as excinfo:
        cg.center([live, stale])

    message = str(excinfo.value)
    assert str(stale.handle) in message
    assert "NodeId(" not in message


@pytest.mark.parametrize("cls", LEAVES, ids=lambda c: c.__name__)
def test_center_with_a_non_finite_mass_is_a_value_error_naming_the_handle(cls):
    mol, node = _one_weighted_node(cls, (1.0, 2.0, 3.0), float("nan"))

    with pytest.raises(ValueError) as excinfo:
        mol.center([node])

    message = str(excinfo.value)
    assert str(node.handle) in message
    assert "NodeId(" not in message


def test_to_frame_columns_are_selected_on_the_block():
    # `to_frame` exports everything; picking columns is the block's own job.
    mol = molrs.core.Atomistic()
    mol.add_atom(atomic_number=8, x=0.0, y=0.0, z=0.0, charge=-0.8)
    atoms = mol.to_frame()["atoms"]
    assert {"atomic_number", "x", "y", "z", "charge"} <= set(atoms.keys())
    assert "mass" not in atoms


def test_perception_is_free_functions():
    # Two verbs: `perceive_*` reports a side table, `assign_*` writes onto the
    # molecule it is given.
    mol = molrs.io.smiles.SmilesIr.parse("C1CC1C").to_atomistic()
    info = molrs.perceive.perceive_rings(mol)
    assert info.ring_sizes() == [3]
    with pytest.raises(TypeError):
        molrs.perceive.RingSet(mol)
    assert not any("is_in_ring" in atom for atom in mol.atoms)
    assert molrs.perceive.assign_rings(mol) is None
    in_ring = sorted(atom["is_in_ring"] for atom in mol.atoms)
    assert in_ring == [0, 1, 1, 1]
