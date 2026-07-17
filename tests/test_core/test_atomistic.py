"""Live-only Atomistic factory API."""

import numpy as np
import pytest

from molpy import Angle, Atom, Atomistic, Bond, Dihedral


def test_def_factories_create_live_interned_refs() -> None:
    struct = Atomistic()
    atoms = struct.def_atoms(
        [
            {"element": "H"},
            {"element": "C", "xyz": [0, 0, 0]},
            {"element": "C"},
            {"element": "H"},
        ]
    )
    bond = struct.def_bond(atoms[0], atoms[1], order=1)
    angle = struct.def_angle(atoms[0], atoms[1], atoms[2], theta=109.5)
    dihedral = struct.def_dihedral(*atoms, phi=180.0)

    assert all(isinstance(atom, Atom) for atom in atoms)
    assert isinstance(bond, Bond)
    assert isinstance(angle, Angle)
    assert isinstance(dihedral, Dihedral)
    assert struct.atoms[0] is atoms[0]
    assert struct.bonds[0] is bond
    assert bond.itom is atoms[0]
    assert bond.jtom is atoms[1]
    assert angle.ktom is atoms[2]
    assert dihedral.ltom is atoms[3]
    assert bond["order"] == 1
    assert angle["theta"] == 109.5
    assert dihedral["phi"] == 180.0


def test_batch_factories() -> None:
    struct = Atomistic()
    atoms = struct.def_atoms([{"element": "C"}, {"element": "H"}, {"element": "H"}])
    bonds = struct.def_bonds(
        [(atoms[0], atoms[1], {"order": 1}), (atoms[0], atoms[2], {"order": 1})]
    )
    angles = struct.def_angles([(atoms[1], atoms[0], atoms[2], {"theta": 109.5})])

    assert np.array_equal(struct.atoms["element"], ["C", "H", "H"])
    assert bonds == list(struct.bonds)
    assert angles == list(struct.angles)


def test_detached_construction_and_add_api_do_not_exist() -> None:
    struct = Atomistic()
    with pytest.raises(TypeError):
        Atom(element="C")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Bond(object(), object())  # type: ignore[call-arg]

    assert not hasattr(struct, "add_atom")
    assert not hasattr(struct, "add_bond")
    assert not hasattr(struct, "add_entity")
    assert not hasattr(struct, "add_link")


def test_cross_world_endpoints_are_rejected() -> None:
    left = Atomistic()
    right = Atomistic()
    a = left.def_atom(element="C")
    b = right.def_atom(element="H")

    with pytest.raises(ValueError, match="belong to this graph"):
        left.def_bond(a, b)
