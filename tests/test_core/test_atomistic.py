"""Live-only Atomistic factory API."""

import numpy as np
import pytest

from molpy import Angle, Atom, Atomistic, Bond, Dihedral


def test_def_factories_create_live_interned_refs() -> None:
    struct = Atomistic()
    atoms = struct.def_atoms(
        [
            {"element": "H"},
            {"element": "C", "x": 0, "y": 0, "z": 0},
            {"element": "C"},
            {"element": "H"},
        ]
    )
    bond = struct.def_bond(atoms[0], atoms[1], bond_type=1, bond_number=1)
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
    assert bond["bond_type"] == 1
    assert bond["bond_number"] == 1
    assert angle["theta"] == 109.5
    assert dihedral["phi"] == 180.0


def test_batch_factories() -> None:
    struct = Atomistic()
    atoms = struct.def_atoms([{"element": "C"}, {"element": "H"}, {"element": "H"}])
    bonds = struct.def_bonds(
        [
            (atoms[0], atoms[1], {"bond_type": 1, "bond_number": 1}),
            (atoms[0], atoms[2], {"bond_type": 1, "bond_number": 1}),
        ]
    )
    angles = struct.def_angles([(atoms[1], atoms[0], atoms[2], {"theta": 109.5})])

    assert np.array_equal(struct.atoms["element"], ["C", "H", "H"])
    assert bonds == list(struct.bonds)
    assert angles == list(struct.angles)


def test_cross_world_endpoints_are_rejected() -> None:
    left = Atomistic()
    right = Atomistic()
    a = left.def_atom(element="C")
    b = right.def_atom(element="H")

    with pytest.raises(ValueError, match="belong to this graph"):
        left.def_bond(a, b)


class TestAtomisticSpatialVerbs:
    """The rigid-body verbs are the native ``translate`` / ``rotate`` / ``scale``."""

    def test_move_is_not_an_atomistic_verb(self) -> None:
        assert not hasattr(Atomistic, "move")

    def test_spatial_verbs_are_the_native_methods(self) -> None:
        import molrs

        for verb in ("translate", "rotate", "scale"):
            assert getattr(Atomistic, verb) is getattr(molrs.Atomistic, verb), verb

    def test_scale_takes_per_axis_factors_and_chains(self) -> None:
        struct = Atomistic()
        atom = struct.def_atom(element="C", x=1.0, y=1.0, z=1.0)

        result = (
            struct.translate([1.0, 0.0, 0.0])
            .scale([2.0, 3.0, 4.0], [0.0, 0.0, 0.0])
            .rotate([0.0, 0.0, 1.0], 0.0)
        )

        assert result is struct
        assert (atom["x"], atom["y"], atom["z"]) == pytest.approx(
            (4.0, 3.0, 4.0), abs=1e-12
        )
