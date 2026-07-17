"""Hydrogen perception is a chemistry operation, not a graph-container method."""

from itertools import combinations
from math import acos, degrees

import molrs
import numpy as np
import pytest

from molpy.core.atomistic import Atomistic


TETRAHEDRAL = [
    np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0),
    np.array([1.0, -1.0, -1.0]) / np.sqrt(3.0),
    np.array([-1.0, 1.0, -1.0]) / np.sqrt(3.0),
    np.array([-1.0, -1.0, 1.0]) / np.sqrt(3.0),
]


def _find_hydrogens(graph: Atomistic) -> Atomistic:
    return Atomistic.adopt(molrs.Perceive().find_hydrogens(graph))


def _carbon_with(degree: int) -> Atomistic:
    graph = Atomistic()
    carbon = graph.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    for direction in TETRAHEDRAL[:degree]:
        neighbor = graph.def_atom(
            element="C", x=direction[0], y=direction[1], z=direction[2]
        )
        graph.def_bond(carbon, neighbor)
    return graph


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_carbon_is_completed_to_four_bonds(degree: int) -> None:
    perceived = _find_hydrogens(_carbon_with(degree))
    carbon = perceived.atoms[0]
    neighbors = perceived.get_neighbors(carbon)

    assert len(neighbors) == 4
    assert sum(atom["element"] == "H" for atom in neighbors) == 4 - degree


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_added_hydrogens_have_tetrahedral_geometry(degree: int) -> None:
    perceived = _find_hydrogens(_carbon_with(degree))
    carbon = perceived.atoms[0]
    origin = np.array(carbon["xyz"])
    vectors = [
        np.array(neighbor["xyz"]) - origin
        for neighbor in perceived.get_neighbors(carbon)
    ]

    for left, right in combinations(vectors, 2):
        cosine = np.dot(left, right) / np.linalg.norm(left) / np.linalg.norm(right)
        assert degrees(acos(np.clip(cosine, -1.0, 1.0))) == pytest.approx(
            109.47, abs=0.2
        )


def test_double_bonded_oxygen_is_not_overcompleted() -> None:
    graph = Atomistic()
    carbon = graph.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    oxygen = graph.def_atom(element="O", x=1.2, y=0.0, z=0.0)
    graph.def_bond(carbon, oxygen, order=2.0)

    perceived = _find_hydrogens(graph)
    assert len(perceived.get_neighbors(perceived.atoms[1])) == 1


def test_operation_is_non_mutating_and_preserves_existing_topology() -> None:
    graph = Atomistic()
    atoms = [graph.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(4)]
    for left, right in zip(atoms, atoms[1:], strict=False):
        graph.def_bond(left, right)
    graph.def_angle(atoms[0], atoms[1], atoms[2], type="c-c-c")
    graph.def_angle(atoms[1], atoms[2], atoms[3])
    graph.def_dihedral(*atoms)

    perceived = _find_hydrogens(graph)

    assert len(graph.atoms) == 4
    assert len(perceived.atoms) > 4
    assert len(perceived.angles) == 2
    assert len(perceived.dihedrals) == 1
    assert perceived.angles[0]["type"] == "c-c-c"


def test_graph_leaves_expose_no_completion_facade() -> None:
    assert not hasattr(Atomistic(), "complete_valence")
    from molpy.core.cg import CoarseGrain

    assert not hasattr(CoarseGrain(), "complete_valence")
