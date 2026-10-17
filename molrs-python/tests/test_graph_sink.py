"""graph-sink-03 — Python bindings for extract / merge / copy contracts."""

from __future__ import annotations

import copy
import itertools

import molrs
import pytest


def _chain(n: int) -> tuple[molrs.core.Atomistic, list[molrs.core.Atom]]:
    mol = molrs.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=6, x=float(i), y=0.0, z=0.0) for i in range(n)]
    for a, b in itertools.pairwise(atoms):
        mol.add_bond(a, b)
    return mol, atoms


def test_copy_preserves_handles():
    mol, atoms = _chain(2)
    c = copy.copy(mol)
    assert [atom.handle for atom in c.atoms] == [atom.handle for atom in atoms]
    assert c.atoms[0]["atomic_number"] == atoms[0]["atomic_number"] == 6


def test_merge_returns_the_merged_atoms_and_empties_other():
    a, _ = _chain(2)
    b, batoms = _chain(2)
    labels = ["first", "second"]
    for atom, label in zip(batoms, labels, strict=True):
        atom["label"] = label
    n_a = a.n_atoms
    n_b = b.n_atoms
    merged = a.merge(b)
    assert isinstance(merged, molrs.core.Refs)
    assert len(merged) == n_b
    assert a.n_atoms == n_a + n_b
    assert b.n_atoms == 0
    # In other's former row order, now views of this graph.
    assert [atom["label"] for atom in merged] == labels
    assert all(atom.world is a for atom in merged)
    assert a.n_bonds == 2


def test_merge_with_itself_is_refused():
    a, _ = _chain(2)
    with pytest.raises(ValueError, match="itself"):
        a.merge(a)


def test_extract_subgraph_maps_consistent():
    mol, atoms = _chain(10)
    result = mol.extract_subgraph([atoms[5]], 2)
    assert isinstance(result, molrs.core.ExtractedSubgraph)
    assert isinstance(result.graph, molrs.core.Atomistic)
    assert set(result.node_map.keys()) == set(result.hops.keys())
    assert result.graph.n_atoms == len(result.node_map)
    assert result.hops[atoms[5]] == 0
    for b in result.boundary:
        assert b in result.hops
        assert b.world is mol
    for parent, child in result.node_map.items():
        assert child.world is result.graph
        assert result.parent_of[child] is parent
        assert child["x"] == parent["x"]


def test_extract_regenerate_topology():
    mol, atoms = _chain(5)
    # no angles on parent
    result = mol.extract_subgraph([atoms[2]], 2, regenerate_topology=True)
    # full chain in ball; angles were perceived on it
    assert result.graph.n_atoms == 5
    assert result.graph.n_angles > 0


def test_induced_subgraph():
    mol, atoms = _chain(4)
    sub, node_map = mol.induced_subgraph([atoms[1], atoms[2]])
    assert sub.n_atoms == 2
    assert set(node_map) == {atoms[1], atoms[2]}
    assert all(child.world is sub for child in node_map.values())
    assert sub.n_bonds == 1
