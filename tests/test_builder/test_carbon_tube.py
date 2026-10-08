"""``mp.builder.CarbonTubeBuilder`` — molrs's, by identity.

The builder makes a :class:`~molpy.Frame`; ``mp.Atomistic.from_frame`` makes
it a graph, and ``generate_topology`` adds angles and dihedrals.
"""

from math import gcd, pi, sqrt

import molrs
import numpy as np
import pytest

import molpy as mp
from molpy.builder import CarbonTubeBuilder


def _graph(builder: CarbonTubeBuilder, **kwargs) -> mp.Atomistic:
    return mp.Atomistic.from_frame(builder.build(**kwargs))


def test_the_builder_is_molrs_s():
    assert CarbonTubeBuilder is molrs.builder.CarbonTubeBuilder


class TestCarbonTubeBuilder:
    @pytest.mark.parametrize("n,m", [(6, 0), (5, 5), (4, 2), (0, 6)])
    def test_periodic_graph_has_exact_graphene_topology(self, n, m):
        cells = 2
        tube = _graph(CarbonTubeBuilder(n, m, cells=cells, periodic=True))
        expected = 4 * (n * n + n * m + m * m) // gcd(2 * m + n, 2 * n + m) * cells
        degrees = {atom: 0 for atom in tube.atoms}
        for bond in tube.bonds:
            for endpoint in bond.endpoints:
                degrees[endpoint] += 1

        assert len(tube.atoms) == expected
        assert len(tube.bonds) == 3 * expected // 2
        assert set(degrees.values()) == {3}

    def test_coordinates_follow_requested_chiral_radius(self):
        n, m, bond_length = 4, 2, 1.42
        tube = _graph(CarbonTubeBuilder(n, m, cells=1, bond_length=bond_length))
        expected_radius = (
            sqrt(3.0) * bond_length * sqrt(n * n + n * m + m * m) / (2.0 * pi)
        )
        radii = np.linalg.norm(tube.atoms["x", "y", "z"][:, :2], axis=1)
        assert radii == pytest.approx(expected_radius)

    def test_nonperiodic_tube_leaves_open_graph_ends(self):
        builder = CarbonTubeBuilder(6, 0, cells=2, periodic=False)
        tube = _graph(builder)
        degrees = {atom: 0 for atom in tube.atoms}
        for bond in tube.bonds:
            for endpoint in bond.endpoints:
                degrees[endpoint] += 1
        assert min(degrees.values()) < 3
        assert max(degrees.values()) == 3
        assert np.array_equal(builder.cell().pbc, [False, False, False])

    def test_periodic_cell_closes_only_the_axis(self):
        builder = CarbonTubeBuilder(5, 5, cells=2, periodic=True)
        box = builder.cell(vacuum=4.0)
        radius = np.linalg.norm(_graph(builder).atoms["x", "y", "z"][0, :2])
        assert np.array_equal(box.pbc, [False, False, True])
        assert box.lengths[0] == pytest.approx(2.0 * radius + 8.0)
        assert box.lengths[1] == pytest.approx(box.lengths[0])

    def test_the_cell_is_never_written_onto_the_graph(self):
        """A graph is topology and chemistry; the cell lives on ``frame.box``."""
        builder = CarbonTubeBuilder(5, 5, cells=2, periodic=True)
        assert "box" not in _graph(builder).props

    def test_length_rounds_up_to_complete_axial_units(self):
        unit = CarbonTubeBuilder(6, 0, cells=1).cell().lengths[2]
        requested = 2.2 * unit
        tube = CarbonTubeBuilder(6, 0, length=requested)
        assert tube.cell().lengths[2] >= requested
        assert tube.cell().lengths[2] < requested + unit

    def test_per_atom_annotations_are_written_without_bonded_typing(self):
        tube = _graph(CarbonTubeBuilder(6, 0), atom_type="CA", charge=-0.125)
        assert {atom.get("element") for atom in tube.atoms} == {"C"}
        assert {atom.get("type") for atom in tube.atoms} == {"CA"}
        assert {atom.get("charge") for atom in tube.atoms} == {-0.125}
        assert not list(tube.angles)
        assert not list(tube.dihedrals)

    def test_topology_is_generated_on_the_graph(self):
        builder = CarbonTubeBuilder(4, 2, cells=1)
        atoms_only = _graph(builder)
        topology = _graph(builder)
        topology.generate_topology(gen_angle=True, gen_dihedral=True)
        assert not list(atoms_only.angles)
        assert not list(atoms_only.dihedrals)
        assert list(topology.angles)
        assert list(topology.dihedrals)

    def test_repeated_builds_return_independent_graphs(self):
        builder = CarbonTubeBuilder(4, 2, cells=2)
        first = _graph(builder, charge=0.0)
        second = _graph(builder, charge=0.0)
        first.atoms[0]["charge"] = 1.0
        assert second.atoms[0].get("charge") == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "args,kwargs,error",
        [
            ((0, 0), {}, ValueError),
            ((1, 1), {}, ValueError),
            ((5.0, 5), {}, TypeError),
            ((5, 5), {"length": 10.0, "cells": 2}, TypeError),
            ((5, 5), {"cells": 0}, ValueError),
            ((5, 5), {"bond_length": 0.0}, ValueError),
            ((5, 5), {"periodic": 1}, TypeError),
            ((5, 5), {"cells": 1, "periodic": True}, ValueError),
        ],
    )
    def test_invalid_shapes_are_rejected(self, args, kwargs, error):
        with pytest.raises(error):
            CarbonTubeBuilder(*args, **kwargs)

    def test_invalid_atom_annotations_are_rejected(self):
        with pytest.raises(ValueError):
            _graph(CarbonTubeBuilder(5, 5), atom_type="")

    def test_negative_vacuum_is_rejected(self):
        with pytest.raises(ValueError):
            CarbonTubeBuilder(5, 5).cell(vacuum=-1.0)
