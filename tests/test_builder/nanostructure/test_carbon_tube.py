"""Unit tests for :mod:`molpy.builder.nanostructure.carbon_tube`."""

from math import gcd, pi, sqrt

import numpy as np
import pytest

from molpy.builder import CarbonTubeBuilder
from molpy.core import fields


class TestCarbonTubeBuilder:
    @pytest.mark.parametrize("n,m", [(6, 0), (5, 5), (4, 2), (0, 6)])
    def test_periodic_graph_has_exact_graphene_topology(self, n, m):
        cells = 2
        tube = CarbonTubeBuilder().build(n, m, cells=cells, periodic=True)
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
        tube = CarbonTubeBuilder().build(n, m, cells=1, bond_length=bond_length)
        expected_radius = (
            sqrt(3.0) * bond_length * sqrt(n * n + n * m + m * m) / (2.0 * pi)
        )
        radii = np.linalg.norm(tube.xyz[:, :2], axis=1)
        assert radii == pytest.approx(expected_radius)

    def test_nonperiodic_tube_leaves_open_graph_ends(self):
        tube = CarbonTubeBuilder().build(6, 0, cells=2, periodic=False)
        degrees = {atom: 0 for atom in tube.atoms}
        for bond in tube.bonds:
            for endpoint in bond.endpoints:
                degrees[endpoint] += 1
        assert min(degrees.values()) < 3
        assert max(degrees.values()) == 3
        assert np.array_equal(tube["box"].pbc, [False, False, False])

    def test_periodic_box_closes_only_the_axis(self):
        tube = CarbonTubeBuilder().build(5, 5, cells=2, periodic=True, vacuum=4.0)
        box = tube["box"]
        radius = np.linalg.norm(tube.xyz[0, :2])
        assert np.array_equal(box.pbc, [False, False, True])
        assert box.lx == pytest.approx(2.0 * radius + 8.0)
        assert box.ly == pytest.approx(box.lx)

    def test_length_rounds_up_to_complete_axial_units(self):
        short = CarbonTubeBuilder().build(6, 0, cells=1)
        requested = 2.2 * short["box"].lz
        tube = CarbonTubeBuilder().build(6, 0, length=requested)
        assert tube["box"].lz >= requested
        assert tube["box"].lz < requested + short["box"].lz

    def test_per_atom_annotations_are_written_without_bonded_typing(self):
        tube = CarbonTubeBuilder().build(6, 0, atom_type="CA", charge=-0.125)
        assert {atom.get(fields.ELEMENT) for atom in tube.atoms} == {"C"}
        assert {atom.get(fields.TYPE) for atom in tube.atoms} == {"CA"}
        assert {atom.get(fields.CHARGE) for atom in tube.atoms} == {-0.125}
        assert not list(tube.angles)
        assert not list(tube.dihedrals)

    def test_topology_finalization_is_optional(self):
        atoms_only = CarbonTubeBuilder().build(4, 2, cells=1)
        topology = CarbonTubeBuilder().build(4, 2, cells=1, finalize="topology")
        assert not list(atoms_only.angles)
        assert not list(atoms_only.dihedrals)
        assert list(topology.angles)
        assert list(topology.dihedrals)

    def test_cached_compilation_still_returns_independent_graphs(self):
        builder = CarbonTubeBuilder()
        before = builder._compile.cache_info()
        first = builder.build(4, 2, cells=2, charge=0.0)
        second = builder.build(4, 2, cells=2, charge=0.0)
        after = builder._compile.cache_info()
        first.atoms[0][fields.CHARGE] = 1.0
        assert second.atoms[0].get(fields.CHARGE) == pytest.approx(0.0)
        assert after.hits >= before.hits + 1

    @pytest.mark.parametrize(
        "args,kwargs,error",
        [
            ((0, 0), {}, ValueError),
            ((1, 1), {}, ValueError),
            ((5.0, 5), {}, TypeError),
            ((5, 5), {"length": 10.0, "cells": 2}, TypeError),
            ((5, 5), {"cells": 0}, ValueError),
            ((5, 5), {"bond_length": 0.0}, ValueError),
            ((5, 5), {"vacuum": -1.0}, ValueError),
            ((5, 5), {"periodic": 1}, TypeError),
            ((5, 5), {"atom_type": ""}, ValueError),
            ((5, 5), {"cells": 1, "periodic": True}, ValueError),
        ],
    )
    def test_invalid_inputs_are_rejected(self, args, kwargs, error):
        with pytest.raises(error):
            CarbonTubeBuilder().build(*args, **kwargs)
