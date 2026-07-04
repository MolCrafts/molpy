"""Geometric distribution regression benchmarks — ADF / DDF / distance-DF, CDF.

The molrs kernel reads the atom tuples to histogram from the frame's core
topology blocks (``bonds`` / ``angles`` / ``dihedrals``), so these forward
``([frame])`` only — no separate ``groups`` index array. Regression sizing: one
small perceived chain.
"""

from __future__ import annotations

import pytest

import molpy as mp

pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
def topo_frame():
    """Small carbon chain with coords + perceived angle/dihedral topology."""
    n = 60
    mol = mp.Atomistic()
    atoms = [
        mol.def_atom(element="C", xyz=[float(i), 0.3 * (i % 2), 0.0]) for i in range(n)
    ]
    for i in range(n - 1):
        mol.def_bond(atoms[i], atoms[i + 1])
    return mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()


def test_distance_distribution(benchmark, topo_frame):
    op = mp.compute.DistanceDistribution(50, 0.0, 6.0)
    result = benchmark(lambda: op([topo_frame]))
    assert result.density.shape == (50,)


def test_angle_distribution(benchmark, topo_frame):
    op = mp.compute.AngleDistribution(60, 0.0, 180.0)
    result = benchmark(lambda: op([topo_frame]))
    assert result.density.shape == (60,)


def test_dihedral_distribution(benchmark, topo_frame):
    op = mp.compute.DihedralDistribution(60)
    result = benchmark(lambda: op([topo_frame]))
    assert result.density.shape == (60,)


def test_combined_distribution(benchmark, topo_frame):
    op = mp.compute.CombinedDistribution(
        [("angle", 30, 0.0, 180.0, True), ("angle", 30, 0.0, 180.0, True)]
    )
    result = benchmark(lambda: op([topo_frame]))
    assert result.ndim == 2
