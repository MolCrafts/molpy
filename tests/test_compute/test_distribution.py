"""Geometric distribution functions (ADF / DDF / distance-DF, CDF) — molrs-backed.

Regression guard: the molrs kernel reads the atom tuples to histogram from each
frame's core topology blocks (``bonds`` / ``angles`` / ``dihedrals``), so the
molpy wrappers forward ``(frames)`` ONLY — no separate ``groups`` index array.
A prior signature passed ``compute(frames, groups)`` and raised ``TypeError``
against molrs; these tests pin the no-``groups`` contract.
"""

from __future__ import annotations

import molpy as mp
from molpy.compute import (
    AngleDistribution,
    CombinedDistribution,
    DihedralDistribution,
    DistanceDistribution,
)
from molpy.compute.base import Compute


def _chain_frame(n: int = 6):
    """A small carbon chain with coordinates + perceived angle/dihedral topology."""
    mol = mp.Atomistic()
    atoms = [
        mol.def_atom(element="C", xyz=[float(i), 0.3 * (i % 2), 0.0]) for i in range(n)
    ]
    for i in range(n - 1):
        mol.def_bond(atoms[i], atoms[i + 1])
    return mol.get_topo(gen_angle=True, gen_dihe=True).to_frame()


def test_distribution_ops_are_compute_subclasses():
    for cls in (
        DistanceDistribution,
        AngleDistribution,
        DihedralDistribution,
        CombinedDistribution,
    ):
        assert issubclass(cls, Compute)


def test_frame_carries_core_topology_blocks():
    frame = _chain_frame()
    for block in ("atoms", "bonds", "angles", "dihedrals"):
        assert block in frame.keys()


def test_distance_distribution_reads_bonds_from_frame():
    frame = _chain_frame()
    # No groups argument — atom pairs come from the frame's `bonds` block.
    result = DistanceDistribution(30, 0.0, 6.0)([frame])
    assert result.density.shape == (30,)


def test_angle_distribution_reads_angles_from_frame():
    frame = _chain_frame()
    result = AngleDistribution(30, 0.0, 180.0)([frame])
    assert result.density.shape == (30,)


def test_dihedral_distribution_reads_dihedrals_from_frame():
    frame = _chain_frame()
    result = DihedralDistribution(30)([frame])
    assert result.density.shape == (30,)


def test_combined_distribution_reads_topology_from_frame():
    frame = _chain_frame()
    # Two angle axes read the same `angles` block, so per-axis sample counts match.
    cdf = CombinedDistribution(
        [("angle", 20, 0.0, 180.0, True), ("angle", 20, 0.0, 180.0, True)]
    )
    result = cdf([frame])
    assert result.ndim == 2
