"""molpy.compute time-correlation benchmarks: VanHove and reorientation.

Van Hove G(r, t) and the first/second Legendre reorientational TCFs — both
consume a small position trajectory. VanHove is frame-only; reorientation is
also frame-only — the ``(tail, head)`` endpoints of each tracked bond vector are
read from each frame's core ``bonds`` topology block.
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import LegendreReorientation, VanHove

pytestmark = pytest.mark.benchmark


def _bonded_chain_frames(n_atoms: int = 20, n_frames: int = 6) -> list["mp.Frame"]:
    """A bonded carbon chain over a few frames; ``(tail, head)`` pairs live in-frame."""
    frames = []
    for t in range(n_frames):
        mol = mp.Atomistic()
        amp = 0.3 + 0.1 * t
        atoms = [
            mol.def_atom(element="C", xyz=[float(i), amp * (i % 2), 0.0])
            for i in range(n_atoms)
        ]
        for i in range(n_atoms - 1):
            mol.def_bond(atoms[i], atoms[i + 1])
        frames.append(mol.get_topo().to_frame())
    return frames


def test_van_hove(benchmark, pos_traj) -> None:
    op = VanHove(n_rbins=50, r_max=10.0, lags=[1, 2, 3])
    out = benchmark(op, pos_traj)
    assert np.asarray(out.g_self).shape[0] >= 1


def test_legendre_reorientation(benchmark) -> None:
    # Bond vectors are read from each frame's core `bonds` block (no `pairs`).
    frames = _bonded_chain_frames()
    op = LegendreReorientation(max_lag=3)
    out = benchmark(op, frames)
    assert np.asarray(out.c1).shape[0] >= 1
