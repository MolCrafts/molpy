"""Legendre reorientational TCFs (``C_1``/``C_2``) — molrs-backed.

Regression guard: the molrs kernel reads the ``(tail, head)`` endpoints of each
tracked bond vector from each frame's core ``bonds`` topology block, so the molpy
wrapper forwards ``(frames)`` ONLY — no separate ``pairs`` endpoint-index array.
A prior signature passed ``compute(frames, pairs)`` and raised ``TypeError``
against molrs; these tests pin the no-``pairs`` contract.
"""

from __future__ import annotations

import molpy as mp
from molpy.compute import LegendreReorientation
from molpy.compute.base import Compute


def _chain_frame(n: int = 6, t: int = 0):
    """A small carbon chain (``n`` atoms, ``n-1`` bonds) with bond topology in-frame.

    ``t`` scales the zig-zag amplitude so the bond vectors reorient between
    successive frames (a nontrivial reorientational correlation).
    """
    mol = mp.Atomistic()
    amp = 0.3 + 0.1 * t
    atoms = [
        mol.def_atom(element="C", xyz=[float(i), amp * (i % 2), 0.0]) for i in range(n)
    ]
    for i in range(n - 1):
        mol.def_bond(atoms[i], atoms[i + 1])
    return mol.get_topo().to_frame()


def test_reorientation_is_compute_subclass():
    assert issubclass(LegendreReorientation, Compute)


def test_frame_carries_bonds_block():
    frame = _chain_frame()
    assert "bonds" in frame.keys()


def test_reorientation_reads_bonds_from_frame():
    # Several time-ordered frames; the (tail, head) pairs come from the first
    # frame's `bonds` block — no `pairs` argument is passed.
    frames = [_chain_frame(t=t) for t in range(4)]
    result = LegendreReorientation(max_lag=2)(frames)
    assert result.c1.shape[0] >= 1
    assert result.c2.shape[0] >= 1
    assert len(result.lags) == result.c1.shape[0]
    assert result.c1.shape == result.c2.shape
