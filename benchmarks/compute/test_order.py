"""molpy.compute bond-orientational order benchmarks.

Steinhardt / Hexatic / Nematic / SolidLiquid — thin shells over
``molrs.compute.order``. Each returns a per-frame list; the bench asserts a
non-empty result so a dispatch regression fails too.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import Hexatic, Nematic, SolidLiquid, Steinhardt

pytestmark = pytest.mark.benchmark


def test_steinhardt(benchmark, cmp_frame, cmp_nlist) -> None:
    op = Steinhardt([4, 6], average=True)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_hexatic(benchmark, cmp_frame, cmp_nlist) -> None:
    out = benchmark(Hexatic(6), cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_solid_liquid(benchmark, cmp_frame, cmp_nlist) -> None:
    op = SolidLiquid(6, q_threshold=0.7, n_threshold=6)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_nematic(benchmark, cmp_frame) -> None:
    # Per-particle directors come from the frame's `orientations` block (one
    # `(head, tail)` atom pair per particle) — no external director array.
    n = len(np.asarray(cmp_frame["atoms"]["x"]))
    idx = np.arange(n, dtype=np.uint32)
    cmp_frame["orientations"] = {"atomi": idx, "atomj": (idx + 1) % n}
    # Nematic returns (order, eigenvalues, director, q_tensor).
    out = benchmark(Nematic(), cmp_frame)
    assert isinstance(out, tuple) and len(out) == 4
