"""molpy.core.Frame benchmarks: construction and block access."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp

pytestmark = pytest.mark.benchmark


def _atom_columns(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "id": np.arange(n, dtype=np.int64),
        "x": rng.random(n, dtype=np.float64),
        "y": rng.random(n, dtype=np.float64),
        "z": rng.random(n, dtype=np.float64),
    }


def test_frame_create(benchmark, n: int) -> None:
    cols = _atom_columns(n)
    frame = benchmark(lambda: mp.Frame(blocks={"atoms": cols}))
    assert frame["atoms"].nrows == n


def test_frame_block_access(benchmark, n: int) -> None:
    frame = mp.Frame(blocks={"atoms": _atom_columns(n)})
    out = benchmark(lambda: frame["atoms"]["x"])
    assert out.shape == (n,)
