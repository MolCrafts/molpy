"""Shared fixtures for the molpy core benchmark suite.

These benches measure molpy's public ``core/`` surface — the thin Python facade
over the molrs Rust kernels (``Box``, ``Atomistic``, ``Frame``). They live under
``benchmarks/`` (not ``tests/``) so the normal ``pytest tests/`` run does not
pick them up; ``pytest benchmarks/`` (and the bench.yml workflow) runs them.

Run::

    pip install -e ".[dev]"          # pulls in pytest-benchmark
    pytest benchmarks/ --benchmark-only
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp

SIZES: list[int] = [1_000, 10_000, 100_000]
SIZE_IDS: list[str] = ["small-1k", "medium-10k", "large-100k"]

BOX_LEN: float = 10.0


@pytest.fixture(params=SIZES, ids=SIZE_IDS)
def n(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def points(n: int) -> np.ndarray:
    """N points spanning ``[-L, 2L]`` per axis so wrap/fractional has work to do."""
    rng = np.random.default_rng(0)
    return (rng.random((n, 3), dtype=np.float64) * 3.0 - 1.0) * BOX_LEN


def make_chain(n: int) -> "mp.Atomistic":
    """Linear carbon chain of ``n`` atoms (``n-1`` bonds)."""
    mol = mp.Atomistic()
    atoms = [mol.def_atom(element="C") for _ in range(n)]
    for i in range(n - 1):
        mol.def_bond(atoms[i], atoms[i + 1])
    return mol
