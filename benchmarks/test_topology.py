"""molpy.core.Atomistic topology benchmarks.

Angle/dihedral perception (``get_topo``) and single-source BFS distances
(``get_topo_distances``) — the core graph operations molpy delegates to the
molrs kernel. Both assert against a linear-chain closed-form reference so a
structural regression fails the bench, not just a perf one.
"""

from __future__ import annotations

import pytest

from conftest import make_chain

pytestmark = pytest.mark.benchmark

# Perception copies the whole graph each call, so cap below the 1M box grid.
SIZES = [1_000, 10_000, 50_000]
SIZE_IDS = ["small-1k", "medium-10k", "large-50k"]


@pytest.fixture(params=SIZES, ids=SIZE_IDS)
def chain_n(request: pytest.FixtureRequest) -> int:
    return request.param


def test_get_topo(benchmark, chain_n: int) -> None:
    mol = make_chain(chain_n)
    topo = benchmark(mol.get_topo, gen_angle=True, gen_dihe=True)
    assert topo.n_relations("angles") == chain_n - 2
    assert topo.n_relations("dihedrals") == chain_n - 3


def test_get_topo_distances(benchmark, chain_n: int) -> None:
    mol = make_chain(chain_n)
    source = next(iter(mol.atoms))
    dists = benchmark(mol.get_topo_distances, source)
    assert sorted(dists.values()) == list(range(chain_n))
