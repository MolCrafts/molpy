"""Transport benchmarks against molrs Computes (array API).

Recipe Trajectory wrappers are gone — bench the same surface as Rust/Python molrs.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import (
    EinsteinConductivity,
    GreenKuboConductivity,
    MSD,
    Onsager,
    Persist,
)

pytestmark = pytest.mark.benchmark


def test_msd(benchmark, pos_traj) -> None:
    series = benchmark(MSD(), pos_traj)
    assert series.mean.shape == (len(pos_traj),)


def test_msd_window(benchmark, pos_traj) -> None:
    series = benchmark(MSD(method="window"), pos_traj)
    assert series.mean.shape == (len(pos_traj),)


def test_einstein_conductivity(benchmark) -> None:
    m = np.zeros((32, 3))
    m[:, 0] = np.linspace(0.0, 1.0, 32)

    def run():
        return EinsteinConductivity().compute(m, 1.0, 10)

    raw = benchmark(run)
    assert raw["msd"].shape[0] == 11


def test_green_kubo_conductivity(benchmark) -> None:
    j = np.ones((32, 3))
    j[:, 1:] = 0.0

    def run():
        return GreenKuboConductivity().compute(j, 1.0, 10)

    raw = benchmark(run)
    assert raw["jacf"].shape[0] == 11


def test_onsager_correlation(benchmark) -> None:
    p = np.zeros((32, 3))
    p[:, 0] = np.arange(32, dtype=np.float64)

    def run():
        return Onsager.correlation(p, p, 1.0, 8)

    out = benchmark(run)
    assert out["correlation"].shape[0] == 9


def test_persist_pair_survival(benchmark) -> None:
    coords_i = np.zeros((16, 2, 3))
    coords_j = np.zeros((16, 2, 3))
    coords_j[:, :, 0] = 1.0
    box = np.tile([[10.0, 10.0, 10.0]], (16, 1))

    def run():
        return Persist.pair_survival_tcf(
            coords_i, coords_j, box, 0.1, 3.5, "intermittent", 1.0, 5, False
        )

    out = benchmark(run)
    assert out["correlation"].shape[0] >= 1
