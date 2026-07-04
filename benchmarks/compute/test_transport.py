"""molpy.compute transport benchmarks: MSD, MCD, PMSD, Onsager, JACF, Persist.

These consume a small trajectory (a few frames, one or two species). Each
asserts the result carries the expected lag axis so a windowing regression
fails the bench.
"""

from __future__ import annotations

import pytest

from molpy.compute import JACF, MSD, MCDCompute, Onsager, Persist, PMSDCompute

pytestmark = pytest.mark.benchmark


def test_msd(benchmark, pos_traj) -> None:
    series = benchmark(MSD(), pos_traj)
    assert series.mean.shape == (len(pos_traj),)


def test_mcd(benchmark, drift_traj) -> None:
    op = MCDCompute(tags=["1"], max_dt=5.0, dt=1.0)
    result = benchmark(op, drift_traj)
    assert result.correlations["1"].shape == (5,)


def test_pmsd(benchmark, drift_traj) -> None:
    op = PMSDCompute(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0)
    result = benchmark(op, drift_traj)
    assert result.pmsd.shape == (5,)


def test_onsager(benchmark, drift_traj) -> None:
    op = Onsager(tags=["1,1", "1,2"], max_dt=5.0, dt=1.0)
    result = benchmark(op, drift_traj)
    assert result.correlations["1,1"].shape == (5,)


def test_jacf(benchmark, current_traj) -> None:
    op = JACF(cation_type=1, anion_type=2, max_dt=5.0, dt=1.0, temperature=300.0)
    result = benchmark(op, current_traj)
    assert result.jacf.shape == (5,)


def test_persist(benchmark, pair_traj) -> None:
    op = Persist(tags=["1,2:continuous:1.0"], max_dt=4.0, dt=1.0)
    result = benchmark(op, pair_traj)
    assert result.correlations["1,2:continuous:1.0"].shape == (4,)
