"""molpy.compute density-field benchmarks: LocalDensity and GaussianDensity."""

from __future__ import annotations

import pytest

from molpy.compute import GaussianDensity, LocalDensity

pytestmark = pytest.mark.benchmark


def test_local_density(benchmark, cmp_frame, cmp_nlist) -> None:
    op = LocalDensity(r_max=3.0, diameter=0.0)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_gaussian_density(benchmark, cmp_frame) -> None:
    op = GaussianDensity(nx=16, ny=16, nz=16, sigma=1.0)
    out = benchmark(op, cmp_frame)
    assert isinstance(out, list) and len(out) >= 1
