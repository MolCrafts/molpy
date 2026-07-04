"""molpy.compute clustering benchmarks: Cluster, ClusterCenters, ClusterProperties."""

from __future__ import annotations

import pytest

from molpy.compute import Cluster, ClusterCenters, ClusterProperties

pytestmark = pytest.mark.benchmark


def test_cluster(benchmark, cmp_frame, cmp_nlist) -> None:
    op = Cluster(min_cluster_size=1)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert out is not None


def test_cluster_centers(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = Cluster(min_cluster_size=1)(cmp_frame, cmp_nlist)
    out = benchmark(ClusterCenters(), cmp_frame, [clusters])
    assert out is not None


def test_cluster_properties(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = Cluster(min_cluster_size=1)(cmp_frame, cmp_nlist)
    out = benchmark(ClusterProperties(), cmp_frame, [clusters])
    assert isinstance(out, list) and "sizes" in out[0]
