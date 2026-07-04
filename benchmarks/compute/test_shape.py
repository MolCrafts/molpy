"""molpy.compute per-cluster shape benchmarks.

CenterOfMass / GyrationTensor / InertiaTensor / RadiusOfGyration reduce the
clusters produced by ``Cluster`` into shape descriptors. The clusters and the
centers each op consumes are built outside the timed region.
"""

from __future__ import annotations

import pytest

from molpy.compute import (
    CenterOfMass,
    Cluster,
    ClusterCenters,
    GyrationTensor,
    InertiaTensor,
    RadiusOfGyration,
)

pytestmark = pytest.mark.benchmark


def _clusters(frame, nlist):
    return Cluster(min_cluster_size=1)(frame, nlist)


def test_center_of_mass(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = _clusters(cmp_frame, cmp_nlist)
    out = benchmark(CenterOfMass(), cmp_frame, [clusters])
    assert out is not None


def test_gyration_tensor(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = _clusters(cmp_frame, cmp_nlist)
    centers = ClusterCenters()(cmp_frame, [clusters])
    out = benchmark(GyrationTensor(), cmp_frame, [clusters], centers)
    assert out is not None


def test_inertia_tensor(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = _clusters(cmp_frame, cmp_nlist)
    com = CenterOfMass()(cmp_frame, [clusters])
    out = benchmark(InertiaTensor(), cmp_frame, [clusters], com)
    assert out is not None


def test_radius_of_gyration(benchmark, cmp_frame, cmp_nlist) -> None:
    clusters = _clusters(cmp_frame, cmp_nlist)
    com = CenterOfMass()(cmp_frame, [clusters])
    out = benchmark(RadiusOfGyration(), cmp_frame, [clusters], com)
    assert out is not None
