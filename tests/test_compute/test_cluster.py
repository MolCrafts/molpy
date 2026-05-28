"""ClusterProperties — molrs-backed thin shell.

Builds clusters with the (already-exposed) ``Cluster`` operator, then feeds the
result to ``ClusterProperties``; parity vs the direct molrs call.
"""

from __future__ import annotations

import numpy as np

import molrs

from molpy.compute import Cluster, ClusterProperties, NeighborList
from molpy.compute.base import Compute

from .parity_helpers import (
    assert_nested_equal,
    frame_coords_snapshot,
    random_periodic_frame,
)


def _frame_nlist_clusters():
    frame = random_periodic_frame()
    nlist = NeighborList(cutoff=3.0)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    return frame, clusters


def test_cluster_properties_is_compute_subclass():
    assert issubclass(ClusterProperties, Compute)


def test_cluster_properties_smoke():
    frame, clusters = _frame_nlist_clusters()
    out = ClusterProperties()(frame, [clusters])
    assert isinstance(out, list) and isinstance(out[0], dict)
    assert "sizes" in out[0] and "radii_of_gyration" in out[0]


def test_cluster_properties_parity_with_molrs_direct():
    frame, clusters = _frame_nlist_clusters()
    mine = ClusterProperties()(frame, [clusters])
    direct = molrs.compute.cluster.ClusterProperties().compute([frame], [clusters])
    assert_nested_equal(mine, direct)


def test_cluster_properties_input_frame_immutable():
    frame, clusters = _frame_nlist_clusters()
    before = frame_coords_snapshot(frame)
    ClusterProperties()(frame, [clusters])
    np.testing.assert_array_equal(before, frame_coords_snapshot(frame))
