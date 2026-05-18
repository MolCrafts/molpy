"""Distance-based clustering — molrs-backed.

``Cluster`` takes (frames, nlists) and returns one ``ClusterResult`` per
frame. ``ClusterCenters`` takes (frames, clusters) and returns the
geometric centers per cluster.

Both wrappers expose two inputs and therefore override ``__call__``
(not the single-input ``_compute`` hook) — mirroring the ``RDF`` pattern.
"""

from __future__ import annotations

from molrs.compute.cluster import Cluster as _MolrsCluster
from molrs.compute.cluster import ClusterCenters as _MolrsClusterCenters

from .base import Compute
from .frame_view import to_molrs_frames


class Cluster(Compute):
    """Group particles into clusters via a neighbor-list connectivity graph.

    Parameters
    ----------
    min_cluster_size : int
        Minimum size for a connected component to count as a cluster.
    """

    def __init__(self, min_cluster_size: int) -> None:
        super().__init__(min_cluster_size=min_cluster_size)
        self._inner = _MolrsCluster(min_cluster_size)

    def __call__(self, frames, neighbors):
        return self._inner.compute(to_molrs_frames(frames), neighbors)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError(
            "Cluster takes two inputs (frames, neighbors); call directly."
        )


class ClusterCenters(Compute):
    """Geometric centers per cluster (unweighted)."""

    def __init__(self) -> None:
        super().__init__()
        self._inner = _MolrsClusterCenters()

    def __call__(self, frames, clusters):
        return self._inner.compute(to_molrs_frames(frames), clusters)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError(
            "ClusterCenters takes two inputs (frames, clusters); call directly."
        )
