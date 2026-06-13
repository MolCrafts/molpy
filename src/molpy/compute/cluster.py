"""Distance-based clustering — molrs-backed.

``Cluster`` takes (frames, nlists) and returns one ``ClusterResult`` per
frame. ``ClusterCenters`` takes (frames, clusters) and returns the
geometric centers per cluster.

Both wrappers take two data inputs — mirroring the ``RDF`` pattern.
"""

from __future__ import annotations

from molrs.compute.cluster import Cluster as _MolrsCluster
from molrs.compute.cluster import ClusterCenters as _MolrsClusterCenters
from molrs.compute.cluster import ClusterProperties as _MolrsClusterProperties

from .base import Compute


class Cluster(Compute):
    """Group particles into clusters via a neighbor-list connectivity graph.

    Parameters
    ----------
    min_cluster_size : int
        Minimum size for a connected component to count as a cluster.
    """

    def __init__(self, min_cluster_size: int) -> None:
        super().__init__(min_cluster_size=min_cluster_size)
        self._impl = _MolrsCluster(min_cluster_size)

    def __call__(self, frames, neighbors):
        return self._impl.compute(frames, neighbors)


class ClusterCenters(Compute):
    """Geometric centers per cluster (unweighted)."""

    def __init__(self) -> None:
        super().__init__()
        self._impl = _MolrsClusterCenters()

    def __call__(self, frames, clusters):
        return self._impl.compute(frames, clusters)


class ClusterProperties(Compute):
    """Per-cluster properties (sizes, centers, masses, gyration tensors, Rg).

    Takes ``(frames, clusters)`` where ``clusters`` is the sequence of
    ``ClusterResult`` returned by :class:`Cluster`. Returns one dict per frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self._impl = _MolrsClusterProperties()

    def __call__(self, frames, clusters):
        return self._impl.compute(frames, clusters)
