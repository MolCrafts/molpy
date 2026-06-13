"""Per-cluster shape descriptors — molrs-backed.

Thin wrappers around ``molrs.compute.cluster``:

- ``CenterOfMass`` (frames, clusters) → mass-weighted centers
- ``GyrationTensor`` (frames, clusters, centers) → 3×3 tensors per cluster
- ``InertiaTensor`` (frames, clusters, com) → 3×3 tensors per cluster
- ``RadiusOfGyration`` (frames, clusters, com) → Rg per cluster
"""

from __future__ import annotations

import numpy as np

from molrs.compute.cluster import (
    CenterOfMass as _MolrsCenterOfMass,
    GyrationTensor as _MolrsGyrationTensor,
    InertiaTensor as _MolrsInertiaTensor,
    RadiusOfGyration as _MolrsRadiusOfGyration,
)

from .base import Compute


def _as_masses(masses):
    if masses is None:
        return None
    return np.ascontiguousarray(masses, dtype=np.float64)


class CenterOfMass(Compute):
    """Mass-weighted center per cluster.

    Parameters
    ----------
    masses : ndarray | None
        Per-particle masses (length N). ``None`` falls back to unit mass.
    """

    def __init__(self, masses=None) -> None:
        super().__init__(masses=masses)
        self._impl = _MolrsCenterOfMass(_as_masses(masses))

    def __call__(self, frames, clusters):
        return self._impl.compute(frames, clusters)


class GyrationTensor(Compute):
    """Gyration tensor per cluster (unweighted)."""

    def __init__(self) -> None:
        super().__init__()
        self._impl = _MolrsGyrationTensor()

    def __call__(self, frames, clusters, centers):
        return self._impl.compute(frames, clusters, centers)


class InertiaTensor(Compute):
    """Inertia tensor per cluster.

    Parameters
    ----------
    masses : ndarray | None
        Per-particle masses (length N). ``None`` falls back to unit mass.
    """

    def __init__(self, masses=None) -> None:
        super().__init__(masses=masses)
        self._impl = _MolrsInertiaTensor(_as_masses(masses))

    def __call__(self, frames, clusters, com):
        return self._impl.compute(frames, clusters, com)


class RadiusOfGyration(Compute):
    """Radius of gyration per cluster.

    Parameters
    ----------
    masses : ndarray | None
        Per-particle masses (length N). ``None`` falls back to unit mass.
    """

    def __init__(self, masses=None) -> None:
        super().__init__(masses=masses)
        self._impl = _MolrsRadiusOfGyration(_as_masses(masses))

    def __call__(self, frames, clusters, com):
        return self._impl.compute(frames, clusters, com)
