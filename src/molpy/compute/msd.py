"""Mean Squared Displacement — molrs-backed.

Returns ``molrs.compute.msd.MSDTimeSeries`` directly. Frame[0] is the
reference; ``series.mean[i]`` is ⟨|r(i) - r(0)|²⟩ averaged over particles.
"""

from __future__ import annotations

from molrs.compute.msd import MSD as _MolrsMSD
from molrs.compute.msd import MSDTimeSeries as _MolrsMSDTimeSeries

from .base import Compute


class MSD(Compute):
    """Mean squared displacement against frame[0].

    Examples
    --------
    >>> series = MSD()(trajectory_frames)
    >>> series.mean.shape    # (n_frames,)
    """

    def __init__(self) -> None:
        super().__init__()
        self._impl = _MolrsMSD()

    def __call__(self, frames) -> _MolrsMSDTimeSeries:
        # Frames are canonical molrs.Frame objects; pass the list through.
        return self._impl.compute(frames)
