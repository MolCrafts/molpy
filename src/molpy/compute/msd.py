"""Mean Squared Displacement — molrs-backed.

Returns ``molrs.compute.msd.MSDTimeSeries`` directly. Frame[0] is the
reference; ``series.mean[i]`` is ⟨|r(i) - r(0)|²⟩ averaged over particles.
"""

from __future__ import annotations

from molrs.compute.msd import MSD as _MolrsMSD
from molrs.compute.msd import MSDTimeSeries as _MolrsMSDTimeSeries

from .base import Compute
from .frame_view import to_molrs_frames


class MSD(Compute):
    """Mean squared displacement against frame[0].

    Examples
    --------
    >>> series = MSD()(trajectory_frames)
    >>> series.mean.shape    # (n_frames,)
    """

    def __init__(self) -> None:
        super().__init__()
        self._inner = _MolrsMSD()

    def _compute(self, frames) -> _MolrsMSDTimeSeries:
        return self._inner.compute(to_molrs_frames(frames))
