"""Pair-survival (persistence) time-correlation functions.

Call ``Persist.pair_survival_tcf(coords_i, coords_j, box_lengths, r0, r1,
method, dt, max_correlation_time, exclude_self=False)`` on per-species
coordinate arrays. ``C(0)`` is the mean coordination number.
"""

from __future__ import annotations

from molrs.compute.transport import Persist

__all__ = ["Persist"]
