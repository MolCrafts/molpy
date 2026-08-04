"""Onsager collective displacement cross-correlation.

Call ``Onsager.correlation(p_i, p_j, dt, max_correlation_time)`` on
pre-assembled collective coordinates ``P_α(t) = Σ_{i∈α} r_i`` (unwrapped,
shape ``(n_frames, 3)``). Returns the raw ``L_ij(τ)`` curve; take the
long-time slope yourself for ``Ω_ij``.
"""

from __future__ import annotations

from molrs.compute.transport import Onsager

__all__ = ["Onsager"]
