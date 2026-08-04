"""Onsager.correlation — raw cross-species displacement correlation."""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import Onsager


def test_onsager_correlation_self_msd_like():
    n = 20
    p = np.zeros((n, 3))
    p[:, 0] = np.arange(n, dtype=np.float64)
    out = Onsager.correlation(p, p, 1.0, 5)
    assert "correlation" in out and "lag_times" in out
    assert (
        out["correlation"][0] == pytest.approx(0.0, abs=1e-10)
        or out["correlation"][0] >= 0.0
    )
