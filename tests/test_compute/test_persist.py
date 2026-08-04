"""Persist.pair_survival_tcf — pair presence autocorrelation."""

from __future__ import annotations

import numpy as np

from molpy.compute import Persist


def test_pair_survival_tcf_runs():
    coords_i = np.zeros((4, 1, 3))
    coords_j = np.zeros((4, 1, 3))
    coords_j[:, 0, 0] = 1.0  # 1 Å away
    box = np.tile(np.array([[10.0, 10.0, 10.0]]), (4, 1))
    out = Persist.pair_survival_tcf(
        coords_i, coords_j, box, 0.5, 3.5, "intermittent", 1.0, 2, False
    )
    assert "correlation" in out
    assert out["correlation"].shape[0] >= 1
