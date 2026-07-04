"""molpy.compute hydrogen-bond detection benchmark.

Per-frame geometric H-bond search from explicit ``(D, H)`` donor pairs and
acceptor indices under the Luzar-Chandler criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import HBondCriterion, HBonds

pytestmark = pytest.mark.benchmark


def test_hbonds(benchmark, cmp_frame) -> None:
    donors = np.array([[i, i + 1] for i in range(0, 200, 2)], dtype=np.int64)
    acceptors = np.arange(300, 400, dtype=np.int64)
    op = HBonds(donors, acceptors, HBondCriterion(dist_cutoff=3.5, angle_cutoff=150.0))
    out = benchmark(op, [cmp_frame])
    assert np.asarray(out.counts).shape == (1,)
