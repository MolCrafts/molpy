"""molpy.compute spatial-distribution (SDF) benchmark.

The 3-D orientation-resolved density on a molecule body-fixed grid. A reference
triplet defines the body frame (Kabsch-aligned to a template); target-atom
density accumulates on the grid.

The 1-D geometric distributions (DistanceDistribution / AngleDistribution /
DihedralDistribution / CombinedDistribution) are benched in
``test_distribution.py`` — they read their atom tuples from the frame's
``bonds`` / ``angles`` / ``dihedrals`` topology blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import SpatialDistribution

pytestmark = pytest.mark.benchmark


def test_spatial_distribution(benchmark, cmp_frame) -> None:
    template = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    op = SpatialDistribution(
        reference=[0, 1, 2],
        template=template,
        target=list(range(600)),
        n=(16, 16, 16),
        extent=(4.0, 4.0, 4.0),
        bulk_density=0.03,
    )
    out = benchmark(op, [cmp_frame])
    assert np.asarray(out.density).size > 0
