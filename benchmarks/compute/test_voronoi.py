"""molpy.compute radical-Voronoi benchmarks: tessellation, domains, voids.

RadicalVoronoi builds the power tessellation from positions + radii + box;
``voronoi_domains`` merges same-label cells and ``voronoi_voids`` aggregates
empty cells. (VoronoiIntegration needs a volumetric density grid and is out of
scope for a regression bench.)
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import RadicalVoronoi, voronoi_domains, voronoi_voids

pytestmark = pytest.mark.benchmark


def test_radical_voronoi(benchmark, voronoi_inputs) -> None:
    positions, radii, box = voronoi_inputs
    cells = benchmark(RadicalVoronoi(), positions, radii, box)
    assert cells.neighbors(0) is not None


def test_voronoi_domains(benchmark, voronoi_inputs) -> None:
    positions, radii, box = voronoi_inputs
    cells = RadicalVoronoi()(positions, radii, box)
    rng = np.random.default_rng(2)
    labels = (rng.random(len(positions)) > 0.5).astype(np.int64)
    out = benchmark(voronoi_domains, cells, labels)
    assert isinstance(out, dict)


def test_voronoi_voids(benchmark, voronoi_inputs) -> None:
    positions, radii, box = voronoi_inputs
    cells = RadicalVoronoi()(positions, radii, box)
    is_void = np.zeros(len(positions), dtype=bool)
    out = benchmark(voronoi_voids, cells, is_void, box.volume)
    assert isinstance(out, dict)
