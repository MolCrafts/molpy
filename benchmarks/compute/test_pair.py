"""molpy.compute pair / neighbor benchmarks: NeighborList and RDF.

The neighbor list is the shared input to most structural ops; g(r) is the
canonical multi-frame accumulator. Each asserts a cheap structural invariant so
a shape or normalization regression fails the bench, not just a perf one.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import RDF, NeighborList

pytestmark = pytest.mark.benchmark


def test_neighborlist(benchmark, cmp_frame) -> None:
    nlist = benchmark(NeighborList(cutoff=3.0), cmp_frame)
    assert nlist.n_pairs > 0
    assert (np.asarray(nlist.distances) <= 3.0 + 1e-9).all()


def test_rdf(benchmark, cmp_frames_nlists) -> None:
    frames, nlists = cmp_frames_nlists
    rdf = RDF(n_bins=50, r_max=3.0)
    result = benchmark(rdf, frames, nlists)
    g = np.asarray(result.rdf)
    assert g.shape == (50,)
    assert np.isfinite(g).all() and (g >= 0.0).all()
