"""molpy.compute ML-primitive benchmarks: PCA and k-means.

Two-component PCA over a list of descriptor rows, then k-means over the PCA
result (the ``PcaResult`` feeds ``KMeans`` directly).
"""

from __future__ import annotations

import pytest

from molpy.compute import KMeans, Pca

pytestmark = pytest.mark.benchmark


def test_pca(benchmark, descriptor_rows) -> None:
    out = benchmark(Pca(), descriptor_rows)
    assert out is not None


def test_kmeans(benchmark, descriptor_rows) -> None:
    pca_result = Pca()(descriptor_rows)
    out = benchmark(KMeans(k=3), pca_result)
    assert out is not None
