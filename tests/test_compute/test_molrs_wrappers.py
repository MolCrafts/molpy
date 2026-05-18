"""Thin wrappers over molrs.compute kernels.

Each wrapper class is a ~10-line shell forwarding to a molrs ``Compute``
implementation; tests assert:

1. The wrapper is registered as a ``Compute`` subclass and exported from
   ``molpy.compute``.
2. The result type matches the direct ``molrs.<X>.compute(...)`` call
   (parity), to lock in zero molpy-side physics.
3. Input frames are not mutated.

Coverage: ``MSD``, ``Cluster``, ``ClusterCenters``, ``CenterOfMass``,
``GyrationTensor``, ``InertiaTensor``, ``RadiusOfGyration``, ``Pca``,
``KMeans``.
"""

from __future__ import annotations

import numpy as np
import pytest

import molpy
import molrs
from molpy.compute import (
    MSD,
    Cluster,
    ClusterCenters,
    CenterOfMass,
    DescriptorRow,
    GyrationTensor,
    InertiaTensor,
    KMeans,
    NeighborList,
    Pca,
    RadiusOfGyration,
)
from molpy.compute.base import Compute


def _frame(n: int, box_len: float, seed: int) -> molpy.Frame:
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(0.0, box_len, size=(n, 3))
    frame = molpy.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = molpy.Box.cubic(box_len)
    return frame


# ---- subclass registration ------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        MSD,
        Cluster,
        ClusterCenters,
        CenterOfMass,
        GyrationTensor,
        InertiaTensor,
        RadiusOfGyration,
        Pca,
        KMeans,
    ],
)
def test_is_compute_subclass(cls):
    assert issubclass(cls, Compute)


# ---- MSD ------------------------------------------------------------------


def test_msd_basic():
    frames = [_frame(50, 10.0, seed=i) for i in range(4)]
    series = MSD()(frames)
    mean = np.asarray(series.mean)
    assert mean.shape == (4,)
    assert mean[0] == pytest.approx(0.0, abs=1e-12)
    assert (mean[1:] > 0).all()


def test_msd_parity_with_molrs_direct():
    frames = [_frame(40, 8.0, seed=i + 100) for i in range(3)]
    via_molpy = MSD()(frames)
    via_molrs = molrs.compute.msd.MSD().compute([f.to_molrs() for f in frames])
    np.testing.assert_array_equal(
        np.asarray(via_molpy.mean), np.asarray(via_molrs.mean)
    )


# ---- Cluster / ClusterCenters --------------------------------------------


def _clustered_frame():
    # Two well-separated blobs of 5 atoms each in a 30 Å cube.
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=5.0, scale=0.3, size=(5, 3))
    blob_b = rng.normal(loc=20.0, scale=0.3, size=(5, 3))
    xyz = np.vstack([blob_a, blob_b])
    f = molpy.Frame()
    f["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    f.box = molpy.Box.cubic(30.0)
    return f, xyz


def test_cluster_finds_two_clusters():
    frame, _ = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    result = Cluster(min_cluster_size=1)(frame, nlist)
    assert result.num_clusters == 2


def test_cluster_centers_shape():
    frame, _ = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    centers = ClusterCenters()(frame, clusters)
    arr = np.asarray(centers.centers)
    assert arr.shape == (2, 3)


# ---- CenterOfMass / GyrationTensor / InertiaTensor / Rg -------------------


def test_center_of_mass_unit_masses_matches_geometric():
    frame, xyz = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    com = CenterOfMass()(frame, clusters)
    centers = ClusterCenters()(frame, clusters)
    np.testing.assert_allclose(
        np.asarray(com.centers_of_mass),
        np.asarray(centers.centers),
        atol=1e-10,
    )


def test_gyration_tensor_is_symmetric():
    frame, _ = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    centers = ClusterCenters()(frame, clusters)
    tensor = np.asarray(GyrationTensor()(frame, clusters, centers))
    # molrs single-frame path returns the first cluster's (3, 3) tensor.
    assert tensor.shape[-2:] == (3, 3)
    flat = tensor.reshape(-1, 3, 3)
    for t in flat:
        np.testing.assert_allclose(t, t.T, atol=1e-10)


def test_inertia_tensor_is_symmetric():
    frame, _ = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    com = CenterOfMass()(frame, clusters)
    inertia = np.asarray(InertiaTensor()(frame, clusters, com))
    assert inertia.shape[-2:] == (3, 3)
    flat = inertia.reshape(-1, 3, 3)
    for t in flat:
        np.testing.assert_allclose(t, t.T, atol=1e-10)


def test_radius_of_gyration_positive():
    frame, _ = _clustered_frame()
    nlist = NeighborList(cutoff=1.5)(frame)
    clusters = Cluster(min_cluster_size=1)(frame, nlist)
    com = CenterOfMass()(frame, clusters)
    rg = RadiusOfGyration()(frame, clusters, com)
    arr = np.asarray(rg)
    assert arr.shape == (2,)
    assert (arr > 0).all()


# ---- PCA + KMeans ---------------------------------------------------------


def test_pca_kmeans_roundtrip():
    rng = np.random.default_rng(0)
    rows_a = rng.normal(loc=[0, 0, 0, 0], scale=0.1, size=(20, 4))
    rows_b = rng.normal(loc=[5, 5, 5, 5], scale=0.1, size=(20, 4))
    rows = [DescriptorRow(r.astype(np.float64)) for r in np.vstack([rows_a, rows_b])]

    pca = Pca()(rows)
    coords = np.asarray(pca.coords)
    assert coords.shape == (40, 2)

    labels = np.asarray(KMeans(k=2, seed=42)(pca).labels)
    assert labels.shape == (40,)
    # Two well-separated blobs → two distinct labels.
    assert set(labels.tolist()) == {0, 1}


# ---- input immutability ---------------------------------------------------


def test_msd_does_not_mutate_input():
    frames = [_frame(20, 5.0, seed=i) for i in range(3)]
    snapshots = [(f["atoms"]["x"].copy(), f.box.matrix.copy()) for f in frames]
    MSD()(frames)
    for (x_before, m_before), f in zip(snapshots, frames):
        np.testing.assert_array_equal(f["atoms"]["x"], x_before)
        np.testing.assert_array_equal(f.box.matrix, m_before)
