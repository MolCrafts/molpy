"""FFI seam of the group, feature and Voronoi computes.

Each class is built from its required arguments, called once on tiny
synthetic data, and its result is passed into the next compute where the core
composes them. The science is tested in Rust (``molrs::compute``).
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.compute import (
    CenterOfMass,
    CenterOfMassResult,
    Cluster,
    ClusterResult,
    FiniteFieldPolarizability,
    GyrationTensor,
    GyrationTensorResult,
    InertiaTensor,
    InertiaTensorResult,
    Kmeans,
    KmeansResult,
    MolecularMoments,
    Pca,
    PcaResult,
    RadicalVoronoi,
    RegularGrid,
    VoronoiCells,
    VoronoiDomainAnalysis,
    VoronoiDomainResult,
    VoronoiFace,
    VoronoiIntegration,
    VoronoiVoidAnalysis,
    VoronoiVoidResult,
)

BOX_LEN = 10.0


def _two_dimers() -> tuple[molrs.core.Box, np.ndarray, molrs.core.Neighbors]:
    """Two bonded pairs far apart: (0, 1) and (2, 3)."""
    box = molrs.core.Box.cube(BOX_LEN)
    pts = np.array(
        [[1.0, 1.0, 1.0], [2.0, 1.0, 1.0], [6.0, 6.0, 6.0], [7.0, 6.0, 6.0]],
        dtype=np.float64,
    )
    pairs = molrs.core.NeighborQuery(box, pts, 1.5).query_self()
    return box, pts, pairs


def _lattice() -> tuple[molrs.core.Box, np.ndarray]:
    """A 2 x 2 x 2 simple-cubic lattice filling a periodic box of side 4."""
    box = molrs.core.Box.cube(4.0)
    grid = np.arange(2) * 2.0 + 1.0
    pts = np.array(
        [[x, y, z] for x in grid for y in grid for z in grid], dtype=np.float64
    )
    return box, pts


class TestCluster:
    def test_pairs_cluster_into_components(self) -> None:
        _, _, pairs = _two_dimers()
        result = Cluster().compute(pairs=pairs)
        assert isinstance(result, ClusterResult)
        assert result.n_clusters == 2
        assert result.cluster_idx.shape == (4,)
        assert sorted(size for _, size in result.sizes()) == [2, 2]

    def test_merge_takes_a_partition(self) -> None:
        _, _, pairs = _two_dimers()
        merged = Cluster().compute(pairs=pairs, merge=np.zeros(4, dtype=np.uint32))
        assert merged.n_clusters == 1


class TestShape:
    def test_cluster_result_is_the_partition_of_every_group_compute(self) -> None:
        box, pts, pairs = _two_dimers()
        clusters = Cluster().compute(pairs=pairs)
        masses = np.ones(4)

        com = CenterOfMass().compute(
            positions=pts, simbox=box, partition=clusters, masses=masses
        )
        assert isinstance(com, CenterOfMassResult)
        assert com.centers.shape == (2, 3)
        assert np.allclose(com.total_masses, [2.0, 2.0])

        gyration = GyrationTensor().compute(
            positions=pts, partition=clusters, weights=masses
        )
        assert isinstance(gyration, GyrationTensorResult)
        assert gyration.tensors.shape == (2, 3, 3)
        assert np.allclose(gyration.radius_of_gyration(), [0.5, 0.5])

        inertia = InertiaTensor().compute(
            positions=pts, partition=clusters, masses=masses
        )
        assert isinstance(inertia, InertiaTensorResult)
        assert inertia.tensors.shape == (2, 3, 3)

    def test_unsigned_labels_are_a_partition(self) -> None:
        _, pts, _ = _two_dimers()
        labels = np.array([0, 0, 1, 1], dtype=np.uint32)
        com = CenterOfMass().compute(positions=pts, partition=labels, masses=np.ones(4))
        assert np.allclose(com.centers[0], [1.5, 1.0, 1.0])

    def test_signed_labels_are_refused(self) -> None:
        _, pts, _ = _two_dimers()
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        with pytest.raises(TypeError, match="partition"):
            CenterOfMass().compute(positions=pts, partition=labels, masses=np.ones(4))


class TestFeatures:
    def test_array_to_pca_to_kmeans_to_partition(self) -> None:
        rng = np.random.default_rng(0)
        features = np.vstack(
            [rng.normal(0.0, 0.1, (5, 3)), rng.normal(5.0, 0.1, (5, 3))]
        )

        pca = Pca(2).compute(features=features)
        assert isinstance(pca, PcaResult)
        assert pca.coords.shape == (10, 2)
        assert pca.variance.shape == (2,)

        kmeans = Kmeans(2).compute(features=pca)
        assert isinstance(kmeans, KmeansResult)
        assert kmeans.labels.shape == (10,)
        assert len(set(kmeans.labels[:5])) == 1
        assert kmeans.labels[0] != kmeans.labels[9]

        com = CenterOfMass().compute(
            positions=np.zeros((10, 3)), partition=kmeans, masses=np.ones(10)
        )
        assert com.centers.shape == (2, 3)

    def test_a_non_matrix_is_refused(self) -> None:
        with pytest.raises(TypeError, match="feature matrix"):
            Pca(1).compute(features="rows")

    def test_bad_configuration_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            Kmeans(0)


class TestVoronoi:
    def test_cells_feed_the_domain_and_void_analyses(self) -> None:
        box, pts = _lattice()
        cells = RadicalVoronoi().compute(positions=pts, simbox=box, radii=np.ones(8))
        assert isinstance(cells, VoronoiCells)
        assert len(cells) == 8
        assert cells.total_volume() == pytest.approx(64.0)
        assert all(isinstance(face, VoronoiFace) for face in cells.faces[0])
        assert set(cells.neighbors(0)) == {face.neighbor for face in cells.faces[0]}

        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        domains = VoronoiDomainAnalysis().compute(cells=cells, partition=labels)
        assert isinstance(domains, VoronoiDomainResult)
        assert sum(domains.sizes) == 8

        voids = VoronoiVoidAnalysis().compute(
            cells=cells, is_void=[True] + [False] * 7, box_volume=64.0
        )
        assert isinstance(voids, VoronoiVoidResult)
        assert voids.void_fraction == pytest.approx(1.0 / 8.0)

    def test_integration_feeds_the_finite_field_polarizability(self) -> None:
        box, pts = _lattice()
        grid = RegularGrid.over_box(box, [8, 8, 8])
        partition = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
        integrate = VoronoiIntegration("angstrom")

        def moments(shift: float) -> MolecularMoments:
            x = (np.arange(8) + 0.5) / 8.0
            density = 0.1 + shift * np.sin(2 * np.pi * x)[:, None, None] * np.ones(
                (8, 8, 8)
            )
            return integrate.compute(
                positions=pts,
                radii=np.ones(8),
                nuclear_charges=np.ones(8),
                partition=partition,
                simbox=box,
                grid=grid,
                density=density,
            )

        plus, minus = moments(0.01), moments(-0.01)
        assert isinstance(plus, MolecularMoments)
        assert plus.dipoles.shape == (2, 3)

        alpha = FiniteFieldPolarizability(0.01).compute(plus=plus, minus=minus)
        assert isinstance(alpha, np.ndarray)
        assert alpha.shape[0] == 2

    def test_unknown_unit_is_the_core_error(self) -> None:
        with pytest.raises(ValueError, match="LengthUnit"):
            VoronoiIntegration("parsec")
