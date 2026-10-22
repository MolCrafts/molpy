"""FFI seam of the order, environment and PMFT computes.

Each class is built from its required arguments only, called once on tiny
synthetic data, and its result read back; results that feed another compute
are passed into it. The numbers themselves are tested in Rust.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest
from molpy import compute as mc

BOX = 3.0


def fcc_points() -> np.ndarray:
    """A 2 x 2 x 2 FCC supercell (32 sites) filling a cube of side 3, slightly
    jittered so per-particle features vary."""
    a = BOX / 2
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(2) for j in range(2) for k in range(2)])
    lattice = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
    jitter = np.random.default_rng(7).normal(scale=0.02, size=lattice.shape)
    return (lattice + jitter).astype(np.float64)


def square_points() -> np.ndarray:
    """A 3 x 3 square lattice in the z = 0 plane, spacing 1."""
    return np.array([[i, j, 0.0] for i in range(3) for j in range(3)], dtype=np.float64)


def neighbors(points: np.ndarray, cutoff: float) -> mp.core.Neighbors:
    box = mp.core.Box(mp.core.Cell.lengths([BOX] * 3))
    return mp.core.NeighborQuery(box, points, cutoff).query_self()


@pytest.fixture
def fcc() -> tuple[np.ndarray, mp.core.Neighbors]:
    points = fcc_points()
    return points, neighbors(points, 1.2)


def identity_quaternions(n: int) -> np.ndarray:
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    return q


class TestOrder:
    def test_steinhardt_result_and_qlm_round_trip(self, fcc):
        points, nb = fcc
        result = mc.Steinhardt([4, 6]).compute(neighbors=nb)
        assert isinstance(result, mc.SteinhardtResult)
        assert list(result.l) == [4, 6]
        assert result.ql.shape == (len(points), 2)
        assert result.wl is None
        assert all(isinstance(t, mc.QlmTable) for t in result.tables)
        assert result.qlm(5) is None

        table = result.qlm(6)
        assert table.order == 6
        assert table.n_particles == len(points)
        assert table.values.dtype == np.complex128
        assert np.allclose(table.ql(), result.ql[:, 1])
        assert table.norms().shape == (len(points),)
        assert table.bond_coherence(nb).ndim == 1

        solid = mc.SolidLiquid().compute(qlm=table, neighbors=nb)
        assert isinstance(solid, mc.SolidLiquidResult)
        assert solid.is_solid.shape == (len(points),)

        coord = mc.ContinuousCoordination(2.0).compute(qlm=table, neighbors=nb)
        assert coord.coord.shape == (len(points),)

    def test_steinhardt_wl_setting(self, fcc):
        _, nb = fcc
        result = mc.Steinhardt([6], wl=True).compute(neighbors=nb)
        assert result.wl.shape == result.ql.shape

    def test_hexatic(self):
        points = square_points()
        result = mc.Hexatic(4).compute(neighbors=neighbors(points, 1.1))
        assert result.k == 4
        assert result.psi.dtype == np.complex128
        assert result.psi.shape == (len(points),)

    def test_nematic_aligned(self):
        directors = np.tile([0.0, 0.0, 1.0], (5, 1))
        result = mc.Nematic().compute(directors=directors)
        assert result.order == pytest.approx(1.0)
        assert result.q_tensor.shape == (3, 3)
        assert result.director.shape == (3,)

    def test_cubatic(self):
        directors = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
        result = mc.Cubatic(n_steps=50, n_chains=1).compute(directors=directors)
        assert result.director_basis.shape == (3, 3)
        assert np.isfinite(result.order)

    def test_core_errors_are_value_errors(self):
        with pytest.raises(ValueError):
            mc.Steinhardt([])
        with pytest.raises(ValueError):
            mc.Nematic().compute(directors=np.zeros((0, 3)))


class TestEnvironment:
    def test_bond_orientational_order(self, fcc):
        _, nb = fcc
        result = mc.BondOrientationalOrder(4, 8).compute(neighbors=nb)
        assert result.bond_order.shape == (4, 8)
        assert result.raw_counts.shape == (4, 8)
        assert len(result.theta_edges) == 5

    def test_local_descriptors_feature_matrix(self, fcc):
        _, nb = fcc
        result = mc.LocalDescriptors(2).compute(neighbors=nb)
        assert result.l_max == 2
        assert result.features.shape[1] == 2 * result.n_sphs
        z = result.descriptor(0, 0, 0)
        assert isinstance(z, complex)
        assert result.descriptor(0, 3, 0) is None

    def test_local_bond_projection(self, fcc):
        points, nb = fcc
        out = mc.LocalBondProjection().compute(
            neighbors=nb,
            proj_vectors=np.array([[0.0, 0.0, 1.0]]),
            orientations=identity_quaternions(len(points)),
        )
        assert out.projections.shape[1] == 1

    def test_environment_match(self, fcc):
        points, nb = fcc
        result = mc.EnvironmentMatch(0.1).compute(neighbors=nb)
        assert isinstance(result, mc.EnvironmentMatchResult)
        assert len(result.cluster_idx) == len(points)
        assert result.n_clusters >= 1

    def test_angular_separation_global(self):
        out = mc.AngularSeparationGlobal().compute(
            query=identity_quaternions(3), reference=identity_quaternions(2)
        )
        assert out.angles.shape == (3, 2)
        assert np.allclose(out.angles, 0.0)

    def test_angular_separation_neighbor(self, fcc):
        points, nb = fcc
        q = identity_quaternions(len(points))
        out = mc.AngularSeparationNeighbor().compute(
            neighbors=nb, query_orientations=q, point_orientations=q
        )
        assert np.allclose(out.angles, 0.0)


class TestPmft:
    @pytest.mark.parametrize(
        ("cls", "args"),
        [
            (mc.PmftR12, (1.2, 4, 3, 3)),
            (mc.PmftXy, (1.2, 1.2, 4, 4)),
            (mc.PmftXyt, (1.2, 1.2, 4, 4, 3)),
        ],
    )
    def test_planar(self, cls, args):
        points = square_points()
        nb = neighbors(points, 1.1)
        result = cls(*args).compute(pairs=nb, orientations=np.zeros(len(points)))
        assert isinstance(result, mc.PmftResult)
        assert result.density.shape == result.pmf.shape == result.raw_counts.shape
        assert len(result.edges) == result.density.ndim
        assert result.raw_counts.sum() > 0

    def test_xyz(self, fcc):
        points, nb = fcc
        result = mc.PmftXyz(1.2, 1.2, 1.2, 4, 4, 4).compute(
            pairs=nb, orientations=identity_quaternions(len(points))
        )
        assert result.density.shape == (4, 4, 4)


class TestFeatureRoundTrip:
    def test_steinhardt_result_feeds_pca(self, fcc):
        pca = getattr(mc, "Pca", None)
        if pca is None:
            pytest.skip("Pca is not bound in this build")
        points, nb = fcc
        steinhardt = mc.Steinhardt([4, 6]).compute(neighbors=nb)
        assert pca(1).compute(features=steinhardt).coords.shape == (len(points), 1)

    def test_local_descriptors_result_feeds_kmeans(self, fcc):
        # Pca would refuse it: the l = 0 columns are constant.
        kmeans = getattr(mc, "Kmeans", None)
        if kmeans is None:
            pytest.skip("Kmeans is not bound in this build")
        _, nb = fcc
        descriptors = mc.LocalDescriptors(2).compute(neighbors=nb)
        labels = kmeans(2).compute(features=descriptors).labels
        assert len(labels) == descriptors.features.shape[0]
