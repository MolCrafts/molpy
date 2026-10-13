"""Smoke tests for the freud-ported analyzers exposed via PyO3.

Each test exercises the Python wrapper end-to-end: construct, build a
NeighborList where required, call compute, and sanity-check the shape
and a known value. Numerical correctness is covered by the Rust unit
tests in `molrs::compute` — these are wiring-level checks.
"""

import molrs
import numpy as np
import pytest
from conftest import make_frame, octahedron_frame
from molrs.compute import Cluster, ClusterProperties
from molrs.compute import GaussianDensity, LocalDensity
from molrs.compute import StaticStructureFactorDebye
from molrs.compute import BondOrientationalOrder
from molrs.compute import Hexatic, Nematic, SolidLiquid, Steinhardt
from molrs.compute import PmftXy


def _nlist(frame, pts, cutoff=1.2):
    nq = molrs.core.NeighborQuery(frame.box, pts, cutoff)
    return nq.query_self()


def test_neighbor_query_unbounded():
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    query = molrs.core.NeighborQuery.unbounded(points, 1.0)
    result = query.query(np.array([[0.0, 0.0, 0.0]]))

    assert np.array_equal(result.point_indices(), np.array([0, 1], dtype=np.uint32))
    assert np.allclose(np.sqrt(result.dist_sq()), [0.0, 0.5])


def test_neighbor_query_rejects_non_positive_cutoff():
    points = np.zeros((1, 3))
    box = molrs.core.Box.cube(1.0)
    with pytest.raises(ValueError, match="positive"):
        molrs.core.NeighborQuery(box, points, 0.0)
    with pytest.raises(ValueError, match="positive"):
        molrs.core.NeighborQuery.unbounded(points, 0.0)


class TestSteinhardt:
    def test_ql_finite_on_octahedron(self):
        frame, pts = octahedron_frame()
        nl = _nlist(frame, pts)
        s = Steinhardt(l=[6])
        out = s.compute(frame, nl)
        assert len(out) == 1
        ql = out[0]["ql"][0]
        # Centre particle has nonzero q_6; outer particles have 1 neighbor each.
        assert ql[0] > 0.0
        assert ql.shape == (7,)


class TestNematic:
    def test_aligned_gives_unity(self):
        # Directors are unit (head - tail) vectors of the "orientations" block
        # (atomi=head, atomj=tail). Five pairs all pointing +z -> order == 1.
        pts = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 1.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 1.0],
                [4.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        frame = make_frame(pts, box_len=20.0)
        ori = molrs.core.Block()
        ori.insert("atomi", np.array([0, 2, 4, 6, 8], dtype=np.uint32))
        ori.insert("atomj", np.array([1, 3, 5, 7, 9], dtype=np.uint32))
        frame["orientations"] = ori
        order, eigs, _director, q = Nematic().compute(frame)
        assert abs(order - 1.0) < 1e-10
        assert eigs.shape == (3,)
        assert q.shape == (3, 3)


class TestHexatic:
    def test_psi_shape(self):
        frame, pts = octahedron_frame()
        nl = _nlist(frame, pts)
        out = Hexatic(k=6).compute(frame, nl)
        assert len(out) == 1
        psi = out[0]
        assert psi.shape == (7, 2)  # real / imag pairs


class TestSolidLiquid:
    def test_returns_arrays(self):
        frame, pts = octahedron_frame()
        nl = _nlist(frame, pts)
        out = SolidLiquid(l=6, q_threshold=-2.0, n_threshold=1).compute(frame, nl)
        n_solid, is_solid = out[0]
        # All bonds count as "solid" with threshold = -2.
        assert n_solid[0] == 6
        assert len(is_solid) == 7


class TestLocalDensity:
    def test_pair_count(self):
        frame, pts = octahedron_frame()
        nl = _nlist(frame, pts, cutoff=2.0)
        num, density = LocalDensity(r_max=2.0).compute(frame, nl)[0]
        # Centre particle has 6 neighbors within r=2.
        assert abs(num[0] - 6.0) < 1e-9
        assert density.shape == (7,)


class TestGaussianDensity:
    def test_integral(self):
        frame, _ = octahedron_frame(box_len=10.0)
        grids = GaussianDensity(40, 40, 40, sigma=0.4).compute(frame)
        g = grids[0]
        assert g.shape == (40, 40, 40)
        voxel = (10.0 / 40) ** 3
        # Integral ≈ N (7 particles), within Gaussian-truncation error.
        assert abs(g.sum() * voxel - 7.0) < 0.5


class TestBondOrientationalOrder:
    def test_octahedron_counts(self):
        frame, pts = octahedron_frame()
        nl = _nlist(frame, pts)
        out = BondOrientationalOrder(8, 8).compute(frame, nl)
        counts, _bo, _t_edges, _p_edges = out[0]
        # 6 unique self-query bonds × 2 (symmetric counterparts) = 12.
        assert counts.sum() == 12


class TestStaticStructureFactorDebye:
    def test_zero_k_equals_n(self):
        frame, _ = octahedron_frame()
        ssf = StaticStructureFactorDebye([0.0])
        out = ssf.compute(frame)
        _k, sk, n = out[0]
        assert n == 7
        assert abs(sk[0] - 7.0) < 1e-10

    def test_linspace_constructor(self):
        ssf = StaticStructureFactorDebye.linspace(0.5, 5.0, 10)
        assert ssf is not None


class TestPmftXy:
    def test_two_particles_two_bins(self):
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        f = make_frame(pts, box_len=10.0)
        nl = _nlist(f, pts, cutoff=1.5)
        out = PmftXy(2.0, 2.0, 8, 8).compute(f, nl)
        counts, _density, _pmf = out[0]
        assert counts.sum() == 2  # one each side, self-query symmetric pair


class TestClusterProperties:
    def test_two_particle_uniform(self):
        pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        f = make_frame(pts, box_len=10.0)
        nl = _nlist(f, pts, cutoff=3.0)
        cl_result = Cluster(1).compute(f, nl)
        out = ClusterProperties().compute(f, [cl_result])
        d = out[0]
        assert d["sizes"] == [2]
        # Centre of mass at (1, 0, 0).
        assert abs(d["centers_of_mass"][0, 0] - 1.0) < 1e-10


class TestMsdMethodSelection:
    """``Msd(method=...)`` picks the estimator; there is no per-mode factory.

    The two are different estimators of the same quantity, so the choice is a
    constructor argument rather than a second constructor to keep in sync.
    """

    @staticmethod
    def _random_walk_frames(n_frames=40, n_particles=6, seed=7):
        rng = np.random.default_rng(seed)
        pos = np.cumsum(rng.normal(size=(n_frames, n_particles, 3)), axis=0)
        frames = []
        for t in range(n_frames):
            f, b = molrs.core.Frame(), molrs.core.Block()
            b["x"] = pos[t, :, 0].copy()
            b["y"] = pos[t, :, 1].copy()
            b["z"] = pos[t, :, 2].copy()
            f["atoms"] = b
            frames.append(f)
        return frames, pos

    def test_default_is_direct(self):
        assert molrs.compute.Msd().method == "direct"

    def test_window_is_selected_by_method(self):
        assert molrs.compute.Msd(method="window").method == "window"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown MSD method"):
            molrs.compute.Msd(method="rolling")

    def test_the_two_methods_disagree(self):
        """Otherwise the parameter would be pinning nothing."""
        frames, _ = self._random_walk_frames()
        direct = np.asarray(molrs.compute.Msd().compute(frames).values)
        window = np.asarray(molrs.compute.Msd(method="window").compute(frames).values)
        assert not np.allclose(direct, window)

    def test_frames_and_positions_give_one_curve(self):
        frames, pos = self._random_walk_frames()
        from_frames = molrs.compute.Msd(method="window").compute(frames, dt=2.0)
        from_array = molrs.compute.Msd(method="window").compute(pos, dt=2.0)
        np.testing.assert_array_equal(from_frames.values, from_array.values)
        np.testing.assert_array_equal(from_frames.per_entity, from_array.per_entity)
        assert from_frames.kind == "msd"
        assert from_frames.per_entity.shape == (len(frames), pos.shape[1])
        assert from_frames.lag_times[1] == 2.0
        assert from_frames.values[0] == 0.0

    def test_window_matches_the_nested_loop(self):
        _, pos = self._random_walk_frames(n_frames=20, n_particles=3)
        got = molrs.compute.Msd(method="window").compute(pos, max_lag=6)
        for lag in range(7):
            d = pos[lag:] - pos[: len(pos) - lag]
            want = (d**2).sum(axis=2).mean()
            assert got.values[lag] == pytest.approx(want, abs=1e-9)
        assert got.counts.tolist() == [20 - lag for lag in range(7)]

    def test_direct_is_the_single_origin_curve(self):
        _, pos = self._random_walk_frames(n_frames=12, n_particles=4)
        got = molrs.compute.Msd().compute(pos, max_lag=5)
        want = ((pos - pos[0]) ** 2).sum(axis=2)
        np.testing.assert_allclose(got.per_entity, want[:6])
        np.testing.assert_allclose(got.values, want[:6].mean(axis=1))
        assert got.counts.tolist() == [1] * 6
        with pytest.raises(ValueError, match="stride"):
            molrs.compute.Msd(stride=2)


class TestAcf:
    """Generic all-time-origins autocorrelation.

    The FFT route is checked against the definition written out longhand, and
    against ``Vacf`` to pin that the two give different numbers on the same
    array — swapping one for the other is the mistake this class exists to
    prevent.
    """

    @staticmethod
    def _series(n_frames=48, n_entities=5, n_components=3, seed=11):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n_frames, n_entities, n_components))

    @staticmethod
    def _direct(series, max_lag):
        n_frames, n_entities, _ = series.shape
        return np.array(
            [
                sum(
                    float((series[tau] * series[tau + lag]).sum())
                    for tau in range(n_frames - lag)
                )
                / (n_entities * (n_frames - lag))
                for lag in range(max_lag + 1)
            ]
        )

    def test_lag_zero_is_the_mean_square(self):
        s = self._series()
        got = np.asarray(molrs.compute.Acf().compute(s, max_lag=3).values)
        assert got[0] == pytest.approx((s**2).sum() / (s.shape[1] * s.shape[0]))

    def test_fft_route_matches_the_direct_definition(self):
        s = self._series(n_frames=32)
        got = molrs.compute.Acf().compute(s, max_lag=8)
        np.testing.assert_allclose(got.values, self._direct(s, 8), atol=1e-10)
        assert got.kind == "acf"
        assert got.lag_times is None
        assert got.counts.tolist() == [32 - lag for lag in range(9)]

    def test_max_lag_is_clamped(self):
        s = self._series(n_frames=10)
        res = molrs.compute.Acf().compute(s, max_lag=999)
        assert len(np.asarray(res.values)) == 10
        assert res.lags[-1] == 9

    def test_a_constant_series_is_flat(self):
        s = np.full((16, 3, 2), 2.0)
        got = np.asarray(molrs.compute.Acf().compute(s, max_lag=6).values)
        assert np.allclose(got, 8.0)  # n_components * value^2

    def test_is_not_the_same_estimator_as_vacf(self):
        """Vacf mean-subtracts and averages over degrees of freedom: the same
        estimator on a different view, so the numbers differ."""
        s = self._series()
        acf = np.asarray(molrs.compute.Acf().compute(s, max_lag=10).values)
        flat = s.reshape(s.shape[0], -1)
        vacf = np.asarray(molrs.compute.Vacf().compute(flat, 1.0, 10).values)
        assert not np.allclose(acf, vacf), "the two estimators must not coincide"

    def test_too_few_frames_raises(self):
        with pytest.raises(ValueError):
            molrs.compute.Acf().compute(np.zeros((1, 2, 3)), max_lag=0)


class TestRustNames:
    """Every ``molrs.compute`` name is the Rust name."""

    def test_dielectric_functions(self):
        charges = np.array([1.0, -1.0])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        m = molrs.compute.dipole_moment(charges, positions)
        np.testing.assert_allclose(m, [-1.0, 0.0, 0.0])
        dipoles = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        j = molrs.compute.current_density(dipoles, 1.0, 2.0)
        assert j.shape == (3, 3)
        assert np.isnan(j[0]).all()
        np.testing.assert_allclose(j[1:, 0], [0.5, 1.0])

    def test_onsager_correlation_is_a_compute(self):
        p = np.cumsum(np.ones((8, 3)), axis=0)
        out = molrs.compute.OnsagerCorrelation().compute(p, p, 1.0, 4)
        assert out.kind == "onsager"
        assert out.lag_times.shape == out.values.shape
        # Straight-line motion: |ΔP|² = 3 t².
        np.testing.assert_allclose(out.values, 3.0 * np.arange(5.0) ** 2)

    def test_distribution_function_reads_the_observable(self):
        with pytest.raises(ValueError, match="unknown observable"):
            molrs.compute.DistributionFunction("torsion", 10)
        with pytest.raises(ValueError, match="required"):
            molrs.compute.DistributionFunction("distance", 10)
        with pytest.raises(ValueError):
            molrs.compute.DistributionFunction("angle", 10, min=0.0)
        molrs.compute.DistributionFunction("angle", 18)
        molrs.compute.DistributionFunction("dihedral", 36)
        molrs.compute.DistributionFunction("distance", 10, 0.0, 5.0)

    def test_route_agreement_is_a_check(self):
        curve = np.linspace(0.0, 1.0, 16)
        out = molrs.compute.RouteAgreement().check({"a": curve, "b": curve})
        assert out["passed"]
        assert out["pairwise_rms"]

    def test_kramers_kronig_is_a_check(self):
        assert isinstance(molrs.compute.KramersKronig(1.0), molrs.compute.KramersKronig)
        assert isinstance(
            molrs.compute.ConductivitySumRule(1.0, 1.0, 300.0),
            molrs.compute.ConductivitySumRule,
        )


class TestKinetic:
    """``kinetic_energy`` / ``kinetic_temperature`` /
    ``center_of_mass_velocity`` — the readings ``md.MdDriver`` thermo prints."""

    def test_kinetic_energy_and_temperature(self):
        mass = np.array([2.0, 1.0])
        vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 2.0]])
        ke = molrs.compute.kinetic_energy(mass, vel)
        assert ke == pytest.approx(5.0)
        assert molrs.compute.kinetic_temperature(ke, 5, 0.5) == pytest.approx(4.0)
        with pytest.raises(ValueError):
            molrs.compute.kinetic_temperature(ke, 0, 0.5)

    def test_center_of_mass_velocity(self):
        mass = np.array([3.0, 1.0])
        vel = np.array([[1.0, 0.0, 0.0], [-3.0, 4.0, 0.0]])
        np.testing.assert_allclose(
            molrs.compute.center_of_mass_velocity(mass, vel), [0.0, 1.0, 0.0]
        )
