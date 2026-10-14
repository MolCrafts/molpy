"""FFI seam of the binned, gridded and pair-dynamics computes.

Each class is built from its required arguments only and called once on tiny
synthetic data; the assertions check that data crosses the boundary and the
result comes back with the right types and shapes. Numerical correctness is
pinned by the Rust tests in ``molrs::compute``.
"""

from __future__ import annotations

import molrs
import molrs.compute as mc
import numpy as np
import pytest

L = 6.0


@pytest.fixture
def box():
    return molrs.core.Box.cube(L)


@pytest.fixture
def points():
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, L, size=(20, 3))
    pts[1] = pts[0] + [0.8, 0.0, 0.0]  # one pair certainly bound
    return pts


@pytest.fixture
def table(box, points):
    return molrs.core.NeighborQuery(box, points, 2.5).query_self()


class TestRdf:
    def test_folds_one_pair_source_per_frame_of_either_kind(self, box, points, table):
        engine = molrs.core.NeighborList(2.5, points, box)
        out = mc.Rdf(10, 2.5, 0.0).compute(pairs=[table, engine])
        assert isinstance(out, mc.RdfResult)
        assert out.rdf.shape == (10,)
        assert out.bin_edges.shape == (11,)
        assert out.n_frames == 2

    def test_rejects_a_non_pair_source(self):
        with pytest.raises(TypeError, match="pair source"):
            mc.Rdf(10, 2.5, 0.0).compute(pairs=[np.zeros((2, 3))])


class TestDensity:
    def test_correlation_function(self, points, table):
        n = len(points)
        out = mc.CorrelationFunction(5, 2.5, 0.0).compute(
            pairs=table, values_a=np.ones(n), values_b=np.ones(n)
        )
        assert isinstance(out, mc.CorrelationFunctionResult)
        assert out.correlation.shape == (5,)
        assert out.bin_counts.dtype == np.uint64

    def test_local_density(self, points, table):
        out = mc.LocalDensity(2.0).compute(pairs=table)
        assert out.density.shape == (len(points),)

    def test_gaussian_density(self, box, points):
        out = mc.GaussianDensity((4, 4, 4), 0.5).compute(positions=points, simbox=box)
        assert out.density.shape == (4, 4, 4)
        assert isinstance(out.grid, mc.RegularGrid)

    def test_sphere_voxelization(self, box, points):
        out = mc.SphereVoxelization((4, 4, 4), 1.0).compute(
            positions=points, simbox=box
        )
        assert out.voxels.shape == (4, 4, 4)
        assert out.voxels.dtype == np.uint8

    def test_spatial_distribution(self, points):
        grid = mc.RegularGrid.centered([4.0, 4.0, 4.0], [4, 4, 4])
        template = points[:3] - points[:3].mean(axis=0)
        out = mc.SpatialDistribution(grid).compute(
            positions=[points, points],
            reference=[[0, 1, 2], [0, 1, 2]],
            template=[template, template],
            target=[[3, 4, 5], [3, 4, 5]],
        )
        assert isinstance(out, mc.SpatialDistributionResult)
        assert out.counts.shape == (4, 4, 4)
        assert out.n_frames == 2
        assert out.g_sdf is None

    def test_frames_lists_must_agree_in_length(self, points):
        grid = mc.RegularGrid.centered([4.0, 4.0, 4.0], [4, 4, 4])
        with pytest.raises(ValueError, match="reference"):
            mc.SpatialDistribution(grid).compute(
                positions=[points, points],
                reference=[[0, 1, 2]],
                template=[points[:3], points[:3]],
                target=[[3], [3]],
            )


class TestDistribution:
    def test_on_an_axis(self, points):
        df = mc.DistributionFunction("distance", axis=mc.Axis(0.0, 5.0, 10))
        out = df.compute(positions=[points], tuples=[[[0, 1], [1, 2]]])
        assert isinstance(out, mc.DistributionResult)
        assert out.density.shape == (10,)
        assert isinstance(out.histogram, mc.Histogram)
        assert out.n_frames == 1

    def test_over_the_natural_range(self, box, points):
        df = mc.DistributionFunction("angle", n_bins=6)
        out = df.compute(positions=[points], simbox=[box], tuples=[[[0, 1, 2]]])
        assert out.density.shape == (6,)

    @pytest.mark.parametrize(
        "kwargs", [{}, {"axis": mc.Axis(0.0, 1.0, 2), "n_bins": 4}]
    )
    def test_exactly_one_constructor_form(self, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            mc.DistributionFunction("distance", **kwargs)

    def test_combined(self, points):
        cd = mc.CombinedDistribution(
            ["distance", "angle"], [mc.Axis(0.0, 5.0, 5), mc.Axis(0.0, np.pi, 6)]
        )
        out = cd.compute(positions=[points], tuples=[[[[0, 1]], [[0, 1, 2]]]])
        assert isinstance(out, mc.CombinedDistributionResult)
        assert out.density.shape == (5, 6)
        assert isinstance(out.marginal(1), mc.DistributionResult)
        assert out.free_energy(300.0).shape == (5, 6)
        assert dict(cd.tunables()) == {"sin_weight": None}

    def test_combined_takes_two_or_three_observables(self):
        with pytest.raises(ValueError, match="only 2-D and 3-D"):
            mc.CombinedDistribution(["distance"], [mc.Axis(0.0, 1.0, 2)])


class TestDiffraction:
    def test_debye(self, points):
        out = mc.StaticStructureFactorDebye([1.0, 2.0]).compute(positions=points)
        assert out.sk.shape == (2,)
        assert out.n_particles == len(points)

    def test_direct_on_explicit_and_isotropic_k(self, box, points):
        explicit = mc.StaticStructureFactorDirect(k_vecs=[[1.0, 0.0, 0.0]])
        assert explicit.compute(positions=points).sk.shape == (1,)
        iso = mc.StaticStructureFactorDirect(k_max=5.0, n_bins=4)
        assert iso.compute(positions=points, simbox=box).sk.shape == (4,)

    @pytest.mark.parametrize(
        "kwargs", [{}, {"k_max": 5.0}, {"k_vecs": [[1.0, 0.0, 0.0]], "n_bins": 4}]
    )
    def test_direct_takes_one_constructor_form(self, kwargs):
        with pytest.raises(ValueError, match="either k_vecs"):
            mc.StaticStructureFactorDirect(**kwargs)

    def test_diffraction_pattern(self, box, points):
        out = mc.DiffractionPattern(16, 0.5).compute(positions=points, simbox=box)
        assert out.diffraction.shape == (16, 16)


class TestDynamics:
    def test_pair_presence_feeds_survival(self, table):
        presence = mc.PairPresence(1.5, False).compute(frames=[table, table, table])
        assert isinstance(presence, mc.Presence)
        assert presence.values.shape[0] == 3
        assert presence.pairs.shape == (presence.values.shape[1], 2)
        curve = mc.Survival("continuous", 2).compute(presence=presence.values, dt=1.0)
        assert isinstance(curve, mc.TimeCorrelation)
        assert curve.n_lags == 3

    def test_van_hove(self, box, points):
        trajectory = np.stack([points, points + 0.1, points + 0.2])
        out = mc.VanHove(mc.Axis(0.0, 3.0, 6), [1]).compute(
            positions=trajectory, boxes=[box, box, box]
        )
        assert isinstance(out, mc.VanHoveResult)
        assert out.g_self.shape == (1, 6)
        assert (
            dict(mc.VanHove(mc.Axis(0.0, 3.0, 6), [1], origins="first").tunables())[
                "origins"
            ]
            == "first"
        )


class TestHBonds:
    def test_bonds_presence_and_survival(self, box):
        frame = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.8, 0.0, 0.0]])
        positions = np.stack([frame, frame])
        table = molrs.core.NeighborQuery(box, frame[[2]], 3.5).query(frame[[0]])
        out = mc.HBonds().compute(
            positions=positions, donors=[(0, 1)], acceptors=[2], pairs=[table, table]
        )
        assert isinstance(out, mc.HBondsResult)
        assert list(out.counts) == [1, 1]
        bond = out.per_frame[0][0]
        assert isinstance(bond, mc.HBond)
        assert (bond.donor, bond.hydrogen, bond.acceptor) == (0, 1, 2)
        assert out.components(0).shape == (3,)
        presence = out.presence()
        assert isinstance(presence, mc.Presence)
        curve = mc.Survival("intermittent", 1).compute(presence=presence.values, dt=1.0)
        assert curve.n_lags == 2

    def test_criterion_fields_override_the_core_default(self):
        default = dict(mc.HBonds().tunables())
        tuned = dict(mc.HBonds(dist_kind="hydrogen_acceptor").tunables())
        assert tuned["dist_kind"] == "hydrogen_acceptor"
        assert tuned["dist_cutoff"] == default["dist_cutoff"]
        with pytest.raises(ValueError, match="HBondDistanceKind"):
            mc.HBonds(dist_kind="nope")
