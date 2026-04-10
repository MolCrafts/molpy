"""Unit tests for polydisperse distribution implementations."""

import numpy as np
import pytest

from molpy.builder.polymer.distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
    create_polydisperse_from_ir,
)


# ---- UniformPolydisperse Tests ----


class TestUniformPolydisperse:
    def test_sample_in_range(self):
        dist = UniformPolydisperse(min_dp=5, max_dp=15)
        rng = np.random.default_rng(42)
        samples = [dist.sample_dp(rng) for _ in range(100)]
        assert all(5 <= s <= 15 for s in samples)

    def test_sample_single_value(self):
        dist = UniformPolydisperse(min_dp=7, max_dp=7)
        rng = np.random.default_rng(0)
        assert dist.sample_dp(rng) == 7

    def test_pmf_inside_range(self):
        dist = UniformPolydisperse(min_dp=1, max_dp=10)
        dp_arr = np.arange(1, 11)
        pmf = dist.dp_pmf(dp_arr)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-10)
        assert np.all(pmf == pytest.approx(0.1, abs=1e-10))

    def test_pmf_outside_range(self):
        dist = UniformPolydisperse(min_dp=5, max_dp=10)
        dp_arr = np.array([1, 2, 3, 4, 11, 12])
        pmf = dist.dp_pmf(dp_arr)
        assert np.all(pmf == 0.0)

    def test_validation_min_dp(self):
        with pytest.raises(ValueError, match="min_dp must be >= 1"):
            UniformPolydisperse(min_dp=0, max_dp=10)

    def test_validation_max_lt_min(self):
        with pytest.raises(ValueError, match="max_dp.*must be >= min_dp"):
            UniformPolydisperse(min_dp=10, max_dp=5)

    def test_satisfies_dp_protocol(self):
        dist = UniformPolydisperse(min_dp=1, max_dp=10)
        assert isinstance(dist, DPDistribution)


# ---- PoissonPolydisperse Tests ----


class TestPoissonPolydisperse:
    def test_sample_positive(self):
        dist = PoissonPolydisperse(lambda_param=10.0)
        rng = np.random.default_rng(42)
        samples = [dist.sample_dp(rng) for _ in range(100)]
        assert all(s >= 1 for s in samples)

    def test_mean_dp(self):
        dist = PoissonPolydisperse(lambda_param=20.0)
        rng = np.random.default_rng(42)
        samples = [dist.sample_dp(rng) for _ in range(1000)]
        mean = sum(samples) / len(samples)
        assert mean == pytest.approx(20.0, abs=3.0)

    def test_pmf_normalized(self):
        dist = PoissonPolydisperse(lambda_param=5.0)
        dp_arr = np.arange(1, 50)
        pmf = dist.dp_pmf(dp_arr)
        assert pmf.sum() == pytest.approx(1.0, abs=0.01)

    def test_pmf_zero_for_nonpositive(self):
        dist = PoissonPolydisperse(lambda_param=5.0)
        dp_arr = np.array([0, -1, -5])
        pmf = dist.dp_pmf(dp_arr)
        assert np.all(pmf == 0.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="lambda_param must be > 0"):
            PoissonPolydisperse(lambda_param=0.0)

    def test_satisfies_dp_protocol(self):
        assert isinstance(PoissonPolydisperse(lambda_param=5.0), DPDistribution)


# ---- FlorySchulzPolydisperse Tests ----


class TestFlorySchulzPolydisperse:
    def test_sample_positive(self):
        dist = FlorySchulzPolydisperse(a=0.5)
        rng = np.random.default_rng(42)
        samples = [dist.sample_dp(rng) for _ in range(100)]
        assert all(s >= 1 for s in samples)

    def test_pmf_positive_for_valid_dp(self):
        dist = FlorySchulzPolydisperse(a=0.3)
        dp_arr = np.arange(1, 20)
        pmf = dist.dp_pmf(dp_arr)
        assert np.all(pmf > 0)

    def test_pmf_zero_for_nonpositive(self):
        dist = FlorySchulzPolydisperse(a=0.5)
        dp_arr = np.array([0, -1])
        pmf = dist.dp_pmf(dp_arr)
        assert np.all(pmf == 0.0)

    def test_validation_bounds(self):
        with pytest.raises(ValueError, match="a must be in"):
            FlorySchulzPolydisperse(a=0.0)
        with pytest.raises(ValueError, match="a must be in"):
            FlorySchulzPolydisperse(a=1.0)
        with pytest.raises(ValueError, match="a must be in"):
            FlorySchulzPolydisperse(a=-0.5)

    def test_satisfies_dp_protocol(self):
        assert isinstance(FlorySchulzPolydisperse(a=0.5), DPDistribution)


# ---- SchulzZimmPolydisperse Tests ----


class TestSchulzZimmPolydisperse:
    def test_sample_positive(self):
        dist = SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0)
        rng = np.random.default_rng(42)
        samples = [dist.sample_mass(rng) for _ in range(100)]
        assert all(s > 0 for s in samples)

    def test_mean_mass(self):
        dist = SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0)
        rng = np.random.default_rng(42)
        samples = [dist.sample_mass(rng) for _ in range(5000)]
        mean = sum(samples) / len(samples)
        # Mean of Gamma(z, theta) = z * theta = Mn
        assert mean == pytest.approx(1000.0, rel=0.1)

    def test_pdi(self):
        dist = SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0)
        assert dist.PDI == pytest.approx(2.0)

    def test_pdf_positive_for_positive_mass(self):
        dist = SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0)
        mass_arr = np.linspace(100, 5000, 50)
        pdf = dist.mass_pdf(mass_arr)
        assert np.all(pdf > 0)

    def test_pdf_zero_for_nonpositive(self):
        dist = SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0)
        mass_arr = np.array([0.0, -100.0])
        pdf = dist.mass_pdf(mass_arr)
        assert np.all(pdf == 0.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="Mw.*must be greater than Mn"):
            SchulzZimmPolydisperse(Mn=2000.0, Mw=1000.0)
        with pytest.raises(ValueError, match="Mw.*must be greater than Mn"):
            SchulzZimmPolydisperse(Mn=1000.0, Mw=1000.0)

    def test_satisfies_mass_protocol(self):
        assert isinstance(
            SchulzZimmPolydisperse(Mn=1000.0, Mw=2000.0), MassDistribution
        )


# ---- Factory Tests ----


class TestCreatePolydisperseFromIR:
    def _make_ir(self, name: str, params: dict):
        """Create a mock DistributionIR."""

        class MockIR:
            pass

        ir = MockIR()
        ir.name = name
        ir.params = params
        return ir

    def test_uniform(self):
        ir = self._make_ir("uniform", {"p0": "5", "p1": "15"})
        dist = create_polydisperse_from_ir(ir)
        assert isinstance(dist, UniformPolydisperse)

    def test_poisson(self):
        ir = self._make_ir("poisson", {"p0": "10.0"})
        dist = create_polydisperse_from_ir(ir)
        assert isinstance(dist, PoissonPolydisperse)

    def test_flory_schulz(self):
        ir = self._make_ir("flory_schulz", {"p0": "0.5"})
        dist = create_polydisperse_from_ir(ir)
        assert isinstance(dist, FlorySchulzPolydisperse)

    def test_schulz_zimm(self):
        ir = self._make_ir("schulz_zimm", {"p0": "1000", "p1": "2000"})
        dist = create_polydisperse_from_ir(ir)
        assert isinstance(dist, SchulzZimmPolydisperse)

    def test_unknown_type_raises(self):
        ir = self._make_ir("unknown", {})
        with pytest.raises(ValueError, match="Unsupported distribution"):
            create_polydisperse_from_ir(ir)

    def test_missing_params_raises(self):
        ir = self._make_ir("uniform", {"p0": "5"})
        with pytest.raises(ValueError, match="requires"):
            create_polydisperse_from_ir(ir)
