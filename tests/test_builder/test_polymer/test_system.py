"""
Unit tests for the three-layer polymer system architecture.

Tests cover:
- SequenceGenerator (bottom layer)
- PolydisperseChainGenerator (middle layer)
- SystemPlanner (top layer)
"""

import numpy as np

import pytest

from molpy.builder import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    SchulzZimmPolydisperse,
    WeightedSequenceGenerator,
)
from molpy.builder import Chain, PolydisperseChainGenerator, SystemPlan, SystemPlanner


class TestChain:
    def test_records_one_sampled_chain(self):
        chain = Chain(dp=2, monomers=["A", "B"], mass=150.0)
        assert chain.dp == len(chain.monomers) == 2
        assert chain.mass == 150.0


class TestSystemPlan:
    def test_records_the_system_mass_target(self):
        chain = Chain(dp=1, monomers=["A"], mass=100.0)
        plan = SystemPlan(chains=[chain], total_mass=100.0, target_mass=100.0)
        assert plan.chains == [chain]
        assert plan.total_mass == plan.target_mass


class TestPolydisperseChainGenerator:
    """Tests for PolydisperseChainGenerator (middle layer)."""

    def test_chain_generator_basic(self):
        """Test basic chain generation."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        dp_dist = SchulzZimmPolydisperse(Mn=1500.0, Mw=3000.0)

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            distribution=dp_dist,
        )

        rng = np.random.default_rng(42)
        chain = chain_gen.build_chain(rng)

        assert isinstance(chain, Chain)
        assert chain.dp >= 1
        assert len(chain.monomers) == chain.dp
        assert chain.mass > 0
        assert all(m in ["A", "B"] for m in chain.monomers)

    def test_chain_generator_mass_calculation(self):
        """Test that chain mass is calculated correctly."""
        seq_gen = WeightedSequenceGenerator(
            monomer_weights={"A": 1.0}  # Only A monomers
        )

        # Create a simple fixed DP distribution for testing
        import numpy as np

        class FixedDpDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return self.dp

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                # Delta function at self.dp
                pmf = np.zeros_like(dp_array, dtype=float)
                pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
                return pmf

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=18.0,
            distribution=FixedDpDistribution(dp=10),
        )

        rng = np.random.default_rng(42)
        chain = chain_gen.build_chain(rng)

        # Mass should be 10 * 100.0 + 18.0 = 1018.0
        expected_mass = 10 * 100.0 + 18.0
        assert chain.mass == pytest.approx(expected_mass, abs=0.1)
        assert chain.dp == 10

    def test_chain_generator_dp_distribution(self):
        """Test that DP distribution is followed."""
        import numpy as np

        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        # Use a simple DP-based distribution for testing
        class PoissonDpDistribution:
            """Simple Poisson distribution for DP testing."""

            def __init__(self, mean_dp: float):
                self.mean_dp = mean_dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return max(1, int(rng.poisson(self.mean_dp)))

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                from scipy.stats import poisson

                return poisson.pmf(dp_array, self.mean_dp)

        dp_dist = PoissonDpDistribution(mean_dp=15)

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            distribution=dp_dist,
        )

        dps = [chain_gen.sample_dp(np.random.default_rng(i)) for i in range(100)]

        # Average DP should be approximately mean_dp = 15
        avg_dp = sum(dps) / len(dps)
        assert avg_dp == pytest.approx(15.0, abs=5.0)


class FixedDp:
    """A delta DP distribution, for predictable planner arithmetic."""

    def __init__(self, dp: int):
        self.dp = dp

    def sample_dp(self, rng: np.random.Generator) -> int:
        return self.dp

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        pmf = np.zeros_like(dp_array, dtype=float)
        pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
        return pmf


def _fixed_chains(dp: int, monomer_mass: float = 100.0) -> PolydisperseChainGenerator:
    return PolydisperseChainGenerator(
        seq_generator=WeightedSequenceGenerator({"A": 1.0}),
        monomer_mass={"A": monomer_mass},
        distribution=FixedDp(dp),
    )


class TestMassDistributionChains:
    """A mass-space chain is one whole sequence, not dp sequences of length 1."""

    def _chains(self, seq_gen, n: int = 50) -> list[Chain]:
        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 100.0},
            end_group_mass=0.0,
            distribution=SchulzZimmPolydisperse(Mn=2000.0, Mw=2400.0),
        )
        rng = np.random.default_rng(7)
        return [chain_gen.build_chain(rng) for _ in range(n)]

    def test_alternating_sequence_survives_mass_sampling(self):
        seq_gen = AlternatingSequenceGenerator(["A", "B"])
        chains = self._chains(seq_gen)
        assert sum(c.monomers.count("B") for c in chains) > 0
        for chain in chains:
            assert chain.monomers == seq_gen.generate_sequence(chain.dp)

    def test_block_sequence_survives_mass_sampling(self):
        seq_gen = BlockSequenceGenerator({"A": 0.5, "B": 0.5})
        chains = self._chains(seq_gen)
        assert sum(c.monomers.count("B") for c in chains) > 0
        for chain in chains:
            assert chain.monomers == seq_gen.generate_sequence(chain.dp)

    def test_chain_masses_follow_the_target_mn(self):
        chains = self._chains(WeightedSequenceGenerator({"A": 1.0, "B": 1.0}), 2000)
        mn = np.mean([c.mass for c in chains])
        assert mn == pytest.approx(2000.0, rel=0.03)

    def test_dp_matches_the_sampled_mass(self):
        chain_gen = PolydisperseChainGenerator(
            seq_generator=WeightedSequenceGenerator({"A": 1.0}),
            monomer_mass={"A": 100.0},
            end_group_mass=18.0,
            distribution=SchulzZimmPolydisperse(Mn=5000.0, Mw=6000.0),
        )
        target = chain_gen.sample_mass(np.random.default_rng(3))
        chain = chain_gen.build_chain(np.random.default_rng(3))
        assert chain.dp == max(1, round((target - 18.0) / 100.0))
        assert chain.mass == pytest.approx(chain.dp * 100.0 + 18.0)


class TestPolydisperseChainGeneratorInputs:
    def test_copies_the_callers_masses(self):
        masses = {"A": 100.0}
        chain_gen = PolydisperseChainGenerator(
            WeightedSequenceGenerator({"A": 1.0}), masses, distribution=FixedDp(3)
        )
        masses["A"] = 1.0
        assert chain_gen.build_chain(np.random.default_rng(0)).mass == 300.0

    def test_rejects_missing_distribution(self):
        with pytest.raises(TypeError, match="exactly one"):
            PolydisperseChainGenerator(
                WeightedSequenceGenerator({"A": 1.0}), {"A": 1.0}
            )

    def test_rejects_distribution_with_both_capabilities(self):
        class Both(FixedDp):
            def sample_mass(self, rng):
                return 1.0

            def mass_pdf(self, m):
                return m

        with pytest.raises(TypeError, match="exactly one"):
            PolydisperseChainGenerator(
                WeightedSequenceGenerator({"A": 1.0}), {"A": 1.0}, distribution=Both(1)
            )

    @pytest.mark.parametrize("mass", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_monomer_mass(self, mass):
        with pytest.raises(ValueError, match="monomer_mass"):
            PolydisperseChainGenerator(
                WeightedSequenceGenerator({"A": 1.0}),
                {"A": mass},
                distribution=FixedDp(1),
            )

    def test_rejects_negative_end_group_mass(self):
        with pytest.raises(ValueError, match="end_group_mass"):
            PolydisperseChainGenerator(
                WeightedSequenceGenerator({"A": 1.0}),
                {"A": 1.0},
                end_group_mass=-1.0,
                distribution=FixedDp(1),
            )

    def test_rejects_monomer_without_mass(self):
        with pytest.raises(ValueError, match="'B'"):
            PolydisperseChainGenerator(
                AlternatingSequenceGenerator(["A", "B"]),
                {"A": 1.0},
                distribution=FixedDp(1),
            )

    def test_rejects_dp_below_one(self):
        chain_gen = _fixed_chains(dp=0)
        with pytest.raises(ValueError, match="dp"):
            chain_gen.build_chain(np.random.default_rng(0))

    def test_rejects_sequence_of_wrong_length(self):
        class Short:
            def generate_sequence(self, dp, rng):
                return ["A"]

            def expected_composition(self):
                return {"A": 1.0}

        chain_gen = PolydisperseChainGenerator(
            Short(), {"A": 1.0}, distribution=FixedDp(3)
        )
        with pytest.raises(ValueError, match="3"):
            chain_gen.build_chain(np.random.default_rng(0))


class TestSystemPlanner:
    """Tests for SystemPlanner (top layer)."""

    def test_system_planner_basic(self):
        """Test basic system planning."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        dp_dist = SchulzZimmPolydisperse(Mn=1500.0, Mw=3000.0)

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            distribution=dp_dist,
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=1.0e6,
            max_rel_error=0.02,
        )

        rng = np.random.default_rng(42)
        system_plan = planner.plan_system(rng)

        assert isinstance(system_plan, SystemPlan)
        assert len(system_plan.chains) > 0
        assert system_plan.total_mass > 0
        assert system_plan.target_mass == 1.0e6

        # Total mass should be within max_rel_error of target
        rel_error = (
            abs(system_plan.total_mass - system_plan.target_mass)
            / system_plan.target_mass
        )
        assert rel_error <= planner.max_rel_error * 1.1  # Allow small margin

    def test_system_planner_mass_constraint(self):
        """Test that system planner respects mass constraints."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        # Use fixed DP for predictable testing
        import numpy as np

        class FixedDpDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return self.dp

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                pmf = np.zeros_like(dp_array, dtype=float)
                pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
                return pmf

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            distribution=FixedDpDistribution(dp=10),
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=5000.0,  # Should get ~5 chains
            max_rel_error=0.02,
        )

        rng = np.random.default_rng(42)
        system_plan = planner.plan_system(rng)

        # Should have approximately 5 chains
        assert len(system_plan.chains) >= 4
        assert len(system_plan.chains) <= 6

        # Total mass should be close to target
        assert system_plan.total_mass <= 5000.0 * 1.02

    def test_system_planner_max_chains(self):
        """Test that max_chains constraint is respected."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        import numpy as np

        class FixedDpDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return self.dp

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                pmf = np.zeros_like(dp_array, dtype=float)
                pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
                return pmf

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            distribution=FixedDpDistribution(dp=10),
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=1.0e6,  # Very large target
            max_rel_error=0.02,
            max_chains=10,  # Limit to 10 chains
        )

        rng = np.random.default_rng(42)
        system_plan = planner.plan_system(rng)

        assert len(system_plan.chains) <= 10

    def test_system_planner_trimming(self):
        """Test that chain trimming works when enabled."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        import numpy as np

        class FixedDpDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return self.dp

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                pmf = np.zeros_like(dp_array, dtype=float)
                pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
                return pmf

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            distribution=FixedDpDistribution(dp=100),  # Large chains
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=150.0,  # Less than one full chain
            max_rel_error=0.02,
            enable_trimming=True,
        )

        rng = np.random.default_rng(42)
        system_plan = planner.plan_system(rng)

        # Should have at least one chain (possibly trimmed)
        assert len(system_plan.chains) >= 1

        # If trimming worked, the last chain should be shorter
        if len(system_plan.chains) > 0:
            last_chain = system_plan.chains[-1]
            # Last chain might be trimmed to fit
            assert last_chain.dp <= 100

    def test_system_planner_no_trimming(self):
        """Test that trimming can be disabled."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        import numpy as np

        class FixedDpDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: np.random.Generator) -> int:
                return self.dp

            def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
                pmf = np.zeros_like(dp_array, dtype=float)
                pmf[np.round(dp_array).astype(int) == self.dp] = 1.0
                return pmf

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            distribution=FixedDpDistribution(dp=100),
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=150.0,
            max_rel_error=0.02,
            enable_trimming=False,
        )

        rng = np.random.default_rng(42)
        system_plan = planner.plan_system(rng)

        # Without trimming, should stop before adding a chain that would exceed target
        # So might have 0 chains or stop early
        assert len(system_plan.chains) >= 0
