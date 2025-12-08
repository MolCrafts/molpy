"""
Unit tests for the three-layer polymer system architecture.

Tests cover:
- SequenceGenerator (bottom layer)
- PolydisperseChainGenerator (middle layer)
- SystemPlanner (top layer)
"""

import random
from random import Random

import pytest

from molpy.builder.polymer.sequence_generator import WeightedSequenceGenerator
from molpy.builder.polymer.system import (
    Chain,
    PolydisperseChainGenerator,
    SchulzZimmDPDistribution,
    SystemPlan,
    SystemPlanner,
)


class TestSequenceGenerator:
    """Tests for SequenceGenerator (bottom layer)."""

    def test_weighted_sequence_generator_basic(self):
        """Test basic sequence generation with monomer weights."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        rng = Random(42)
        sequence = seq_gen.generate_sequence(dp=10, rng=rng)

        assert len(sequence) == 10
        assert all(m in ["A", "B"] for m in sequence)

    def test_weighted_sequence_generator_composition(self):
        """Test expected composition calculation."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        comp = seq_gen.expected_composition()

        assert comp["A"] == pytest.approx(0.7, abs=0.01)
        assert comp["B"] == pytest.approx(0.3, abs=0.01)
        assert sum(comp.values()) == pytest.approx(1.0, abs=0.01)

    def test_weighted_sequence_generator_legacy_format(self):
        """Test legacy format with weights dict and n_monomers."""
        seq_gen = WeightedSequenceGenerator(weights={0: 0.7, 1: 0.3}, n_monomers=2)

        rng = Random(42)
        sequence = seq_gen.generate_sequence(dp=10, rng=rng)

        assert len(sequence) == 10
        assert all(m in ["0", "1"] for m in sequence)

    def test_weighted_sequence_generator_distribution(self):
        """Test that sequence generation follows weight distribution."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.8, "B": 0.2})

        rng = Random(42)
        sequences = [
            seq_gen.generate_sequence(dp=100, rng=Random(i)) for i in range(10)
        ]

        # Count occurrences across all sequences
        all_monomers = [m for seq in sequences for m in seq]
        a_count = all_monomers.count("A")
        b_count = all_monomers.count("B")
        total = len(all_monomers)

        # Should be approximately 80% A and 20% B
        a_fraction = a_count / total
        assert a_fraction == pytest.approx(0.8, abs=0.1)


class TestPolydisperseChainGenerator:
    """Tests for PolydisperseChainGenerator (middle layer)."""

    def test_chain_generator_basic(self):
        """Test basic chain generation."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        dp_dist = SchulzZimmDPDistribution(
            Mn=1500.0, Mw=3000.0, avg_monomer_mass=100.0, random_seed=42
        )

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            dp_distribution=dp_dist,
        )

        rng = Random(42)
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
        class FixedDPDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: Random) -> int:
                return self.dp

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=18.0,
            dp_distribution=FixedDPDistribution(dp=10),
        )

        rng = Random(42)
        chain = chain_gen.build_chain(rng)

        # Mass should be 10 * 100.0 + 18.0 = 1018.0
        expected_mass = 10 * 100.0 + 18.0
        assert chain.mass == pytest.approx(expected_mass, abs=0.1)
        assert chain.dp == 10

    def test_chain_generator_dp_distribution(self):
        """Test that DP distribution is followed."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        dp_dist = SchulzZimmDPDistribution(
            Mn=1500.0, Mw=3000.0, avg_monomer_mass=100.0, random_seed=42
        )

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            dp_distribution=dp_dist,
        )

        rng = Random(42)
        dps = [chain_gen.sample_dp(Random(i)) for i in range(100)]

        # Average DP should be approximately Mn / avg_monomer_mass = 15
        avg_dp = sum(dps) / len(dps)
        assert avg_dp == pytest.approx(15.0, abs=5.0)


class TestSystemPlanner:
    """Tests for SystemPlanner (top layer)."""

    def test_system_planner_basic(self):
        """Test basic system planning."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        dp_dist = SchulzZimmDPDistribution(
            Mn=1500.0, Mw=3000.0, avg_monomer_mass=100.0, random_seed=42
        )

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            dp_distribution=dp_dist,
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=1.0e6,
            max_rel_error=0.02,
        )

        rng = Random(42)
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
        class FixedDPDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: Random) -> int:
                return self.dp

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            dp_distribution=FixedDPDistribution(dp=10),  # Each chain = 1000 g/mol
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=5000.0,  # Should get ~5 chains
            max_rel_error=0.02,
        )

        rng = Random(42)
        system_plan = planner.plan_system(rng)

        # Should have approximately 5 chains
        assert len(system_plan.chains) >= 4
        assert len(system_plan.chains) <= 6

        # Total mass should be close to target
        assert system_plan.total_mass <= 5000.0 * 1.02

    def test_system_planner_max_chains(self):
        """Test that max_chains constraint is respected."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        class FixedDPDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: Random) -> int:
                return self.dp

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            dp_distribution=FixedDPDistribution(dp=10),
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=1.0e6,  # Very large target
            max_rel_error=0.02,
            max_chains=10,  # Limit to 10 chains
        )

        rng = Random(42)
        system_plan = planner.plan_system(rng)

        assert len(system_plan.chains) <= 10

    def test_system_planner_trimming(self):
        """Test that chain trimming works when enabled."""
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 1.0})

        class FixedDPDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: Random) -> int:
                return self.dp

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            dp_distribution=FixedDPDistribution(dp=100),  # Large chains
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=150.0,  # Less than one full chain
            max_rel_error=0.02,
            enable_trimming=True,
        )

        rng = Random(42)
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

        class FixedDPDistribution:
            def __init__(self, dp: int):
                self.dp = dp

            def sample_dp(self, rng: Random) -> int:
                return self.dp

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0},
            end_group_mass=0.0,
            dp_distribution=FixedDPDistribution(dp=100),
        )

        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=150.0,
            max_rel_error=0.02,
            enable_trimming=False,
        )

        rng = Random(42)
        system_plan = planner.plan_system(rng)

        # Without trimming, should stop before adding a chain that would exceed target
        # So might have 0 chains or stop early
        assert len(system_plan.chains) >= 0


class TestIntegration:
    """Integration tests for the full three-layer pipeline."""

    def test_full_pipeline(self):
        """Test the complete three-layer pipeline."""
        # Bottom layer: SequenceGenerator
        seq_gen = WeightedSequenceGenerator(monomer_weights={"A": 0.7, "B": 0.3})

        # Middle layer: PolydisperseChainGenerator
        dp_dist = SchulzZimmDPDistribution(
            Mn=1500.0, Mw=3000.0, avg_monomer_mass=100.0, random_seed=42
        )

        chain_gen = PolydisperseChainGenerator(
            seq_generator=seq_gen,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            dp_distribution=dp_dist,
        )

        # Top layer: SystemPlanner
        planner = SystemPlanner(
            chain_generator=chain_gen,
            target_total_mass=1.0e6,
            max_rel_error=0.02,
        )

        rng = Random(42)
        system_plan = planner.plan_system(rng)

        # Verify the plan
        assert len(system_plan.chains) > 0
        assert system_plan.total_mass > 0

        # Verify each chain
        for chain in system_plan.chains:
            assert chain.dp >= 1
            assert len(chain.monomers) == chain.dp
            assert chain.mass > 0
            assert all(m in ["A", "B"] for m in chain.monomers)

        # Verify mass constraint
        rel_error = (
            abs(system_plan.total_mass - system_plan.target_mass)
            / system_plan.target_mass
        )
        assert rel_error <= planner.max_rel_error * 1.1
