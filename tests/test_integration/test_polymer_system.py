"""Full statistical polymer-planning pipeline."""

import numpy as np
import pytest

from molpy.builder.polymer.distributions import SchulzZimmPolydisperse
from molpy.builder.polymer.sequences import WeightedSequenceGenerator
from molpy.builder.polymer.system import PolydisperseChainGenerator, SystemPlanner

pytestmark = pytest.mark.integration


class TestPolymerSystemWorkflow:
    def test_sequence_distribution_and_system_planning(self):
        sequence = WeightedSequenceGenerator({"A": 0.7, "B": 0.3})
        chains = PolydisperseChainGenerator(
            seq_generator=sequence,
            monomer_mass={"A": 100.0, "B": 150.0},
            end_group_mass=18.0,
            distribution=SchulzZimmPolydisperse(Mn=1500.0, Mw=3000.0),
        )
        planner = SystemPlanner(
            chain_generator=chains,
            target_total_mass=1.0e6,
            max_rel_error=0.02,
        )

        plan = planner.plan_system(np.random.default_rng(42))

        assert plan.chains
        assert all(chain.dp == len(chain.monomers) for chain in plan.chains)
        assert all(
            monomer in {"A", "B"} for chain in plan.chains for monomer in chain.monomers
        )
        relative_error = abs(plan.total_mass - plan.target_mass) / plan.target_mass
        assert relative_error <= planner.max_rel_error * 1.1
