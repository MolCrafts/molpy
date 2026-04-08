"""
System-level polymer generation module.

This module provides the top two layers of the three-layer architecture:
- SystemPlanner (top, system level): manages total system mass constraints
- PolydisperseChainGenerator (middle, chain level): manages chain length distribution and building chains

The data flow is:
- SystemPlanner keeps asking for chains from PolydisperseChainGenerator until total system mass is satisfied
- PolydisperseChainGenerator keeps asking for monomer sequences from SequenceGenerator when building each chain
- Each time SystemPlanner receives a chain, it updates the running total mass and decides whether to accept, stop, or trim

Distribution implementations live in :mod:`molpy.builder.polymer.distributions`.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .distributions import DPDistribution, MassDistribution
from .sequences import SequenceGenerator


@dataclass
class Chain:
    """
    Represents a single polymer chain.

    Attributes:
        dp: Degree of polymerization (number of monomers)
        monomers: List of monomer identifiers in the chain
        mass: Total mass of the chain (g/mol)
    """

    dp: int
    monomers: list[str]
    mass: float


@dataclass
class SystemPlan:
    """
    Represents a complete system plan with all chains.

    Attributes:
        chains: List of all chains in the system
        total_mass: Total mass of all chains (g/mol)
        target_mass: Target total mass that was requested (g/mol)
    """

    chains: list[Chain]
    total_mass: float
    target_mass: float


class PolydisperseChainGenerator:
    """
    Middle layer: Chain-level generator.

    Responsible for:
    - Sampling chain size:
        - Either in DP-space via a DPDistribution (sample_dp)
        - Or in mass-space via a MassDistribution (sample_mass)
    - Using a SequenceGenerator to build the chain sequence
    - Computing the mass of a chain using monomer mass table and optional end-group mass

    Does NOT know anything about total system mass. Only returns one chain at a time.
    """

    def __init__(
        self,
        seq_generator: SequenceGenerator,
        monomer_mass: dict[str, float],
        end_group_mass: float = 0.0,
        distribution: DPDistribution | MassDistribution | None = None,
    ):
        """
        Initialize polydisperse chain generator.

        Args:
            seq_generator: Sequence generator for generating monomer sequences
            monomer_mass: Dictionary mapping monomer identifiers to their masses (g/mol)
            end_group_mass: Mass of end groups (g/mol), default 0.0
            distribution: Distribution implementing DPDistribution or MassDistribution protocol
        """
        self.seq_generator = seq_generator
        self.monomer_mass = monomer_mass
        self.end_group_mass = end_group_mass

        if distribution is None:
            raise ValueError("distribution must be provided")
        self.distribution = distribution

    def sample_dp(self, rng: np.random.Generator) -> int:
        """
        Sample a degree of polymerization from the distribution.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        if self.distribution is None:
            raise ValueError("distribution must be set")

        # Determine what capabilities the distribution actually provides
        has_sample_dp = callable(getattr(self.distribution, "sample_dp", None))
        has_sample_mass = callable(getattr(self.distribution, "sample_mass", None))

        # DP-based distributions may only be used via sample_dp
        if has_sample_dp and not has_sample_mass:
            return self.distribution.sample_dp(rng)
        # Mass-distributions are not valid for DP sampling; caller should use sample_mass path.
        raise TypeError(
            f"Distribution {type(self.distribution).__name__} does not support 'sample_dp'. "
            "Use mass-based sampling via 'sample_mass' and the corresponding build_chain logic."
        )

    def sample_mass(self, rng: np.random.Generator) -> float:
        """
        Sample a target chain mass from a mass-based distribution.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Target chain mass in g/mol (>= 0)
        """
        if self.distribution is None:
            raise ValueError("distribution must be set")

        has_sample_mass = callable(getattr(self.distribution, "sample_mass", None))
        if not has_sample_mass:
            raise TypeError(
                f"Distribution {type(self.distribution).__name__} does not support 'sample_mass'. "
                "Use DP-based sampling via 'sample_dp' instead."
            )

        target_mass = float(self.distribution.sample_mass(rng))
        return max(0.0, target_mass)

    def build_chain(self, rng: np.random.Generator) -> Chain:
        """
        Sample DP, generate monomer sequence, and compute mass.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Chain object with dp, monomers, and mass
        """
        has_sample_dp = callable(getattr(self.distribution, "sample_dp", None))
        has_sample_mass = callable(getattr(self.distribution, "sample_mass", None))

        # Pure DPDistribution path: sample DP once and build fixed-length chain
        if has_sample_dp and not has_sample_mass:
            dp = self.sample_dp(rng)
            monomers = self.seq_generator.generate_sequence(dp, rng)
            mass = self._compute_mass(monomers)
            return Chain(dp=dp, monomers=monomers, mass=mass)

        # Pure MassDistribution path: sample a target mass and grow chain incrementally
        if has_sample_mass and not has_sample_dp:
            target_mass = self.sample_mass(rng)

            monomers: list[str] = []
            current_mass = self.end_group_mass

            # Conservative safety cap to prevent pathological infinite loops
            max_steps = 10_000
            steps = 0

            while steps < max_steps:
                steps += 1

                # Propose adding a single monomer
                next_label = self.seq_generator.generate_sequence(1, rng)[0]
                proposed_mass = current_mass + self.monomer_mass.get(next_label, 0.0)

                # Always accept at least one monomer, even if it overshoots
                if not monomers:
                    monomers.append(next_label)
                    current_mass = proposed_mass
                    if current_mass >= target_mass:
                        break
                    continue

                # For subsequent monomers, reject if this would overshoot the target
                if proposed_mass > target_mass:
                    break

                monomers.append(next_label)
                current_mass = proposed_mass

                # If we have reached or are extremely close to the target, stop
                if current_mass >= target_mass:
                    break

            dp = len(monomers)
            mass = current_mass
            return Chain(dp=dp, monomers=monomers, mass=mass)

        # Either the distribution implements neither method or both, which is invalid
        raise TypeError(
            f"Distribution {type(self.distribution).__name__} must implement exactly one of "
            "'sample_dp' or 'sample_mass'."
        )

    def _compute_mass(self, monomers: list[str]) -> float:
        """
        Compute the mass of a chain given its monomer sequence.

        Args:
            monomers: List of monomer identifiers

        Returns:
            Total chain mass (g/mol)
        """
        monomer_mass_sum = sum(self.monomer_mass.get(m, 0.0) for m in monomers)
        return monomer_mass_sum + self.end_group_mass


class SystemPlanner:
    """
    Top layer: System-level planner.

    Responsible for:
    - Enforcing a target total mass for the overall system
    - Iteratively requesting chains from PolydisperseChainGenerator
    - Maintaining a running sum of total mass
    - Stopping when mass reaches target window, and optionally trimming the final chain

    Does NOT micromanage sequence probabilities or DP distribution; only orchestrates at the ensemble level.
    """

    def __init__(
        self,
        chain_generator: PolydisperseChainGenerator,
        target_total_mass: float,
        max_rel_error: float = 0.02,
        max_chains: int | None = None,
        enable_trimming: bool = True,
    ):
        """
        Initialize system planner.

        Args:
            chain_generator: Chain generator for building chains
            target_total_mass: Target total system mass (g/mol)
            max_rel_error: Maximum relative error allowed (default 0.02 = 2%)
            max_chains: Maximum number of chains to generate (None = no limit)
            enable_trimming: Whether to enable chain trimming to better hit target mass
        """
        self.chain_generator = chain_generator
        self.target_total_mass = target_total_mass
        self.max_rel_error = max_rel_error
        self.max_chains = max_chains
        self.enable_trimming = enable_trimming

    def plan_system(self, rng: np.random.Generator) -> SystemPlan:
        """
        Repeatedly ask chain_generator for new chains until accumulated mass
        reaches target_total_mass within max_rel_error.

        Args:
            rng: np.random.Generator number generator

        Returns:
            SystemPlan with all chains and total mass
        """
        total_mass = 0.0
        chains: list[Chain] = []
        max_allowed_mass = self.target_total_mass * (1 + self.max_rel_error)

        while total_mass < self.target_total_mass:
            # Check max_chains constraint
            if self.max_chains is not None and len(chains) >= self.max_chains:
                break

            # Generate next chain
            chain = self.chain_generator.build_chain(rng)

            # Check if adding this chain would exceed the maximum allowed mass
            if total_mass + chain.mass <= max_allowed_mass:
                chains.append(chain)
                total_mass += chain.mass
            else:
                # Try to trim this chain if enabled
                if self.enable_trimming:
                    remaining_mass = self.target_total_mass - total_mass
                    trimmed = self._try_trim_chain(chain, remaining_mass, rng)
                    if trimmed is not None:
                        chains.append(trimmed)
                        total_mass += trimmed.mass
                break

        return SystemPlan(
            chains=chains,
            total_mass=total_mass,
            target_mass=self.target_total_mass,
        )

    def _try_trim_chain(
        self,
        chain: Chain,
        remaining_mass: float,
        rng: np.random.Generator,
    ) -> Chain | None:
        """
        Optional trimming logic: reduce chain.dp, regenerate sequence,
        so that mass ~= remaining_mass.

        May return None if trimming is not desired or not possible.

        Args:
            chain: Original chain that would exceed the target
            remaining_mass: Remaining mass needed to reach target
            rng: np.random.Generator number generator

        Returns:
            Trimmed chain, or None if trimming is not possible
        """
        if remaining_mass <= 0:
            return None

        # Estimate average monomer mass from the chain
        if len(chain.monomers) == 0:
            return None

        # Use expected composition to estimate average monomer mass
        expected_comp = self.chain_generator.seq_generator.expected_composition()
        avg_monomer_mass = sum(
            expected_comp.get(m, 0.0) * self.chain_generator.monomer_mass.get(m, 0.0)
            for m in expected_comp.keys()
        )

        if avg_monomer_mass <= 0:
            # Fallback: use actual chain average
            avg_monomer_mass = (
                chain.mass / len(chain.monomers) if len(chain.monomers) > 0 else 0
            )

        if avg_monomer_mass <= 0:
            return None

        # Estimate DP needed for remaining mass
        # mass = dp * avg_monomer_mass + end_group_mass
        # dp = (mass - end_group_mass) / avg_monomer_mass
        estimated_dp = int(
            (remaining_mass - self.chain_generator.end_group_mass) / avg_monomer_mass
        )

        # Ensure at least 1 monomer
        estimated_dp = max(1, estimated_dp)

        # Don't trim if estimated DP is not significantly smaller
        if estimated_dp >= chain.dp:
            return None

        # Generate a new shorter chain
        trimmed_monomers = self.chain_generator.seq_generator.generate_sequence(
            estimated_dp, rng
        )
        trimmed_mass = self.chain_generator._compute_mass(trimmed_monomers)

        # Only accept if it's closer to target than without trimming
        if trimmed_mass <= remaining_mass * (1 + self.max_rel_error):
            return Chain(
                dp=estimated_dp,
                monomers=trimmed_monomers,
                mass=trimmed_mass,
            )

        return None
