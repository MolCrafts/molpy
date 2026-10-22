"""
System-level polymer generation module.

This module provides the top two layers of the three-layer architecture:
- SystemPlanner (top, system level): manages total system mass constraints
- PolydisperseChainGenerator (middle, chain level): manages chain length distribution and building chains

The data flow is:
- SystemPlanner keeps asking for chains from PolydisperseChainGenerator until total system mass is satisfied
- PolydisperseChainGenerator samples each chain's dp (directly, or from a sampled mass) and asks SequenceGenerator for that whole sequence in one call
- Each time SystemPlanner receives a chain, it updates the running total mass and decides whether to accept, stop, or trim
- SystemPlanner raises instead of returning a plan outside max_rel_error of the target

Distribution implementations live in :mod:`molpy.builder.polymer.distributions`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from molpy.builder.polymer.distributions import DpDistribution, MassDistribution
from molpy.builder.polymer.sequences import SequenceGenerator


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
        - Either in DP-space via a DpDistribution (sample_dp)
        - Or in mass-space via a MassDistribution (sample_mass), whose sampled
          mass is turned into the nearest dp at the mean monomer mass
    - Asking the SequenceGenerator for the whole sequence of that dp in one call
    - Computing the mass of a chain using monomer mass table and optional end-group mass

    Does NOT know anything about total system mass. Only returns one chain at a time.
    """

    def __init__(
        self,
        seq_generator: SequenceGenerator,
        monomer_mass: Mapping[str, float],
        end_group_mass: float = 0.0,
        distribution: DpDistribution | MassDistribution | None = None,
    ):
        """
        Initialize polydisperse chain generator.

        Args:
            seq_generator: Sequence generator for generating monomer sequences
            monomer_mass: Mapping of monomer identifiers to their masses (g/mol,
                finite and > 0); it must cover every monomer of
                ``seq_generator.expected_composition()``. The mapping is copied.
            end_group_mass: Mass of end groups (g/mol, >= 0), default 0.0
            distribution: Distribution implementing exactly one of the
                DpDistribution or MassDistribution protocols

        Raises:
            TypeError: if ``distribution`` implements neither or both protocols.
            ValueError: if a mass is not finite and positive (``end_group_mass``:
                non-negative), or a monomer of the sequence generator has no mass.
        """
        # DpDistribution / MassDistribution are runtime_checkable Protocols:
        # ask the type system, do not sniff for a method name.
        samples_dp = isinstance(distribution, DpDistribution)
        samples_mass = isinstance(distribution, MassDistribution)
        if samples_dp == samples_mass:
            raise TypeError(
                f"distribution {type(distribution).__name__} must implement exactly "
                "one of DpDistribution ('sample_dp') or MassDistribution ('sample_mass')"
            )

        masses = dict(monomer_mass)
        if not masses:
            raise ValueError("monomer_mass must be a non-empty mapping")
        for monomer_id, mass in masses.items():
            if not (math.isfinite(mass) and mass > 0):
                raise ValueError(
                    f"monomer_mass[{monomer_id!r}] must be finite and > 0, got {mass}"
                )
        if not (math.isfinite(end_group_mass) and end_group_mass >= 0):
            raise ValueError(
                f"end_group_mass must be finite and >= 0, got {end_group_mass}"
            )

        composition = seq_generator.expected_composition()
        missing = sorted(set(composition) - set(masses))
        if missing:
            raise ValueError(f"monomer_mass has no entry for monomers {missing}")
        mean_monomer_mass = sum(f * masses[m] for m, f in composition.items())
        if not mean_monomer_mass > 0:
            raise ValueError(
                f"{type(seq_generator).__name__}.expected_composition() gives a "
                f"non-positive mean monomer mass ({mean_monomer_mass})"
            )

        self.seq_generator = seq_generator
        self.monomer_mass = masses
        self.end_group_mass = float(end_group_mass)
        self.distribution = distribution
        self._mean_monomer_mass = mean_monomer_mass

    def sample_dp(self, rng: np.random.Generator) -> int:
        """
        Sample a degree of polymerization from a DP-based distribution.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        if not isinstance(self.distribution, DpDistribution):
            raise TypeError(
                f"Distribution {type(self.distribution).__name__} does not support "
                "'sample_dp'. Use mass-based sampling via 'sample_mass'."
            )
        return self.distribution.sample_dp(rng)

    def sample_mass(self, rng: np.random.Generator) -> float:
        """
        Sample a target chain mass from a mass-based distribution.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Target chain mass in g/mol (>= 0)
        """
        if not isinstance(self.distribution, MassDistribution):
            raise TypeError(
                f"Distribution {type(self.distribution).__name__} does not support "
                "'sample_mass'. Use DP-based sampling via 'sample_dp' instead."
            )
        return max(0.0, float(self.distribution.sample_mass(rng)))

    def build_chain(self, rng: np.random.Generator) -> Chain:
        """
        Sample a DP, generate the whole monomer sequence once, and compute mass.

        A mass-space distribution's sampled mass M becomes the dp nearest to
        ``(M - end_group_mass) / mean_monomer_mass`` (at least 1), where the mean
        monomer mass is weighted by ``seq_generator.expected_composition()``.
        The sequence is then generated in one call, so alternating and block
        generators keep their pattern; the chain's mass is that of its actual
        monomers.

        Args:
            rng: np.random.Generator number generator

        Returns:
            Chain object with dp, monomers, and mass
        """
        if isinstance(self.distribution, MassDistribution):
            target_mass = self.sample_mass(rng)
            dp = max(
                1,
                round((target_mass - self.end_group_mass) / self._mean_monomer_mass),
            )
        else:
            dp = self.sample_dp(rng)
            if dp < 1:
                raise ValueError(
                    f"{type(self.distribution).__name__} sampled dp={dp}; "
                    "dp must be >= 1"
                )
        return self._chain(dp, rng)

    def _chain(self, dp: int, rng: np.random.Generator) -> Chain:
        """Build the chain of one whole ``dp``-long sequence."""
        monomers = list(self.seq_generator.generate_sequence(dp, rng))
        if len(monomers) != dp:
            raise ValueError(
                f"{type(self.seq_generator).__name__} returned {len(monomers)} "
                f"monomers for dp={dp}"
            )
        return Chain(dp=dp, monomers=monomers, mass=self._compute_mass(monomers))

    def _compute_mass(self, monomers: list[str]) -> float:
        """
        Compute the mass of a chain given its monomer sequence.

        Args:
            monomers: List of monomer identifiers

        Returns:
            Total chain mass (g/mol)
        """
        monomer_mass_sum = sum(self.monomer_mass[m] for m in monomers)
        return monomer_mass_sum + self.end_group_mass


class SystemPlanner:
    """
    Top layer: System-level planner.

    Responsible for:
    - Enforcing a target total mass for the overall system
    - Iteratively requesting chains from PolydisperseChainGenerator
    - Maintaining a running sum of total mass
    - Stopping when mass reaches target window, and optionally trimming the
      chain that would overshoot it
    - Raising when the plan cannot reach the target within ``max_rel_error``

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
            target_total_mass: Target total system mass (g/mol, finite and > 0)
            max_rel_error: Maximum relative error allowed (default 0.02 = 2%,
                finite and >= 0)
            max_chains: Maximum number of chains to generate (None = no limit,
                otherwise >= 1). Reaching it short of the target is an error.
            enable_trimming: Whether to enable chain trimming to better hit target mass

        Raises:
            ValueError: if an argument is out of range.
        """
        if not (math.isfinite(target_total_mass) and target_total_mass > 0):
            raise ValueError(
                f"target_total_mass must be finite and > 0, got {target_total_mass}"
            )
        if not (math.isfinite(max_rel_error) and max_rel_error >= 0):
            raise ValueError(
                f"max_rel_error must be finite and >= 0, got {max_rel_error}"
            )
        if max_chains is not None and max_chains < 1:
            raise ValueError(f"max_chains must be None or >= 1, got {max_chains}")

        self.chain_generator = chain_generator
        self.target_total_mass = target_total_mass
        self.max_rel_error = max_rel_error
        self.max_chains = max_chains
        self.enable_trimming = enable_trimming

    def plan_system(self, rng: np.random.Generator) -> SystemPlan:
        """
        Repeatedly ask chain_generator for new chains until accumulated mass
        reaches target_total_mass within max_rel_error.

        A chain that would push the total past ``target * (1 + max_rel_error)``
        is trimmed (when enabled) and sampling goes on; when it cannot be
        trimmed, or ``max_chains`` is reached, sampling stops.

        Args:
            rng: np.random.Generator number generator

        Returns:
            SystemPlan with all chains and total mass; its ``total_mass`` is
            always within ``max_rel_error`` of ``target_mass``.

        Raises:
            RuntimeError: if sampling stopped with the total mass outside
                ``max_rel_error`` of the target. The message gives the
                planned mass, the target and why sampling stopped.
        """
        target = self.target_total_mass
        max_allowed_mass = target * (1 + self.max_rel_error)
        total_mass = 0.0
        chains: list[Chain] = []
        stopped_because = "sampling stopped"

        while total_mass < target:
            if self.max_chains is not None and len(chains) >= self.max_chains:
                stopped_because = f"max_chains={self.max_chains} was reached"
                break

            chain = self.chain_generator.build_chain(rng)
            if total_mass + chain.mass > max_allowed_mass:
                if not self.enable_trimming:
                    stopped_because = (
                        f"the next chain ({chain.mass:.6g} g/mol) overshoots and "
                        "trimming is disabled"
                    )
                    break
                trimmed = self._try_trim_chain(chain, target - total_mass, rng)
                if trimmed is None:
                    stopped_because = (
                        f"the next chain ({chain.mass:.6g} g/mol) overshoots and "
                        "cannot be trimmed to fit"
                    )
                    break
                chain = trimmed

            chains.append(chain)
            total_mass += chain.mass

        rel_error = abs(total_mass - target) / target
        if rel_error > self.max_rel_error:
            raise RuntimeError(
                f"planned {total_mass:.6g} g/mol in {len(chains)} chains against a "
                f"target of {target:.6g} g/mol (relative error {rel_error:.3g} > "
                f"max_rel_error={self.max_rel_error}): {stopped_because}"
            )

        return SystemPlan(
            chains=chains,
            total_mass=total_mass,
            target_mass=target,
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

        Args:
            chain: Original chain that would exceed the target
            remaining_mass: Remaining mass needed to reach target (> 0)
            rng: np.random.Generator number generator

        Returns:
            Trimmed chain, or None if no shorter chain fits
        """
        chain_generator = self.chain_generator
        # mass = dp * mean_monomer_mass + end_group_mass, rounded down to fit
        estimated_dp = max(
            1,
            int(
                (remaining_mass - chain_generator.end_group_mass)
                / chain_generator._mean_monomer_mass
            ),
        )
        if estimated_dp >= chain.dp:
            return None

        trimmed = chain_generator._chain(estimated_dp, rng)
        if trimmed.mass <= remaining_mass * (1 + self.max_rel_error):
            return trimmed
        return None
