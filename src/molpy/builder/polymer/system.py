"""
System-level polymer generation module.

This module provides the top two layers of the three-layer architecture:
- SystemPlanner (top, system level): manages total system mass constraints
- PolydisperseChainGenerator (middle, chain level): manages chain length distribution and building chains

The data flow is:
- SystemPlanner keeps asking for chains from PolydisperseChainGenerator until total system mass is satisfied
- PolydisperseChainGenerator keeps asking for monomer sequences from SequenceGenerator when building each chain
- Each time SystemPlanner receives a chain, it updates the running total mass and decides whether to accept, stop, or trim
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from random import Random
from typing import Protocol, runtime_checkable

import numpy as np
from molpy.parser.smiles.gbigsmiles_ir import DistributionIR

from .sequence_generator import SequenceGenerator

# ============================================================================
# Capability-Based Distribution Protocols
# ============================================================================


@runtime_checkable
class DPDistribution(Protocol):
    """Protocol for distributions that sample degree of polymerization directly.

    Distributions implementing this protocol can sample DP values without
    requiring monomer mass information. This is suitable for distributions
    defined in DP space (e.g., Poisson, Uniform).
    """

    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample degree of polymerization from distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        ...

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """Probability mass function for DP values.

        Args:
            dp_array: Array of DP values

        Returns:
            Array of probability mass values
        """
        ...


@runtime_checkable
class MassDistribution(Protocol):
    """Protocol for distributions that sample molecular weight directly.

    Distributions implementing this protocol sample mass values directly
    from the distribution without converting through DP. This is suitable
    for distributions defined in mass space (e.g., Schulz-Zimm, Flory-Schulz).
    """

    def sample_mass(self, rng: np.random.Generator) -> float:
        """Sample molecular weight from distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Molecular weight (g/mol, > 0)
        """
        ...

    def mass_pdf(self, mass_array: np.ndarray) -> np.ndarray:
        """Probability density function for mass values.

        Args:
            mass_array: Array of mass values (g/mol)

        Returns:
            Array of probability density values
        """
        ...


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

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization from the distribution.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        if self.distribution is None:
            raise ValueError("distribution must be set")

        # Convert Random to numpy Generator
        seed = rng.randint(0, 2**31 - 1)
        np_rng = np.random.Generator(np.random.PCG64(seed))

        # Determine what capabilities the distribution actually provides
        has_sample_dp = callable(getattr(self.distribution, "sample_dp", None))
        has_sample_mass = callable(getattr(self.distribution, "sample_mass", None))

        # DP-based distributions may only be used via sample_dp
        if has_sample_dp and not has_sample_mass:
            return self.distribution.sample_dp(np_rng)
        # Mass-distributions are not valid for DP sampling; caller should use sample_mass path.
        raise TypeError(
            f"Distribution {type(self.distribution).__name__} does not support 'sample_dp'. "
            "Use mass-based sampling via 'sample_mass' and the corresponding build_chain logic."
        )

    def sample_mass(self, rng: Random) -> float:
        """
        Sample a target chain mass from a mass-based distribution.

        Args:
            rng: Random number generator

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

        seed = rng.randint(0, 2**31 - 1)
        np_rng = np.random.Generator(np.random.PCG64(seed))
        target_mass = float(self.distribution.sample_mass(np_rng))
        return max(0.0, target_mass)

    def build_chain(self, rng: Random) -> Chain:
        """
        Sample DP, generate monomer sequence, and compute mass.

        Args:
            rng: Random number generator

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


class SchulzZimmPolydisperse(MassDistribution):
    """
    Schulz-Zimm molecular weight distribution for polydisperse polymer chains.

    Implements :class:`MassDistribution` - sampling is done directly in
    molecular-weight (:math:`M`) space.

    Following the notation in the paper, the probability density
    (often also called “PMF” there) is

    .. math::

        \\operatorname{PMF}(M)
        = \\frac{z^{z+1}}{\\Gamma(z+1)}
          \\frac{M^{z-1}}{M_n^{z}}
          \\exp\\left(-\\frac{z M}{M_n}\\right),

    where

    .. math::

        z = \\frac{M_n}{M_w - M_n},

    :math:`M_n` is the number-average molecular weight, and
    :math:`M_w` is the weight-average molecular weight.

    This expression is mathematically equivalent to a Gamma distribution
    with shape :math:`z` and scale

    .. math::

        \\theta = \\frac{M_n}{z} = M_w - M_n,

    which satisfies the prescribed :math:`M_n` and :math:`M_w` with
    polydispersity index :math:`\\text{PDI} = M_w / M_n`.

    Parameters
    ----------
    Mn:
        Number-average molecular weight :math:`M_n` (g/mol).
    Mw:
        Weight-average molecular weight :math:`M_w` (g/mol), must satisfy
        :math:`M_w > M_n`.
    random_seed:
        Optional random seed used when sampling.
    """

    def __init__(
        self,
        Mn: float,
        Mw: float,
        random_seed: int | None = None,
    ):
        """
        Initialize Schulz-Zimm polydisperse distribution.

        Args:
            Mn: Number-average molecular weight (g/mol)
            Mw: Weight-average molecular weight (g/mol)
            random_seed: Random seed for reproducibility (optional)
        """
        if Mw <= Mn:
            raise ValueError(
                f"Mw ({Mw}) must be greater than Mn ({Mn}) for valid Schulz-Zimm distribution"
            )

        self.Mn = Mn
        self.Mw = Mw
        self.random_seed = random_seed

        # Calculate Gamma-equivalent parameters.
        # Using the paper's notation:
        #   z = Mn / (Mw - Mn)
        # and the Schulz-Zimm PDF can be written as a Gamma distribution
        # with shape parameter z and scale theta = Mn / z = Mw - Mn.
        self.z = Mn / (Mw - Mn)
        self.theta = Mw - Mn
        self.PDI = Mw / Mn

    # MassDistribution protocol implementation
    def sample_mass(self, rng: np.random.Generator) -> float:
        """Sample molecular weight directly from Schulz-Zimm distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Molecular weight (g/mol)
        """
        # Schulz-Zimm samples from Gamma distribution directly
        return rng.gamma(shape=self.z, scale=self.theta)

    def mass_pdf(self, mass_array: np.ndarray) -> np.ndarray:
        """Probability density function for mass values.

        This implements the Gamma PDF described in the class docstring and
        returns :math:`f(M)` evaluated at the entries of ``mass_array``.

        Args
        ----
        mass_array:
            Array of molecular weights :math:`M` (g/mol).

        Returns
        -------
        numpy.ndarray
            Array of probability density values with the same shape as
            ``mass_array``.
        """
        M = np.asarray(mass_array, dtype=float)
        if self.Mn <= 0 or self.Mw <= self.Mn:
            return np.zeros_like(M)

        z = self.z
        theta = self.theta
        out = np.zeros_like(M, dtype=float)

        mask = M > 0
        Ms = M[mask]

        log_pdf = (
            (z - 1.0) * np.log(Ms) - Ms / theta - z * np.log(theta) - math.lgamma(z)
        )
        out[mask] = np.exp(log_pdf)
        return out


class UniformPolydisperse(DPDistribution):
    """
    Uniform distribution over degree of polymerization (DP).

    Implements :class:`DPDistribution` - sampling is done directly in
    DP space.  All integer DP values between :math:`N_{\\min}` and
    :math:`N_{\\max}` (inclusive) are equally likely.

    The probability mass function (PMF) is

    .. math::

        P(N = k) =
        \\begin{cases}
            \\dfrac{1}{N_{\\max} - N_{\\min} + 1},
                & N_{\\min} \\le k \\le N_{\\max}, \\\\
            0, & \\text{otherwise},
        \\end{cases}

    where :math:`N` denotes the degree of polymerization.

    Parameters
    ----------
    min_dp:
        Lower bound :math:`N_{\\min}` for the degree of polymerization
        (must be :math:`\\ge 1`).
    max_dp:
        Upper bound :math:`N_{\\max}` for the degree of polymerization
        (must satisfy :math:`N_{\\max} \\ge N_{\\min}`).
    random_seed:
        Optional random seed used when sampling.
    """

    def __init__(
        self,
        min_dp: int,
        max_dp: int,
        random_seed: int | None = None,
    ):
        """
        Initialize uniform DP distribution.

        Args:
            min_dp: Minimum degree of polymerization (must be >= 1)
            max_dp: Maximum degree of polymerization (must be >= min_dp)
            random_seed: Random seed for reproducible sampling (optional)

        Raises:
            ValueError: If min_dp < 1 or max_dp < min_dp
        """
        if min_dp < 1:
            raise ValueError(f"min_dp must be >= 1, got {min_dp}")
        if max_dp < min_dp:
            raise ValueError(f"max_dp ({max_dp}) must be >= min_dp ({min_dp})")

        self.min_dp = min_dp
        self.max_dp = max_dp
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """
        Compute the probability mass function (PMF) over DP values.

        The PMF assigns equal probability to all integer DP values between
        min_dp and max_dp (inclusive), and zero probability outside this range.

        Args:
            dp_array: Array of DP values (typically integer, but can be float)

        Returns:
            Array of PMF values, same shape as dp_array.
            PMF[i] = 1 / (max_dp - min_dp + 1) if min_dp <= dp_array[i] <= max_dp,
                    0 otherwise.
        """
        dp_array = np.asarray(dp_array, dtype=float)
        pmf = np.zeros_like(dp_array, dtype=float)

        # Count number of valid integer DP values in the range
        n_valid = self.max_dp - self.min_dp + 1
        uniform_prob = 1.0 / n_valid

        # Assign probability to DP values within the valid range
        # Use np.round to handle float DP values (rounds to nearest integer)
        dp_rounded = np.round(dp_array).astype(int)
        mask = (dp_rounded >= self.min_dp) & (dp_rounded <= self.max_dp)
        pmf[mask] = uniform_prob

        return pmf

    # DPDistribution protocol implementation
    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample degree of polymerization from uniform distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        return int(rng.integers(self.min_dp, self.max_dp + 1))


class PoissonPolydisperse(DPDistribution):
    """
    Poisson distribution for the degree of polymerization (DP).

    Implements :class:`DPDistribution` - sampling is done directly in
    DP space.  The number of repeat units is modeled as a Poisson process
    with mean :math:`\\lambda`.

    The (untruncated) Poisson probability mass function is

    .. math::

        P(N = k)
        = \\frac{\\lambda^{k} e^{-\\lambda}}{k!},
        \\qquad k = 0, 1, 2, \\dots

    In this implementation we restrict to :math:`k \\ge 1` when sampling
    chains, i.e. a sampled value :math:`k = 0` is mapped to :math:`k = 1`
    so that every chain contains at least one monomer.

    Parameters
    ----------
    lambda_param:
        Mean :math:`\\lambda` of the Poisson distribution.
    random_seed:
        Optional random seed used when sampling.
    """

    def __init__(
        self,
        lambda_param: float,  # Mean of Poisson distribution
        random_seed: int | None = None,
    ):
        """
        Initialize Poisson DP distribution.

        Args:
            lambda_param: Mean (lambda) parameter of Poisson distribution
            random_seed: Random seed for reproducibility (optional)
        """
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be > 0, got {lambda_param}")

        self.lambda_param = lambda_param
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """
        Compute the probability mass function (PMF) over DP values.

        Poisson PMF: P(k; λ) = (λ^k * e^(-λ)) / k! for k >= 1

        Args:
            dp_array: Array of DP values

        Returns:
            Array of PMF values
        """
        k = np.asarray(dp_array, dtype=int)
        pmf = np.zeros_like(k, dtype=float)

        lam = float(self.lambda_param)
        if lam <= 0:
            return pmf

        mask = k >= 1
        ks = k[mask]

        # log P(k) = k log lam - lam - log(k!)
        log_p = ks * np.log(lam) - lam - np.array([math.lgamma(int(x) + 1) for x in ks])
        p = np.exp(log_p)

        # zero-truncated normalization: divide by (1 - e^{-lam})
        p /= 1.0 - np.exp(-lam)

        pmf[mask] = p
        return pmf

    # DPDistribution protocol implementation
    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample degree of polymerization from Poisson distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        dp = rng.poisson(self.lambda_param)
        return max(1, int(dp))


class FlorySchulzPolydisperse(DPDistribution):
    """
    Flory-Schulz distribution for degree of polymerization (DP).

    Implements :class:`DPDistribution` - sampling is done directly in DP
    space.  In this formulation the Flory-Schulz distribution is a geometric
    distribution over the chain length :math:`N`, commonly used for
    step-growth polymerization.

    The probability mass function (PMF) is

    .. math::

        P(N = k) = (1 - p)^{k-1} p, \\qquad k = 1, 2, \\dots,

    where :math:`p \\in (0, 1)` is related to the extent of reaction.

    Parameters
    ----------
    p:
        Success probability :math:`p` in the geometric PMF above
        (:math:`0 < p < 1`).
    random_seed:
        Optional random seed used when sampling.
    """

    def __init__(
        self,
        a: float,  # Success probability (0 < a < 1)
        random_seed: int | None = None,
    ):
        """
        Initialize Flory-Schulz DP distribution.

        Args:
            p: Success probability (0 < p < 1), related to extent of reaction
            random_seed: Random seed for reproducibility (optional)
        """
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}")

        self.a = a
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        k = np.asarray(dp_array, dtype=int)
        pmf = np.zeros_like(k, dtype=float)
        mask = k >= 1
        a = self.a
        pmf[mask] = (a * a) * k[mask] * (1.0 - a) ** (k[mask] - 1)
        return pmf

    # DPDistribution protocol implementation
    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample degree of polymerization from Flory-Schulz distribution.

        Args:
            rng: NumPy random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        a = self.a
        return max(1, int(rng.geometric(p=a) + rng.geometric(p=a) - 1))


def create_polydisperse_from_ir(
    distribution_ir: DistributionIR,
    random_seed: int | None = None,
) -> DPDistribution | MassDistribution:
    """
    Create a Polydisperse instance from DistributionIR.

    Args:
        distribution_ir: DistributionIR from parser
        random_seed: Random seed for reproducibility

    Returns:
        Polydisperse instance

    Raises:
        ValueError: If distribution type is not supported or parameters are invalid
    """
    dist_name = distribution_ir.name
    params = distribution_ir.params

    if dist_name == "schulz_zimm":
        if "p0" not in params or "p1" not in params:
            raise ValueError(
                f"schulz_zimm requires 'p0' (Mn) and 'p1' (Mw) parameters, got {params}"
            )
        Mn = float(params["p0"])
        Mw = float(params["p1"])
        return SchulzZimmPolydisperse(Mn=Mn, Mw=Mw, random_seed=random_seed)

    elif dist_name == "uniform":
        if "p0" not in params or "p1" not in params:
            raise ValueError(
                f"uniform requires 'p0' (min_dp) and 'p1' (max_dp) parameters, got {params}"
            )
        min_dp = int(params["p0"])
        max_dp = int(params["p1"])
        return UniformPolydisperse(
            min_dp=min_dp,
            max_dp=max_dp,
            random_seed=random_seed,
        )

    elif dist_name == "poisson":
        if "p0" not in params:
            raise ValueError(f"poisson requires 'p0' (lambda) parameter, got {params}")
        lambda_param = float(params["p0"])
        return PoissonPolydisperse(
            lambda_param=lambda_param,
            random_seed=random_seed,
        )

    elif dist_name == "flory_schulz":
        if "p0" not in params:
            raise ValueError(f"flory_schulz requires 'p0' (a) parameter, got {params}")
        a = float(params["p0"])
        return FlorySchulzPolydisperse(a=a, random_seed=random_seed)

    else:
        raise ValueError(
            f"Unsupported distribution type: {dist_name}. Supported types: schulz_zimm, uniform, poisson, flory_schulz"
        )


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

    def plan_system(self, rng: Random) -> SystemPlan:
        """
        Repeatedly ask chain_generator for new chains until accumulated mass
        reaches target_total_mass within max_rel_error.

        Args:
            rng: Random number generator

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
        rng: Random,
    ) -> Chain | None:
        """
        Optional trimming logic: reduce chain.dp, regenerate sequence,
        so that mass ~= remaining_mass.

        May return None if trimming is not desired or not possible.

        Args:
            chain: Original chain that would exceed the target
            remaining_mass: Remaining mass needed to reach target
            rng: Random number generator

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
