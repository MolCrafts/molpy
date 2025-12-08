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

from dataclasses import dataclass
from random import Random
from typing import Protocol
import numpy as np

from .sequence_generator import SequenceGenerator


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
    - Sampling chain length (DP) from a specified distribution
    - Using a SequenceGenerator to build the chain sequence
    - Computing the mass of a chain using monomer mass table and optional end-group mass

    Does NOT know anything about total system mass. Only returns one chain at a time.
    """

    def __init__(
        self,
        seq_generator: SequenceGenerator,
        monomer_mass: dict[str, float],
        end_group_mass: float = 0.0,
        dp_distribution: DPDistribution | None = None,
    ):
        """
        Initialize polydisperse chain generator.

        Args:
            seq_generator: Sequence generator for generating monomer sequences
            monomer_mass: Dictionary mapping monomer identifiers to their masses (g/mol)
            end_group_mass: Mass of end groups (g/mol), default 0.0
            dp_distribution: Distribution for sampling degree of polymerization
        """
        self.seq_generator = seq_generator
        self.monomer_mass = monomer_mass
        self.end_group_mass = end_group_mass

        if dp_distribution is None:
            raise ValueError("dp_distribution must be provided")
        self.dp_distribution = dp_distribution

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization from the distribution.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        if self.dp_distribution is None:
            raise ValueError("dp_distribution must be set")
        return self.dp_distribution.sample_dp(rng)

    def build_chain(self, rng: Random) -> Chain:
        """
        Sample DP, generate monomer sequence, and compute mass.

        Args:
            rng: Random number generator

        Returns:
            Chain object with dp, monomers, and mass
        """
        dp = self.sample_dp(rng)
        monomers = self.seq_generator.generate_sequence(dp, rng)
        mass = self._compute_mass(monomers)
        return Chain(dp=dp, monomers=monomers, mass=mass)

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


class DPDistribution(Protocol):
    """
    Protocol for degree of polymerization distributions.

    Used by PolydisperseChainGenerator to sample chain lengths.
    """

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        ...


class SchulzZimmDPDistribution:
    """
    Adapter that wraps SchulzZimm distribution to conform to DPDistribution protocol.

    This allows the existing SchulzZimm class to be used with PolydisperseChainGenerator.
    """

    def __init__(
        self,
        Mn: float,
        Mw: float,
        avg_monomer_mass: float,
        random_seed: int | None = None,
    ):
        """
        Initialize Schulz-Zimm DP distribution adapter.

        Args:
            Mn: Number-average molecular weight (g/mol)
            Mw: Weight-average molecular weight (g/mol)
            avg_monomer_mass: Average monomer molecular weight (g/mol)
            random_seed: Random seed for reproducibility (optional)
        """
        from .polydisperse import SchulzZimm

        self.sz_dist = SchulzZimm(Mn=Mn, Mw=Mw, random_seed=random_seed)
        self.avg_monomer_mass = avg_monomer_mass
        # Expose Mn, Mw, PDI for unified interface
        self.Mn = Mn
        self.Mw = Mw
        self.PDI = self.sz_dist.PDI

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization using Schulz-Zimm distribution.

        Args:
            rng: Random number generator (converted to numpy seed)

        Returns:
            Degree of polymerization (>= 1)
        """
        import numpy as np

        # Convert Random to numpy seed by sampling an integer from Random
        # and using it as seed for numpy
        # Random.randint(a, b) returns a random integer N such that a <= N <= b
        seed = rng.randint(0, 2**31 - 1)
        return self.sz_dist.sample_length(
            monomer_mw=self.avg_monomer_mass,
            random_seed=seed,
        )

    def molecular_weight_pdf(
        self,
        M: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Schulz-Zimm molecular weight distribution probability density function.

        Uses numpy only (no scipy dependency). The distribution is a Gamma distribution
        with shape parameter k and scale parameter theta.

        Args:
            M: Array of molecular weight values (g/mol)

        Returns:
            Array of probability density values
        """
        import numpy as np

        if self.sz_dist.Mw <= self.sz_dist.Mn or self.sz_dist.Mn <= 0:
            return np.zeros_like(M)

        k = self.sz_dist.k  # Shape parameter
        theta = self.sz_dist.theta  # Scale parameter

        # Gamma distribution PDF: f(x) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k))
        # For numerical stability, compute in log space
        # log(f(x)) = (k-1)*log(x) - x/theta - k*log(theta) - log(Gamma(k))

        # Compute log(Gamma(k)) using Stirling's approximation for k > 0
        def log_gamma(z: float) -> float:
            """Compute log(Gamma(z)) using Stirling's approximation."""
            if z <= 0:
                return np.nan
            # For small z, use recursion to reach z >= 12, then use Stirling's approximation
            if z < 12:
                # Recursively compute: log(Gamma(z)) = log(Gamma(z+1)) - log(z)
                # Keep recursing until we reach z >= 12
                result = 0.0
                current_z = z
                while current_z < 12:
                    result -= np.log(current_z)
                    current_z += 1.0
                # Now current_z >= 12, use Stirling's approximation
                return (
                    result
                    + (current_z - 0.5) * np.log(current_z)
                    - current_z
                    + 0.5 * np.log(2 * np.pi)
                    + 1.0 / (12 * current_z)
                )
            # Stirling's approximation for large z
            return (z - 0.5) * np.log(z) - z + 0.5 * np.log(2 * np.pi) + 1.0 / (12 * z)

        # Handle edge cases
        M_safe = np.where(M > 0, M, np.nan)

        # Compute log PDF
        log_pdf = (
            (k - 1) * np.log(M_safe) - M_safe / theta - k * np.log(theta) - log_gamma(k)
        )

        # Convert back from log space
        pdf = np.exp(log_pdf)

        # Handle edge cases
        pdf = np.where(np.isfinite(pdf), pdf, 0.0)
        pdf = np.where(M > 0, pdf, 0.0)

        return pdf


class UniformDPDistribution:
    """
    Uniform distribution for degree of polymerization.

    All chain lengths between min_dp and max_dp are equally likely.
    """

    def __init__(
        self,
        min_dp: int,
        max_dp: int,
        avg_monomer_mass: float,
        random_seed: int | None = None,
    ):
        """
        Initialize uniform DP distribution.

        Args:
            min_dp: Minimum degree of polymerization
            max_dp: Maximum degree of polymerization
            avg_monomer_mass: Average monomer molecular weight (g/mol)
            random_seed: Random seed for reproducibility (optional)
        """
        if min_dp < 1:
            raise ValueError(f"min_dp must be >= 1, got {min_dp}")
        if max_dp < min_dp:
            raise ValueError(f"max_dp ({max_dp}) must be >= min_dp ({min_dp})")

        self.min_dp = min_dp
        self.max_dp = max_dp
        self.avg_monomer_mass = avg_monomer_mass
        self.random_seed = random_seed

        # Calculate molecular weight range
        self.min_mw = min_dp * avg_monomer_mass
        self.max_mw = max_dp * avg_monomer_mass
        self.Mn = (self.min_mw + self.max_mw) / 2.0
        self.Mw = self.Mn  # Uniform distribution has PDI = 1
        self.PDI = 1.0

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization from uniform distribution.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization
        """
        return rng.randint(self.min_dp, self.max_dp + 1)

    def molecular_weight_pdf(self, M: np.ndarray) -> np.ndarray:
        """
        Calculate uniform molecular weight distribution PDF.

        Args:
            M: Array of molecular weight values (g/mol)

        Returns:
            Array of probability density values
        """
        import numpy as np

        pdf = np.zeros_like(M, dtype=float)
        mask = (M >= self.min_mw) & (M <= self.max_mw)
        if np.any(mask):
            pdf[mask] = 1.0 / (self.max_mw - self.min_mw)
        return pdf


class PoissonDPDistribution:
    """
    Poisson distribution for degree of polymerization.

    The Poisson distribution models the number of events (monomers) in a fixed interval.
    """

    def __init__(
        self,
        lambda_param: float,  # Mean of Poisson distribution
        avg_monomer_mass: float,
        random_seed: int | None = None,
    ):
        """
        Initialize Poisson DP distribution.

        Args:
            lambda_param: Mean (lambda) parameter of Poisson distribution
            avg_monomer_mass: Average monomer molecular weight (g/mol)
            random_seed: Random seed for reproducibility (optional)
        """
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be > 0, got {lambda_param}")

        self.lambda_param = lambda_param
        self.avg_monomer_mass = avg_monomer_mass
        self.random_seed = random_seed

        # For Poisson, mean = lambda, variance = lambda
        # Mn = lambda * avg_monomer_mass
        # Mw = (lambda + 1) * avg_monomer_mass (approximately, for large lambda)
        self.Mn = lambda_param * avg_monomer_mass
        self.Mw = (lambda_param + 1.0) * avg_monomer_mass
        self.PDI = self.Mw / self.Mn if self.Mn > 0 else 1.0

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization from Poisson distribution.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        import numpy as np

        seed = rng.randint(0, 2**31 - 1)
        rng_np = np.random.RandomState(seed)
        dp = rng_np.poisson(self.lambda_param)
        return max(1, dp)

    def molecular_weight_pdf(self, M: np.ndarray) -> np.ndarray:
        """
        Calculate Poisson molecular weight distribution PDF.

        Args:
            M: Array of molecular weight values (g/mol)

        Returns:
            Array of probability density values
        """
        import numpy as np

        # Convert molecular weight to DP
        dp_values = M / self.avg_monomer_mass

        # Poisson PMF: P(k) = lambda^k * exp(-lambda) / k!
        # For PDF, we need to account for the conversion from DP to MW
        pdf = np.zeros_like(M, dtype=float)

        # Compute log(Gamma(k+1)) = log(k!) for k >= 0
        def log_gamma(z: float) -> float:
            if z <= 0:
                return np.nan
            if z < 12:
                result = 0.0
                current_z = z
                while current_z < 12:
                    result -= np.log(current_z)
                    current_z += 1.0
                return (
                    result
                    + (current_z - 0.5) * np.log(current_z)
                    - current_z
                    + 0.5 * np.log(2 * np.pi)
                    + 1.0 / (12 * current_z)
                )
            return (z - 0.5) * np.log(z) - z + 0.5 * np.log(2 * np.pi) + 1.0 / (12 * z)

        # For each molecular weight, compute corresponding DP and Poisson probability
        for i, mw in enumerate(M):
            if mw <= 0:
                continue
            dp = mw / self.avg_monomer_mass
            k = int(np.round(dp))
            if k < 1:
                continue

            # Poisson PMF: P(k) = lambda^k * exp(-lambda) / k!
            # In log space: log(P(k)) = k*log(lambda) - lambda - log(k!)
            log_pmf = (
                k * np.log(self.lambda_param) - self.lambda_param - log_gamma(k + 1)
            )
            pmf = np.exp(log_pmf)

            # Convert from PMF (per integer DP) to PDF (per unit MW)
            # PDF = PMF / (avg_monomer_mass)
            pdf[i] = pmf / self.avg_monomer_mass

        return pdf


class FlorySchulzDPDistribution:
    """
    Flory-Schulz distribution for degree of polymerization.

    This is a geometric distribution, which is the discrete analog of exponential distribution.
    It's commonly used for step-growth polymerization.
    """

    def __init__(
        self,
        p: float,  # Success probability (0 < p < 1)
        avg_monomer_mass: float,
        random_seed: int | None = None,
    ):
        """
        Initialize Flory-Schulz DP distribution.

        Args:
            p: Success probability (0 < p < 1), related to extent of reaction
            avg_monomer_mass: Average monomer molecular weight (g/mol)
            random_seed: Random seed for reproducibility (optional)
        """
        if not (0 < p < 1):
            raise ValueError(f"p must be in (0, 1), got {p}")

        self.p = p
        self.avg_monomer_mass = avg_monomer_mass
        self.random_seed = random_seed

        # For geometric distribution: mean = (1-p)/p, variance = (1-p)/p^2
        # Mn = mean * avg_monomer_mass = (1-p)/p * avg_monomer_mass
        # Mw = (2-p)/(1-p) * Mn (for Flory-Schulz)
        mean_dp = (1.0 - p) / p
        self.Mn = mean_dp * avg_monomer_mass
        self.Mw = (2.0 - p) / (1.0 - p) * self.Mn
        self.PDI = self.Mw / self.Mn if self.Mn > 0 else 1.0

    def sample_dp(self, rng: Random) -> int:
        """
        Sample a degree of polymerization from Flory-Schulz distribution.

        Args:
            rng: Random number generator

        Returns:
            Degree of polymerization (>= 1)
        """
        import numpy as np

        seed = rng.randint(0, 2**31 - 1)
        rng_np = np.random.RandomState(seed)
        # Geometric distribution: number of failures before first success
        # We want number of successes (DP), so use geometric with p
        # For DP >= 1, we use: DP = geometric(p) + 1
        dp = rng_np.geometric(p=self.p)
        return max(1, dp)

    def molecular_weight_pdf(self, M: np.ndarray) -> np.ndarray:
        """
        Calculate Flory-Schulz molecular weight distribution PDF.

        Args:
            M: Array of molecular weight values (g/mol)

        Returns:
            Array of probability density values
        """
        import numpy as np

        # Convert molecular weight to DP
        dp_values = M / self.avg_monomer_mass

        # Flory-Schulz (geometric) PMF: P(k) = (1-p)^(k-1) * p for k >= 1
        # In log space: log(P(k)) = (k-1)*log(1-p) + log(p)
        pdf = np.zeros_like(M, dtype=float)

        log_one_minus_p = np.log(1.0 - self.p)
        log_p = np.log(self.p)

        for i, mw in enumerate(M):
            if mw <= 0:
                continue
            dp = mw / self.avg_monomer_mass
            k = int(np.round(dp))
            if k < 1:
                continue

            # Flory-Schulz PMF: P(k) = (1-p)^(k-1) * p
            log_pmf = (k - 1) * log_one_minus_p + log_p
            pmf = np.exp(log_pmf)

            # Convert from PMF (per integer DP) to PDF (per unit MW)
            pdf[i] = pmf / self.avg_monomer_mass

        return pdf


def create_dp_distribution_from_ir(
    distribution_ir: DistributionIR,
    avg_monomer_mass: float,
    random_seed: int | None = None,
) -> DPDistribution:
    """
    Create a DPDistribution instance from DistributionIR.

    Args:
        distribution_ir: DistributionIR from parser
        avg_monomer_mass: Average monomer molecular weight (g/mol)
        random_seed: Random seed for reproducibility

    Returns:
        DPDistribution instance

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
        return SchulzZimmDPDistribution(
            Mn=Mn, Mw=Mw, avg_monomer_mass=avg_monomer_mass, random_seed=random_seed
        )

    elif dist_name == "uniform":
        if "p0" not in params or "p1" not in params:
            raise ValueError(
                f"uniform requires 'p0' (min_dp) and 'p1' (max_dp) parameters, got {params}"
            )
        min_dp = int(params["p0"])
        max_dp = int(params["p1"])
        return UniformDPDistribution(
            min_dp=min_dp,
            max_dp=max_dp,
            avg_monomer_mass=avg_monomer_mass,
            random_seed=random_seed,
        )

    elif dist_name == "poisson":
        if "p0" not in params:
            raise ValueError(f"poisson requires 'p0' (lambda) parameter, got {params}")
        lambda_param = float(params["p0"])
        return PoissonDPDistribution(
            lambda_param=lambda_param,
            avg_monomer_mass=avg_monomer_mass,
            random_seed=random_seed,
        )

    elif dist_name == "flory_schulz":
        if "p0" not in params:
            raise ValueError(f"flory_schulz requires 'p0' (p) parameter, got {params}")
        p = float(params["p0"])
        return FlorySchulzDPDistribution(
            p=p, avg_monomer_mass=avg_monomer_mass, random_seed=random_seed
        )

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
        so that mass ≈ remaining_mass.

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
