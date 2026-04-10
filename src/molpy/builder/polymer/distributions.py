"""Polydisperse distribution implementations for polymer chain generation.

This module provides distributions for sampling degree of polymerization (DP)
or molecular weight during polymer system assembly.

Distribution Types:
- DPDistribution: Sample DP directly (Poisson, Uniform, Flory-Schulz)
- MassDistribution: Sample molecular weight directly (Schulz-Zimm)
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from molpy.parser.smiles.gbigsmiles_ir import DistributionIR

__all__ = [
    "DPDistribution",
    "MassDistribution",
    "UniformPolydisperse",
    "PoissonPolydisperse",
    "FlorySchulzPolydisperse",
    "SchulzZimmPolydisperse",
    "create_polydisperse_from_ir",
]


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
    for distributions defined in mass space (e.g., Schulz-Zimm).
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


# ============================================================================
# Distribution Implementations
# ============================================================================


class SchulzZimmPolydisperse:
    """Schulz-Zimm molecular weight distribution for polydisperse polymer chains.

    Implements :class:`MassDistribution` - sampling is done directly in
    molecular-weight space.

    The probability density is:

    .. math::

        f(M) = \\frac{z^{z+1}}{\\Gamma(z+1)}
               \\frac{M^{z-1}}{M_n^{z}}
               \\exp\\left(-\\frac{z M}{M_n}\\right),

    where z = Mn / (Mw - Mn). This is equivalent to a Gamma distribution
    with shape z and scale theta = Mw - Mn.

    Args:
        Mn: Number-average molecular weight (g/mol).
        Mw: Weight-average molecular weight (g/mol), must satisfy Mw > Mn.
        random_seed: Optional random seed.
    """

    def __init__(self, Mn: float, Mw: float, random_seed: int | None = None):
        if Mw <= Mn:
            raise ValueError(
                f"Mw ({Mw}) must be greater than Mn ({Mn}) for valid Schulz-Zimm distribution"
            )

        self.Mn = Mn
        self.Mw = Mw
        self.random_seed = random_seed
        self.z = Mn / (Mw - Mn)
        self.theta = Mw - Mn
        self.PDI = Mw / Mn

    def sample_mass(self, rng: np.random.Generator) -> float:
        """Sample molecular weight from Schulz-Zimm (Gamma) distribution."""
        return float(rng.gamma(shape=self.z, scale=self.theta))

    def mass_pdf(self, mass_array: np.ndarray) -> np.ndarray:
        """Probability density function for mass values."""
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


class UniformPolydisperse:
    """Uniform distribution over degree of polymerization (DP).

    All integer DP values between min_dp and max_dp (inclusive) are equally likely.

    Args:
        min_dp: Lower bound (>= 1).
        max_dp: Upper bound (>= min_dp).
        random_seed: Optional random seed.
    """

    def __init__(self, min_dp: int, max_dp: int, random_seed: int | None = None):
        if min_dp < 1:
            raise ValueError(f"min_dp must be >= 1, got {min_dp}")
        if max_dp < min_dp:
            raise ValueError(f"max_dp ({max_dp}) must be >= min_dp ({min_dp})")

        self.min_dp = min_dp
        self.max_dp = max_dp
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """PMF: equal probability for all integer DP in [min_dp, max_dp]."""
        dp_array = np.asarray(dp_array, dtype=float)
        pmf = np.zeros_like(dp_array, dtype=float)
        n_valid = self.max_dp - self.min_dp + 1
        uniform_prob = 1.0 / n_valid
        dp_rounded = np.round(dp_array).astype(int)
        mask = (dp_rounded >= self.min_dp) & (dp_rounded <= self.max_dp)
        pmf[mask] = uniform_prob
        return pmf

    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample DP uniformly from [min_dp, max_dp]."""
        return int(rng.integers(self.min_dp, self.max_dp + 1))


class PoissonPolydisperse:
    """Poisson distribution for the degree of polymerization (DP).

    Zero-truncated: sampled k=0 is mapped to k=1.

    Args:
        lambda_param: Mean of the Poisson distribution (> 0).
        random_seed: Optional random seed.
    """

    def __init__(self, lambda_param: float, random_seed: int | None = None):
        if lambda_param <= 0:
            raise ValueError(f"lambda_param must be > 0, got {lambda_param}")
        self.lambda_param = lambda_param
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """Zero-truncated Poisson PMF."""
        k = np.asarray(dp_array, dtype=int)
        pmf = np.zeros_like(k, dtype=float)

        lam = float(self.lambda_param)
        if lam <= 0:
            return pmf

        mask = k >= 1
        ks = k[mask]

        log_p = ks * np.log(lam) - lam - np.array([math.lgamma(int(x) + 1) for x in ks])
        p = np.exp(log_p)
        p /= 1.0 - np.exp(-lam)

        pmf[mask] = p
        return pmf

    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample DP from zero-truncated Poisson distribution (>= 1)."""
        while True:
            dp = int(rng.poisson(self.lambda_param))
            if dp >= 1:
                return dp


class FlorySchulzPolydisperse:
    """Flory-Schulz (geometric) distribution for degree of polymerization.

    PMF: P(N = k) = a^2 * k * (1 - a)^(k-1), k = 1, 2, ...

    Args:
        a: Probability parameter (0 < a < 1), related to extent of reaction.
        random_seed: Optional random seed.
    """

    def __init__(self, a: float, random_seed: int | None = None):
        if not (0 < a < 1):
            raise ValueError(f"a must be in (0, 1), got {a}")
        self.a = a
        self.random_seed = random_seed

    def dp_pmf(self, dp_array: np.ndarray) -> np.ndarray:
        """Flory-Schulz PMF."""
        k = np.asarray(dp_array, dtype=int)
        pmf = np.zeros_like(k, dtype=float)
        mask = k >= 1
        a = self.a
        pmf[mask] = (a * a) * k[mask] * (1.0 - a) ** (k[mask] - 1)
        return pmf

    def sample_dp(self, rng: np.random.Generator) -> int:
        """Sample DP from Flory-Schulz distribution (>= 1)."""
        a = self.a
        return max(1, int(rng.geometric(p=a) + rng.geometric(p=a) - 1))


# ============================================================================
# Factory
# ============================================================================


def create_polydisperse_from_ir(
    distribution_ir: DistributionIR,
    random_seed: int | None = None,
) -> DPDistribution | MassDistribution:
    """Create a distribution instance from a parsed DistributionIR.

    Args:
        distribution_ir: DistributionIR from parser.
        random_seed: Random seed for reproducibility.

    Returns:
        Distribution instance.

    Raises:
        ValueError: If distribution type is not supported or parameters are invalid.
    """
    dist_name = distribution_ir.name
    params = distribution_ir.params

    if dist_name == "schulz_zimm":
        if "p0" not in params or "p1" not in params:
            raise ValueError(
                f"schulz_zimm requires 'p0' (Mn) and 'p1' (Mw) parameters, got {params}"
            )
        return SchulzZimmPolydisperse(
            Mn=float(params["p0"]),
            Mw=float(params["p1"]),
            random_seed=random_seed,
        )

    if dist_name == "uniform":
        if "p0" not in params or "p1" not in params:
            raise ValueError(
                f"uniform requires 'p0' (min_dp) and 'p1' (max_dp) parameters, got {params}"
            )
        return UniformPolydisperse(
            min_dp=int(params["p0"]),
            max_dp=int(params["p1"]),
            random_seed=random_seed,
        )

    if dist_name == "poisson":
        if "p0" not in params:
            raise ValueError(f"poisson requires 'p0' (lambda) parameter, got {params}")
        return PoissonPolydisperse(
            lambda_param=float(params["p0"]),
            random_seed=random_seed,
        )

    if dist_name == "flory_schulz":
        if "p0" not in params:
            raise ValueError(f"flory_schulz requires 'p0' (a) parameter, got {params}")
        return FlorySchulzPolydisperse(a=float(params["p0"]), random_seed=random_seed)

    raise ValueError(
        f"Unsupported distribution type: {dist_name}. "
        "Supported types: schulz_zimm, uniform, poisson, flory_schulz"
    )
