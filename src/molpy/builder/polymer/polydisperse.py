"""
Polydisperse polymer generation module.

This module provides base classes and implementations for generating
polydisperse polymer ensembles based on molecular weight distributions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from molpy.core.atomistic import Atomistic

# Import sequence generator
from .sequence_generator import SequenceGenerator, WeightedSequenceGenerator


class Polydisperse(ABC):
    """
    Abstract base class for polydisperse polymer generation.

    This class defines the interface for generating polymers with
    different chain lengths according to a molecular weight distribution.

    Subclasses should implement the distribution-specific sampling logic.
    """

    def __init__(self, random_seed: int | None = None):
        """
        Initialize polydisperse generator.

        Args:
            random_seed: Random seed for reproducible sampling (optional)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def _get_rng(self, random_seed: int | None = None):
        """
        Get a numpy random number generator for sampling.

        Args:
            random_seed: Optional random seed for this specific sample.
                        If None, uses the instance's random_seed if set,
                        otherwise uses the global numpy random state.

        Returns:
            A numpy RandomState or Generator instance for random number generation.
        """
        if random_seed is not None:
            return np.random.RandomState(random_seed)
        elif self.random_seed is not None:
            return np.random.RandomState(self.random_seed)
        else:
            return np.random

    @abstractmethod
    def sample_length(
        self,
        monomer_mw: float,
        random_seed: int | None = None,
    ) -> int:
        """
        Sample a single polymer chain length (number of monomers) from the distribution.

        Args:
            monomer_mw: Average molecular weight of a monomer (g/mol)
            random_seed: Optional random seed for this specific sample

        Returns:
            Number of monomers in the chain (must be >= 1)
        """
        pass

    def generate_lengths(
        self,
        n_samples: int,
        monomer_mw: float,
        random_seed: int | None = None,
    ) -> List[int]:
        """
        Generate multiple chain length samples from the distribution.

        Args:
            n_samples: Number of length samples to generate
            monomer_mw: Average molecular weight of a monomer (g/mol)
            random_seed: Optional random seed for this batch (if None, uses different seed for each sample)

        Returns:
            List of chain lengths (number of monomers)
        """
        lengths = []
        for i in range(n_samples):
            # Use different seed for each sample if no global seed provided
            sample_seed = (
                random_seed
                if random_seed is not None
                else (self.random_seed + i if self.random_seed is not None else None)
            )
            length = self.sample_length(monomer_mw, random_seed=sample_seed)
            lengths.append(length)

        return lengths

    def generate_sequences(
        self,
        sequence_generator: SequenceGenerator,
        n_polymers: int,
        monomer_mw: float,
        random_seed: int | None = None,
    ) -> List[List[int]]:
        """
        Generate multiple polymer sequences using a sequence generator and chain length distribution.

        This method:
        1. Samples chain lengths from the distribution
        2. For each length, uses the sequence generator to generate a sequence

        Args:
            sequence_generator: SequenceGenerator instance for generating monomer sequences
            n_polymers: Number of polymer sequences to generate
            monomer_mw: Average molecular weight of a structure (g/mol)
            random_seed: Optional random seed for this batch

        Returns:
            List of sequences, where each sequence is a list of structure indices

        .. deprecated::
            This method uses the old SequenceGenerator interface.
            For new code, use the three-layer architecture:
            - SystemPlanner (top): manages total system mass constraints
            - PolydisperseChainGenerator (middle): manages chain length distribution
            - SequenceGenerator (bottom, Protocol-based): manages monomer selection
        """
        import warnings

        warnings.warn(
            "Polydisperse.generate_sequences() is deprecated. "
            "Use the new three-layer architecture: SystemPlanner → PolydisperseChainGenerator → SequenceGenerator. "
            "See example_polymer_system.py for usage.",
            DeprecationWarning,
            stacklevel=2,
        )
        if random_seed is not None:
            np.random.seed(random_seed)
        elif self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Sample chain lengths
        lengths = self.generate_lengths(n_polymers, monomer_mw, random_seed)

        # Generate sequences for each length using the sequence generator
        sequences = []
        for i, length in enumerate(lengths):
            # Use different seed for each sequence if no global seed
            seq_seed = (
                random_seed
                if random_seed is not None
                else (self.random_seed + i if self.random_seed is not None else None)
            )
            # Legacy: convert to new interface if needed
            # For WeightedSequenceGenerator, we need to adapt
            if hasattr(sequence_generator, "generate"):
                # Old interface (legacy code - type check suppressed)
                sequence = sequence_generator.generate(length, random_seed=seq_seed)  # type: ignore[attr-defined]
            elif hasattr(sequence_generator, "generate_sequence"):
                # New interface - convert random_seed to Random
                from random import Random

                rng = Random(seq_seed) if seq_seed is not None else Random()
                sequence = sequence_generator.generate_sequence(length, rng)
                # Convert string identifiers back to int indices for legacy compatibility
                sequence = [int(m) if m.isdigit() else m for m in sequence]
            else:
                raise ValueError(
                    "SequenceGenerator must implement either 'generate' or 'generate_sequence'"
                )
            sequences.append(sequence)

        return sequences


class SchulzZimm(Polydisperse):
    """
    Schulz-Zimm distribution for polydisperse polymer generation.

    The Schulz-Zimm distribution is a special case of the Gamma distribution
    commonly used to model molecular weight distributions in polymers.

    Parameters:
        Mn: Number-average molecular weight (g/mol)
        Mw: Weight-average molecular weight (g/mol)

    The distribution parameters k (shape) and theta (scale) are calculated as:
        k = Mn / (Mw - Mn)
        theta = (Mw - Mn) / Mn

    Note: Mw must be greater than Mn for a valid distribution.
    """

    def __init__(
        self,
        Mn: float,
        Mw: float,
        random_seed: int | None = None,
    ):
        """
        Initialize Schulz-Zimm distribution.

        Args:
            Mn: Number-average molecular weight (g/mol)
            Mw: Weight-average molecular weight (g/mol)
            random_seed: Random seed for reproducible sampling (optional)

        Raises:
            ValueError: If Mw <= Mn (invalid distribution)
        """
        if Mw <= Mn:
            raise ValueError(
                f"Mw ({Mw}) must be greater than Mn ({Mn}) for valid Schulz-Zimm distribution"
            )

        super().__init__(random_seed)
        self.Mn = Mn
        self.Mw = Mw

        # Calculate distribution parameters
        # Schulz-Zimm distribution: M ~ Gamma(k, theta)
        # From: Mn = k * theta, Mw = (k + 1) * theta
        # Solving: k = Mn / (Mw - Mn), theta = (Mw - Mn)
        # Note: theta here is the scale parameter for Gamma distribution
        # The mean of Gamma(k, scale) = k * scale = Mn
        self.k = Mn / (Mw - Mn)
        self.theta = Mw - Mn  # Scale parameter (has units of molecular weight)

        # Calculate polydispersity index
        self.PDI = Mw / Mn

    def sample_length(
        self,
        monomer_mw: float,
        random_seed: int | None = None,
    ) -> int:
        """
        Sample a polymer chain length from Schulz-Zimm distribution.

        The molecular weight is sampled from the Gamma distribution,
        then converted to number of monomers.

        Args:
            monomer_mw: Average molecular weight of a monomer (g/mol)
            random_seed: Optional random seed for this specific sample

        Returns:
            Number of monomers in the chain (>= 1)
        """
        rng = self._get_rng(random_seed)

        # Sample molecular weight from Gamma distribution
        # Schulz-Zimm is a Gamma distribution with:
        #   shape = k
        #   scale = theta (where theta = Mw - Mn)
        # The mean of Gamma(k, scale) = k * scale = k * (Mw - Mn) = Mn (correct!)
        # The variance = k * scale^2 = k * (Mw - Mn)^2
        polymer_mw = rng.gamma(shape=self.k, scale=self.theta)

        # Convert to number of monomers
        n_monomers = int(np.round(polymer_mw / monomer_mw))

        # Ensure at least 1 monomer
        return max(1, n_monomers)

    def __repr__(self) -> str:
        """String representation of SchulzZimm distribution."""
        return (
            f"SchulzZimm(Mn={self.Mn:.1f}, Mw={self.Mw:.1f}, "
            f"PDI={self.PDI:.3f}, k={self.k:.3f}, theta={self.theta:.1f})"
        )
