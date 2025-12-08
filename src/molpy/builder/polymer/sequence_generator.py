"""
Sequence generator module for polymer sequence generation.

This module provides generators that produce sequences of monomer identifiers
based on connection probabilities and weights.

The SequenceGenerator is the bottom layer in the three-layer architecture:
- SystemPlanner (top, system level)
- PolydisperseChainGenerator (middle, chain level)
- SequenceGenerator (bottom, monomer level)
"""

from typing import Protocol, Iterator, List, Dict
from random import Random


class SequenceGenerator(Protocol):
    """
    Protocol for sequence generators.

    A sequence generator controls how monomers are arranged in a single chain.
    It encapsulates monomer reaction/selection probabilities and generates
    monomer sequences given a desired degree of polymerization (DP).

    This is the bottom layer in the three-layer architecture, responsible
    only for local monomer selection and reaction probabilities.
    """

    def generate_sequence(
        self,
        dp: int,
        rng: Random,
    ) -> List[str]:
        """
        Generate a monomer sequence of specified degree of polymerization.

        Args:
            dp: Degree of polymerization (number of monomers)
            rng: Random number generator for reproducible sampling

        Returns:
            List of monomer identifiers (or monomer keys as strings)
        """
        ...

    def expected_composition(self) -> Dict[str, float]:
        """
        Optional: Return expected long-chain monomer fractions.

        Used for rough mass estimates. Returns a dictionary mapping
        monomer identifiers to their expected fraction in long chains.

        Returns:
            Dictionary mapping monomer identifiers to expected fractions
        """
        ...


class WeightedSequenceGenerator:
    """
    Sequence generator based on monomer weights/proportions.

    This generator selects monomers based on their relative weights.
    Each selection is independent (no memory of previous selections).

    Conforms to the SequenceGenerator protocol.
    """

    def __init__(
        self,
        weights: Dict[int, float] | None = None,
        n_monomers: int | None = None,
        monomer_weights: Dict[str, float] | None = None,
    ):
        """
        Initialize weighted sequence generator.

        Args:
            weights: Dictionary mapping monomer index to selection weight (legacy format)
            n_monomers: Total number of available monomers (legacy format)
            monomer_weights: Dictionary mapping monomer identifier to selection weight (new format)

        Note: Either (weights, n_monomers) or monomer_weights should be provided.
        If monomer_weights is provided, it takes precedence.
        """
        if monomer_weights is not None:
            # New format: use string identifiers
            self.monomer_weights = monomer_weights
            self.monomer_ids = sorted(monomer_weights.keys())
            self.n_monomers = len(self.monomer_ids)

            # Normalize weights to probabilities
            total_weight = sum(monomer_weights.values())
            if total_weight > 0:
                self.probs = [
                    monomer_weights[mid] / total_weight for mid in self.monomer_ids
                ]
            else:
                # Equal probability if all weights are zero
                self.probs = [1.0 / self.n_monomers] * self.n_monomers
        elif weights is not None and n_monomers is not None:
            # Legacy format: convert indices to string identifiers
            self.n_monomers = n_monomers
            self.monomer_ids = [str(i) for i in range(n_monomers)]
            self.monomer_weights = {
                str(i): weights.get(i, 1.0) for i in range(n_monomers)
            }

            # Normalize weights to probabilities
            monomer_weights_list = [weights.get(i, 1.0) for i in range(n_monomers)]
            total_weight = sum(monomer_weights_list)
            if total_weight > 0:
                self.probs = [w / total_weight for w in monomer_weights_list]
            else:
                # Equal probability if all weights are zero
                self.probs = [1.0 / n_monomers] * n_monomers
        else:
            raise ValueError(
                "Either (weights, n_monomers) or monomer_weights must be provided"
            )

    def generate_sequence(
        self,
        dp: int,
        rng: Random,
    ) -> List[str]:
        """
        Generate a sequence of specified degree of polymerization.

        Args:
            dp: Degree of polymerization (number of monomers)
            rng: Random number generator for reproducible sampling

        Returns:
            List of monomer identifiers (strings)
        """
        # Use rng.choices for weighted random selection
        sequence = rng.choices(
            self.monomer_ids,
            weights=[self.monomer_weights[mid] for mid in self.monomer_ids],
            k=dp,
        )
        return sequence

    def expected_composition(self) -> Dict[str, float]:
        """
        Return expected long-chain monomer fractions.

        Returns:
            Dictionary mapping monomer identifiers to their expected fractions
        """
        total_weight = sum(self.monomer_weights.values())
        if total_weight > 0:
            return {
                mid: self.monomer_weights[mid] / total_weight
                for mid in self.monomer_ids
            }
        else:
            # Equal probability if all weights are zero
            equal_prob = 1.0 / len(self.monomer_ids)
            return {mid: equal_prob for mid in self.monomer_ids}

    def __repr__(self) -> str:
        """String representation of WeightedSequenceGenerator."""
        return (
            f"WeightedSequenceGenerator(n_monomers={self.n_monomers}, "
            f"monomer_weights={self.monomer_weights})"
        )
