"""Sequence generation for polymer assembly.

This module provides generators that produce sequences of monomer identifiers
based on connection probabilities and weights.

The SequenceGenerator is the bottom layer in the three-layer architecture:
- SystemPlanner (top, system level)
- PolydisperseChainGenerator (middle, chain level)
- SequenceGenerator (bottom, monomer level)
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = [
    "SequenceGenerator",
    "WeightedSequenceGenerator",
]


class SequenceGenerator(Protocol):
    """Protocol for sequence generators.

    A sequence generator controls how monomers are arranged in a single chain.
    """

    def generate_sequence(self, dp: int, rng: np.random.Generator) -> list[str]:
        """Generate a monomer sequence of specified degree of polymerization.

        Args:
            dp: Degree of polymerization (number of monomers)
            rng: numpy random Generator

        Returns:
            List of monomer identifiers (strings)
        """
        ...

    def expected_composition(self) -> dict[str, float]:
        """Return expected long-chain monomer fractions.

        Returns:
            Dictionary mapping monomer identifiers to expected fractions
        """
        ...


class WeightedSequenceGenerator:
    """Sequence generator based on monomer weights/proportions.

    Each selection is independent (no memory of previous selections).
    """

    def __init__(self, monomer_weights: dict[str, float]):
        if not monomer_weights:
            raise ValueError("monomer_weights must be a non-empty dictionary")
        for monomer_id in monomer_weights:
            if not isinstance(monomer_id, str) or not monomer_id:
                raise ValueError(
                    "monomer_weights keys must be non-empty string monomer identifiers"
                )

        self.monomer_weights = monomer_weights
        self.monomer_ids = sorted(monomer_weights.keys())
        self.n_monomers = len(self.monomer_ids)

    def generate_sequence(self, dp: int, rng: np.random.Generator) -> list[str]:
        """Generate a sequence of specified degree of polymerization.

        Args:
            dp: Degree of polymerization (number of monomers)
            rng: numpy random Generator

        Returns:
            List of monomer identifiers
        """
        weights = np.array([self.monomer_weights[mid] for mid in self.monomer_ids])
        probs = weights / weights.sum()
        indices = rng.choice(len(self.monomer_ids), size=dp, p=probs)
        return [self.monomer_ids[i] for i in indices]

    def expected_composition(self) -> dict[str, float]:
        """Return expected long-chain monomer fractions."""
        total_weight = float(sum(self.monomer_weights.values()))
        if total_weight > 0:
            return {
                mid: float(self.monomer_weights[mid]) / total_weight
                for mid in self.monomer_ids
            }
        n_monomers = len(self.monomer_ids)
        if n_monomers == 0:
            return {}
        equal_prob = 1.0 / n_monomers
        return {mid: equal_prob for mid in self.monomer_ids}

    def __repr__(self) -> str:
        return (
            f"WeightedSequenceGenerator(n_monomers={self.n_monomers}, "
            f"monomer_weights={self.monomer_weights})"
        )
