"""
Growth kernel for G-BigSMILES stochastic polymer growth.

This module provides the GrowthKernel protocol and implementations for
controlling port-level stochastic growth decisions.
"""

from typing import Protocol, Sequence

import numpy as np

from molpy.core.atomistic import Atomistic
from molpy.builder.polymer.port_utils import PortInfo
from molpy.builder.polymer.types import MonomerPlacement, MonomerTemplate


class GrowthKernel(Protocol):
    """Protocol for local transition function in port-level stochastic growth.

    A GrowthKernel decides which monomer (if any) to add next for a given
    reactive port on the growing polymer. This encapsulates the reaction
    probability logic from G-BigSMILES notation.
    """

    def choose_next_for_port(
        self,
        polymer: Atomistic,
        port: PortInfo,
        candidates: Sequence[MonomerTemplate],
        rng: np.random.Generator | None = None,
    ) -> MonomerPlacement | None:
        """Choose next monomer for a given port.

        Args:
            polymer: Current polymer structure
            port: Port to extend from
            candidates: Available monomer templates
            rng: Random number generator for sampling

        Returns:
            MonomerPlacement: Add this template at target port
            None: Terminate this port (implicit end-group)
        """
        ...


class ProbabilityTableKernel:
    """GrowthKernel based on G-BigSMILES probability tables.

    This kernel uses pre-computed probability tables that map each port
    descriptor to weighted choices over (template, target_descriptor_id) pairs.
    Weights are integers that are normalized to probabilities during sampling.
    """

    def __init__(
        self,
        probability_tables: dict[int, list[tuple[MonomerTemplate, int, int]]],
        end_group_templates: dict[int, MonomerTemplate] | None = None,
    ):
        """Initialize probability table kernel.

        Args:
            probability_tables: Maps descriptor_id -> [(template, target_desc, integer_weight)]
                Integer weights are normalized to probabilities during sampling.
            end_group_templates: Maps descriptor_id -> end-group template (no ports)
        """
        self.tables = probability_tables
        self.end_groups = end_group_templates or {}

    def choose_next_for_port(
        self,
        polymer: Atomistic,
        port: PortInfo,
        candidates: Sequence[MonomerTemplate],
        rng: np.random.Generator | None = None,
    ) -> MonomerPlacement | None:
        """Choose next monomer based on probability table.

        Args:
            polymer: Current polymer structure
            port: Port to extend from
            candidates: Available monomer templates
            rng: Random number generator (uses default if None)

        Returns:
            MonomerPlacement or None (terminate)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Extract descriptor ID from port metadata
        descriptor_id = self._get_descriptor_id(port)

        # Get probability table for this descriptor
        if descriptor_id not in self.tables:
            return None  # Terminate

        options = self.tables[descriptor_id]

        # Filter by available candidates and non-zero weights
        valid_options = [
            (tmpl, target_desc, weight)
            for tmpl, target_desc, weight in options
            if tmpl in candidates and weight > 0
        ]

        if not valid_options:
            return None  # Terminate

        # Normalize integer weights to probabilities and sample
        templates, target_descs, weights = zip(*valid_options)
        weights_array = np.array(weights, dtype=float)
        weights_array /= weights_array.sum()  # Normalize to probabilities

        idx = rng.choice(len(valid_options), p=weights_array)
        return MonomerPlacement(
            template=templates[idx], target_descriptor_id=target_descs[idx]
        )

    def _get_descriptor_id(self, port: PortInfo) -> int:
        """Extract descriptor ID from port metadata.

        Args:
            port: PortInfo object

        Returns:
            Descriptor ID (defaults to 0 if not found)
        """
        # Check for descriptor_id in port data
        if "descriptor_id" in port.data:
            return port.data["descriptor_id"]

        # Fallback: try to infer from port name
        # This is a temporary solution until parser sets descriptor_id
        port_name = port.name
        if port_name == "<":
            return 0
        elif port_name == ">":
            return 1
        else:
            # For other ports, use hash of name
            return hash(port_name) % 1000
