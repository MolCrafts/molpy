"""
Stochastic polymer growth for G-BigSMILES notation.

This module implements breadth-first search (BFS) stochastic growth for building
polymers with complex topologies using template-based monomer placement.

Architecture:
- MonomerTemplate: Template for instantiable monomers with port descriptors
- PortDescriptor: Metadata for reactive ports on templates
- MonomerPlacement: Decision for next monomer to add
- StochasticChain: Result of stochastic growth (polymer + metadata)
- GrowthKernel: Protocol for choosing next monomer at a port
- ProbabilityTableKernel: G-BigSMILES probability table implementation
- StochasticChainGenerator: BFS-based chain builder

Example:
    >>> from molpy.builder.polymer.stochastic import (
    ...     MonomerTemplate, PortDescriptor, ProbabilityTableKernel,
    ...     StochasticChainGenerator
    ... )
    >>> import numpy as np
    >>>
    >>> # Define monomer template
    >>> template = MonomerTemplate(
    ...     label="EO",
    ...     structure=eo_monomer,
    ...     port_descriptors={
    ...         0: PortDescriptor(0, "<", role="left"),
    ...         1: PortDescriptor(1, ">", role="right"),
    ...     },
    ...     mass=44.05,
    ... )
    >>>
    >>> # Create growth kernel with probability table
    >>> kernel = ProbabilityTableKernel(
    ...     probability_tables={
    ...         0: [(template, 1, 10)],  # descriptor 0 -> template at port 1, weight 10
    ...     }
    ... )
    >>>
    >>> # Build stochastic chain
    >>> generator = StochasticChainGenerator(kernel, [template], connector, distribution)
    >>> chain = generator.build_chain(rng=np.random.default_rng(42))
    >>> print(f"Built polymer with DP={chain.dp}, mass={chain.mass:.1f} g/mol")
"""

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, Sequence

import numpy as np

from molpy.core.atomistic import Atomistic

# Import from consolidated modules
from .port_utils import get_all_ports

if TYPE_CHECKING:
    from .connectors import Connector
    from .distributions import DPDistribution, MassDistribution

__all__ = [
    # Data structures
    "PortDescriptor",
    "MonomerTemplate",
    "MonomerPlacement",
    "StochasticChain",
    # Growth kernel
    "GrowthKernel",
    "ProbabilityTableKernel",
    # Chain generator
    "StochasticChainGenerator",
]


# ==============================================================================
# Data Structures
# ==============================================================================


@dataclass
class PortDescriptor:
    """
    Descriptor for a reactive port on a monomer template.

    Port descriptors identify ports with unique IDs and store metadata
    about port behavior (role, bond type, compatibility).

    Attributes:
        descriptor_id: Unique ID within template (e.g., 0, 1, 2)
        port_name: Port name on atom (e.g., "<", ">", "branch")
        role: Port role (e.g., "left", "right", "branch")
        bond_kind: Bond type (e.g., "-", "=", "#")
        compat: Compatibility set for port matching

    Example:
        >>> desc = PortDescriptor(
        ...     descriptor_id=0,
        ...     port_name="<",
        ...     role="left",
        ...     bond_kind="-",
        ...     compat={"donor"}
        ... )
        >>> print(f"Descriptor {desc.descriptor_id}: port '{desc.port_name}' ({desc.role})")
    """

    descriptor_id: int
    port_name: str
    role: str | None = None
    bond_kind: str | None = None
    compat: set[str] | None = None


@dataclass
class MonomerTemplate:
    """
    Template for a monomer with port descriptors and metadata.

    This represents a monomer type that can be instantiated multiple times
    during stochastic growth. Each instantiation creates a fresh copy of
    the structure.

    Attributes:
        label: Monomer label (e.g., "EO2", "PS")
        structure: Base Atomistic structure (will be copied on instantiation)
        port_descriptors: Mapping from descriptor_id to PortDescriptor
        mass: Molecular weight (g/mol)
        metadata: Additional metadata (optional)

    Example:
        >>> template = MonomerTemplate(
        ...     label="EO",
        ...     structure=eo_monomer,
        ...     port_descriptors={
        ...         0: PortDescriptor(0, "<", role="left"),
        ...         1: PortDescriptor(1, ">", role="right"),
        ...     },
        ...     mass=44.05,
        ... )
        >>> fresh_copy = template.instantiate()
        >>> print(f"Template: {template.label}, mass={template.mass} g/mol")
    """

    label: str
    structure: Atomistic
    port_descriptors: dict[int, PortDescriptor]
    mass: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def instantiate(self) -> Atomistic:
        """
        Create a fresh copy of the structure.

        Each instantiation is independent with separate atoms and bonds,
        allowing the same template to be used multiple times in a polymer.

        Returns:
            New Atomistic instance with independent atoms and bonds

        Example:
            >>> template = MonomerTemplate(label="EO", structure=eo_monomer, ...)
            >>> copy1 = template.instantiate()
            >>> copy2 = template.instantiate()
            >>> copy1 is not copy2  # Different objects
            True
        """
        return self.structure.copy()

    def get_port_by_descriptor(self, descriptor_id: int) -> PortDescriptor | None:
        """
        Get port descriptor for a specific descriptor ID.

        Args:
            descriptor_id: Descriptor ID to look up

        Returns:
            PortDescriptor if found, None otherwise

        Example:
            >>> template = MonomerTemplate(...)
            >>> left_port = template.get_port_by_descriptor(0)
            >>> if left_port:
            ...     print(f"Port: {left_port.port_name}, role: {left_port.role}")
        """
        return self.port_descriptors.get(descriptor_id)

    def get_all_descriptors(self) -> list[PortDescriptor]:
        """
        Get all port descriptors for this template.

        Returns:
            List of all PortDescriptor objects sorted by descriptor_id

        Example:
            >>> template = MonomerTemplate(...)
            >>> descriptors = template.get_all_descriptors()
            >>> for desc in descriptors:
            ...     print(f"Port {desc.descriptor_id}: {desc.port_name}")
        """
        return list(self.port_descriptors.values())


@dataclass
class MonomerPlacement:
    """
    Decision for next monomer placement during stochastic growth.

    Represents the output of a GrowthKernel's decision: which template
    to add and which port on that template to connect.

    Attributes:
        template: MonomerTemplate to add
        target_descriptor_id: Which port descriptor on the new monomer to connect

    Example:
        >>> placement = MonomerPlacement(
        ...     template=eo_template,
        ...     target_descriptor_id=1  # Connect via port descriptor 1
        ... )
        >>> print(f"Add {placement.template.label} at port {placement.target_descriptor_id}")
    """

    template: "MonomerTemplate"
    target_descriptor_id: int


@dataclass
class StochasticChain:
    """
    Result of stochastic BFS growth.

    Contains the assembled polymer structure along with metadata about
    the growth process.

    Attributes:
        polymer: The assembled Atomistic structure
        dp: Degree of polymerization (number of monomers added)
        mass: Total molecular weight (g/mol)
        growth_history: Metadata for each monomer addition step

    Example:
        >>> chain = StochasticChain(
        ...     polymer=final_structure,
        ...     dp=25,
        ...     mass=1101.25,
        ...     growth_history=[...]
        ... )
        >>> print(f"Built polymer: DP={chain.dp}, mass={chain.mass:.1f} g/mol")
    """

    polymer: Atomistic
    dp: int
    mass: float
    growth_history: list[dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# Growth Kernel
# ==============================================================================


class GrowthKernel(Protocol):
    """
    Protocol for local transition function in port-level stochastic growth.

    A GrowthKernel decides which monomer (if any) to add next for a given
    reactive port on the growing polymer. This encapsulates the reaction
    probability logic from G-BigSMILES notation.

    Example:
        >>> class CustomKernel:
        ...     def choose_next_for_port(self, polymer, port, candidates, rng):
        ...         # Always choose first candidate
        ...         if candidates:
        ...             return MonomerPlacement(candidates[0], target_descriptor_id=0)
        ...         return None  # Terminate
    """

    def choose_next_for_port(
        self,
        polymer: Atomistic,
        port: "Atom",
        candidates: Sequence[MonomerTemplate],
        rng: np.random.Generator | None = None,
    ) -> MonomerPlacement | None:
        """
        Choose next monomer for a given port.

        Args:
            polymer: Current polymer structure
            port: Port to extend from
            candidates: Available monomer templates
            rng: Random number generator for sampling

        Returns:
            MonomerPlacement: Add this template at target port
            None: Terminate this port (implicit end-group)

        Example:
            >>> kernel = ProbabilityTableKernel(tables={...})
            >>> placement = kernel.choose_next_for_port(
            ...     polymer=current_polymer,
            ...     port=active_port,
            ...     candidates=[template1, template2],
            ...     rng=np.random.default_rng(42)
            ... )
            >>> if placement:
            ...     print(f"Add {placement.template.label}")
            ... else:
            ...     print("Terminate port")
        """
        ...


class ProbabilityTableKernel:
    """
    GrowthKernel based on G-BigSMILES probability tables.

    This kernel uses pre-computed probability tables that map each port
    descriptor to weighted choices over (template, target_descriptor_id) pairs.
    Weights are integers that are normalized to probabilities during sampling.

    Example:
        >>> # Create kernel with probability table
        >>> kernel = ProbabilityTableKernel(
        ...     probability_tables={
        ...         0: [  # For descriptor 0
        ...             (template_eo, 1, 10),   # EO at port 1, weight 10
        ...             (template_ps, 0, 5),    # PS at port 0, weight 5
        ...         ],
        ...         1: [  # For descriptor 1
        ...             (template_eo, 0, 10),   # EO at port 0, weight 10
        ...         ],
        ...     },
        ...     end_group_templates={
        ...         0: end_group_template,  # End-group for descriptor 0
        ...     }
        ... )
    """

    def __init__(
        self,
        probability_tables: dict[int, list[tuple[MonomerTemplate, int, int]]],
        end_group_templates: dict[int, MonomerTemplate] | None = None,
    ):
        """
        Initialize probability table kernel.

        Args:
            probability_tables: Maps descriptor_id -> [(template, target_desc, integer_weight)]
                Integer weights are normalized to probabilities during sampling.
            end_group_templates: Maps descriptor_id -> end-group template (no ports)

        Example:
            >>> tables = {
            ...     0: [(eo_template, 1, 10), (ps_template, 0, 5)],
            ...     1: [(eo_template, 0, 10)],
            ... }
            >>> kernel = ProbabilityTableKernel(tables)
        """
        self.tables = probability_tables
        self.end_groups = end_group_templates or {}

    def choose_next_for_port(
        self,
        polymer: Atomistic,
        port: "Atom",
        candidates: Sequence[MonomerTemplate],
        rng: np.random.Generator | None = None,
    ) -> MonomerPlacement | None:
        """
        Choose next monomer based on probability table.

        Looks up the port's descriptor ID in the probability table,
        filters by available candidates, and samples according to weights.

        Args:
            polymer: Current polymer structure
            port: Port to extend from
            candidates: Available monomer templates
            rng: Random number generator (uses default if None)

        Returns:
            MonomerPlacement or None (terminate)

        Example:
            >>> kernel = ProbabilityTableKernel(tables={...})
            >>> placement = kernel.choose_next_for_port(
            ...     polymer, port, [eo_template, ps_template],
            ...     rng=np.random.default_rng(42)
            ... )
            >>> if placement:
            ...     print(f"Chose {placement.template.label}")
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

    def _get_descriptor_id(self, port: "Atom") -> int:
        """Extract descriptor ID from port atom."""
        desc_id = port.get("port_descriptor_id")
        if desc_id is not None:
            return desc_id

        port_name = port.get("port", "")
        if port_name == "<":
            return 0
        elif port_name == ">":
            return 1
        else:
            return hash(port_name) % 1000


# ==============================================================================
# Stochastic Chain Generator
# ==============================================================================


class StochasticChainGenerator:
    """
    BFS-based stochastic chain generator for G-BigSMILES polymers.

    This generator builds polymer chains using breadth-first search (BFS)
    growth, where each added monomer introduces new reactive ports that
    enter a queue. Growth proceeds until chain mass or DP target is reached.

    Key features:
    - Template-based monomer placement
    - Port-resolved stochastic growth
    - Mass-based or DP-based stopping criteria
    - BFS port queue management

    Example:
        >>> from molpy.builder.polymer.stochastic import StochasticChainGenerator
        >>> from molpy.builder.polymer.distributions import UniformPolydisperse
        >>>
        >>> # Create generator
        >>> distribution = UniformPolydisperse(dp_low=10, dp_high=20)
        >>> generator = StochasticChainGenerator(
        ...     growth_kernel=kernel,
        ...     monomer_templates=[eo_template, ps_template],
        ...     connector=reacter_connector,
        ...     distribution=distribution,
        ...     seed_template=eo_template
        ... )
        >>>
        >>> # Build chain
        >>> import numpy as np
        >>> chain = generator.build_chain(rng=np.random.default_rng(42))
        >>> print(f"Built polymer: DP={chain.dp}, mass={chain.mass:.1f} g/mol")
    """

    def __init__(
        self,
        growth_kernel: GrowthKernel,
        monomer_templates: Sequence[MonomerTemplate],
        connector: "Connector",  # Forward reference (was ReacterConnector)
        distribution: "DPDistribution | MassDistribution",  # Forward reference
        seed_template: MonomerTemplate | None = None,
    ):
        """
        Initialize stochastic chain generator.

        Args:
            growth_kernel: GrowthKernel for choosing next monomers
            monomer_templates: Available monomer templates
            connector: Connector for applying reactions
            distribution: Distribution for sampling target (DP or mass)
            seed_template: Template for seed monomer (uses first template if None)

        Example:
            >>> generator = StochasticChainGenerator(
            ...     growth_kernel=kernel,
            ...     monomer_templates=[template1, template2],
            ...     connector=reacter_conn,
            ...     distribution=dp_dist,
            ... )
        """
        self.kernel = growth_kernel
        self.templates = list(monomer_templates)
        self.connector = connector
        self.distribution = distribution
        self.seed_template = seed_template or self.templates[0]

        # Import here to avoid circular dependency
        from .distributions import MassDistribution

        # Determine stopping criterion based on distribution capabilities
        self.use_mass_criterion = isinstance(distribution, MassDistribution)

    def build_chain(self, rng: np.random.Generator | None = None) -> StochasticChain:
        """Build a single polymer chain using BFS growth.

        The algorithm:
        1. Sample target DP or mass from distribution
        2. Initialize with seed monomer
        3. Maintain BFS queue of active ports
        4. For each port: choose next monomer, connect, add new ports to queue
        5. Stop when target reached or all ports terminated

        Args:
            rng: NumPy random number generator (creates default if None)

        Returns:
            StochasticChain with polymer, DP, mass, and growth history
        """
        import logging

        logger = logging.getLogger(__name__)

        if rng is None:
            rng = np.random.default_rng()

        # 1. Sample target from distribution
        if self.use_mass_criterion:
            target_mass = self.distribution.sample_mass(rng)  # type: ignore
            target_dp = None
        else:
            target_dp = self.distribution.sample_dp(rng)  # type: ignore
            target_mass = None

        # 2. Initialize with seed monomer
        polymer = self.seed_template.instantiate()
        mass = self.seed_template.mass
        dp = 1

        # 3. Initialize active ports queue (BFS)
        # Track ports by identity to avoid duplicates
        active_ports: deque["Atom"] = deque()
        seen_port_targets: set[int] = set()
        for port_list in get_all_ports(polymer).values():
            for port_info in port_list:
                port_id = id(port_info.target)
                if port_id not in seen_port_targets:
                    seen_port_targets.add(port_id)
                    active_ports.append(port_info)

        growth_history = []

        # 4. BFS growth loop
        while active_ports:
            if target_mass is not None and mass >= target_mass:
                break
            if target_dp is not None and dp >= target_dp:
                break

            port = active_ports.popleft()

            placement = self.kernel.choose_next_for_port(
                polymer, port, self.templates, rng
            )

            if placement is None:
                continue

            new_monomer = placement.template.instantiate()
            target_descriptor = placement.template.get_port_by_descriptor(
                placement.target_descriptor_id
            )

            if target_descriptor is None:
                continue

            target_port_name = target_descriptor.port_name
            new_monomer_ports = get_all_ports(new_monomer)

            if target_port_name not in new_monomer_ports:
                continue

            target_port = new_monomer_ports[target_port_name][0]

            # Collect port targets from new monomer BEFORE connection
            new_port_targets: set[int] = set()
            for pname, plist in new_monomer_ports.items():
                for patom in plist:
                    if patom is not target_port:
                        new_port_targets.add(id(patom))

            # Apply reaction via Connector
            from .core import AssemblyError

            try:
                result = self.connector.connect(
                    polymer,
                    new_monomer,
                    left_type=self.seed_template.label,
                    right_type=placement.template.label,
                    port_atom_L=port,
                    port_atom_R=target_port,
                )
            except AssemblyError as e:
                logger.warning(
                    "Connection failed for %s at port %s: %s",
                    placement.template.label,
                    port.get("port"),
                    e,
                )
                continue

            polymer = result.product
            mass += placement.template.mass
            dp += 1

            # Add only NEW ports from the merged product to the queue
            for pname, plist in get_all_ports(polymer).items():
                for patom in plist:
                    port_id = id(patom)
                    if port_id not in seen_port_targets:
                        seen_port_targets.add(port_id)
                        active_ports.append(patom)

            growth_history.append(
                {
                    "template": placement.template.label,
                    "port_used": port.get("port"),
                    "target_descriptor": placement.target_descriptor_id,
                    "target_port": target_port.get("port"),
                }
            )

        return StochasticChain(
            polymer=polymer,
            dp=dp,
            mass=mass,
            growth_history=growth_history,
        )

    def _choose_seed(self, rng: np.random.Generator) -> MonomerTemplate:
        """
        Choose seed monomer template.

        Args:
            rng: Random number generator

        Returns:
            MonomerTemplate for seed

        Example:
            >>> generator = StochasticChainGenerator(...)
            >>> seed = generator._choose_seed(np.random.default_rng(42))
            >>> print(f"Seed: {seed.label}")
        """
        if self.seed_template is not None:
            return self.seed_template

        # Default: choose first template
        return self.templates[0]
