"""
Connector abstraction for polymer assembly.

Connectors decide which ports to connect between adjacent monomers
and optionally execute chemical reactions during connection.
"""

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from molpy.core.atomistic import Atomistic
from molpy.core.entity import Entity
from molpy.reacter.base import Reacter
from molpy.typifier.atomistic import TypifierBase

from .errors import AmbiguousPortsError, MissingConnectorRule, NoCompatiblePortsError
from .port_utils import PortInfo, get_all_port_info
from .types import ConnectionMetadata, ConnectionResult

BondKind = Literal["-", "=", "#", ":"]


class ConnectorContext(dict[str, Any]):
    """
    Shared context passed to connectors during linear build.

    Contains information like:
    - step: int (current connection step index)
    - sequence: str (full sequence being built)
    - left_label: str (label of left monomer)
    - right_label: str (label of right monomer)
    - audit: list (accumulated connection records)
    """

    pass


class Connector:
    """
    Abstract base for port selection between two adjacent Atomistic structures.

    This is topology-only: connectors decide WHICH ports to connect,
    not HOW to position them geometrically.
    """

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],  # unconsumed ports
        right_ports: Mapping[str, list[PortInfo]],  # unconsumed ports
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """
        Select which ports to connect between left and right structures.

        Args:
            left: Left Atomistic structure in the sequence
            right: Right Atomistic structure in the sequence
            left_ports: Available (unconsumed) ports on left structure (port name -> list of PortInfo)
            right_ports: Available (unconsumed) ports on right structure (port name -> list of PortInfo)
            ctx: Shared context with step info, sequence, etc.

        Returns:
            Tuple of (left_port_name, left_port_index, right_port_name, right_port_index, optional_bond_kind_override)
            The indices specify which port to use when multiple ports have the same name.

        Raises:
            AmbiguousPortsError: Cannot uniquely determine ports
            NoCompatiblePortsError: No valid port pair found
            MissingConnectorRule: Required rule not found (TableConnector)
        """
        raise NotImplementedError("Subclasses must implement select_ports()")


class AutoConnector(Connector):
    """
    BigSMILES-guided automatic port selection.

    Strategy:
    1. If left has port with role='right' and right has role='left' -> use those
    2. Else if each side has exactly one unconsumed port -> use that pair
    3. Else raise AmbiguousPortsError

    This implements the common case where:
    - BigSMILES uses [<] for "left" role and [>] for "right" role
    - We connect left's "right" port to right's "left" port

    Ports are stored directly on atoms using the "port" or "ports" attribute.
    """

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],
        right_ports: Mapping[str, list[PortInfo]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """Select ports using BigSMILES role heuristics."""

        # Strategy 1: Try role-based selection (BigSMILES < and >)
        # Case 1a: left has role='right', right has role='left' (normal chain extension)
        # Flatten lists and find ports with matching roles
        left_right_role = [
            (name, idx)
            for name, port_list in left_ports.items()
            for idx, p in enumerate(port_list)
            if p.role == "right"
        ]
        right_left_role = [
            (name, idx)
            for name, port_list in right_ports.items()
            for idx, p in enumerate(port_list)
            if p.role == "left"
        ]

        if len(left_right_role) == 1 and len(right_left_role) == 1:
            return (
                left_right_role[0][0],
                left_right_role[0][1],
                right_left_role[0][0],
                right_left_role[0][1],
                None,
            )

        # Case 1b: left has role='left', right has role='right' (terminus connection)
        # E.g., HO[*:t] (left terminus) connects to CC[>] (right-directed monomer)
        left_left_role = [
            (name, idx)
            for name, port_list in left_ports.items()
            for idx, p in enumerate(port_list)
            if p.role == "left"
        ]
        right_right_role = [
            (name, idx)
            for name, port_list in right_ports.items()
            for idx, p in enumerate(port_list)
            if p.role == "right"
        ]

        if len(left_left_role) == 1 and len(right_right_role) == 1:
            return (
                left_left_role[0][0],
                left_left_role[0][1],
                right_right_role[0][0],
                right_right_role[0][1],
                None,
            )

        # Strategy 2: If both sides have the same port name(s) (e.g., all $), randomly select one
        # Since all ports are equivalent, just pick the first available one
        if len(left_ports) == 1 and len(right_ports) == 1:
            left_name = next(iter(left_ports.keys()))
            right_name = next(iter(right_ports.keys()))
            # If same port name or both are $, they're compatible - use first available
            if left_name == right_name or (left_name == "$" and right_name == "$"):
                # Use the first port in each list (all are equivalent)
                return (left_name, 0, right_name, 0, None)

        # Strategy 3: Try to match port names - if same name exists on both sides, use it
        common_port_names = set(left_ports.keys()) & set(right_ports.keys())
        if common_port_names:
            # Use first common port name
            port_name = next(iter(common_port_names))
            return (port_name, 0, port_name, 0, None)

        # Strategy 4: If both sides have $ ports, use them (all ports are equivalent)
        if "$" in left_ports and "$" in right_ports:
            return ("$", 0, "$", 0, None)

        # Strategy 5: Ambiguous - cannot decide
        # Count total ports (flattened)
        left_total = sum(len(port_list) for port_list in left_ports.values())
        right_total = sum(len(port_list) for port_list in right_ports.values())

        raise AmbiguousPortsError(
            f"Cannot auto-select ports between {ctx.get('left_label')} and {ctx.get('right_label')}: "
            f"left has {left_total} available ports across {len(left_ports)} port names {list(left_ports.keys())}, "
            f"right has {right_total} available ports across {len(right_ports)} port names {list(right_ports.keys())}. "
            "Use TableConnector or CallbackConnector to specify explicit rules."
        )


class TableConnector(Connector):
    """
    Rule-based port selection using a lookup table.

    Maps (left_label, right_label) -> (left_port, right_port [, bond_kind])

    Example:
        rules = {
            ("A", "B"): ("1", "2"),
            ("B", "A"): ("3", "1", "="),  # with bond kind override
            ("T", "A"): ("t", "1"),
        }
        connector = TableConnector(rules, fallback=AutoConnector())
    """

    def __init__(
        self,
        rules: Mapping[tuple[str, str], tuple[str, str] | tuple[str, str, BondKind]],
        fallback: Connector | None = None,
    ):
        """
        Initialize table connector.

        Args:
            rules: Mapping from (left_label, right_label) to port specifications
            fallback: Optional connector to try if pair not in rules
        """
        self.rules = dict(rules)  # Convert to dict for internal use
        self.fallback = fallback

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],
        right_ports: Mapping[str, list[PortInfo]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """Select ports using table lookup."""

        left_label = ctx.get("left_label", "")
        right_label = ctx.get("right_label", "")
        key = (left_label, right_label)

        if key in self.rules:
            rule = self.rules[key]
            # Use first port (index 0) when multiple ports have the same name
            if len(rule) == 2:
                return (rule[0], 0, rule[1], 0, None)
            else:
                return (rule[0], 0, rule[1], 0, rule[2])

        # Try fallback if available
        if self.fallback is not None:
            return self.fallback.select_ports(left, right, left_ports, right_ports, ctx)

        # No rule and no fallback
        raise MissingConnectorRule(
            f"No rule found for ({left_label}, {right_label}) and no fallback connector"
        )


class CallbackConnector(Connector):
    """
    User-defined callback for port selection.

    Example:
        def my_selector(left, right, left_ports, right_ports, ctx):
            # Custom logic here
            return ("port_out", "port_in", "-")

        connector = CallbackConnector(my_selector)
    """

    def __init__(
        self,
        fn: Callable[
            [
                Atomistic,
                Atomistic,
                Mapping[str, PortInfo],
                Mapping[str, PortInfo],
                ConnectorContext,
            ],
            tuple[str, str] | tuple[str, str, BondKind],
        ],
    ):
        """
        Initialize callback connector.

        Args:
            fn: Callable that takes (left, right, left_ports, right_ports, ctx)
                and returns (left_port, right_port [, bond_kind])
        """
        self.fn = fn

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],
        right_ports: Mapping[str, list[PortInfo]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """Select ports using user callback."""

        result = self.fn(left, right, left_ports, right_ports, ctx)

        # Callback can return either (name, idx, name, idx) or (name, idx, name, idx, bond_kind)
        if len(result) == 4:
            return (result[0], result[1], result[2], result[3], None)
        else:
            return (result[0], result[1], result[2], result[3], result[4])


class ChainConnector(Connector):
    """
    Try a list of connectors in order; first one that succeeds wins.

    Example:
        connector = ChainConnector([
            TableConnector(specific_rules),
            AutoConnector(),
        ])
    """

    def __init__(self, connectors: Iterable[Connector]):
        """
        Initialize chain connector.

        Args:
            connectors: List of connectors to try in order
        """
        self.connectors = list(connectors)

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],
        right_ports: Mapping[str, list[PortInfo]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """Try connectors in order until one succeeds."""

        errors = []
        for connector in self.connectors:
            try:
                return connector.select_ports(left, right, left_ports, right_ports, ctx)
            except (
                AmbiguousPortsError,
                MissingConnectorRule,
                NoCompatiblePortsError,
            ) as e:
                errors.append(f"{type(connector).__name__}: {e}")
                continue

        # All connectors failed
        raise AmbiguousPortsError(
            f"All connectors failed for ({ctx.get('left_label')}, {ctx.get('right_label')}): "
            f"{'; '.join(errors)}"
        )


class ReacterConnector(Connector):
    """
    Connector that uses chemical reactions (Reacter) for polymer assembly.

    This connector integrates port selection and chemical reaction execution.
    It manages multiple Reacter instances and port mapping strategies for
    different structure pairs.

    **Port Selection Strategy:**
    Port selection is handled via a `port_map` which must be a dict:
    - **Dict**: Explicit mapping {('A','B'): ('1','2'), ...}

    There is NO 'auto' mode - port selection must be explicit via port_map.
    Ports are stored directly on atoms using the "port" or "ports" attribute.

    Attributes:
        default: Default Reacter for most connections
        overrides: Dict mapping (left_type, right_type) -> specialized Reacter
        port_map: Dict mapping (left_type, right_type) -> (port_L, port_R)

    Example:
        >>> from molpy.reacter import Reacter
        >>> from molpy.reacter.selectors import select_port_atom, select_one_hydrogen
        >>> from molpy.reacter.transformers import form_single_bond
        >>>
        >>> default_reacter = Reacter(
        ...     name="C-C_coupling",
        ...     port_selector_left=select_port_atom,
        ...     port_selector_right=select_port_atom,
        ...     leaving_selector_left=select_one_hydrogen,
        ...     leaving_selector_right=select_one_hydrogen,
        ...     bond_former=form_single_bond,
        ... )
        >>>
        >>> # Explicit port mapping for all structure pairs
        >>> connector = ReacterConnector(
        ...     default=default_reacter,
        ...     port_map={
        ...         ('A', 'B'): ('port_1', 'port_2'),
        ...         ('B', 'C'): ('port_3', 'port_4'),
        ...     },
        ...     overrides={('B', 'C'): special_reacter},
        ... )
    """

    def __init__(
        self,
        default: "Reacter",
        port_map: dict[tuple[str, str], tuple[str, str]],
        overrides: dict[tuple[str, str], "Reacter"] | None = None,
    ):
        """
        Initialize ReacterConnector.

        Args:
            default: Default Reacter for most connections
            port_map: Mapping from (left_type, right_type) to (port_L, port_R)
            overrides: Optional mapping from (left_type, right_type) to
                specialized Reacter instances

        Raises:
            TypeError: If port_map is not a dict
        """
        if not isinstance(port_map, dict):
            raise TypeError(f"port_map must be dict, got {type(port_map).__name__}")

        self.default = default
        self.overrides = overrides or {}
        self.port_map = port_map
        self._history: list = []  # List of ReactionResult objects

    def get_reacter(self, left_type: str, right_type: str) -> "Reacter":
        """
        Get appropriate reacter for a structure pair.

        Args:
            left_type: Type label of left structure (e.g., 'A', 'B')
            right_type: Type label of right structure

        Returns:
            The appropriate Reacter (override if exists, else default)
        """
        key = (left_type, right_type)
        return self.overrides.get(key, self.default)

    def select_ports(
        self,
        left: Atomistic,
        right: Atomistic,
        left_ports: Mapping[str, list[PortInfo]],
        right_ports: Mapping[str, list[PortInfo]],
        ctx: ConnectorContext,
    ) -> tuple[str, int, str, int, BondKind | None]:
        """
        Select ports using the configured port_map.

        Args:
            left: Left Atomistic structure
            right: Right Atomistic structure
            left_ports: Available ports on left (port name -> list of PortInfo)
            right_ports: Available ports on right (port name -> list of PortInfo)
            ctx: Connector context with structure type information

        Returns:
            Tuple of (port_L, port_L_idx, port_R, port_R_idx, None)
            Uses index 0 when multiple ports share the same name.

        Raises:
            ValueError: If port mapping not found or ports invalid
        """
        # Get monomer types from context
        left_type = ctx.get("left_label", "")
        right_type = ctx.get("right_label", "")

        # Look up explicit mapping
        key = (left_type, right_type)
        if key not in self.port_map:
            raise ValueError(
                f"No port mapping defined for ({left_type}, {right_type}). "
                f"Available mappings: {list(self.port_map.keys())}"
            )
        port_L, port_R = self.port_map[key]

        # Validate ports exist
        if port_L not in left_ports:
            raise ValueError(
                f"Selected port '{port_L}' not found in left structure ({left_type})"
            )
        if port_R not in right_ports:
            raise ValueError(
                f"Selected port '{port_R}' not found in right structure ({right_type})"
            )

        # Use first port (index 0) when multiple ports share the same name
        return port_L, 0, port_R, 0, None  # bond_kind determined by reacter

    def connect(
        self,
        left: Atomistic,
        right: Atomistic,
        left_type: str,
        right_type: str,
        port_atom_L: Entity,
        port_atom_R: Entity,
        typifier: TypifierBase | None = None,
    ) -> ConnectionResult:
        """
        Execute chemical reaction between two Atomistic structures.

        This method performs the full chemical reaction including:
        1. Selecting appropriate reacter based on structure types
        2. Executing reaction (merging, bond making, removing leaving groups)
        3. Computing new topology (angles, dihedrals)
        4. Collecting metadata for retypification

        Args:
            left: Left Atomistic structure
            right: Right Atomistic structure
            left_type: Type label of left structure
            right_type: Type label of right structure
            port_atom_L: Port atom from left structure
            port_atom_R: Port atom from right structure
            typifier: Optional typifier

        Returns:
            ConnectionResult containing product and metadata
        """
        from molpy.reacter.base import ReactionResult

        # Select reacter
        reacter = self.get_reacter(left_type, right_type)

        # Execute reaction
        product_set: ReactionResult = reacter.run(
            left,
            right,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            compute_topology=True,
            typifier=typifier,
        )

        # Store in history
        self._history.append(product_set)

        # Create metadata dataclass
        port_name_L = port_atom_L.get("port", "unknown")
        port_name_R = port_atom_R.get("port", "unknown")
        metadata = ConnectionMetadata(
            port_L=port_name_L,
            port_R=port_name_R,
            reaction_name=reacter.name,
            formed_bonds=product_set.topology_changes.new_bonds,
            new_angles=product_set.topology_changes.new_angles,
            new_dihedrals=product_set.topology_changes.new_dihedrals,
            modified_atoms=product_set.topology_changes.modified_atoms,
            requires_retype=product_set.metadata.requires_retype,
            entity_maps=product_set.metadata.entity_maps,
        )

        return ConnectionResult(
            product=product_set.product_info.product, metadata=metadata
        )

    def get_history(self) -> list:
        """Get all reaction history (list of ReactionResult)."""
        return self._history

    def get_all_modified_atoms(self) -> set[Entity]:
        """Get all atoms modified across all reactions.

        Returns:
            Set of all atoms that were modified during reactions
        """
        all_atoms: set[Entity] = set()
        for result in self._history:
            all_atoms.update(result.topology_changes.modified_atoms)
        return all_atoms

    def needs_retypification(self) -> bool:
        """Check if any reactions require retypification.

        Returns:
            True if any reaction requires retypification
        """
        return any(r.metadata.requires_retype for r in self._history)


# Alias for backward compatibility
TopologyConnector = AutoConnector
