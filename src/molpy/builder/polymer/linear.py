"""
Simplified linear polymer builder.

Key: Retypify ONCE after all connections (not per step).
"""

import warnings
from collections.abc import Mapping

from molpy.core.atomistic import Atom, Atomistic
from molpy.typifier.atomistic import TypifierBase

from .connectors import ConnectorContext, ReacterConnector
from .errors import NoCompatiblePortsError, SequenceError
from .placer import Placer
from .port_utils import get_all_port_info, set_port_metadata
from .types import ConnectionMetadata, PolymerBuildResult


def linear(
    *,
    sequence: str,
    library: Mapping[str, Atomistic],
    connector: ReacterConnector,
    typifier: TypifierBase | None = None,
    placer: Placer | None = None,
) -> PolymerBuildResult:
    """
    Assemble linear polymer using chemical reactions.

    Retypify happens ONCE after all connections complete.

    Args:
        sequence: String of structure labels (e.g., "ABCBD")
        library: Mapping from labels to Atomistic structures
        connector: ReacterConnector for port selection and chemical reactions
        typifier: Optional typifier for automatic retypification
        placer: Optional Placer for positioning structures before connection

    Returns:
        Assembled Atomistic structure with connection history
    """
    # Validate sequence
    if len(sequence) < 2:
        raise SequenceError(f"Sequence must have at least 2 labels, got: {sequence!r}")

    for label in sequence:
        if label not in library:
            raise SequenceError(f"Label {label!r} not found in library")

    # Step 1: Copy first structure
    current = library[sequence[0]].copy()

    # Track connection history
    connection_history: list[ConnectionMetadata] = []

    # Step 2: Iteratively add monomers
    for step in range(len(sequence) - 1):
        left_label = sequence[step]
        right_label = sequence[step + 1]

        # Copy next structure
        right = library[right_label].copy()

        # Get unconsumed ports
        left_ports = get_all_port_info(current)
        right_ports = get_all_port_info(right)

        if not left_ports:
            raise NoCompatiblePortsError(
                f"Step {step}: Left ({left_label}) has no available ports"
            )
        if not right_ports:
            raise NoCompatiblePortsError(
                f"Step {step}: Right ({right_label}) has no available ports"
            )

        # Build context
        ctx = ConnectorContext(
            step=step,
            sequence=sequence,
            left_label=left_label,
            right_label=right_label,
            audit=[],
        )

        # Select ports
        left_port_name, right_port_name, _ = connector.select_ports(
            current,
            right,
            left_ports,
            right_ports,
            ctx,
        )

        # Get the actual PortInfo objects for placer
        left_port = left_ports[left_port_name]
        right_port = right_ports[right_port_name]

        # Position right structure (if placer provided)
        if placer is not None:
            placer.place_monomer(
                current,
                right,
                left_port,
                right_port,
            )

        # CRITICAL: Save port targets BEFORE connection
        # Since reacter.run() uses copy(), we need to track which atoms in the product
        # correspond to the original port targets

        # Save left port targets (atoms) before connection
        left_port_targets: dict[str, Atom] = {
            name: port.target for name, port in left_ports.items()
        }

        # Save right port targets before connection
        right_port_targets: dict[str, Atom] = {
            name: port.target for name, port in right_ports.items()
        }

        # Execute reaction
        connection_result = connector.connect(
            current,
            right,
            left_label,
            right_label,
            left_port_name,
            right_port_name,
            typifier=typifier,
        )

        current = connection_result.product
        metadata = connection_result.metadata

        # Store connection metadata
        connection_history.append(metadata)

        # Transfer unused ports using entity map from reaction
        # The entity_maps in metadata contain the mapping from original atoms to product atoms
        atoms_in_product = set(current.atoms)

        # Build combined entity map from metadata
        entity_map: dict[Atom, Atom] = {}
        if metadata.entity_maps:
            for emap in metadata.entity_maps:
                entity_map.update(emap)

        # Transfer unused left ports
        for port_name, original_target in left_port_targets.items():
            if port_name == left_port_name:
                continue  # Skip the port that was used

            # Find the corresponding atom in product using entity map
            new_target = entity_map.get(original_target)

            # Only add port if we found a valid target in product
            if new_target is not None and new_target in atoms_in_product:
                # Get original port metadata
                original_port = left_ports[port_name]
                # Mark port on atom
                new_target["port"] = port_name
                # Set port metadata
                set_port_metadata(
                    new_target,
                    port_name,
                    role=original_port.role,
                    bond_kind=original_port.bond_kind,
                    compat=original_port.compat,
                    priority=original_port.priority,
                )

        # Transfer unused right ports
        for port_name, original_target in right_port_targets.items():
            if port_name == right_port_name:
                continue  # Skip the port that was used

            # Find the corresponding atom in product using entity map
            new_target = entity_map.get(original_target)

            # Only add port if we found a valid target in product
            if new_target is not None and new_target in atoms_in_product:
                # Get original port metadata
                original_port = right_ports[port_name]
                # Mark port on atom
                new_target["port"] = port_name
                # Set port metadata
                set_port_metadata(
                    new_target,
                    port_name,
                    role=original_port.role,
                    bond_kind=original_port.bond_kind,
                    compat=original_port.compat,
                    priority=original_port.priority,
                )

    return PolymerBuildResult(
        polymer=current,
        connection_history=connection_history,
        total_steps=len(sequence) - 1,
    )
