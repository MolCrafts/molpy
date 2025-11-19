"""
Simplified linear polymer builder.

Key: Retypify ONCE after all connections (not per step).
"""

import warnings
from collections.abc import Mapping
from typing import cast

from molpy.core.wrappers.base import Wrapper
from molpy.core.wrappers.monomer import Monomer
from molpy.core.wrappers.polymer import Polymer
from molpy.typifier.atomistic import OplsAtomisticTypifier

from .connectors import ConnectorContext, ReacterConnector
from .errors import NoCompatiblePortsError, SequenceError
from .placer import Placer


def linear(
    *,
    sequence: str,
    library: Mapping[str, Monomer],
    connector: ReacterConnector,
    typifier: "OplsAtomisticTypifier | None" = None,
    auto_retypify: bool = True,
    placer: "Placer | None" = None,
) -> Polymer:
    """
    Assemble linear polymer using chemical reactions.

    Retypify happens ONCE after all connections complete.

    Args:
        sequence: String of monomer labels (e.g., "ABCBD")
        library: Mapping from labels to Monomer objects
        connector: ReacterConnector for port selection and chemical reactions
        typifier: Optional OPLS typifier for automatic retypification
        auto_retypify: Whether to automatically retypify after assembly
        placer: Optional Placer for positioning monomers before connection

    Returns:
        Assembled Polymer object
    """
    # Validate sequence
    if len(sequence) < 2:
        raise SequenceError(f"Sequence must have at least 2 labels, got: {sequence!r}")

    for label in sequence:
        if label not in library:
            raise SequenceError(f"Label {label!r} not found in library")

    # Step 1: Copy first monomer
    first_copy = library[sequence[0]].copy()

    # Get the Monomer (unwrap if it's a Wrapper like RDKitWrapper)
    if isinstance(first_copy, Wrapper) and not isinstance(first_copy, Monomer):
        first_monomer = first_copy.inner
        if not isinstance(first_monomer, Monomer):
            raise TypeError(
                f"Expected Monomer or Wrapper[Monomer], got {type(first_copy)}"
            )
    else:
        first_monomer = first_copy

    current_polymer = Polymer.from_atomistic(first_copy)

    # Transfer ports from the Monomer
    for pname, p in first_monomer.ports.items():
        current_polymer.set_port(
            pname,
            p.target,
            role=p.role,
            bond_kind=p.bond_kind,
            compat=p.compat,
            priority=p.priority,
        )

    # Track if retypification needed
    needs_retypification = False

    # Step 2: Iteratively add monomers
    for step in range(len(sequence) - 1):
        left_label = sequence[step]
        right_label = sequence[step + 1]

        # Copy next monomer
        right_copy = library[right_label].copy()

        # Get the Monomer (unwrap if it's a Wrapper like RDKitWrapper)
        if isinstance(right_copy, Wrapper) and not isinstance(right_copy, Monomer):
            right_monomer = right_copy.inner
            if not isinstance(right_monomer, Monomer):
                raise TypeError(
                    f"Expected Monomer or Wrapper[Monomer], got {type(right_copy)}"
                )
        else:
            right_monomer = right_copy

        # Get unconsumed ports
        left_ports = current_polymer.ports
        right_ports = right_monomer.ports

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
            cast(Monomer, current_polymer),
            right_monomer,
            left_ports,
            right_ports,
            ctx,
        )

        # Get the actual Port objects for placer
        left_port = left_ports[left_port_name]
        right_port = right_ports[right_port_name]

        # Position right monomer (if placer provided)
        if placer is not None:
            placer.place_monomer(
                cast(Monomer, current_polymer),
                right_monomer,
                left_port,
                right_port,
            )

        # Execute reaction
        product_assembly, metadata = connector.connect(
            cast(Monomer, current_polymer),
            right_monomer,
            left_label,
            right_label,
            left_port_name,
            right_port_name,
        )

        # Check if retypification needed
        if metadata.get("needs_retypification"):
            needs_retypification = True

        # Wrap product in new Polymer
        current_polymer = Polymer.from_atomistic(product_assembly)

        # Transfer unused ports
        atoms_in_product = set(product_assembly.atoms)

        # Transfer unused left ports
        for port_name, port in left_ports.items():
            if port_name == left_port_name:
                continue
            if port.target in atoms_in_product:
                current_polymer.set_port(
                    port_name,
                    port.target,
                    role=port.role,
                    bond_kind=port.bond_kind,
                    compat=port.compat,
                    priority=port.priority,
                )

        # Transfer unused right ports
        for port_name, port in right_monomer.ports.items():
            if port_name == right_port_name:
                continue
            if port.target in atoms_in_product:
                current_polymer.set_port(
                    port_name,
                    port.target,
                    role=port.role,
                    bond_kind=port.bond_kind,
                    compat=port.compat,
                    priority=port.priority,
                )

    # Step 3: Retypify ONCE after all connections
    if auto_retypify and typifier and needs_retypification:
        final_product = current_polymer

        # Clear existing angles/dihedrals to avoid duplication

        for angle in list(final_product.angles):
            final_product.links.remove(angle)
        for dihedral in list(final_product.dihedrals):
            final_product.links.remove(dihedral)

        # Generate angles and dihedrals
        final_product.get_topo(gen_angle=True, gen_dihe=True)

        # Ensure atom types are up to date before running higher-order typers
        atom_typifier = getattr(typifier, "atom_typifier", None)
        if atom_typifier is not None:
            atom_typifier.typify(final_product)

        # Typify everything (best-effort)
        try:
            typifier.typify(final_product)
        except Exception as exc:  # pragma: no cover - warn but continue
            warnings.warn(
                f"linear(): typifier {typifier.__class__.__name__} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return current_polymer
