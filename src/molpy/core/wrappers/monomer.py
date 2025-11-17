"""
Monomer wrapper for Struct objects.

Provides port definition management for monomer units.
"""

from typing import Any, Self, TypeVar

from ..entity import Entity, Struct
from .base import Wrapper

T = TypeVar("T", bound=Struct)


class Port(Entity):
    """
    Port represents a connection point on a monomer.

    A Port is an Entity that wraps a target entity (e.g., an atom)
    and provides a named interface for connecting monomers.

    **Minimal Usage (for manual Reacter):**
        Only `name` and `target` are required.

    **Extended Usage (for automatic builders):**
        Optional metadata (role, bond_kind, etc.) can be provided for
        automatic port selection in ``molpy.builder.polymer.linear()``.

    Attributes:
        name: Port identifier (e.g., 'in', 'out', 'port_1')
        target: The underlying entity this port points to
        role: (Optional) Port role for auto-selection ('left'/'right')
        bond_kind: (Optional) Bond type ('-', '=', '#', ':')
        compat: (Optional) Compatibility spec
        multiplicity: (Optional) Connection count limit
        priority: (Optional) Selection priority

    Example:
        >>> # Minimal usage for Reacter
        >>> port = Port('head', carbon_atom)
        >>>
        >>> # Extended usage for automatic builders
        >>> port = Port('head', carbon_atom, role='left', bond_kind='-')
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        *,
        role: str | None = None,
        bond_kind: str | None = None,
        compat: set[str] | str | None = None,
        multiplicity: int | None = None,
        priority: int | None = None,
    ):
        """
        Create a port.

        Args:
            name: Port identifier
            target: The entity this port points to
            role: (Optional) Port role for auto-selection
            bond_kind: (Optional) Bond type
            compat: (Optional) Compatibility specification
            multiplicity: (Optional) Connection count limit
            priority: (Optional) Selection priority
        """
        super().__init__()
        self.data["name"] = name
        self.data["target"] = target

        # Optional metadata for automatic builders
        if role is not None:
            self.data["role"] = role
        if bond_kind is not None:
            self.data["bond_kind"] = bond_kind
        if compat is not None:
            self.data["compat"] = compat
        if multiplicity is not None:
            self.data["multiplicity"] = multiplicity
        if priority is not None:
            self.data["priority"] = priority

    @property
    def name(self) -> str:
        """Port identifier."""
        return self.data["name"]

    @property
    def target(self) -> Entity:
        """The entity this port points to."""
        return self.data["target"]

    @property
    def role(self) -> str | None:
        """Port role (for automatic builders)."""
        return self.data.get("role")

    @property
    def bond_kind(self) -> str | None:
        """Bond type (for automatic builders)."""
        return self.data.get("bond_kind")

    @property
    def compat(self) -> set[str] | str | None:
        """Compatibility specification (for automatic builders)."""
        return self.data.get("compat")

    @property
    def multiplicity(self) -> int | None:
        """Connection count limit (for automatic builders)."""
        return self.data.get("multiplicity")

    @multiplicity.setter
    def multiplicity(self, value: int) -> None:
        """Update multiplicity."""
        self.data["multiplicity"] = value

    @property
    def priority(self) -> int | None:
        """Selection priority (for automatic builders)."""
        return self.data.get("priority")

    def __repr__(self) -> str:
        return f"<Port {self.name!r} -> {self.target}>"


class Monomer(Wrapper[T]):
    """Wrapper representing a monomer unit with port definitions.

    Monomer wraps a Struct and adds port definition management
    for tracking reactive sites and connection points.
    """

    def __init__(self, wrapped: T | Wrapper[T], **props: Any):
        """Initialize monomer wrapper.

        Args:
            wrapped: Struct instance or Wrapper to wrap
            **props: Additional properties
        """
        super().__init__(wrapped, **props)
        self._port_defs: dict[str, Port] = {}

    def define_port(
        self,
        name: str,
        target: Entity,
        *,
        role: str | None = None,
        bond_kind: str | None = None,
        compat: set[str] | str | None = None,
        multiplicity: int | None = None,
        priority: int | None = None,
    ) -> Self:
        """Define a connection port on this monomer.

        **Minimal usage (for manual Reacter):**
            monomer.define_port('head', carbon_atom)

        **Extended usage (for automatic builders):**
            monomer.define_port('head', carbon_atom, role='left', bond_kind='-')

        Args:
            name: Port identifier
            target: Entity at this port (typically an Atom)
            role: (Optional) Port role for auto-selection
            bond_kind: (Optional) Bond type
            compat: (Optional) Compatibility specification
            multiplicity: (Optional) Connection count limit
            priority: (Optional) Selection priority

        Returns:
            Self for method chaining
        """
        port = Port(
            name,
            target,
            role=role,
            bond_kind=bond_kind,
            compat=compat,
            multiplicity=multiplicity,
            priority=priority,
        )
        self._port_defs[name] = port
        return self

    def remove_port_def(self, name: str) -> Self:
        """Remove a port definition.

        Args:
            name: Port identifier to remove

        Returns:
            Self for method chaining
        """
        self._port_defs.pop(name, None)
        return self

    def get_port_def(self, name: str) -> Port | None:
        """Get port definition by name.

        Args:
            name: Port identifier

        Returns:
            Port object or None if not found
        """
        return self._port_defs.get(name)

    def get_port(self, name: str) -> Port | None:
        """Get port by name (alias for get_port_def for compatibility).

        Args:
            name: Port identifier

        Returns:
            Port object or None if not found
        """
        return self.get_port_def(name)

    def set_port(
        self,
        name: str,
        target: Entity,
        *,
        role: str | None = None,
        bond_kind: str | None = None,
        compat: set[str] | str | None = None,
        multiplicity: int | None = None,
        priority: int | None = None,
    ) -> Self:
        """Set a port (alias for define_port for API consistency with Polymer).

        Args:
            name: Port identifier
            target: Entity at this port
            role: (Optional) Port role
            bond_kind: (Optional) Bond type
            compat: (Optional) Compatibility specification
            multiplicity: (Optional) Connection count limit
            priority: (Optional) Selection priority

        Returns:
            Self for method chaining
        """
        return self.define_port(
            name,
            target,
            role=role,
            bond_kind=bond_kind,
            compat=compat,
            multiplicity=multiplicity,
            priority=priority,
        )

    @property
    def port_defs(self) -> dict[str, Port]:
        """Access port definitions dictionary (directly modifiable)."""
        return self._port_defs

    @property
    def ports(self) -> dict[str, Port]:
        """Access port definitions dictionary (alias for port_defs)."""
        return self._port_defs

    def copy(self) -> Self:
        """Create a deep copy with properly remapped port targets.

        Overrides Wrapper.copy() to ensure port targets point to
        copied atoms rather than original atoms.

        Returns:
            New Monomer with copied assembly and remapped ports
        """
        import copy as copy_module

        # Deep copy the inner struct
        old_inner = self.inner
        new_inner = copy_module.deepcopy(old_inner)

        # Build entity map: old entity -> new entity
        entity_map = {}

        # Map atoms
        old_atoms = list(old_inner.atoms)
        new_atoms = list(new_inner.atoms)
        for old_atom, new_atom in zip(old_atoms, new_atoms):
            entity_map[old_atom] = new_atom

        # Create new wrapper
        new_monomer = type(self)(new_inner)

        # Remap ports: create new Port objects with mapped targets
        for port_name, old_port in self._port_defs.items():
            old_target = old_port.target
            new_target = entity_map.get(old_target, old_target)

            new_monomer.define_port(
                port_name,
                new_target,
                role=old_port.role,
                bond_kind=old_port.bond_kind,
                compat=old_port.compat,
                multiplicity=old_port.multiplicity,
                priority=old_port.priority,
            )

        return new_monomer

    def __repr__(self) -> str:
        """Repr showing monomer port definitions."""
        port_names = list(self._port_defs.keys())
        return f"<Monomer port_defs={port_names} wrapping {self.inner!r}>"

    def port_names(self) -> list[str]:
        """Return defined port names."""
        return list(self._port_defs.keys())
