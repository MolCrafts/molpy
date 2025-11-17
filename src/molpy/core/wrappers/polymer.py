"""
Polymer wrapper for Struct objects.

Provides port management for polymer structures.
"""

from typing import Any, Self, TypeVar

from ..entity import Entity, Struct
from .base import Wrapper

T = TypeVar("T", bound=Struct)


class Polymer(Wrapper[T]):
    """Wrapper representing a polymer with named connection ports.

    Polymer wraps a Struct and adds port management functionality
    for tracking connection points (head, tail, reactive sites, etc.).
    """

    def __init__(self, wrapped: T | Wrapper[T], **props: Any):
        """Initialize polymer wrapper.

        Args:
            wrapped: Struct instance or Wrapper to wrap
            **props: Additional properties
        """
        super().__init__(wrapped, **props)
        self._ports: dict[str, Entity] = {}

    def set_port(
        self,
        name: str,
        target: Entity,
        *,
        role: str | None = None,
        bond_kind: str | None = None,
        compat: set | str | None = None,
        multiplicity: int | None = None,
        priority: int | None = None,
    ) -> Self:
        """Create or update a port pointing to an entity in the wrapped assembly.

        **Minimal usage:**
            polymer.set_port('head', atom)

        **Extended usage (for builders):**
            polymer.set_port('head', atom, role='left', bond_kind='-')

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
        from .monomer import Port

        port = Port(
            name,
            target,
            role=role,
            bond_kind=bond_kind,
            compat=compat,
            multiplicity=multiplicity,
            priority=priority,
        )
        self._ports[name] = port
        return self

    def add_port(self, name: str, target: Entity) -> Self:
        """Add a port (alias for set_port).

        Args:
            name: Port identifier
            target: Entity at this port

        Returns:
            Self for method chaining
        """
        return self.set_port(name, target)

    def remove_port(self, name: str) -> Self:
        """Remove a port.

        Args:
            name: Port identifier to remove

        Returns:
            Self for method chaining
        """
        self._ports.pop(name, None)
        return self

    def get_port(self, name: str):
        """Get port by name.

        Args:
            name: Port identifier

        Returns:
            Port object or None if not found
        """
        return self._ports.get(name)

    def get_port_def(self, name: str):
        """Get port definition by name (alias for get_port).

        Args:
            name: Port identifier

        Returns:
            Port object or None if not found
        """
        return self.get_port(name)

    @property
    def ports(self) -> dict[str, Entity]:
        """Access ports dictionary (directly modifiable)."""
        return self._ports

    def __repr__(self) -> str:
        """Repr showing polymer ports."""
        port_names = list(self._ports.keys())
        return f"<Polymer ports={port_names} wrapping {self.inner!r}>"
