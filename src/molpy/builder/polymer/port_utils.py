"""
Utility functions for working with ports on Atomistic structures.

Ports are now stored directly on atoms using the "port" or "ports" attribute,
replacing the previous Monomer.ports dictionary approach.
"""

from typing import Any

from molpy.core.atomistic import Atom, Atomistic


def get_ports(struct: Atomistic) -> dict[str, list[Atom]]:
    """
    Get all ports from an Atomistic structure.

    Returns a dictionary mapping port names to lists of atoms that have those ports.
    Multiple atoms can have the same port name (e.g., BigSMILES [$]CC[$]CC[$]).

    Args:
        struct: Atomistic structure to extract ports from

    Returns:
        Dictionary mapping port name -> list of atom entities

    Example:
        >>> # Mark ports on atoms
        >>> atom1["port"] = "head"
        >>> atom2["port"] = "tail"
        >>> atom3["port"] = "head"  # Same name as atom1
        >>> ports = get_ports(struct)
        >>> # ports = {"head": [atom1, atom3], "tail": [atom2]}
    """
    ports: dict[str, list[Atom]] = {}
    for atom in struct.atoms:
        # Check single port marker
        port_name = atom.get("port")
        if port_name is not None:
            if port_name not in ports:
                ports[port_name] = []
            ports[port_name].append(atom)
    return ports


def get_port_atom(struct: Atomistic, port_name: str) -> Atom | None:
    """
    Get the first atom entity for a specific port name.

    Note: If multiple atoms have the same port name, this returns only the first one.
    Use get_ports() to get all atoms with a given port name.

    Args:
        struct: Atomistic structure to search
        port_name: Name of the port to find

    Returns:
        First atom entity with the port, or None if not found
    """
    for atom in struct.atoms:
        if atom.get("port") == port_name:
            return atom
    return None


def get_port_metadata(atom: Atom, port_name: str) -> dict[str, Any]:
    """
    Get metadata for a port from an atom.

    Port metadata can be stored on the atom using keys like:
    - port_role_{port_name}
    - port_bond_kind_{port_name}
    - port_compat_{port_name}
    - port_multiplicity_{port_name}
    - port_priority_{port_name}

    Args:
        atom: Atom entity that has the port
        port_name: Name of the port

    Returns:
        Dictionary with port metadata (role, bond_kind, compat, etc.)
    """
    metadata: dict[str, Any] = {}

    # Check for metadata keys
    role_key = f"port_role_{port_name}"
    if role_key in atom:
        metadata["role"] = atom[role_key]

    bond_kind_key = f"port_bond_kind_{port_name}"
    if bond_kind_key in atom:
        metadata["bond_kind"] = atom[bond_kind_key]

    compat_key = f"port_compat_{port_name}"
    if compat_key in atom:
        metadata["compat"] = atom[compat_key]

    multiplicity_key = f"port_multiplicity_{port_name}"
    if multiplicity_key in atom:
        metadata["multiplicity"] = atom[multiplicity_key]

    priority_key = f"port_priority_{port_name}"
    if priority_key in atom:
        metadata["priority"] = atom[priority_key]

    return metadata


def set_port_metadata(
    atom: Atom,
    port_name: str,
    role: str | None = None,
    bond_kind: str | None = None,
    compat: set[str] | str | None = None,
    multiplicity: int | None = None,
    priority: int | None = None,
) -> None:
    """
    Set metadata for a port on an atom.

    Args:
        atom: Atom entity to set metadata on
        port_name: Name of the port
        role: Port role (e.g., 'left', 'right')
        bond_kind: Bond type (e.g., '-', '=', '#')
        compat: Compatibility specification
        multiplicity: Connection count limit
        priority: Selection priority
    """
    if role is not None:
        atom[f"port_role_{port_name}"] = role
    if bond_kind is not None:
        atom[f"port_bond_kind_{port_name}"] = bond_kind
    if compat is not None:
        atom[f"port_compat_{port_name}"] = compat
    if multiplicity is not None:
        atom[f"port_multiplicity_{port_name}"] = multiplicity
    if priority is not None:
        atom[f"port_priority_{port_name}"] = priority


class PortInfo:
    """
    Simple container for port information, similar to the old Port class.

    This is used for compatibility with code that expects Port-like objects.
    """

    def __init__(self, name: str, target: Atom, **metadata: Any):
        """
        Create a PortInfo object.

        Args:
            name: Port name
            target: Target atom entity
            **metadata: Optional metadata (role, bond_kind, compat, etc.)
        """
        self.name = name
        self.target = target
        self.data = metadata

    @property
    def role(self) -> str | None:
        """Port role."""
        return self.data.get("role")

    @property
    def bond_kind(self) -> str | None:
        """Bond type."""
        return self.data.get("bond_kind")

    @property
    def compat(self) -> set[str] | str | None:
        """Compatibility specification."""
        return self.data.get("compat")

    @property
    def multiplicity(self) -> int | None:
        """Connection count limit."""
        return self.data.get("multiplicity")

    @property
    def priority(self) -> int | None:
        """Selection priority."""
        return self.data.get("priority")

    def __repr__(self) -> str:
        return f"<PortInfo {self.name!r} -> {self.target}>"


def get_port_info(struct: Atomistic, port_name: str) -> PortInfo | None:
    """
    Get the first PortInfo object for a port name.

    Note: If multiple atoms have the same port name, this returns only the first one.
    Use get_all_port_info() to get all ports with a given name.

    Args:
        struct: Atomistic structure
        port_name: Port name

    Returns:
        First PortInfo object or None if port not found
    """
    atom = get_port_atom(struct, port_name)
    if atom is None:
        return None

    metadata = get_port_metadata(atom, port_name)
    return PortInfo(port_name, atom, **metadata)


def get_all_port_info(struct: Atomistic) -> dict[str, list[PortInfo]]:
    """
    Get all PortInfo objects from an Atomistic structure.

    Returns a dictionary mapping port names to lists of PortInfo objects.
    Multiple atoms can have the same port name (e.g., BigSMILES [$]CC[$]CC[$]).

    Args:
        struct: Atomistic structure

    Returns:
        Dictionary mapping port name -> list of PortInfo objects
    """
    ports: dict[str, list[PortInfo]] = {}
    for atom in struct.atoms:
        # Check single port marker
        port_name = atom.get("port")
        if port_name is not None:
            metadata = get_port_metadata(atom, port_name)
            if port_name not in ports:
                ports[port_name] = []
            ports[port_name].append(PortInfo(port_name, atom, **metadata))
    return ports

