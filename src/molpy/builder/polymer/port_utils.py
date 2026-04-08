"""
Utility functions for working with ports on Atomistic structures.

A port is simply ``atom["port"] = "<"`` — a string stored on the atom.
Role (left/right/terminal) is derived from the port name per BigSMILES convention.
No additional metadata keys are needed.
"""

from molpy.core.atomistic import Atom, Atomistic


def port_role(name: str) -> str:
    """Derive port role from its name (BigSMILES convention).

    ``<`` (and ``<N``) → ``"left"`` (head),
    ``>`` (and ``>N``) → ``"right"`` (tail),
    everything else → ``"terminal"`` (symmetric).
    """
    if name.startswith("<"):
        return "left"
    if name.startswith(">"):
        return "right"
    return "terminal"


def ports_compatible(a: str, b: str) -> bool:
    """Check whether two port names are compatible for connection.

    Rules:
    - ``>`` connects to ``<`` (directional pair)
    - Identical names connect to each other (e.g. ``$``, ``$1``)
    - ``>`` does NOT connect to ``>``; ``<`` does NOT connect to ``<``
    """
    if {a, b} == {">", "<"}:
        return True
    if a == b and a not in {">", "<"}:
        return True
    return False


def get_ports(struct: Atomistic) -> dict[str, list[Atom]]:
    """Get all ports from an Atomistic structure.

    Returns a dictionary mapping port names to lists of atoms with those ports.
    """
    ports: dict[str, list[Atom]] = {}
    for atom in struct.atoms:
        port_name = atom.get("port")
        if port_name is not None:
            if port_name not in ports:
                ports[port_name] = []
            ports[port_name].append(atom)
    return ports


# Alias used by stochastic modules
get_all_ports = get_ports


def get_port_atom(struct: Atomistic, port_name: str) -> Atom | None:
    """Get the first atom with a specific port name."""
    for atom in struct.atoms:
        if atom.get("port") == port_name:
            return atom
    return None


def get_ports_on_node(struct: Atomistic, node_id: int) -> dict[str, list[Atom]]:
    """Get all ports on atoms belonging to a specific monomer_node_id."""
    ports: dict[str, list[Atom]] = {}
    for atom in struct.atoms:
        if atom.get("monomer_node_id") != node_id:
            continue
        port_name = atom.get("port")
        if port_name is not None:
            if port_name not in ports:
                ports[port_name] = []
            ports[port_name].append(atom)
    return ports


def cleanup_build_markers(struct: Atomistic) -> None:
    """Remove build-time markers from all atoms.

    Called after polymer building is complete to keep Atomistic clean.
    Removes: ``port``, ``monomer_node_id``, ``port_descriptor_id``.
    """
    build_keys = ("port", "monomer_node_id", "port_descriptor_id")
    for atom in struct.atoms:
        for key in build_keys:
            if key in atom:
                del atom[key]
