"""
Selector functions for identifying reaction sites and leaving groups.

Selectors are composable functions that identify specific atoms
in a structure for use in reactions.

**Selector Types:**

1. **Port Selectors** (port_selector_left, port_selector_right):
   - Signature: `(struct: Atomistic, port_atom: Entity) -> Entity`
   - Transform a port atom to its actual reaction site
   - Example: O port → C neighbor for dehydration

2. **Leaving Selectors** (leaving_selector_left, leaving_selector_right):
   - Signature: `(struct: Atomistic, reaction_site: Entity) -> list[Entity]`
   - Identify atoms to remove from the reaction site
   - Example: C → [O, H] for hydroxyl removal

**Naming:**
- `struct`: The Atomistic structure containing atoms
- `port_atom`: The specific atom with port marker
- `reaction_site`: The atom where the bond will form
"""

from collections.abc import Callable

from molpy.core.atomistic import Atom, Atomistic

from molpy.reacter.utils import find_neighbors, get_bond_between

# Generic selector type alias used by presets and other modules.
Selector = Callable[..., Atom | list[Atom]]

# =============================================================================
# Port Selectors - Transform port atom to reaction site
# =============================================================================


def select_port(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Port selector - returns the port atom as the reaction site.

    Use this when the port atom itself is the reaction site.

    Args:
        struct: Atomistic structure containing the atoms
        port_atom: The atom with port marker

    Returns:
        The same port_atom
    """
    return port_atom


def select_c_neighbor(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Select C neighbor of the port atom as reaction site.

    Useful when port is on O but reaction site is adjacent C.

    Args:
        struct: Atomistic structure
        port_atom: Port atom (typically O)

    Returns:
        First C neighbor of port_atom
    """
    c_neighbors = [
        a for a in find_neighbors(struct, port_atom, element="C") if isinstance(a, Atom)
    ]
    return c_neighbors[0]


def select_o_neighbor(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Select O neighbor of the port atom as reaction site.

    Useful when port is on C but reaction site is adjacent O.

    Args:
        struct: Atomistic structure
        port_atom: Port atom (typically C)

    Returns:
        First O neighbor of port_atom
    """
    o_neighbors = [
        a for a in find_neighbors(struct, port_atom, element="O") if isinstance(a, Atom)
    ]
    return o_neighbors[0]


def select_dehydration_left(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Left-side selector for dehydration reactions.

    Returns C as reaction site, handling ports on either O or C:
    - If port on O: returns C neighbor
    - If port on C: returns C itself

    Args:
        struct: Atomistic structure
        port_atom: Port atom (O or C)

    Returns:
        C atom as reaction site
    """
    elem = port_atom.get("element") or port_atom.get("symbol")
    if elem == "O":
        c_neighbors = [
            a
            for a in find_neighbors(struct, port_atom, element="C")
            if isinstance(a, Atom)
        ]
        return c_neighbors[0]
    else:  # C
        return port_atom


def select_dehydration_right(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Right-side selector for dehydration reactions.

    Returns O as reaction site, handling ports on either O or C:
    - If port on O: returns O itself
    - If port on C: returns O neighbor

    Args:
        struct: Atomistic structure
        port_atom: Port atom (O or C)

    Returns:
        O atom as reaction site
    """
    if (port_atom.get("element") or port_atom.get("symbol")) == "O":
        return port_atom
    else:  # C
        o_neighbors = [
            a
            for a in find_neighbors(struct, port_atom, element="O")
            if isinstance(a, Atom)
        ]
        return o_neighbors[0]


# =============================================================================
# Leaving Group Selectors - Identify atoms to remove
# =============================================================================


def select_one_hydrogen(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select one hydrogen atom bonded to the reaction site.

    Useful for condensation reactions where H is eliminated.

    Args:
        struct: Atomistic structure
        reaction_site: Atom to find H neighbor of

    Returns:
        List with one H atom, or empty list if none
    """
    h_neighbors = [
        a
        for a in find_neighbors(struct, reaction_site, element="H")
        if isinstance(a, Atom)
    ]
    if h_neighbors:
        return [h_neighbors[0]]
    return []


def select_all_hydrogens(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select all hydrogen atoms bonded to the reaction site.

    Args:
        struct: Atomistic structure
        reaction_site: Atom to find H neighbors of

    Returns:
        List of all H atoms bonded to reaction site
    """
    return [
        a
        for a in find_neighbors(struct, reaction_site, element="H")
        if isinstance(a, Atom)
    ]


def select_dummy_atoms(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select dummy atoms (*) bonded to the reaction site.

    Useful for BigSMILES-style reactions where * marks connection points.

    Args:
        struct: Atomistic structure
        reaction_site: Atom to find dummy neighbors of

    Returns:
        List of dummy atoms (symbol='*')
    """
    neighbors = [
        a for a in find_neighbors(struct, reaction_site) if isinstance(a, Atom)
    ]
    return [n for n in neighbors if (n.get("element") or n.get("symbol")) == "*"]


def select_hydroxyl_group(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select hydroxyl group (-OH) bonded to the reaction site.

    Finds single-bonded O neighbor, then finds H bonded to that O.
    Used for esterification and dehydration reactions.

    Args:
        struct: Atomistic structure
        reaction_site: Atom (typically C) with -OH attached

    Returns:
        [O, H] atoms from hydroxyl group
    """

    o_neighbors = [
        a
        for a in find_neighbors(struct, reaction_site, element="O")
        if isinstance(a, Atom)
    ]

    if not o_neighbors:
        raise ValueError("No oxygen neighbors found for reaction site")

    # Find single-bonded oxygen (hydroxyl, not carbonyl)
    hydroxyl_o = None
    for o in o_neighbors:
        bond = get_bond_between(struct, reaction_site, o)
        if bond and bond.get("order") == 1:
            hydroxyl_o = o
            break

    if hydroxyl_o is None:
        raise ValueError("No single-bonded oxygen (hydroxyl) found for reaction site")

    h_neighbors = [
        a
        for a in find_neighbors(struct, hydroxyl_o, element="H")
        if isinstance(a, Atom)
    ]

    if not h_neighbors:
        raise ValueError("No hydrogen found bonded to hydroxyl oxygen")

    return [hydroxyl_o, h_neighbors[0]]


def select_hydroxyl_h_only(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select only the H from hydroxyl group bonded to reaction site.

    The O remains (becomes part of the new bond).

    Args:
        struct: Atomistic structure
        reaction_site: Atom with -OH attached (typically O itself)

    Returns:
        [H] if found, otherwise []
    """
    # If reaction_site is O, look for H directly bonded
    if reaction_site.get("element") == "O":
        h_neighbors = [
            a
            for a in find_neighbors(struct, reaction_site, element="H")
            if isinstance(a, Atom)
        ]
        if h_neighbors:
            return [h_neighbors[0]]
        return []

    # If reaction_site is C, look for O neighbor then H
    from .utils import get_bond_between

    o_neighbors = [
        a
        for a in find_neighbors(struct, reaction_site, element="O")
        if isinstance(a, Atom)
    ]
    for o in o_neighbors:
        bond = get_bond_between(struct, reaction_site, o)
        if not bond or bond.get("order") != 1:
            continue
        h_neighbors = [
            a for a in find_neighbors(struct, o, element="H") if isinstance(a, Atom)
        ]
        if h_neighbors:
            return [h_neighbors[0]]
    return []


def select_none(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """
    Select no leaving group - returns empty list.

    Useful for addition reactions where nothing is eliminated.

    Args:
        struct: Atomistic structure (ignored)
        reaction_site: Reaction site atom (ignored)

    Returns:
        Empty list
    """
    return []


# =============================================================================
# Utility - Find port atom by name (used by PolymerBuilder)
# =============================================================================


def find_port_atom(struct: Atomistic, port_name: str) -> Atom:
    """
    Find an atom with the specified port marker.

    This is a utility function, not a selector. Use it to find
    port atoms before passing them to selectors.

    Args:
        struct: Atomistic structure containing ports
        port_name: Name of port to find

    Returns:
        Atom with matching port

    Raises:
        ValueError: If port not found
    """
    for atom in struct.atoms:
        if atom.get("port") == port_name:
            return atom  # type: ignore[return-value]
        ports = atom.get("ports")
        if isinstance(ports, list) and port_name in ports:
            return atom

    raise ValueError(f"Port '{port_name}' not found")


def find_port_atom_by_node(struct: Atomistic, port_name: str, node_id: int) -> Atom:
    """
    Find port atom for a specific node ID.

    Useful when structure has multiple nodes and you need
    to find the port for a specific one.

    Args:
        struct: Atomistic structure
        port_name: Name of port
        node_id: The monomer_node_id to match

    Returns:
        Atom with matching port and node_id

    Raises:
        ValueError: If port not found for the specified node
    """
    for atom in struct.atoms:
        if atom.get("monomer_node_id") != node_id:
            continue
        if atom.get("port") == port_name:
            return atom  # type: ignore[return-value]
        ports = atom.get("ports")
        if isinstance(ports, list) and port_name in ports:
            return atom

    raise ValueError(f"Port '{port_name}' not found for node {node_id}")


def find_port(struct: Atomistic, port_name: str, *, node_id: int | None = None) -> Atom:
    """Find a port atom, optionally filtering by node ID.

    Convenience wrapper: delegates to find_port_atom or
    find_port_atom_by_node depending on whether node_id is given.

    Args:
        struct: Atomistic structure containing ports.
        port_name: Name of port to find.
        node_id: If given, restrict search to atoms with this monomer_node_id.

    Returns:
        Atom with matching port.

    Raises:
        ValueError: If port not found.
    """
    if node_id is not None:
        return find_port_atom_by_node(struct, port_name, node_id)
    return find_port_atom(struct, port_name)


# =============================================================================
# High-level convenience selectors (used by presets and builder)
# =============================================================================


def select_self(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Identity anchor selector -- returns the port atom as the reaction site.

    This is the simplest anchor selector: the port atom itself is where
    the new bond will form.

    Args:
        struct: Atomistic structure (unused, kept for interface compatibility)
        port_atom: The port atom entity

    Returns:
        The same port atom
    """
    return port_atom


def select_hydrogens(n: int | None = None) -> Callable[[Atomistic, Atom], list[Atom]]:
    """
    Factory that returns a leaving selector picking *n* hydrogen neighbors.

    Args:
        n: Number of hydrogens to select.
            ``None`` means all hydrogens bonded to the site.

    Returns:
        A leaving selector ``(struct, site) -> list[Atom]``
    """

    def _selector(struct: Atomistic, site: Atom) -> list[Atom]:
        h_atoms = [
            a for a in find_neighbors(struct, site, element="H") if isinstance(a, Atom)
        ]
        if n is None:
            return h_atoms
        return h_atoms[:n]

    return _selector


def select_neighbor(
    element: str,
) -> Callable[[Atomistic, Atom], Atom]:
    """
    Factory that returns an anchor selector picking a neighbor of a given element.

    Args:
        element: Element symbol to look for (e.g. ``"C"``, ``"O"``).

    Returns:
        An anchor selector ``(struct, port_atom) -> Atom``

    Raises:
        ValueError: If no neighbor with the requested element is found.
    """

    def _selector(struct: Atomistic, port_atom: Atom) -> Atom:
        neighbors = [
            a
            for a in find_neighbors(struct, port_atom, element=element)
            if isinstance(a, Atom)
        ]
        if not neighbors:
            raise ValueError(f"No {element} neighbor found for atom {port_atom}")
        return neighbors[0]

    return _selector
