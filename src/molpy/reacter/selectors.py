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

from molpy.core.atomistic import Atom, Atomistic

from molpy.reacter.utils import find_neighbors, get_bond_between

# =============================================================================
# Port Selectors - Transform port atom to reaction site
# =============================================================================


def select_identity(struct: Atomistic, port_atom: Atom) -> Atom:
    """
    Identity selector - returns the port atom as the reaction site.

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

    Raises:
        ValueError: If no C neighbor found
    """
    c_neighbors = [
        a for a in find_neighbors(struct, port_atom, element="C") if isinstance(a, Atom)
    ]
    if not c_neighbors:
        raise ValueError(f"No C neighbor found for {port_atom.get('symbol')} port atom")
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

    Raises:
        ValueError: If no O neighbor found
    """
    o_neighbors = [
        a for a in find_neighbors(struct, port_atom, element="O") if isinstance(a, Atom)
    ]
    if not o_neighbors:
        raise ValueError(f"No O neighbor found for {port_atom.get('symbol')} port atom")
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
    symbol = port_atom.get("symbol")

    if symbol == "O":
        c_neighbors = [
            a
            for a in find_neighbors(struct, port_atom, element="C")
            if isinstance(a, Atom)
        ]
        if not c_neighbors:
            raise ValueError("No C neighbor found for O port atom")
        return c_neighbors[0]
    elif symbol == "C":
        return port_atom
    else:
        raise ValueError(f"Unexpected atom type '{symbol}' for dehydration port")


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
    symbol = port_atom.get("symbol")

    if symbol == "O":
        return port_atom
    elif symbol == "C":
        o_neighbors = [
            a
            for a in find_neighbors(struct, port_atom, element="O")
            if isinstance(a, Atom)
        ]
        if not o_neighbors:
            raise ValueError("No O neighbor found for C port atom")
        return o_neighbors[0]
    else:
        raise ValueError(f"Unexpected atom type '{symbol}' for dehydration port")


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
    return [n for n in neighbors if n.get("symbol") == "*"]


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

    Raises:
        ValueError: If no hydroxyl found
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
        raise ValueError("No hydroxyl oxygen found bonded to reaction site")

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
    if reaction_site.get("symbol") == "O":
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

    raise ValueError(
        f"Port '{port_name}' not found in structure. "
        f"Atoms must have 'port' or 'ports' attribute set to '{port_name}'"
    )


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
        ValueError: If not found
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
