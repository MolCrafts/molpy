"""
Selector functions for identifying reaction sites and leaving groups.

Selectors are composable functions that identify specific atoms
in a monomer structure for use in reactions.

**Important Concepts:**
    - **Anchor**: The atom marked in SMILES (e.g., `[*:1]`), accessible via
      `Port.target` or `Port.anchor`. This is the reference point for the port.
    - **Reaction Site**: The actual atom where the reaction occurs, selected by
      a port selector function. This may differ from the anchor (e.g., in EO
      polymerization, the anchor is O but the reaction site is C or H).

Port selectors (e.g., `select_port_atom`) return the actual reaction site atom,
which may be the anchor itself or a different atom (e.g., a neighbor of the anchor).
"""

from molpy.core.atomistic import Atomistic
from molpy.core.entity import Entity

from .utils import find_neighbors


def select_port_atom(assembly: Atomistic, port_name: str) -> Entity:
    """
    Select reaction site atom from an Atomistic structure by port name.

    This is the standard selector for reactions that connect directly at the port atom.
    It searches for atoms with the specified port name marked in their "port" or "ports" attribute.

    Note: This selector returns the atom itself. Other selectors may return
    different atoms (e.g., neighbors of the port atom) as the actual reaction site.

    Args:
        assembly: Atomistic structure containing the port
        port_name: Name of the port to use

    Returns:
        The atom entity marked with the port name

    Raises:
        ValueError: If port not found

    Example:
        >>> # Mark an atom with a port
        >>> atom["port"] = "port_1"
        >>> # Or use multiple ports
        >>> atom["ports"] = ["port_1", "port_2"]
        >>> reaction_site = select_port_atom(assembly, "port_1")
        >>> print(reaction_site.get('symbol'))  # 'C' (the port atom)
    """
    # Search for atom with matching port name
    for atom in assembly.atoms:
        # Check single port marker
        if atom.get("port") == port_name:
            return atom
        # Check multiple ports marker
        ports = atom.get("ports")
        if isinstance(ports, list) and port_name in ports:
            return atom

    raise ValueError(
        f"Port '{port_name}' not found in assembly. "
        f"Atoms must have 'port' or 'ports' attribute set to '{port_name}'"
    )


def select_prev_atom(assembly: Atomistic, port_name: str) -> Entity:
    """
    Select the previous atom of port atom.
    """
    port_atom = select_port_atom(assembly, port_name)
    neighbors = find_neighbors(assembly, port_atom, element="C")
    return neighbors[0]


def select_one_hydrogen(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select one hydrogen atom bonded to the port atom.

    Useful for condensation reactions where H is eliminated.
    Returns the first H neighbor found, or empty list if none.

    Args:
        assembly: Atomistic structure containing the atoms
        port_atom: Port atom to find H neighbors of

    Returns:
        List containing one H atom, or empty list

    Example:
        >>> leaving = select_one_hydrogen(assembly, carbon_atom)
        >>> # [H_atom] or []
    """
    h_neighbors = find_neighbors(assembly, port_atom, element="H")
    if h_neighbors:
        return [h_neighbors[0]]
    return []


def select_all_hydrogens(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select all hydrogen atoms bonded to the port atom.

    Useful for reactions that eliminate all hydrogens from a carbon.

    Args:
        assembly: Atomistic structure containing the atoms
        port_atom: Port atom to find H neighbors of

    Returns:
        List of all H atoms bonded to port atom

    Example:
        >>> leaving = select_all_hydrogens(assembly, carbon_atom)
        >>> # [H1, H2, H3] for CH3
    """
    return find_neighbors(assembly, port_atom, element="H")


def select_dummy_atoms(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select dummy atoms (*) bonded to the port atom.

    Useful for BigSMILES-style reactions where * marks connection points.

    Args:
        assembly: Atomistic structure containing the atoms
        port_atom: Port atom (usually ignored, kept for signature compatibility)

    Returns:
        List of dummy atoms (symbol='*') bonded to port atom

    Example:
        >>> leaving = select_dummy_atoms(assembly, carbon_atom)
        >>> # [*_atom]
    """
    neighbors = find_neighbors(assembly, port_atom)
    return [n for n in neighbors if n.get("symbol") == "*"]


def select_hydroxyl_group(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select hydroxyl group (-OH) bonded to the port atom.

    Useful for esterification and condensation reactions.
    Finds single-bonded O neighbor (not double-bonded C=O), then finds H bonded to that O.

    Args:
        assembly: Atomistic structure containing the atoms
        port_atom: Port atom (e.g., C in -COOH)

    Returns:
        List containing O and H atoms [O, H], or empty list

    Example:
        >>> leaving = select_hydroxyl_group(assembly, carboxyl_carbon)
        >>> # [O_atom, H_atom] from -OH group, not from C=O
    """
    from .utils import get_bond_between

    # Find all O neighbors
    o_neighbors = find_neighbors(assembly, port_atom, element="O")
    if not o_neighbors:
        raise ValueError("No oxygen neighbors found for port atom")

    # Find single-bonded oxygen (hydroxyl, not carbonyl)
    hydroxyl_o = None
    for o in o_neighbors:
        bond = get_bond_between(assembly, port_atom, o)
        if bond and bond.get("order") == 1:  # Single bond = hydroxyl
            hydroxyl_o = o
            break

    if hydroxyl_o is None:
        raise ValueError("No hydroxyl oxygen found bonded to port atom")

    # Find H bonded to O
    h_neighbors = find_neighbors(assembly, hydroxyl_o, element="H")
    if not h_neighbors:
        raise ValueError("No hydrogen found bonded to hydroxyl oxygen")

    return [hydroxyl_o, h_neighbors[0]]


def select_hydroxyl_h_only(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select only the hydrogen from a hydroxyl group bonded to the port atom.

    This is useful when the hydroxyl oxygen must remain (e.g., port anchor sits
    on the neighboring atom and the O should participate in the final bond).

    Args:
        assembly: Atomistic structure containing the atoms
        port_atom: Port atom

    Returns:
        [H_atom] if found, otherwise [].
    """
    from .utils import get_bond_between

    # Find O neighbor
    o_neighbors = find_neighbors(assembly, port_atom, element="O")
    for o in o_neighbors:
        bond = get_bond_between(assembly, port_atom, o)
        if not bond or bond.get("order") != 1:
            continue
        h_neighbors = find_neighbors(assembly, o, element="H")
        if h_neighbors:
            return [h_neighbors[0]]
    return []


def select_none(assembly: Atomistic, port_atom: Entity) -> list[Entity]:
    """
    Select no leaving group - returns empty list.

    Useful for addition reactions where nothing is eliminated.

    Args:
        assembly: Atomistic structure (ignored)
        port_atom: Port atom (ignored)

    Returns:
        Empty list

    Example:
        >>> reacter = Reacter(
        ...     leaving_selector_left=select_none,
        ...     leaving_selector_right=select_one_hydrogen,
        ...     ...
        ... )
    """
    return []
