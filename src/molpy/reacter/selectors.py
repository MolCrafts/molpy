"""
Selector functions for identifying anchor atoms and leaving groups.

Selectors are composable functions that identify specific atoms
in a monomer structure for use in reactions.
"""

from molpy import Atomistic
from molpy.core.entity import Entity
from molpy.core.wrappers.monomer import Monomer

from .utils import find_neighbors


def select_port_atom(monomer: Monomer, port_name: str) -> Entity:
    """
    Select port atom from a port's target.

    This is the standard selector for reactions that connect via ports.
    It simply returns the atom that the port points to.

    Args:
        monomer: Monomer containing the port
        port_name: Name of the port to use

    Returns:
        The atom entity targeted by the port

    Raises:
        ValueError: If port not found

    Example:
        >>> port_atom = select_port_atom(monomer, "1")
        >>> print(port_atom.get('symbol'))  # 'C'
    """
    port = monomer.get_port(port_name)
    if port is None:
        raise ValueError(f"Port '{port_name}' not found in monomer")
    return port.target


def select_one_hydrogen(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select one hydrogen atom bonded to the port atom.

    Useful for condensation reactions where H is eliminated.
    Returns the first H neighbor found, or empty list if none.

    Args:
        monomer: Monomer containing the atoms
        port_atom: Port atom to find H neighbors of

    Returns:
        List containing one H atom, or empty list

    Example:
        >>> leaving = select_one_hydrogen(monomer, carbon_atom)
        >>> # [H_atom] or []
    """
    if not isinstance(monomer, Atomistic):
        return []

    h_neighbors = find_neighbors(monomer, port_atom, element="H")
    if h_neighbors:
        return [h_neighbors[0]]
    return []


def select_all_hydrogens(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select all hydrogen atoms bonded to the port atom.

    Useful for reactions that eliminate all hydrogens from a carbon.

    Args:
        monomer: Monomer containing the atoms
        port_atom: Port atom to find H neighbors of

    Returns:
        List of all H atoms bonded to port atom

    Example:
        >>> leaving = select_all_hydrogens(monomer, carbon_atom)
        >>> # [H1, H2, H3] for CH3
    """
    if not isinstance(monomer, Atomistic):
        return []

    return find_neighbors(monomer, port_atom, element="H")


def select_dummy_atoms(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select dummy atoms (*) bonded to the port atom.

    Useful for BigSMILES-style reactions where * marks connection points.

    Args:
        monomer: Monomer containing the atoms
        port_atom: Port atom (usually ignored, kept for signature compatibility)

    Returns:
        List of dummy atoms (symbol='*') bonded to port atom

    Example:
        >>> leaving = select_dummy_atoms(monomer, carbon_atom)
        >>> # [*_atom]
    """
    if not isinstance(monomer, Atomistic):
        return []

    neighbors = find_neighbors(monomer, port_atom)
    return [n for n in neighbors if n.get("symbol") == "*"]


def select_hydroxyl_group(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select hydroxyl group (-OH) bonded to the port atom.

    Useful for esterification and condensation reactions.
    Finds single-bonded O neighbor (not double-bonded C=O), then finds H bonded to that O.

    Args:
        monomer: Monomer containing the atoms
        port_atom: Port atom (e.g., C in -COOH)

    Returns:
        List containing O and H atoms [O, H], or empty list

    Example:
        >>> leaving = select_hydroxyl_group(monomer, carboxyl_carbon)
        >>> # [O_atom, H_atom] from -OH group, not from C=O
    """
    if not isinstance(monomer, Atomistic):
        return []

    from .utils import get_bond_between

    # Find all O neighbors
    o_neighbors = find_neighbors(monomer, port_atom, element="O")
    if not o_neighbors:
        return []

    # Find single-bonded oxygen (hydroxyl, not carbonyl)
    hydroxyl_o = None
    for o in o_neighbors:
        bond = get_bond_between(monomer, port_atom, o)
        if bond and bond.get("order") == 1:  # Single bond = hydroxyl
            hydroxyl_o = o
            break

    if hydroxyl_o is None:
        return []

    # Find H bonded to O
    h_neighbors = find_neighbors(monomer, hydroxyl_o, element="H")
    if not h_neighbors:
        return [hydroxyl_o]  # Just remove O if no H found

    return [hydroxyl_o, h_neighbors[0]]


def select_hydroxyl_h_only(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select only the hydrogen from a hydroxyl group bonded to the port atom.

    This is useful when the hydroxyl oxygen must remain (e.g., port anchor sits
    on the neighboring atom and the O should participate in the final bond).

    Returns:
        [H_atom] if found, otherwise [].
    """
    if not isinstance(monomer, Atomistic):
        return []

    from .utils import get_bond_between

    # Find O neighbor
    o_neighbors = find_neighbors(monomer, port_atom, element="O")
    for o in o_neighbors:
        bond = get_bond_between(monomer, port_atom, o)
        if not bond or bond.get("order") != 1:
            continue
        h_neighbors = find_neighbors(monomer, o, element="H")
        if h_neighbors:
            return [h_neighbors[0]]
    return []


def select_none(monomer: Monomer, port_atom: Entity) -> list[Entity]:
    """
    Select no leaving group - returns empty list.

    Useful for addition reactions where nothing is eliminated.

    Args:
        monomer: Monomer (ignored)
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
