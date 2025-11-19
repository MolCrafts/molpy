"""
Selector functions for identifying anchor atoms and leaving groups.

Selectors are composable functions that identify specific atoms
in a monomer structure for use in reactions.
"""

from molpy import Atomistic
from molpy.core.entity import Entity
from molpy.core.wrappers.monomer import Monomer

from .utils import find_neighbors


def port_anchor_selector(monomer: Monomer, port_name: str) -> Entity:
    """
    Select anchor atom from a port's target.

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
        >>> anchor = port_anchor_selector(monomer, "1")
        >>> print(anchor.get('symbol'))  # 'C'
    """
    port = monomer.get_port(port_name)
    if port is None:
        raise ValueError(f"Port '{port_name}' not found in monomer")
    return port.target


def remove_one_H(monomer: Monomer, anchor: Entity) -> list[Entity]:
    """
    Remove one hydrogen atom bonded to the anchor.

    Useful for condensation reactions where H is eliminated.
    Returns the first H neighbor found, or empty list if none.

    Args:
        monomer: Monomer containing the atoms
        anchor: Anchor atom to find H neighbors of

    Returns:
        List containing one H atom, or empty list

    Example:
        >>> leaving = remove_one_H(monomer, carbon_atom)
        >>> # [H_atom] or []
    """
    if not isinstance(monomer, Atomistic):
        return []

    h_neighbors = find_neighbors(monomer, anchor, element="H")
    if h_neighbors:
        return [h_neighbors[0]]
    return []


def remove_all_H(monomer: Monomer, anchor: Entity) -> list[Entity]:
    """
    Remove all hydrogen atoms bonded to the anchor.

    Useful for reactions that eliminate all hydrogens from a carbon.

    Args:
        monomer: Monomer containing the atoms
        anchor: Anchor atom to find H neighbors of

    Returns:
        List of all H atoms bonded to anchor

    Example:
        >>> leaving = remove_all_H(monomer, carbon_atom)
        >>> # [H1, H2, H3] for CH3
    """
    if not isinstance(monomer, Atomistic):
        return []

    return find_neighbors(monomer, anchor, element="H")


def remove_dummy_atoms(monomer: Monomer, anchor: Entity) -> list[Entity]:
    """
    Remove dummy atoms (*) bonded to the anchor.

    Useful for BigSMILES-style reactions where * marks connection points.

    Args:
        monomer: Monomer containing the atoms
        anchor: Anchor atom (usually ignored, kept for signature compatibility)

    Returns:
        List of dummy atoms (symbol='*') bonded to anchor

    Example:
        >>> leaving = remove_dummy_atoms(monomer, carbon_atom)
        >>> # [*_atom]
    """
    if not isinstance(monomer, Atomistic):
        return []

    neighbors = find_neighbors(monomer, anchor)
    return [n for n in neighbors if n.get("symbol") == "*"]


def remove_OH(monomer: Monomer, anchor: Entity) -> list[Entity]:
    """
    Remove hydroxyl group (-OH) bonded to the anchor.

    Useful for esterification and condensation reactions.
    Finds single-bonded O neighbor (not double-bonded C=O), then finds H bonded to that O.

    Args:
        monomer: Monomer containing the atoms
        anchor: Anchor atom (e.g., C in -COOH)

    Returns:
        List containing O and H atoms [O, H], or empty list

    Example:
        >>> leaving = remove_OH(monomer, carboxyl_carbon)
        >>> # [O_atom, H_atom] from -OH group, not from C=O
    """
    if not isinstance(monomer, Atomistic):
        return []

    from .utils import get_bond_between

    # Find all O neighbors
    o_neighbors = find_neighbors(monomer, anchor, element="O")
    if not o_neighbors:
        return []

    # Find single-bonded oxygen (hydroxyl, not carbonyl)
    hydroxyl_o = None
    for o in o_neighbors:
        bond = get_bond_between(monomer, anchor, o)
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


def no_leaving_group(monomer: Monomer, anchor: Entity) -> list[Entity]:
    """
    No leaving group - returns empty list.

    Useful for addition reactions where nothing is eliminated.

    Args:
        monomer: Monomer (ignored)
        anchor: Anchor atom (ignored)

    Returns:
        Empty list

    Example:
        >>> reacter = Reacter(
        ...     leaving_left=no_leaving_group,
        ...     leaving_right=remove_one_H,
        ...     ...
        ... )
    """
    return []
