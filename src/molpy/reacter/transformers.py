"""
Bond-making transformer functions for reactions.

Transformers create or modify bonds between anchor atoms in the
product assembly.
"""

from typing import cast

from molpy import Atom, Atomistic, Bond
from molpy.core.entity import Entity

from .base import BondFormer
from .utils import get_bond_between


def form_single_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """
    Create a single bond between two atoms.

    If a bond already exists, updates it to single bond (order=1).

    Args:
        assembly: Struct to add bond to
        i: First atom
        j: Second atom

    Returns:
        The created or updated Bond object

    Side effects:
        Adds Bond(i, j, order=1) to assembly.links

    Example:
        >>> bond = form_single_bond(merged, carbon1, carbon2)
    """
    # Check if bond exists
    existing = get_bond_between(assembly, i, j)

    if existing is not None:
        # Update existing bond
        existing["order"] = 1
        existing["kind"] = "-"
        return existing
    else:
        # Create new bond
        bond = Bond(cast(Atom, i), cast(Atom, j), order=1, kind="-")
        assembly.add_link(bond, include_endpoints=False)
        return bond


def form_double_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """
    Create a double bond between two atoms.

    If a bond already exists, updates it to double bond (order=2).

    Args:
        assembly: Struct to add bond to
        i: First atom
        j: Second atom

    Returns:
        The created or updated Bond object

    Side effects:
        Adds Bond(i, j, order=2) to assembly.links
    """
    existing = get_bond_between(assembly, i, j)

    if existing is not None:
        existing["order"] = 2
        existing["kind"] = "="
        return existing
    else:
        bond = Bond(cast(Atom, i), cast(Atom, j), order=2, kind="=")
        assembly.add_link(bond, include_endpoints=False)
        return bond


def form_triple_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """
    Create a triple bond between two atoms.

    If a bond already exists, updates it to triple bond (order=3).

    Args:
        assembly: Struct to add bond to
        i: First atom
        j: Second atom

    Returns:
        The created or updated Bond object

    Side effects:
        Adds Bond(i, j, order=3) to assembly.links
    """
    existing = get_bond_between(assembly, i, j)

    if existing is not None:
        existing["order"] = 3
        existing["kind"] = "#"
        return existing
    else:
        bond = Bond(cast(Atom, i), cast(Atom, j), order=3, kind="#")
        assembly.add_link(bond, include_endpoints=False)
        return bond


def form_aromatic_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """
    Create an aromatic bond between two atoms.

    If a bond already exists, updates it to aromatic (order=1.5 by convention).

    Args:
        assembly: Struct to add bond to
        i: First atom
        j: Second atom

    Returns:
        The created or updated Bond object

    Side effects:
        Adds Bond(i, j, order=1.5, kind=':') to assembly.links
    """
    existing = get_bond_between(assembly, i, j)

    if existing is not None:
        existing["order"] = 1.5
        existing["kind"] = ":"
        existing["aromatic"] = True
        return existing
    else:
        bond = Bond(cast(Atom, i), cast(Atom, j), order=1.5, kind=":", aromatic=True)
        assembly.add_link(bond, include_endpoints=False)
        return bond


def create_bond_former(order: int) -> BondFormer:
    """
    Factory function to create bond former with specific order.

    Args:
        order: Bond order (1, 2, 3, or 1.5 for aromatic)

    Returns:
        BondFormer function that creates bonds with specified order

    Example:
        >>> double_bond_former = create_bond_former(2)
        >>> reacter = Reacter(
        ...     bond_former=double_bond_former,
        ...     ...
        ... )
    """

    def bond_former(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
        existing = get_bond_between(assembly, i, j)

        # Determine kind symbol
        kind_map = {1: "-", 2: "=", 3: "#", 1.5: ":"}
        kind = kind_map.get(order, "-")

        if existing is not None:
            existing["order"] = order
            existing["kind"] = kind
            if order == 1.5:
                existing["aromatic"] = True
            return existing
        else:
            attrs = {"order": order, "kind": kind}
            if order == 1.5:
                attrs["aromatic"] = True
            bond = Bond(cast(Atom, i), cast(Atom, j), **attrs)
            assembly.add_link(bond, include_endpoints=False)
            return bond

    return bond_former


def skip_bond_formation(assembly: Atomistic, i: Entity, j: Entity) -> None:
    """
    Do not create any bond.

    Useful for reactions that only remove atoms without forming new bonds.

    Args:
        assembly: Struct (ignored)
        i: First atom (ignored)
        j: Second atom (ignored)

    Side effects:
        None

    Example:
        >>> reacter = Reacter(
        ...     bond_former=skip_bond_formation,  # Just remove leaving groups
        ...     ...
        ... )
    """
    return None


def break_bond(assembly: Atomistic, i: Entity, j: Entity) -> None:
    """
    Remove existing bond between two atoms.

    Opposite of bond makers - used for bond-breaking reactions.

    Args:
        assembly: Struct containing the bond
        i: First atom
        j: Second atom

    Side effects:
        Removes bond from assembly.links

    Example:
        >>> break_bond(assembly, carbon1, oxygen1)
        >>> # Breaks C-O bond
    """
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        assembly.remove_link(existing)
