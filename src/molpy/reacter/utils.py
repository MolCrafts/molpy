"""
Utility functions and type aliases for reactions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.core.entity import Entity

# ── Type aliases for reaction components ─────────────────────────

AnchorSelector = Callable[[Atomistic, Atom], Atom]
"""Select anchor atom given a port atom."""

LeavingSelector = Callable[[Atomistic, Atom], list[Atom]]
"""Select leaving group atoms given an anchor atom."""

BondFormer = Callable[[Atomistic, Atom, Atom], Bond | None]
"""Create or modify bonds between two anchor atoms in an assembly."""


def find_neighbors(
    assembly: Atomistic,
    atom: Entity,
    *,
    element: str | None = None,
) -> list[Entity]:
    """
    Find neighboring atoms of a given atom.

    Args:
        assembly: Atomistic assembly containing the atom
        atom: Atom entity to find neighbors of
        element: Optional element symbol to filter by (e.g., 'H', 'C')

    Returns:
        List of neighboring atom entities

    Example:
        >>> h_neighbors = find_neighbors(asm, carbon_atom, element='H')
        >>> all_neighbors = find_neighbors(asm, carbon_atom)
    """
    neighbors: list[Entity] = []

    # Look through all bonds
    for bond in assembly.links.bucket(Bond):
        # Use identity check (is) not equality check (==)
        if any(ep is atom for ep in bond.endpoints):
            # Found a bond involving this atom
            for endpoint in bond.endpoints:
                if endpoint is not atom:
                    # Filter by element if specified
                    if (
                        element is None
                        or endpoint.get("element") == element
                        or endpoint.get("symbol") == element
                    ):
                        neighbors.append(endpoint)

    return neighbors


def get_bond_between(
    assembly: Atomistic,
    i: Entity,
    j: Entity,
) -> Bond | None:
    """
    Find existing bond between two atoms.

    Args:
        assembly: Atomistic assembly containing the atoms
        i: First atom entity
        j: Second atom entity

    Returns:
        Bond entity if found, None otherwise

    Example:
        >>> bond = get_bond_between(asm, itom, jtom)
        >>> if bond:
        ...     print(f"Bond order: {bond.get('order', 1)}")
    """
    for bond in assembly.links.bucket(Bond):
        endpoints = bond.endpoints
        # Use identity check (is) not equality check (==)
        if any(ep is i for ep in endpoints) and any(ep is j for ep in endpoints):
            return bond
    return None


def count_bonds(assembly: Atomistic, atom: Entity) -> int:
    """
    Count the number of bonds connected to an atom.

    Args:
        assembly: Atomistic assembly containing the atom
        atom: Atom entity to count bonds for

    Returns:
        Number of bonds

    Example:
        >>> valence = count_bonds(asm, carbon_atom)
        >>> print(f"Carbon has {valence} bonds")
    """
    count = 0
    for bond in assembly.links.bucket(Bond):
        # Use identity check (is) not equality check (==)
        if any(ep is atom for ep in bond.endpoints):
            count += 1
    return count


def remove_dummy_atoms(assembly: Atomistic) -> list[Entity]:
    """
    Remove all dummy atoms (element '*') from assembly.

    Args:
        assembly: Atomistic assembly to clean

    Returns:
        List of removed dummy atoms

    Example:
        >>> removed = remove_dummy_atoms(merged_asm)
        >>> print(f"Removed {len(removed)} dummy atoms")
    """
    dummy_atoms: list[Entity] = []

    for atom in assembly.entities.bucket(Atom):
        if (atom.get("element") or atom.get("symbol") or "") == "*":
            dummy_atoms.append(atom)

    if dummy_atoms:
        assembly.remove_entity(*dummy_atoms, drop_incident_links=True)

    return dummy_atoms


def create_atom_mapping(pre_atoms: list, post_atoms: list) -> dict[int, int]:
    """Create atom mapping between pre-reaction and post-reaction states.

    Creates a mapping of atom IDs from pre-reaction template to post-reaction
    template. This is used to generate map files for LAMMPS fix bond/react.

    Args:
        pre_atoms: List of atoms in pre-reaction state
        post_atoms: List of atoms in post-reaction state

    Returns:
        Dictionary mapping pre-reaction atom indices to post-reaction indices
        (1-indexed for LAMMPS)

    Note:
        Atoms that are deleted during the reaction will not appear in the mapping.
        Atoms are matched by their 'id' attribute if present, otherwise by position.

    Example:
        >>> from molpy.reacter.utils import create_atom_mapping
        >>> mapping = create_atom_mapping(pre_atoms, post_atoms)
        >>> # Write to map file
        >>> with open("reaction.map", 'w') as f:
        ...     for pre_id, post_id in sorted(mapping.items()):
        ...         f.write(f"{pre_id} {post_id}\\n")
    """
    mapping = {}

    # Try matching by atom 'id' first
    pre_id_map = {}
    for i, atom in enumerate(pre_atoms):
        atom_id = atom.get("id")
        if atom_id is not None:
            pre_id_map[atom_id] = i + 1

    post_id_map = {}
    for i, atom in enumerate(post_atoms):
        atom_id = atom.get("id")
        if atom_id is not None:
            post_id_map[atom_id] = i + 1

    # Match atoms by 'id' attribute
    for atom_id, pre_idx in pre_id_map.items():
        if atom_id in post_id_map:
            mapping[pre_idx] = post_id_map[atom_id]

    # If no matches by ID, try matching by position (for atoms without IDs)
    if not mapping:
        import numpy as np

        for i, pre_atom in enumerate(pre_atoms):
            pre_pos = np.array(
                [
                    pre_atom["x"],
                    pre_atom["y"],
                    pre_atom["z"],
                ]
            )

            for j, post_atom in enumerate(post_atoms):
                post_pos = np.array(
                    [
                        post_atom["x"],
                        post_atom["y"],
                        post_atom["z"],
                    ]
                )

                # Check if positions are close (within 0.01 Angstrom)
                dist_sq = np.sum((pre_pos - post_pos) ** 2)
                if dist_sq < 1e-4:
                    mapping[i + 1] = j + 1
                    break

    return mapping


# ===================================================================
#               Bond formers (moved from transformers.py)
# ===================================================================


def form_single_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """Create a single bond between two atoms."""
    if i is j:
        return None
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        existing["order"] = 1
        existing["kind"] = "-"
        return existing
    bond = Bond(cast(Atom, i), cast(Atom, j), order=1, kind="-")
    assembly.add_link(bond, include_endpoints=False)
    return bond


def form_double_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """Create a double bond between two atoms."""
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        existing["order"] = 2
        existing["kind"] = "="
        return existing
    bond = Bond(cast(Atom, i), cast(Atom, j), order=2, kind="=")
    assembly.add_link(bond, include_endpoints=False)
    return bond


def form_triple_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """Create a triple bond between two atoms."""
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        existing["order"] = 3
        existing["kind"] = "#"
        return existing
    bond = Bond(cast(Atom, i), cast(Atom, j), order=3, kind="#")
    assembly.add_link(bond, include_endpoints=False)
    return bond


def form_aromatic_bond(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
    """Create an aromatic bond between two atoms."""
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        existing["order"] = 1.5
        existing["kind"] = ":"
        existing["aromatic"] = True
        return existing
    bond = Bond(cast(Atom, i), cast(Atom, j), order=1.5, kind=":", aromatic=True)
    assembly.add_link(bond, include_endpoints=False)
    return bond


def create_bond_former(order: int) -> BondFormer:
    """Factory function to create bond former with specific order."""
    kind_map = {1: "-", 2: "=", 3: "#", 1.5: ":"}
    kind = kind_map.get(order, "-")

    def bond_former(assembly: Atomistic, i: Entity, j: Entity) -> Bond | None:
        existing = get_bond_between(assembly, i, j)
        if existing is not None:
            existing["order"] = order
            existing["kind"] = kind
            if order == 1.5:
                existing["aromatic"] = True
            return existing
        attrs = {"order": order, "kind": kind}
        if order == 1.5:
            attrs["aromatic"] = True
        bond = Bond(cast(Atom, i), cast(Atom, j), **attrs)
        assembly.add_link(bond, include_endpoints=False)
        return bond

    return bond_former


def skip_bond_formation(assembly: Atomistic, i: Entity, j: Entity) -> None:
    """Do not create any bond. For reactions that only remove atoms."""
    return None


def break_bond(assembly: Atomistic, i: Entity, j: Entity) -> None:
    """Remove existing bond between two atoms."""
    existing = get_bond_between(assembly, i, j)
    if existing is not None:
        assembly.remove_link(existing)
