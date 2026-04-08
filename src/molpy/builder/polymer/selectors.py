"""Builder-specific port marker utilities for polymer assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom, Atomistic

# Type alias — also available from molpy.reacter.selectors
LeavingGroupSelector = Callable[["Atomistic", "Atom"], list["Atom"]]


def process_port_markers(monomer: "Atomistic") -> "Atomistic":
    """Process port markers in monomer structure.

    Converts port marker notation [>] and [<] to port attributes on real atoms:
    1. Find wildcard atoms (*) with port attribute
    2. Find atoms bonded to these wildcards
    3. Transfer port attribute to bonded atoms
    4. Remove wildcard atoms

    Args:
        monomer: Atomistic structure with port markers from parse_smiles("[>]...[<]")

    Returns:
        Atomistic structure with port attributes on real atoms, wildcards removed
    """
    from molpy.core.atomistic import Atomistic as _Atomistic

    # Find wildcard atoms with port attribute
    port_markers: list[tuple["Atom", str]] = []
    for atom in monomer.atoms:
        if atom.get("element") == "*" and atom.get("port"):
            port_markers.append((atom, atom.get("port")))

    if not port_markers:
        return monomer

    # Find atoms connected to each port marker
    port_transfers: list[tuple["Atom", "Atom", str]] = []
    for wildcard, port_value in port_markers:
        connected_atom = None
        for bond in monomer.bonds:
            if bond.itom == wildcard:
                connected_atom = bond.jtom
                break
            elif bond.jtom == wildcard:
                connected_atom = bond.itom
                break

        if connected_atom:
            port_transfers.append((wildcard, connected_atom, port_value))

    # Create new Atomistic without wildcard atoms
    new_struct = _Atomistic()

    atom_mapping = {}
    for atom in monomer.atoms:
        if atom.get("element") != "*" or not atom.get("port"):
            atom_data = dict(atom.items())
            new_atom = new_struct.def_atom(**atom_data)
            atom_mapping[atom] = new_atom

    # Transfer port attributes to connected atoms
    for wildcard, real_atom, port_value in port_transfers:
        if real_atom in atom_mapping:
            new_atom = atom_mapping[real_atom]
            new_atom["port"] = port_value

    # Add bonds (excluding bonds to wildcards)
    for bond in monomer.bonds:
        if bond.itom in atom_mapping and bond.jtom in atom_mapping:
            new_itom = atom_mapping[bond.itom]
            new_jtom = atom_mapping[bond.jtom]
            new_struct.def_bond(
                new_itom,
                new_jtom,
                order=bond.get("order"),
                kind=bond.get("kind"),
            )

    return new_struct
