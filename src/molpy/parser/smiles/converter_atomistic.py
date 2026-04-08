"""Convert SmilesGraphIR to Atomistic structures.

This module handles the graph→Atomistic conversion for pure SMILES structures.
"""

from dataclasses import asdict

from molpy.core.atomistic import Atom, Atomistic

from .smiles_ir import SmilesAtomIR, SmilesBondIR, SmilesGraphIR


def _convert_bond_order_to_kind(order: int | float) -> str:
    """Convert numeric bond order to kind symbol string.

    Args:
        order: Bond order (1, 2, 3, or 1.5 for aromatic)

    Returns:
        Bond kind symbol ("-", "=", "#", or ":")
    """
    kind_map = {1: "-", 2: "=", 3: "#", 1.5: ":"}
    return kind_map.get(float(order), "-")


def _build_atomistic_from_graph(
    atoms_ir: list[SmilesAtomIR],
    bonds_ir: list[SmilesBondIR],
    *,
    skip_indices: set[int] | None = None,
) -> tuple[Atomistic, dict[int, Atom]]:
    """Build Atomistic structure from IR atoms and bonds.

    Shared helper that handles atom creation and bond creation
    using _convert_bond_order_to_kind.

    Args:
        atoms_ir: List of SmilesAtomIR from graph
        bonds_ir: List of SmilesBondIR from graph
        skip_indices: Optional set of atom indices to skip

    Returns:
        Tuple of (Atomistic structure, mapping from atom_ir id to Atom)
    """
    skip = skip_indices or set()
    struct = Atomistic()

    atom_ir_id_to_atom: dict[int, Atom] = {}
    atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(atoms_ir)}

    for idx, atom_ir in enumerate(atoms_ir):
        if idx in skip:
            continue
        atom_data = asdict(atom_ir)
        extras = atom_data.pop("extras", {}) or {}
        atom_data.update(extras)
        atom = struct.def_atom(**atom_data)
        atom_ir_id_to_atom[id(atom_ir)] = atom

    bonds_added: set[tuple[int, int]] = set()

    for bond_ir in bonds_ir:
        itom_id = id(bond_ir.itom)
        jtom_id = id(bond_ir.jtom)

        if itom_id not in atom_ir_id_to_atom or jtom_id not in atom_ir_id_to_atom:
            continue

        atom_i = atom_ir_id_to_atom[itom_id]
        atom_j = atom_ir_id_to_atom[jtom_id]

        if atom_i is atom_j:
            continue

        bond_key = (
            (id(atom_i), id(atom_j))
            if id(atom_i) < id(atom_j)
            else (id(atom_j), id(atom_i))
        )
        if bond_key not in bonds_added:
            bond_order = bond_ir.order
            bond_kind = _convert_bond_order_to_kind(bond_order)
            struct.def_bond(
                atom_i, atom_j, order=bond_order, kind=bond_kind, stereo=bond_ir.stereo
            )
            bonds_added.add(bond_key)

    return struct, atom_ir_id_to_atom


def smilesir_to_atomistic(ir: SmilesGraphIR) -> Atomistic:
    """
    Convert SmilesGraphIR to Atomistic structure (topology only, no 3D coordinates).

    Handles port markers [>] and [<]: wildcard atoms with port in extras are removed,
    and their port attribute is transferred to connected real atoms.

    Args:
        ir: SmilesGraphIR from parse_smiles()

    Returns:
        Atomistic structure with atoms and bonds (no 3D coordinates)
        Port markers are transferred to real atoms, wildcards removed

    Examples:
        >>> from molpy.parser.smiles import parse_smiles, smilesir_to_atomistic
        >>> ir = parse_smiles("CCO")
        >>> struct = smilesir_to_atomistic(ir)
        >>> len(struct.atoms)
        3
        >>> len(struct.bonds)
        2
    """
    port_transfers: dict[int, str] = {}
    wildcard_indices = set()

    atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(ir.atoms)}

    for idx, atom_ir in enumerate(ir.atoms):
        if atom_ir.element == "*" and atom_ir.extras.get("port"):
            wildcard_indices.add(idx)
            port_symbol = atom_ir.extras["port"]

            for bond_ir in ir.bonds:
                connected_idx = None
                if id(bond_ir.itom) == id(atom_ir):
                    connected_idx = atom_ir_to_idx.get(id(bond_ir.jtom))
                elif id(bond_ir.jtom) == id(atom_ir):
                    connected_idx = atom_ir_to_idx.get(id(bond_ir.itom))

                if connected_idx is not None and connected_idx not in wildcard_indices:
                    port_transfers[connected_idx] = port_symbol
                    break

    struct = Atomistic()
    idx_to_atom = {}

    for idx, atom_ir in enumerate(ir.atoms):
        if idx in wildcard_indices:
            continue

        atom_data = asdict(atom_ir)
        extras = atom_data.pop("extras", {}) or {}
        atom_data.update(extras)

        if idx in port_transfers:
            atom_data["port"] = port_transfers[idx]

        new_atom = struct.def_atom(**atom_data)
        idx_to_atom[idx] = new_atom

    wildcard_connections: dict[int, list[int]] = {idx: [] for idx in wildcard_indices}

    for bond_ir in ir.bonds:
        i = atom_ir_to_idx.get(id(bond_ir.itom))
        j = atom_ir_to_idx.get(id(bond_ir.jtom))

        if i is None or j is None:
            continue

        if i in wildcard_indices and j not in wildcard_indices:
            wildcard_connections[i].append(j)
        elif j in wildcard_indices and i not in wildcard_indices:
            wildcard_connections[j].append(i)

    bonds_added = set()

    for bond_ir in ir.bonds:
        i = atom_ir_to_idx.get(id(bond_ir.itom))
        j = atom_ir_to_idx.get(id(bond_ir.jtom))

        if i is None or j is None or i == j:
            continue

        if i not in wildcard_indices and j not in wildcard_indices:
            atom_i = idx_to_atom.get(i)
            atom_j = idx_to_atom.get(j)

            if atom_i and atom_j:
                bond_key = (
                    (id(atom_i), id(atom_j))
                    if id(atom_i) < id(atom_j)
                    else (id(atom_j), id(atom_i))
                )
                if bond_key not in bonds_added:
                    bond_order = bond_ir.order
                    bond_kind = _convert_bond_order_to_kind(
                        float(bond_order) if bond_order != "ar" else 1.5
                    )
                    struct.def_bond(
                        atom_i,
                        atom_j,
                        order=bond_order,
                        kind=bond_kind,
                        stereo=bond_ir.stereo,
                    )
                    bonds_added.add(bond_key)

        elif i in wildcard_indices:
            for other_idx in wildcard_connections[i]:
                if other_idx != j:
                    atom_j = idx_to_atom.get(j)
                    atom_other = idx_to_atom.get(other_idx)

                    if atom_j and atom_other:
                        bond_key = (
                            (id(atom_j), id(atom_other))
                            if id(atom_j) < id(atom_other)
                            else (id(atom_other), id(atom_j))
                        )
                        if bond_key not in bonds_added:
                            bond_order = bond_ir.order
                            bond_kind = _convert_bond_order_to_kind(
                                float(bond_order) if bond_order != "ar" else 1.5
                            )
                            struct.def_bond(
                                atom_j,
                                atom_other,
                                order=bond_order,
                                kind=bond_kind,
                                stereo=bond_ir.stereo,
                            )
                            bonds_added.add(bond_key)

    return struct
