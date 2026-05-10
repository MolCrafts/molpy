"""Conversion helpers between :class:`molpy.Atomistic` and ``molrs.Atomistic``.

The molrs Python extension exposes its own ``Atomistic`` class with a
columnar Frame interface. molpy keeps its richer Entity-based ``Atomistic``;
Rust-backed molpy modules marshal both directions at the boundary.
"""

from __future__ import annotations

from typing import Any

import molrs

from .atomistic import Atomistic


def to_molrs(mol: Atomistic) -> "molrs.Atomistic":
    """Build a ``molrs.Atomistic`` mirroring ``mol``'s atoms and bonds."""
    out = molrs.Atomistic()
    indices: dict[int, int] = {}
    for atom in mol.atoms:
        symbol = str(atom.get("element") or atom.get("symbol") or "C")
        x = atom.get("x")
        y = atom.get("y")
        z = atom.get("z")
        if x is None or y is None or z is None:
            idx = out.add_atom(symbol)
        else:
            idx = out.add_atom(symbol, float(x), float(y), float(z))
        indices[id(atom)] = idx

    for bond in mol.bonds:
        i = indices.get(id(bond.itom))
        j = indices.get(id(bond.jtom))
        if i is None or j is None:
            continue
        out.add_bond(i, j)
        order = bond.get("order")
        if order is not None:
            out.set_bond_order(i, j, float(order))
    return out


def from_molrs(
    mol_rs: "molrs.Atomistic", template: Atomistic | None = None
) -> Atomistic:
    """Build a fresh :class:`molpy.Atomistic` from ``mol_rs``.

    If ``template`` is provided and its atom count matches, per-atom
    attributes other than coordinates are copied across by index so that
    downstream molpy code sees the same atoms it gave us.
    """
    frame = mol_rs.to_frame()
    atoms_block = frame["atoms"]
    bonds_block = frame["bonds"]

    elements = list(atoms_block.view("element"))
    xs = list(atoms_block.view("x"))
    ys = list(atoms_block.view("y"))
    zs = list(atoms_block.view("z"))

    template_atoms = list(template.atoms) if template is not None else []
    use_template = len(template_atoms) == len(elements)

    out = Atomistic()
    new_atoms = []
    for i, sym in enumerate(elements):
        attrs: dict[str, Any] = {}
        if use_template:
            for k, v in template_atoms[i].data.items():
                if k in {"x", "y", "z"}:
                    continue
                attrs[k] = v
        attrs["element"] = str(sym)
        attrs["x"] = float(xs[i])
        attrs["y"] = float(ys[i])
        attrs["z"] = float(zs[i])
        new_atoms.append(out.def_atom(**attrs))

    bond_keys = bonds_block.keys()
    if "atomi" in bond_keys and "atomj" in bond_keys:
        i_arr = list(bonds_block.view("atomi"))
        j_arr = list(bonds_block.view("atomj"))
        order_arr = (
            list(bonds_block.view("order"))
            if "order" in bond_keys
            else [1.0] * len(i_arr)
        )
        for i, j, order in zip(i_arr, j_arr, order_arr):
            ii = int(i)
            jj = int(j)
            if 0 <= ii < len(new_atoms) and 0 <= jj < len(new_atoms):
                out.def_bond(new_atoms[ii], new_atoms[jj], order=float(order))

    return out
