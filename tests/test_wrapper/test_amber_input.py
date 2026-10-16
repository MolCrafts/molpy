"""The mol2 an AmberTools program is handed carries the graph's bond orders."""

from __future__ import annotations

from pathlib import Path

import molpy as mp
from molpy.wrapper._amber_input import antechamber_input_mol2


def _bond_orders(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index("@<TRIPOS>BOND") + 1
    orders = []
    for line in lines[start:]:
        if line.startswith("@"):
            break
        if line.strip():
            orders.append(line.split()[3])
    return orders


def _benzoic_acid() -> mp.Atomistic:
    graph = mp.io.smiles.SmilesIr("c1ccccc1C(=O)O").to_atomistic()
    return mp.conformer.Conformer(seed=1).generate(graph)[0]


def test_the_canonical_bond_orders_reach_the_mol2(tmp_path):
    path = tmp_path / "input.mol2"
    antechamber_input_mol2(_benzoic_acid(), path, rename=True)
    orders = _bond_orders(path)
    assert orders.count("ar") == 6
    assert orders.count("2") == 1  # C=O
    assert orders.count("1") == 8  # C-C, C-O and the six X-H


def test_a_force_field_bond_label_does_not_shadow_the_bond_order(tmp_path):
    graph = _benzoic_acid()
    for bond in graph.bonds:
        bond["type"] = "c3-f"
    path = tmp_path / "input.mol2"
    antechamber_input_mol2(graph, path, rename=True)
    orders = _bond_orders(path)
    assert "c3-f" not in orders
    assert orders.count("ar") == 6
    assert orders.count("2") == 1


def test_a_graph_read_from_mol2_keeps_its_bond_orders(tmp_path):
    first = tmp_path / "first.mol2"
    antechamber_input_mol2(_benzoic_acid(), first, rename=True)
    graph = mp.Atomistic.from_frame(mp.io.read_mol2(first))
    second = tmp_path / "second.mol2"
    antechamber_input_mol2(graph, second, rename=False)
    assert _bond_orders(second) == _bond_orders(first)
