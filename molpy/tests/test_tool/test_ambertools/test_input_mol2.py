"""The mol2 an AmberTools program is handed carries the graph's bond orders."""

from __future__ import annotations

import molpy as mp
from molpy.tool.ambertools import input_mol2


def _bond_orders(text: str) -> list[str]:
    lines = text.splitlines()
    start = lines.index("@<TRIPOS>BOND") + 1
    orders = []
    for line in lines[start:]:
        if line.startswith("@"):
            break
        if line.strip():
            orders.append(line.split()[3])
    return orders


def _benzoic_acid() -> mp.Atomistic:
    graph = mp.io.smiles.read_smiles("c1ccccc1C(=O)O").to_atomistic()
    return mp.conformer.Conformer(seed=1).generate(graph)[0]


def test_the_canonical_bond_orders_reach_the_mol2():
    orders = _bond_orders(input_mol2(_benzoic_acid(), rename=True))
    assert orders.count("ar") == 6
    assert orders.count("2") == 1  # C=O
    assert orders.count("1") == 8  # C-C, C-O and the six X-H


def test_a_force_field_bond_label_does_not_shadow_the_bond_order():
    graph = _benzoic_acid()
    for bond in graph.bonds:
        bond["type"] = "c3-f"
    orders = _bond_orders(input_mol2(graph, rename=True))
    assert "c3-f" not in orders
    assert orders.count("ar") == 6
    assert orders.count("2") == 1


def test_a_graph_read_from_mol2_keeps_its_bond_orders():
    first = input_mol2(_benzoic_acid(), rename=True)
    frame = mp.io.read_mol2(mp.io.Text(first))
    frame["atoms"]["element"] = frame["atoms"]["type"]
    mp.Element.fill_atomic_numbers(frame)
    graph = mp.Atomistic.from_frame(frame)
    second = input_mol2(graph, rename=False)
    assert _bond_orders(second) == _bond_orders(first)
