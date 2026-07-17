"""T2/T3 — Block / sequence copolymer via build_sequence.

Guide: docs/user-guide/topology/02_block.md
Run:   python topology/02_block.py
"""

from molpy.core import fields

from eo_kit import eo_builder, ethylene_glycol, report


def main() -> None:
    # Two labels, same chemistry — sequence is purely architectural
    a = ethylene_glycol(seed=42)
    b = ethylene_glycol(seed=43)
    builder = eo_builder(extra={"A": a, "B": b})
    block = builder.build_sequence(["A"] * 6 + ["B"] * 4)
    report("block-A6B4", block)

    order = []
    seen: set[int] = set()
    for atom in block.atoms:
        rid = int(atom[fields.RES_ID])
        if rid not in seen:
            seen.add(rid)
            order.append(str(atom[fields.RES_NAME]))
    # residue id order is contiguous 1..N
    by_id = {int(a[fields.RES_ID]): str(a[fields.RES_NAME]) for a in block.atoms}
    seq = [by_id[i] for i in sorted(by_id)]
    print(f"  residue sequence: {''.join(seq)}")


if __name__ == "__main__":
    main()
