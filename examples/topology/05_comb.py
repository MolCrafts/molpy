"""T8 — Comb: branch units on a backbone with short grafts.

Uses hand-written CGSmiles through the sole entry ``build`` — comb shapes are
expressed as residue graphs with multifunctional ``BR`` nodes.

Guide: docs/user-guide/topology/05_comb.md
Run:   python topology/05_comb.py
"""

from eo_kit import branch_unit, eo_builder, report


def main() -> None:
    builder = eo_builder(extra={"BR": branch_unit()})
    # Backbone EO–BR–EO–BR–EO with a one-unit graft on each BR
    comb = builder.build("{[#EO][#BR]([#EO])[#EO][#BR]([#EO])[#EO]}")
    report("comb", comb)
    print("  topology: EO-BR(EO)-EO-BR(EO)-EO")


if __name__ == "__main__":
    main()
