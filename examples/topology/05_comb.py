"""Comb: an EO backbone with two BR branch points, each carrying a graft.

Guide: docs/user-guide/topology/05_comb.md
Run:   python topology/05_comb.py

BR's graft port is labelled ``g``, so only GR (whose ``<`` is labelled ``g``)
can join it; the unlabelled backbone ports join EO as usual.
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    graft = "[#GR][#EO]"
    sites = mp.CGSmilesIR(
        f"{{[#EO][#BR]({graft})[#EO][#BR]({graft})[#EO]}}"
    ).to_coarsegrain()
    comb = mp.Assembler(library(), mp.GrowthPlacer()).assemble(sites, mp.Atomistic)
    report("comb", comb)


if __name__ == "__main__":
    main()
