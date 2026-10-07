"""Block copolymer: six EO then four PO units.

Guide: docs/user-guide/topology/02_block.md
Run:   python topology/02_block.py
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    sites = mp.io.cgsmiles.CgSmilesIr("{[#EO]|6[#PO]|4}").to_coarsegrain()
    block = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(
        sites, mp.Atomistic
    )
    report("EO6-b-PO4", block)


if __name__ == "__main__":
    main()
