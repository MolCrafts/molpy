"""Linear homopolymer: ten EO units grown along a chain.

Guide: docs/user-guide/topology/01_linear.md
Run:   python topology/01_linear.py
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    sites = mp.io.CGSmilesIR("{[#EO]|10}").to_coarsegrain()
    chain = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
    report("linear-10", chain)


if __name__ == "__main__":
    main()
