"""Telechelic: a six-unit EO chain capped by a methyl and a methoxy end.

Guide: docs/user-guide/topology/06_telechelic.md
Run:   python topology/06_telechelic.py
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    sites = mp.io.CGSmilesIR("{[#CAPA][#EO]|6[#CAPB]}").to_coarsegrain()
    tele = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
    report("telechelic", tele)


if __name__ == "__main__":
    main()
