"""Star: a three-arm core with three EO arms of three units.

Guide: docs/user-guide/topology/04_star.md
Run:   python topology/04_star.py
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    arm = "[#EO][#EO][#EO]"
    sites = mp.io.smiles.CGSmilesIR(f"{{[#X3]({arm})({arm}){arm}}}").to_coarsegrain()
    star = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(
        sites, mp.Atomistic
    )
    report("star-3x3", star)


if __name__ == "__main__":
    main()
