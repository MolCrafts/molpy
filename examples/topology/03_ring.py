"""Macrocycle: six EO units closed into a ring.

Guide: docs/user-guide/topology/03_ring.md
Run:   python topology/03_ring.py

The growth placer lays the ring out as an open chain; the closing bond joins
its two ends at whatever distance they grew to, and a minimisation closes it.
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    sites = mp.io.cgsmiles.CgSmilesIr(
        "{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}"
    ).to_coarsegrain()
    ring = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(
        sites, mp.Atomistic
    )
    report("ring-6", ring)


if __name__ == "__main__":
    main()
