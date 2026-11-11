"""Macrocycle: six EO units closed into a ring.

Guide: docs/user-guide/topology/03_ring.md
Run:   python topology/03_ring.py

The growth placer grows the ring from both ends; the unit that closes it is
placed by one rigid fit to both of its neighbours. Six rigid EO units cannot
close a ring exactly — the fit misses by about 0.66 Å rms — and a
minimisation closes the rest.
"""

import molpy as mp
from eo_kit import library, report


def main() -> None:
    sites = mp.io.cgsmiles.read_cgsmiles(
        "{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}"
    ).to_coarsegrain()
    ring = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites)
    report("ring-6", ring)


if __name__ == "__main__":
    main()
