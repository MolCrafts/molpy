"""T4 — Macrocycle: build_ring → closed residue cycle.

Guide: docs/user-guide/topology/03_ring.md
Run:   python topology/03_ring.py
"""

from eo_kit import eo_builder, report


def main() -> None:
    builder = eo_builder()
    ring = builder.build_ring("EO", 6)
    report("ring-6", ring)
    # cyclic: as many bonds as atoms for this simple condensation product
    print(
        f"  cyclic bond/atom check: {len(list(ring.bonds))} bonds / {ring.n_atoms} atoms"
    )


if __name__ == "__main__":
    main()
