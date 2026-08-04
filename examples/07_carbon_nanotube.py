"""Build zigzag, armchair, and chiral carbon-nanotube topologies."""

from molpy.builder import CarbonTubeBuilder


def main() -> None:
    zigzag = CarbonTubeBuilder(8, 0, length=20.0).build()
    periodic = CarbonTubeBuilder(6, 6, cells=3, periodic=True)
    armchair = periodic.build()
    chiral = CarbonTubeBuilder(6, 3, cells=2).build(finalize="topology")

    print("zigzag", len(zigzag.atoms), "atoms", len(zigzag.bonds), "bonds")
    print("armchair", len(armchair.atoms), "atoms", periodic.cell().pbc)
    print("chiral", len(chiral.angles), "angles", len(chiral.dihedrals), "dihedrals")


if __name__ == "__main__":
    main()
