"""Build zigzag, armchair, and chiral carbon-nanotube topologies."""

from molpy.builder import CarbonTubeBuilder


def main() -> None:
    builder = CarbonTubeBuilder()
    zigzag = builder.build(8, 0, length=20.0)
    armchair = builder.build(6, 6, cells=3, periodic=True)
    chiral = builder.build(6, 3, cells=2, finalize="topology")

    print("zigzag", len(zigzag.atoms), "atoms", len(zigzag.bonds), "bonds")
    print("armchair", len(armchair.atoms), "atoms", armchair["box"].pbc)
    print("chiral", len(chiral.angles), "angles", len(chiral.dihedrals), "dihedrals")


if __name__ == "__main__":
    main()
