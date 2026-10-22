"""Build zigzag, armchair, and chiral carbon-nanotube topologies."""

import molpy as mp
from molpy.builder import CarbonTubeBuilder


def main() -> None:
    zigzag = mp.Atomistic.from_frame(CarbonTubeBuilder(8, 0, length=20.0).build())
    periodic = CarbonTubeBuilder(6, 6, cells=3, periodic=True)
    armchair = mp.Atomistic.from_frame(periodic.build())
    chiral = mp.Atomistic.from_frame(CarbonTubeBuilder(6, 3, cells=2).build())
    chiral.generate_topology(gen_angle=True, gen_dihedral=True)

    print("zigzag", len(zigzag.atoms), "atoms", len(zigzag.bonds), "bonds")
    print("armchair", len(armchair.atoms), "atoms", periodic.cell().pbc)
    print("chiral", len(chiral.angles), "angles", len(chiral.dihedrals), "dihedrals")


if __name__ == "__main__":
    main()
