"""Offline crosslinking: turn a bundle of chains into a network.

A ``Crosslinker`` takes a Daylight reaction SMARTS and rewrites one graph into
a new, crosslinked one — matching reactive sites, pairing them (by distance or
topology), and forming bonds. It is offline: the network is built as a graph,
not during an MD run. The assembler forms the bonds; you relax them yourself (the call sequence is relax:

    build chains -> RandomCrosslinker.apply (match + pair + bond)
    -> LBFGS/SoftPotential relaxes the new (over-stretched) bonds

Here the network is exported to LAMMPS as geometry (no force field). To attach
GAFF types + charges, see 06_crosslinked_gel_gaff.

Guide: docs/user-guide/04_crosslinking.md
Run:   python 04_crosslinking.py
"""

from pathlib import Path

import numpy as np

import molpy as mp
from molpy.builder.assembly import GraphAssembler, RandomSelector
from molpy.conformer import Conformer
from molpy.io import write_lammps_data
from molpy.parser import parse_molecule

OUT = Path("crosslink_output")

# Radical C-C coupling: bond two CH2 carbons on different chains, dropping one
# H from each.
CROSSLINK = "[C;H2:1][H].[C;H2:2][H] >> [C:1][C:2]"


def _relax(gel: mp.Atomistic) -> mp.Atomistic:
    """Converge the freshly formed bonds; they started at a guessed length."""
    import molrs

    from molpy.optimize import LBFGS, SoftPotential

    frame = gel.to_frame()
    result = LBFGS(SoftPotential()).run(frame, fmax=0.05, steps=200)
    return mp.Atomistic.adopt(molrs.Atomistic.from_frame(result.frame))


def main() -> None:
    # A 2x2 bundle of PEO chains (from SMILES), placed close enough to react.
    peo, _ = Conformer(add_hydrogens=True, seed=42).generate(
        parse_molecule("OCCOCCOCCO")
    )
    system = mp.Atomistic()
    mol = 0
    for iy in range(2):
        for iz in range(2):
            copy = peo.copy()
            copy.move([0.0, iy * 4.0, iz * 4.0], entity_type=mp.Atom)
            for atom in copy.atoms:
                atom["mol_id"] = mol
            system.merge(copy)
            mol += 1
    n_atoms0 = len(list(system.atoms))
    print(f"{mol} PEO chains: {n_atoms0} atoms")

    # Crosslink across chains (inter-chain only), then relax the new bonds.
    gel = GraphAssembler(mp.Reaction(CROSSLINK)).assemble(
        system,
        RandomSelector(
            conversion=1.0,
            seed=3,
            cutoff=5.0,
            exclude_same_molecule=True,
            max_per_molecule=2,
        ),
    )
    gel = _relax(gel)
    n_xlink = (n_atoms0 - len(list(gel.atoms))) // 2  # each crosslink drops 2 H
    print(f"crosslinked network: {len(list(gel.atoms))} atoms, {n_xlink} crosslinks")

    # Export geometry to LAMMPS. A .data file needs an atom-type index; without a
    # force field, use one type per element (real GAFF types are in 06).
    elem_type = {"C": 1, "O": 2, "H": 3}
    for atom in gel.atoms:
        atom["type"] = elem_type[atom.get("element")]
        atom["charge"] = 0.0  # no force field yet -> zero charges
    frame = gel.to_frame()
    frame["atoms"]["mol_id"] = np.ones(frame["atoms"].nrows, dtype=int)  # one network
    xyz = np.column_stack(
        [frame["atoms"]["x"], frame["atoms"]["y"], frame["atoms"]["z"]]
    )
    frame.box = mp.Box.cubic(
        float((xyz.max(0) - xyz.min(0)).max()) + 10.0, origin=xyz.min(0) - 5.0
    )
    OUT.mkdir(exist_ok=True)
    write_lammps_data(OUT / "network.data", frame)
    print(f"wrote LAMMPS geometry -> {OUT / 'network.data'}")


if __name__ == "__main__":
    main()
