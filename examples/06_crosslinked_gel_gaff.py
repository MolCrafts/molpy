"""Crosslinked PEO gel: from SMILES to a GAFF-parameterised LAMMPS system.

The full pipeline, composing real classes end to end:

    SMILES -> PEO chains -> crosslinked network -> GAFF -> LAMMPS

1. Parse a PEO oligomer from SMILES and embed it in 3D with molpy's native
   (molrs) conformer generator.
2. Replicate it into a small bundle of independent chains, packed close
   enough that CH2 sites reach across chains.
3. Crosslink CH2 sites across chains with ``RandomCrosslinker`` (offline,
   graph-level) and relax the freshly formed bonds (SoftPotential).
4. Parameterise the whole network with GAFF via AmberTools (antechamber +
   parmchk2 + tleap): atom types, AM1-BCC charges, and a force field.
5. Write a complete LAMMPS system (``system.data`` + ``system.ff``).

Requires AmberTools (antechamber/parmchk2/tleap). Set ``AMBER_ENV`` to the
conda env that provides them; the GAFF step shells out via
``conda run -n <AMBER_ENV>``.

Run:   python 03_crosslinked_gel_gaff.py
Notes: antechamber runs AM1-BCC on the whole ~90-atom network (~2-3 min).
       Large networks do not fit whole-network AM1-BCC — type per monomer /
       per junction region instead.
"""

from pathlib import Path

import numpy as np

import molpy as mp
from molpy.builder.ambertools import AmberTools
from molpy.builder.crosslink import RandomCrosslinker, crosslink_gel
from molpy.conformer import Conformer
from molpy.io import write_lammps_system
from molpy.parser import parse_molecule

AMBER_ENV = "AmberTools25"  # conda env providing antechamber/parmchk2/tleap
CHAIN_SMILES = "OCCOCCOCCO"  # HO-(CH2CH2O)3-H, a PEO oligomer
GRID = (2, 2)  # chain bundle: 2x2 = 4 chains
GAP = 4.0  # Angstrom spacing between chains (within crosslink cutoff)
OUT = Path("peo_gel_output")

# Radical C-C coupling: bond two CH2 carbons on different chains, dropping one
# H from each (a standard idealised PEO crosslink).
CROSSLINK = "[C;H2:1][H].[C;H2:2][H] >> [C:1][C:2]"


def main() -> None:
    # 1. PEO chain straight from SMILES; Conformer (molpy's native molrs
    #    embedder) produces a physical 3D conformer.
    peo, _ = Conformer(add_hydrogens=True, seed=42).generate(
        parse_molecule(CHAIN_SMILES)
    )

    # 2. Replicate into a grid of independent chains (each its own mol_id).
    system = mp.Atomistic()
    mol = 0
    for iy in range(GRID[0]):
        for iz in range(GRID[1]):
            copy = peo.copy()
            copy.move([0.0, iy * GAP, iz * GAP], entity_type=mp.Atom)
            for atom in copy.atoms:
                atom["mol_id"] = mol
            system.merge(copy)
            mol += 1
    n_atoms0 = len(list(system.atoms))
    print(f"[1-2] {mol} PEO chains from SMILES {CHAIN_SMILES!r}: {n_atoms0} atoms")

    # 3. Crosslink across chains, then relax the new (over-stretched) bonds with
    #    the force-field-free SoftPotential (crosslink_gel does apply + relax).
    crosslinker = RandomCrosslinker(
        CROSSLINK,
        conversion=1.0,
        seed=3,
        cutoff=5.0,
        exclude_same_molecule=True,  # inter-chain only
        max_per_molecule=2,
    )
    gel = crosslink_gel(system, crosslinker, relax=True)
    n_xlink = (n_atoms0 - len(list(gel.atoms))) // 2  # each crosslink drops 2 H
    print(
        f"[3]   crosslinked network: {len(list(gel.atoms))} atoms, {n_xlink} crosslinks"
    )

    # 4. GAFF-type the whole network via AmberTools (antechamber -> parmchk2 ->
    #    tleap -> prmtop -> molrs ForceField).
    result = AmberTools(env=AMBER_ENV, env_manager="conda").parameterize(gel)
    frame = result.frame
    # The crosslinked network is one connected molecule -> single mol_id.
    frame["atoms"]["mol_id"] = np.ones(frame["atoms"].nrows, dtype=int)
    types = sorted(str(t) for t in set(frame["atoms"]["type"]))
    qsum = float(np.sum(frame["atoms"]["charge"]))
    print(f"[4]   GAFF types {types}  net charge {qsum:+.2f}")

    # 5. Box the system with padding, then write LAMMPS data + force field.
    xyz = np.column_stack(
        [frame["atoms"]["x"], frame["atoms"]["y"], frame["atoms"]["z"]]
    )
    length = float((xyz.max(0) - xyz.min(0)).max()) + 10.0
    frame.box = mp.Box.cubic(length, origin=xyz.min(0) - 5.0)
    paths = write_lammps_system(OUT, frame, result.forcefield)
    print(f"[5]   wrote {paths['data'].name} + {paths['ff'].name} -> {OUT}/")


if __name__ == "__main__":
    main()
