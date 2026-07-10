"""Build a polymer chain by composing the real builder classes.

MolPy has no ``polymer()`` dispatcher — you assemble a chain yourself, which
keeps the data flow explicit: parse a repeat-unit monomer, embed it in 3D, mark
the atoms that may react, then hand a monomer library + a reaction to
``PolymerBuilder``. Each piece (parser, conformer, builder) is a real class.

A repeat unit is an ordinary capped molecule with a couple of its atoms named.
There is no port system, no direction, no ``Connector``: the reaction SMARTS is
the only place the chemistry is written down, and it binds to the names you set.

Guide: docs/user-guide/02_polymer_stepwise.md
Run:   python 02_build_polymer.py
"""

from collections import Counter

import molpy as mp
from molpy.builder.assembly import MonomerLibrary, PolymerBuilder, ResiduePlacer
from molpy.conformer import Conformer
from molpy.core import fields
from molpy.parser import parse_molecule

# Ether condensation: an "a"-site hydroxyl oxygen and the carbon carrying a
# "b"-site hydroxyl become an ether bridge, dropping H2O. Atoms on the left that
# do not reappear on the right are the leaving groups.
ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"


def ethylene_glycol() -> mp.Atomistic:
    """The capped, real molecule an ethylene-oxide repeat unit comes from."""
    eo, _ = Conformer(add_hydrogens=True, seed=42).generate(parse_molecule("OCCO"))
    oxygens = [a for a in eo.atoms if a.get(fields.ELEMENT) == "O"]
    oxygens[0][fields.SITE] = "a"
    oxygens[1][fields.SITE] = "b"
    return eo


def main() -> None:
    # 1. Repeat unit: mark only the two atoms that may react. Every other atom is
    #    left alone — SITE is a sparse annotation, not a column to fill in.
    eo = ethylene_glycol()

    # 2. Assemble PEO-10. The builder stamps out one residue per repeat unit and
    #    bonds the adjacent ones; ResiduePlacer lays them out in space (the bond
    #    length it uses is a guess, converged by a later optimisation).
    builder = PolymerBuilder(
        MonomerLibrary({"EO": eo}), mp.Reaction(ETHER), placer=ResiduePlacer()
    )
    chain = builder.build("{[#EO]|10}")

    atoms = list(chain.atoms)
    counts = Counter(a.get(fields.ELEMENT) for a in atoms)
    print(f"PEO-10 chain: {len(atoms)} atoms, {len(list(chain.bonds))} bonds")
    print(f"  composition: {dict(counts)}")

    # Each repeat unit is a residue, and that survives into a PDB or a prmtop.
    residues = sorted({int(a[fields.RES_ID]) for a in atoms})
    print(f"  residues: {residues[0]}..{residues[-1]} ({len(residues)} units)")

    # The result is a plain Atomistic, so every downstream step (crosslink,
    # typify, optimize, export) accepts it.
    print(f"  type: {type(chain).__name__}")


if __name__ == "__main__":
    main()
