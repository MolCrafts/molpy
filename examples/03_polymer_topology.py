"""Non-linear polymer topologies: build() takes any CGSmiles graph.

One monomer library and one reaction build linear, ring, and branched
architectures — the topology lives entirely in the string:

    "{[#EO]|4}"                  linear
    "{[#EO]1[#EO][#EO]1}"        ring (…1…1 closes it)
    "{[#EO][#EO3]([#EO])[#EO]}"  branch (EO3 is a 3-arm junction)

Nothing in the builder knows which of those it is building. A site carries no
direction and no role: a linear chain is a topology whose edges form a path, a
branch point is a residue with three sites, and a ring closure is one more edge
between residues that are already connected. The reaction is the same in all
three; only how many sites a monomer carries changes.

Guide: docs/user-guide/02_assembly.md
Run:   python 03_polymer_topology.py
"""

import molpy as mp
from molpy.builder.assembly import MonomerLibrary, PolymerBuilder, ResiduePlacer
from molpy.conformer import Conformer
from molpy.core import fields
from molpy.parser import parse_molecule

ETHER = "[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"

TOPOLOGIES = {
    "linear": "{[#EO]|4}",
    "ring": "{[#EO]1[#EO][#EO]1}",
    "branch": "{[#EO][#EO3]([#EO])[#EO]}",
}


def _monomer(smiles: str, sites: dict[int, str]) -> mp.Atomistic:
    """Embed ``smiles`` in 3D and name its reactive oxygens."""
    monomer, _ = Conformer(add_hydrogens=True, seed=42).generate(parse_molecule(smiles))
    oxygens = [a for a in monomer.atoms if a.get(fields.ELEMENT) == "O"]
    for index, name in sites.items():
        oxygens[index][fields.SITE] = name
    return monomer


def main() -> None:
    library = MonomerLibrary(
        {
            # ethylene glycol: two hydroxyls -> two bonds, a chain unit
            "EO": _monomer("OCCO", {0: "a", 1: "b"}),
            # glycerol: three hydroxyls -> three bonds, a branch point
            "EO3": _monomer("OCC(O)CO", {0: "a", 1: "a", 2: "b"}),
        }
    )
    builder = PolymerBuilder(library, mp.Reaction(ETHER), placer=ResiduePlacer())

    # Same builder, three architectures — only the CGSmiles graph changes.
    for name, expr in TOPOLOGIES.items():
        polymer = builder.build(expr)
        atoms = len(list(polymer.atoms))
        bonds = len(list(polymer.bonds))
        shape = "cyclic" if bonds >= atoms else "acyclic"
        print(f"{name:7s} {expr:28s} -> {atoms:3d} atoms, {bonds:3d} bonds ({shape})")


if __name__ == "__main__":
    main()
