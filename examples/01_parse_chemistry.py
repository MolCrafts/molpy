"""Parse chemistry strings into ``Atomistic`` structures.

Every MolPy modeling task starts from a structure. The ``parser`` module turns the
common line notations — SMILES, SMARTS, BigSMILES, CGSmiles — into the ``Atomistic``
graph the rest of MolPy edits, typifies, and exports.

Guide: docs/user-guide/01_parsing_chemistry.md
Run:   python 01_parse_chemistry.py
"""

from molpy.parser import (
    parse_bigsmiles,
    parse_cgsmiles,
    parse_molecule,
    parse_smarts,
)


def main() -> None:
    # A concrete molecule: SMILES -> a heavy-atom Atomistic (add hydrogens later
    # with a 3D-embedding step; see 03_conformers in the guide).
    ethanol = parse_molecule("CCO")
    print(
        f"ethanol (CCO): {len(list(ethanol.atoms))} atoms, "
        f"{len(list(ethanol.bonds))} bonds"
    )  # 3 atoms, 2 bonds

    # A substructure query: SMARTS parses to a query pattern, not a molecule.
    hydroxyl = parse_smarts("[C][OH]")
    print(f"SMARTS '[C][OH]' -> {type(hydroxyl).__name__}")

    # A polymer repeat unit: BigSMILES describes a stochastic object; the bonding
    # descriptors <, >, $ mark where repeat units connect.
    peo_unit = parse_bigsmiles("{[>][<]CCO[>][<]}")
    print(f"BigSMILES PEO unit -> {type(peo_unit).__name__}")

    # A coarse-grained topology: CGSmiles names beads with [#Name].
    cg = parse_cgsmiles("{[#A][#B][#A]}")
    print(f"CGSmiles '[#A][#B][#A]' -> {type(cg).__name__}")


if __name__ == "__main__":
    main()
