"""Chemistry notation via molrs (SMILES / SMARTS).

A polymer topology is a CGsmiles string: ``mp.io.cgsmiles.CgSmilesIr(...)
.to_coarsegrain()`` gives the site graph that ``mp.builder.Assembler`` grows (see
``topology/``).
"""

import molpy as mp


def main() -> None:
    ir = mp.io.smiles.SmilesIr("CCO")
    print("ethanol IR components:", ir.n_components)
    mol = mp.io.smiles.SmilesIr("c1ccccc1").to_atomistic()
    print("benzene atoms:", mol.n_atoms)

    pat = mp.perceive.SmartsPattern("[#6]")
    print("SMARTS query atoms:", pat.n_query_atoms)


if __name__ == "__main__":
    main()
