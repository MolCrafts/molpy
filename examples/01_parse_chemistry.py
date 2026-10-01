"""Chemistry notation via molrs (SMILES / SMARTS).

Lark-based BigSMILES / CGSmiles / G-BigSMILES parsers have been removed from
molpy. A polymer topology is a CGsmiles string: ``mp.CGSmilesIR(...)
.to_coarsegrain()`` gives the site graph that ``mp.Assembler`` grows (see
``topology/``).
"""

import molpy as mp


def main() -> None:
    ir = mp.SmilesIR("CCO")
    print("ethanol IR components:", ir.n_components)
    mol = mp.io.read_smiles("c1ccccc1")
    print("benzene atoms:", mol.n_atoms)

    pat = mp.SmartsPattern("[#6]")
    print("SMARTS query atoms:", pat.num_query_atoms)


if __name__ == "__main__":
    main()
