"""Chemistry notation via molrs (SMILES / SMARTS).

Lark-based BigSMILES / CGSmiles / G-BigSMILES parsers have been removed from
molpy. A polymer topology is a CGsmiles string: ``mp.io.CGSmilesIR(...)
.to_coarsegrain()`` gives the site graph that ``mp.builder.Assembler`` grows (see
``topology/``).
"""

import molpy as mp


def main() -> None:
    ir = mp.io.SmilesIR("CCO")
    print("ethanol IR components:", ir.n_components)
    mol = mp.io.SmilesIR("c1ccccc1").to_atomistic()
    print("benzene atoms:", mol.n_atoms)

    pat = mp.SmartsPattern("[#6]")
    print("SMARTS query atoms:", pat.num_query_atoms)


if __name__ == "__main__":
    main()
