from molpy.atom import Atom
from molpy.group import Group

def read_ASE_atoms(ase_atoms) -> Group:
    symbols = ase_atoms.get_chemical_symbols()
    positions = ase_atoms.get_positions()
    atoms_num = len(symbols)
    name = ase_atoms.get_chemical_formula()
    g = Group(name);
    for s, pos in zip(symbols, positions):
        a = Atom(s, element=s, position = pos)
        g.addAtom(a)
    return g
    