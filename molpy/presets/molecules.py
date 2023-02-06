# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-03
# version: 0.0.1

from molpy.core.entity import Molecule, Atom, Residue
import numpy as np

def tip3p()->Molecule:
    xyz = np.array([
        [4.125, 13.679, 13.761],
        [4.025, 14.428, 14.348],
        [4.670, 13.062, 14.249]
    ])
    ids = [1, 2, 3]
    name = ['O', 'H1', 'H2']
    residueName = ['HOH', 'HOH', 'HOH']
    element = ['O', 'H', 'H']
    m = Molecule(name='tip3p')
    r = Residue(name='HOH')
    natoms = 3
    for i in range(natoms):
        atom = Atom(name=name[i], id=ids[i], xyz=xyz[i], element=element[i], residue=residueName[i])
        r.add_atom(atom)

    connect = np.array([[0, 1], [0, 2]])
    r.topology.add_bonds(connect)
    m.add_residue(r)
    return m
    
def CH2()->Residue:
    xyz = np.array([
        [0.000, 0.000, 0.000],
        [0.000, 1.089, 0.000],
        [1.026, -0.545, 0.000]
    ])
    ids = [1, 2, 3]
    name = ['C', 'H1', 'H2']
    residueName = ['CH2', 'CH2', 'CH2']
    element = ['C', 'H', 'H']
    m = Residue(name='CH2')
    natoms = 3
    for i in range(natoms):
        atom = Atom(name=name[i], id=ids[i], xyz=xyz[i], element=element[i], residue=residueName[i])
        m.add_atom(atom)

    connect = np.array([[0, 1], [0, 2]])
    m.topology.add_bonds(connect)

    return m