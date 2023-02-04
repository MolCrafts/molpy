# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-02-03
# version: 0.0.1

from molpy.core.entity import Molecule
import numpy as np

def tip3p():
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
    m.atoms['xyz'] = xyz
    m.atoms['id'] = ids
    m.atoms['name'] = name
    m.atoms['residue'] = residueName
    m.atoms['element'] = element

    connect = np.array([[0, 1], [0, 2]])
    m.topology.add_bonds(connect)

    return m
    