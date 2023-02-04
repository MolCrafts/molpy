# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .typing import List, Optional
from .box import Box
from .entity import Molecule
from .forcefield import Forcefield
from .io_utils import box2cell, atom2atom
import numpy as np

class System:


    def __init__(self, name:str):

        self.name = name
        self.box = Box()
        self.molecules:List[Molecule] = []
        self.forcefield = Forcefield()

    def add_molecule(self, molecule:Molecule):

        self.molecules.append(molecule)

    def write(self, path:str, format:str):

        from chemfiles import Frame, Trajectory

        frame = Frame()

        cur_natoms = 0
        for molecule in self.molecules:
            for atom in molecule.atoms:
                _atom = atom2atom(atom)
                # _atom.mass = 20
                frame.add_atom(_atom, atom.xyz)
            for bond in molecule.bond_index:
                frame.add_bond(*(bond+cur_natoms))

            cur_natoms += molecule.natoms

        frame.cell = box2cell(self.box)

        with Trajectory(path, 'w', format) as traj:
            traj.write(frame)

    @property
    def atoms(self):
        
        _atoms = []
        for molecule in self.molecules:
            _atoms.extend(molecule.atoms)

        return _atoms

    @property
    def natoms(self):
            
        return sum([molecule.natoms for molecule in self.molecules])

    @property
    def xyz(self):
            
        xyz = np.zeros((self.natoms, 3))
        cur_natoms = 0
        for molecule in self.molecules:
            xyz[cur_natoms:cur_natoms+molecule.natoms, :] = molecule.xyz
            cur_natoms += molecule.natoms

        return xyz