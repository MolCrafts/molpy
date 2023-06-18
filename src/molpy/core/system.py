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

    def write(self, path:str, format:str, isBonds:bool=True, isAngles:bool=True, isDihedrals:bool=True, isImpropers:bool=True):

        from chemfiles import Frame, Trajectory

        frame = Frame()

        for atom in self.forcefield.render_atoms(self.atoms):

            _atom = atom2atom(atom)
            frame.add_atom(_atom, atom.xyz)

        self.forcefield.render_bonds(self.bonds)

        for bond in self.bonds:
            frame.add_bond(bond.i, bond.j)

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
    def bonds(self):

        bonds = []
        bond_idx = self.bond_idx
        for molecule in self.molecules:
            bonds.extend(molecule.bonds)

        for bond, idx in zip(bonds, bond_idx):
            bond.i, bond.j = idx[0], idx[1]

        return bonds

    @property
    def bond_idx(self):

        bond_idx = []
        cur_natoms = 0
        for molecule in self.molecules:
            bond_idx.extend(molecule.connect+cur_natoms)
            cur_natoms += molecule.natoms

        return bond_idx

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
