# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .typing import List, Optional
from .box import Box
from .entity import Molecule
from .forcefield import Forcefield

from chemfiles import Frame, Atom, Trajectory, UnitCell, Topology


class System:

    def __init__(self, name:str):

        self.name = name
        self.box = Box()
        self.molecules:List[Molecule] = []
        self.forcefield = Forcefield()

    def add_molecule(self, molecule:Molecule):

        self.molecules.append(molecule)

    def write(self, path:str, format:str):

        frame = Frame()

    @property
    def atoms(self):
        
        _atoms = []
        for molecule in self.molecules:
            _atoms.extend(molecule.atoms)

        return _atoms
