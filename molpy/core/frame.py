# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .struct import StaticSOA
from .topology import Topology
from .entity import Residue
import numpy as np
# from .box import Box

class Frame:

    def __init__(self, ):

        self.atoms = StaticSOA()
        self.topology = Topology()
        self.box = None

    @classmethod
    def from_chemfile_frame(cls, chemfile_frame):

        molpy_frame = cls()
        
        # load box
        molpy_frame.box = chemfile_frame.cell.matrix

        positions = chemfile_frame.positions
        molpy_frame.atoms['positions'] = positions

        an_atom = chemfile_frame.atoms[0]

        atom_list_properties = an_atom.list_properties()
        for prop in atom_list_properties:
            molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]

        # load Intrinsic properties
        Intrinsic_props = ['name', 'atomic_number', 'charge', 'mass', 'type',] 
        for prop in Intrinsic_props:
            if hasattr(an_atom, prop):
                molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]

        assert molpy_frame.atoms.length == molpy_frame.atoms.size

        # load topology
        ## load bond
        bonds = chemfile_frame.topology.bonds
        angles = chemfile_frame.topology.angles
        dihedrals = chemfile_frame.topology.dihedrals
        impropers = chemfile_frame.topology.impropers

        molpy_frame.topology.add_bonds(bonds)
        molpy_frame.topology.add_angles(angles)
        molpy_frame.topology.add_dihedrals(dihedrals)
        molpy_frame.topology.add_impropers(impropers)

        # load residue

        residues = chemfile_frame.topology.residues
        for residue in residues:
            molpy_frame.topology.add_residue(residue.id, residue.name, np.array(residue.atoms), **{k:residue[k] for k in residue.list_properties()})
        

        return molpy_frame

    @property
    def natoms(self):
        return self.atoms.size

    @property
    def nbonds(self):
        return self.topology.nbonds

    @property
    def nangles(self):
        return self.topology.nangles

    @property
    def ndihedrals(self):
        return self.topology.ndihedrals

    @property
    def nimpropers(self):
        return self.topology.nimpropers

    @property
    def nresidues(self):
        return self.topology.nresidues

    @property
    def positions(self):
        return self.atoms['positions']

    @property
    def properties(self):
        return self.atoms.keys()

    def __getitem__(self, key):
        return self.atoms[key]

    def get_residue(self, name):
        residue_dict = self.topology.get_residue(name)
        mask = residue_dict.pop('mask')
        residue_dict['atoms'] = self.atoms[mask]
        return Residue.from_dict(residue_dict)
