# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .struct import StaticSOA
from .topology import Topology
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
        molpy_frame.box = cell = chemfile_frame.cell.matrix

        positions = chemfile_frame.positions
        molpy_frame.atoms['positions'] = positions

        an_atom = chemfile_frame.atoms[0]

        natoms = molpy_frame.atoms.size
        atom_list_properties = an_atom.list_properties()

        # load properties
        internal_props = ['name', 'atomic_number', 'charge', 'mass', 'type',] 
        for prop in internal_props:
            if hasattr(an_atom, prop):
                molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]

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

        # TODO: residue

        bonds_orders = chemfile_frame.topology.bonds_orders

        return molpy_frame

    @property
    def natoms(self):
        return self.atoms.size

    @property
    def properties(self):
        return 

    def __getitem__(self, key):
        return self.atoms[key]

