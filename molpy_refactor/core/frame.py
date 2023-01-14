# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .struct import StaticSOA

class Frame:

    def __init__(self, ):

        self.atoms = StaticSOA()

    @classmethod
    def from_chemfile_frame(cls, chemfile_frame):

        molpy_frame = cls()
        
        positions = chemfile_frame.positions
        molpy_frame.atoms['positions'] = positions

        an_atom = chemfile_frame.atoms[0]

        natoms = molpy_frame.atoms.size
        atom_list_properties = an_atom.list_properties()

        for prop in atom_list_properties:
            molpy_frame.atoms.set_empty_like(prop, natoms, an_atom[prop])

        for i, atom in enumerate(chemfile_frame.atoms):
            for prop in atom_list_properties:
                molpy_frame.atoms[prop][i] = atom[prop]

        # load topology
        ## load bond
        bonds = chemfile_frame.topology.bonds
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

