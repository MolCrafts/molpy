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

        # load properties
        some_properties = ['charge', 'mass', 'type',] # 'atomic_number', 'full_name', 'name', '']
        for prop in some_properties:
            if hasattr(prop, an_atom):
                molpy_frame.atoms[prop] = [getattr(atom, prop) for atom in chemfile_frame.atoms]


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

