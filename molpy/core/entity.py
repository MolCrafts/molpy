# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

class Residue:

    def __init__(self, id, name, atoms, **prop):

        self.id = id
        self.name = name
        self.atoms = atoms
        self.props = prop

    @classmethod
    def from_dict(cls, dict):
            
        return cls(**dict)

    @property
    def natoms(self):
        return len(self.atoms)