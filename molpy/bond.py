# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-21
# version: 0.0.1

from molpy.base import Item

class Bond(Item):
    
    def __init__(self, atom1, atom2, **attr) -> None:
        super().__init__(f'{str(atom1)}-{str(atom2)}')
        
        self.atoms = tuple(sorted((atom1, atom2)))
        self.update(*attr)
        
    def __hash__(self):
        """ Warning: in the base class Item, hash == hash(id(self))
        """
        return hash(self.atoms)
    
    def update(self, **attr):
        
        for at in attr:
            setattr(self, at, attr[at])