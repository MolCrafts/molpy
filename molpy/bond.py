# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-21
# version: 0.0.1

from molpy.base import Item

class Bond(Item):
    
    def __init__(self, atom1, atom2, **attr) -> None:
        self.atoms = tuple(sorted((atom1, atom2)))
        super().__init__(f'{str(self.atoms[0])}-{str(self.atoms[1])}')
        self.atomType1 = getattr(atom1, 'type', None)
        self.atomType2 = getattr(atom2, 'type', None)
        self.update(*attr)
        
    def __hash__(self):
        """ Warning: in the base class Item, hash == hash(id(self))
        """
        
        if self.atomType1 and self.atomType2:
            return hash(self.atomType1+self.atomType2)
        else:
            return hash(self.atoms)
        
    def __id__(self):
        return id(self)
        
    def __eq__(self, o):
        return hash(self) == hash(o)
    
    def update(self, **attr):
        
        for at in attr:
            setattr(self, at, attr[at])