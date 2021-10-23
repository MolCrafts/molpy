# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-21
# version: 0.0.1

from molpy.item import Item

class Bond(Item):
    def __init__(self, atom1, atom2, **attr) -> None:
        self._atoms = tuple(sorted((atom1, atom2)))
        super().__init__(f'{str(self._atoms[0])}-{str(self._atoms[1])}')
        self._atomType1 = getattr(atom1, 'type', atom1.name)
        self._atomType2 = getattr(atom2, 'type', atom2.name)
        self.update(**attr)
        
        if attr.get('type', None):
            self.type = '-'.join([self._atomType1, self._atomType2])
        
    def __hash__(self):
        """ Warning: in the base class Item, hash == hash(id(self))
        """
        
        if self.atomType1 and self.atomType2:
            return hash(self.atomType1+self.atomType2)
        else:
            return hash(self._atoms)
        
    def __id__(self):
        return id(self)
        
    def __eq__(self, o):
        return hash(self) == hash(o)
    
    def update(self, **attr):
        
        for at in attr:
            setattr(self, at, attr[at])
            
    @property
    def properties(self):
        return dict((k, v) for k, v in super().properties.items() if k not in ['_atoms', '_uuid'])
        
    @property
    def atomType1(self):
        return self._atomType1
    
    @property
    def atomType2(self):
        return self._atomType2
    
    def serialize(self):
        props = super().serialize(exclude=[])