# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-21
# version: 0.0.1

from molpy.base import Edge

class Bond(Edge):
    
    def __init__(self, atom, btom, **attr) -> None:
        if 'name' in attr:
            name = attr['name']
        else:
            name = f'< Bond {atom.name}-{btom.name} >'
        super().__init__(name, **attr)
        self._atom = atom
        self._btom = btom
    
    def __iter__(self):
        return iter((self._atom, self._btom))
    
    def __repr__(self) -> str:
        return self.name
    
    def __getattr__(self, key):
        if 'bondType' not in self.__dict__:
            raise KeyError(f'{key} not in {self} and its bondType')
        return getattr(self.bondType, key)
    
    @property
    def atom(self):
        return self._atom
    
    @property
    def btom(self):
        return self._btom