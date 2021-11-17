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
        super().__init__(name)
        self._atom = atom
        self._btom = btom
        self.update(attr)
    
    def __iter__(self):
        return iter((self._atom, self._btom))
    
    def __repr__(self) -> str:
        return self.name