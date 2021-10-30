# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-30
# version: 0.0.1

from molpy.base import Item

class Dihedral(Item):
    
    def __init__(self, itom, jtom, ktom, ltom, **attr) -> None:
        super().__init__(f'< Dihedral {itom.name}-{jtom.name}-{ktom.name}-{ltom.name} >')
        self._itom = itom
        self._jtom = jtom
        self._ktom = ktom
        self.update(attr)
        
    def __iter__(self):
        return iter(self._itom, self._jtom, self._ktom)
    
    def __repr__(self) -> str:
        return self.name