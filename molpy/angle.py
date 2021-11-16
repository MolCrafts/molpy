# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-30
# version: 0.0.1

from molpy.base import Item

class Angle(Item):

    def __init__(self, itom, jtom, ktom, **attr) -> None:
        """Angle class

        Args:
            itom (Atom): atom at edge
            jtom (Atom): atom at vertex
            ktom (Atom): atom at edge
        """
        super().__init__(f'< Angle {itom.name}-{jtom.name}-{ktom.name} >')
        self._itom = itom
        self._jtom = jtom
        self._ktom = ktom
        self.update(attr)
        
    def __iter__(self):
        return iter([self._itom, self._jtom, self._ktom])
    
    def __repr__(self) -> str:
        return self.name