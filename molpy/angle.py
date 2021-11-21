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
        
    @property
    def itom(self):
        return self._itom
    
    @property
    def jtom(self):
        return self._jtom
    
    @property
    def ktom(self):
        return self._ktom
        
    def __iter__(self):
        return iter([self._itom, self._jtom, self._ktom])
    
    def __repr__(self) -> str:
        return self.name
    
    def __getattr__(self, name):
        
        if 'angleType' in super().__getattribute__('__dict__'):
            return getattr(self.angleType, name)
        else:
            raise KeyError(f'{self} has no {name}')
    
    def atomNameEqualTo(self, angle):
        
        if self.jtom.name == angle.jtomName:
            if self.itom.name == angle.itomName or self.ktom.name == angle.ktomName:
                return True
            
        return False