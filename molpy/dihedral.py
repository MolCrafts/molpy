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
        self._ltom = ltom
        self.update(attr)
        
    def __iter__(self):
        return iter(self._itom, self._jtom, self._ktom)
    
    def __repr__(self) -> str:
        return self.name
    
    @property
    def itom(self):
        return self._itom
    
    @property
    def jtom(self):
        return self._jtom
    
    @property
    def ktom(self):
        return self._ktom
    
    @property
    def ltom(self):
        return self._ltom
        
    def __iter__(self):
        return iter([self._itom, self._jtom, self._ktom, self._ltom])
    
    def __repr__(self) -> str:
        return self.name
    
    def atomNameEqualTo(self, dihe):
        
        if (self.jtom.name == dihe.jtomName) and (self.ktom.name == dihe.ktomName):
            if (self.itom.name == dihe.itomName) and (self.ltom.name == dihe.ltomName):
                return True
            
        if (self.jtom.name == dihe.ktomName) and (self.ktom.name == dihe.jtomName):
            if (self.itom.name == dihe.ltomName) and (self.ltom.name == dihe.itomName):
                return True
        
        return False
        
    