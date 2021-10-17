# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.abc import Item

class Atom(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        self._bondedAtoms = self._container

    def serialize(self):
        pass

    def deserialize(self, tmp):
        pass

    def bondto(self, atom):
        
        if atom not in self._bondedAtoms:
            self._bondedAtoms.append(atom)
        if self not in atom._bondedAtom:
            atom._bondedAtoms.append(self)
            
    def moveTo(self, vec):
        pass
    
    def moveBy(self, vec):
        pass
    
    def rotateWithEuler(self, ref, alpha, beta, gamma):
        pass
    
    def rotateWithQuaternion(self, ):
        pass