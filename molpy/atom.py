# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.item import Item
from molpy.element import Element
from molpy.bond import Bond
import numpy as np

class Atom(Item):
    """ Atom is the class which contains properties bind to the atom.
    """
    def __init__(self, name, **properties) -> None:
        super().__init__(name)
        self._bondInfo = {}
        for k, v in properties.items():
            self.set(k, v)

    def deserialize(self, o):
        return super().deserialize(o)
    
    def serialize(self):
        return super().serialize()

    def bondto(self, atom, **bondType):
        """ Form a chemical bond between two atoms.

        Args:
            atom (Atom): atom to be bonded
        """
        # check bond
        bond = self._bondInfo.get(atom, Bond(self, atom, **bondType))
        bond.update(**bondType)
        
        if atom not in self._bondInfo:
            self._bondInfo[atom] = bond
        if self not in atom._bondInfo:
            atom._bondInfo[self] = bond
            
        return bond
            
    def removeBond(self, atom):
        if atom in self._bondInfo:
            del self._bondInfo[atom]
        if self in atom._bondInfo:
            del atom._bondInfo[self]
            
    @property
    def bondedAtoms(self):
        return self._bondInfo.keys()
    
    @property
    def bonds(self):
        return dict(self._bondInfo)
    
    @property
    def element(self):
        return self._element
    
    @element.setter
    def element(self, symbol):
        self._element = Element.getBySymbol(symbol)
            
    def moveTo(self, vec: np.ndarray):
        self.check_properties(position=np.ndarray)
        self.position = vec
    
    def moveBy(self, vec):
        self.check_properties(position=np.ndarray)
        self.position.__iadd__(vec)
    
    def rotateWithEuler(self, ref, alpha, beta, gamma):
        pass
    
    def rotateWithQuaternion(self, ):
        pass