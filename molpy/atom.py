# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.abc import Item
from molpy.element import Element
import numpy as np

class Atom(Item):
    """ Atom is the class which contains properties bind to the atom.
    """
    def __init__(self, name, **properties) -> None:
        super().__init__(name)
        self._bondedAtoms = {}
        for k, v in properties.items():
            self.set(k, v)

    def serialize(self):
        pass

    def deserialize(self, tmp):
        pass

    def bondto(self, atom, bondType=None):
        """ Form a chemical bond between two atoms.

        Args:
            atom (Atom): atom to be bonded
        """
        if atom not in self._bondedAtoms:
            self._bondedAtoms[atom] = bondType
        if self not in atom._bondedAtoms:
            atom._bondedAtoms[self] = bondType
            
    @property
    def bondedAtoms(self):
        return self._bondedAtoms
    
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