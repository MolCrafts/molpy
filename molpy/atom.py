# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.abc import Item
import numpy as np

class Atom(Item):
    """ Atom is the class which contains properties bind to the atom.
    """
    def __init__(self, name) -> None:
        super().__init__(name)
        self._bondedAtoms = self._container

    def serialize(self):
        pass

    def deserialize(self, tmp):
        pass

    def bondto(self, atom):
        """ Form a chemical bond between two atoms.

        Args:
            atom (Atom): atom to be bonded
        """
        if atom not in self._bondedAtoms:
            self._bondedAtoms.append(atom)
        if self not in atom._bondedAtoms:
            atom._bondedAtoms.append(self)
            
    @property
    def bondedAtoms(self):
        return self._bondedAtoms
            
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