# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from copy import deepcopy
from molpy.base import Node
from molpy.element import Element
from molpy.bond import Bond
import numpy as np

class Atom(Node):
    """ Atom is the class which contains properties bind to the atom.
    """
    def __init__(self, name) -> None:
        super().__init__(name)
        self._bondInfo = {} # bondInfo = {Atom: Bond}

    def bondto(self, atom, **attr):
        
        bond = self._bondInfo.get(atom, Bond(self, atom, **attr))
        bond.update(attr)
        
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
        return list(self._bondInfo.keys())
    
    @property
    def bonds(self):
        return dict(self._bondInfo)
    
    @property
    def element(self):
        return self._element
    
    @element.setter
    def element(self, symbol):
        # TODO:
        self._element = Element.getBySymbol(symbol)
        
    @property
    def atomType(self):
        return self._atomType
    
    @atomType.setter
    def atomType(self, v):
        # TODO:
        self._atomType = v 
        
    def copy(self):
        
        atom = Atom(self.name)
        atom.update(self._attr)
        
        return atom
        