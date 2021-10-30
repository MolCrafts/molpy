# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.base import Node
from molpy.element import Element
from molpy.bond import Bond
import numpy as np

class Atom(Node):
    """ Atom describes all the properties attached on an atom. 
    """
    def __init__(self, name) -> None:
        """Initialize an atom.

        Args:
            name (str): Highly recommand set atom name uniquely, which can help you find any atom in a group or system
        """
        super().__init__(name)
        self._bondInfo = {} # bondInfo = {Atom: Bond}

    def bondto(self, atom, **attr):
        """basic method to set bonded atom. E.g. H1.bondto(O, r0=0.99*mp.unit.angstrom)

        Args:
            atom (Atom): another atom to be bonded

        Returns:
            Bond: bond formed
        """
        bond = self._bondInfo.get(atom, Bond(self, atom, **attr))
        bond.update(attr)
        
        if atom not in self._bondInfo:
            self._bondInfo[atom] = bond
        if self not in atom._bondInfo:
            atom._bondInfo[self] = bond
            
        return bond
            
    def removeBond(self, atom):
        """Remove Bond between this atom and specific atom

        Args:
            atom (Atom): another atom
        """
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
        """Return a new atom which has same properties with this one, but It total another instance. We don't recommand you to use deepcopy() to duplicate.

        Returns:
            Atom: new atom instance
        """
        atom = Atom(self.name)
        atom.update(self._attr)
        
        return atom
        