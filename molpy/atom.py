# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1


from molpy.base import Node
from molpy.element import Element
from molpy.bond import Bond
import numpy as np
from copy import deepcopy


class Atom(Node):
    """Atom describes all the properties attached on an atom."""
    # __slots__ = ('_bondInfo', "_element", "_position", "properties"
    # "key", "parent", "atomType", "type", "serial", "occupancy", "bfactor", "segid",
    # "RecordName", "altLoc", "charge", "resName", "chainID", "resSeq", "iCode")
    def __init__(self, name, **attr) -> None:
        """Initialize an atom.

        Args:
            name (str): Highly recommand set atom name uniquely, which can help you find any atom in a group or system
        """
        super().__init__(name, **attr)
        self._bondInfo = {}  # bondInfo = {Atom: Bond}

    def __getattr__(self, name):
        if 'atomType' in self.__dict__:
            return getattr(self.atomType, name)
        elif 'element' in self.__dict__:
            return getattr(self.element, name)
        else:
            raise AttributeError

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos

    @property
    def x(self):
        return float(self._position[0])

    @property
    def y(self):
        return float(self._position[1])

    @property
    def z(self):
        return float(self._position[2])

    def bondto(self, atom, **attr):
        """basic method to set bonded atom. E.g. H1.bondto(O, r0=0.99*mp.unit.angstrom)

        Args:
            atom (Atom): another atom to be bonded

        Returns:
            Bond: bond formed
        """
        if atom in self._bondInfo:
            # TODO: update attr
            bond = self._bondInfo[atom]
        else:
            bond = Bond(self, atom, **attr)

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
    def nbondedAtoms(self):
        return len(self.bondedAtoms)

    @property
    def bonds(self):
        return list(self._bondInfo.values())
    
    def getBond(self, btom):
        return self._bondInfo[btom]
    
    def getBondByAtomName(self, btomName):
        for btom in self.bondedAtoms:
            if btom.name == btomName:
                return self._bondInfo[btom]

    def getName(self):
        return self.name

    def getElement(self):
        return self._element
    
    def getSymbol(self):
        return self._element.symbol
    
    def setElement(self, symbol):
        self._element = Element.getBySymbol(symbol)

    element = property(fget=getElement, fset=setElement)

    def getPosition(self):
        return self._position

    def setPosition(self, position):
        position = np.array(position)
        if position.shape != (3,):
            assert ValueError(f"shape of position is wrong")
        self._position = position

    position = property(getPosition, setPosition)

    def move(self, vec):
        
        self._position.__iadd__(vec)
        return self