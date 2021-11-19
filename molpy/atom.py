# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

from molpy.base import Node
from molpy.element import Element
from molpy.bond import Bond
import numpy as np


class Atom(Node):
    """Atom describes all the properties attached on an atom."""

    def __init__(self, name, **attr) -> None:
        """Initialize an atom.

        Args:
            name (str): Highly recommand set atom name uniquely, which can help you find any atom in a group or system
        """
        super().__init__(name)
        self._bondInfo = {}  # bondInfo = {Atom: Bond}
        self.update(attr)
        self._position = None
        self._atomType = None
        self._element = None

    # def __getattribute__(self, key):
    #     """ if key in __dict__:
    #             return atom.key
    #         else:
    #             return __getattr__(key)

    #     Args:
    #         key (str): attribute
    #     """
    #     pass

    def __getattr__(self, key):
        return getattr(self.atomType, key)

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
    def nbondedAtoms(self):
        return len(self.bondedAtoms)

    @property
    def bonds(self):
        return dict(self._bondInfo)

    def getElement(self):
        return self._element

    def setElement(self, symbol):
        self._element = Element.getBySymbol(symbol)

    element = property(fget=getElement, fset=setElement)

    def copy(self):
        """Return a new atom which has same properties with this one, but It total another instance. We don't recommand you to use deepcopy() to duplicate.

        Returns:
            Atom: new atom instance
        """
        atom = Atom(self.name)
        atom.update(self._attr)

        return atom

    def getPosition(self):
        return self._position

    def setPosition(self, position):
        position = np.asarray(position)
        if position.shape != (3,):
            assert ValueError(f"shape of position is wrong")
        self._position = position

    position = property(getPosition, setPosition)

    def setAtomType(self, atomType):
        ele = getattr(atomType, "element", None)
        if ele is None:
            pass
        else:
            self.element = ele
        self._atomType = atomType

    def getAtomType(self):
        return self._atomType

    atomType = property(getAtomType, setAtomType)
