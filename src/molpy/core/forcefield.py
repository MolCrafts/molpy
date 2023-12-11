# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

from collections import defaultdict
from typing import Sequence

from .potential import Potential


class ItemType:

    def __init__(self, name:str):
        self._name = name

    def __hash__(self):
        return hash(self._name)
    
    @property
    def name(self):
        return self._name
    
    @property
    def id(self):
        return hash(self)
    
class AtomType:

    def __init__(self, name:str):
        super().__init__(name)

class BondType:

    def __init__(self, atype1:AtomType, atype2:AtomType):
        super().__init__(f'{atype1.name}-{atype2.name}')
        self._atype1 = atype1
        self._atype2 = atype2

class AngleType:

    def __init__(self, atype1:AtomType, atype2:AtomType, atype3:AtomType):
        super().__init__(f'{atype1.name}-{atype2.name}-{atype3.name}')
        self._atype1 = atype1
        self._atype2 = atype2
        self._atype3 = atype3

class ForceField:

    def __init__(self):
        self._atomtypes = []
        self._bondtypes = []
        self._angletypes = []

        self._bondstyles:dict[str, list[BondType]] = defaultdict(list)

    def def_atomtype(self, tname:str):
        at = AtomType(tname)
        self._atomtypes.append(at)
        return at

    def def_bondtype(self, atype1, atype2):
        bt = BondType(atype1, atype2)
        self._bondtypes.append(bt)

        return bt
    
    def def_angletype(self, atype1, atype2, atype3):
        at = AngleType(atype1, atype2, atype3)
        self._angletypes.append(at)
        return at

