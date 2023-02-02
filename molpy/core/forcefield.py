# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-16
# version: 0.0.1

from typing import Iterable, Sequence
from .typing import Dict, Any, ArrayLike, Tuple, List
import numpy as np

class ItemType:

    def __init__(self, identity, **properties):

        self.identity = identity
        self.data:Dict[str, Any] = {}
        self.data.update(properties)
        self._hash = hash(self.identity)

    def __eq__(self, other):
        return self._hash == hash(other)

    def __lt__(self, other):
        return self._hash < hash(other)

    def __gt__(self, other):
        return self._hash > hash(other)

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.identity})>"

    def __getitem__(self, key:str):
        return self.data[key]

class AtomType(ItemType):

    pass

class BondType(ItemType):

    pass

AtomTypeName = str
AtomTypes = Tuple[AtomType, ...]

class Template:

    def __init__(self, ):

        self.atomTypes:Dict[AtomTypeName, AtomType] = {}
        self.bondTypes:Dict[AtomTypes, BondType] = {}

    def def_atomType(self, name:str, **properties):
        at = AtomType(name, **properties)
        self.atomTypes[name] = at
        return at

    def get_atomType(self, name:str)->AtomType:
        return self.atomTypes[name]

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        bt = BondType(identity, **properties)
        self.bondTypes[identity] = bt
        return bt

    def get_bondType(self, atomtype1:AtomType, atomtype2:AtomType)->BondType:
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        return self.bondTypes[identity]

class Forcefield:

    def __init__(self):

        self._parameters:Template = Template()

    def def_atomType(self, name:str, **properties):
        return self._parameters.def_atomType(name, **properties)

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        return self._parameters.def_bondType(atomtype1, atomtype2, **properties)

    def get_atomType(self, name:str)->AtomType:
        return self._parameters.get_atomType(name)

    def get_bondType(self, atomtype1:AtomType, atomtype2:AtomType)->BondType:
        return self._parameters.get_bondType(atomtype1, atomtype2)

    def match_atomTypes(self, names:Iterable)->List[AtomType | None]:
        return list(map(self._parameters.atomTypes.get, names))

    def match_bondTypes(self, atomTypes:Iterable[Tuple[AtomType, AtomType]])->List[BondType | None]:

        sort_fn = lambda x: (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])
        return list(map(self._parameters.bondTypes.get, map(sort_fn, atomTypes)))
