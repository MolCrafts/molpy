# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-16
# version: 0.0.1

from .typing import Dict, Any, Hashable
Identity = Hashable

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


class Template:

    def __init__(self, ):

        self.atomTypes:Dict[Identity, AtomType] = {}
        self.bondTypes:Dict[Identity, BondType] = {}

    def def_atomType(self, name:str, **properties):
        at = AtomType(name, **properties)
        self.atomTypes[name] = at
        return at

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        bt = BondType(identity, **properties)
        self.bondTypes[identity] = bt
        return bt

class ForceField:

    def __init__(self):

        self._parameters:Template = Template()

    def def_atomType(self, name:str, **properties):
        return self._parameters.def_atomType(name, **properties)

    def def_bondType(self, atomtype1:AtomType, atomtype2:AtomType, **properties):
        return self._parameters.def_bondType(atomtype1, atomtype2, **properties)

    def match_atomType(self, name:str=None)->AtomType:
        return self._parameters.atomTypes[name]

    def match_bondType(self, atomtype1:AtomType, atomtype2:AtomType)->BondType:
        if atomtype1 > atomtype2:
            atomtype1, atomtype2 = atomtype2, atomtype1
        identity = (atomtype1, atomtype2)
        return self._parameters.bondTypes[identity]

