# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-06-18
# version: 0.0.1

from typing import Any

from itemtype import AtomType, BondType

ITEMTYPE = {
    'AtomType': AtomType,
    'BondType': BondType
}

class Item:

    def __init__(self):
        self._id = hash(self)
        self._properties = dict()
        self._type = None

    def __eq__(self, rhs):
        return self.id == rhs.id
    
    @property
    def id(self):
        return self._id
        

class Atom(Item):

    def __init__(self):

        super().__init__()

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, value):
        if not isinstance(value, AtomType):
            raise TypeError(f"type must be an instance of AtomType, but got {type(value)}")
        self._type = value 

    def __getiitem__(self, key:str):
        return self._properties.get(key, self._type[key] if self._type else None)
    
    def __setitem__(self, key, value):
        self._properties[key] = value
        

class Bond(Item):

    def __init__(self, itom, jtom):
        super().__init__()
        self.itom = itom
        self.jtom = jtom

    @property
    def itom(self):
        return self._itom
    
    @itom.setter
    def itom(self, value):
        if not isinstance(value, Atom):
            raise TypeError(f"itom must be an instance of Atom, but got {type(value)}")
        self._itom = value

    @property
    def jtom(self):
        return self._jtom
    
    @jtom.setter
    def jtom(self, value):
        if not isinstance(value, Atom):
            raise TypeError(f"jtom must be an instance of Atom, but got {type(value)}")
        self._jtom = value

    @property
    def type(self):
        return self._type
    
    @type.setter
    def type(self, value):
        if not isinstance(value, BondType):
            raise TypeError(f"type must be an instance of BondType, but got {type(value)}")
        self._type = value 

    def __getiitem__(self, key:str):
        return self._properties.get(key, self._type[key] if self._type else None)
    
    def __setitem__(self, key, value):
        self._properties[key] = value
        