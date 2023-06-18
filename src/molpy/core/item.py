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
    
    @property
    def id(self):
        return self._id

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if isinstance(value, ITEMTYPE[f'{self.__class__.__name__}Type']):
            raise TypeError(f"type must be an instance of {self.__class__.__name__}, but got {type(value)}")
        self._type = value

    def get(self, key:str):

        return self._properties.get(key)
    
    def set(self, key:str, value:Any):

        self._properties[key] = value
    
    def __repr__(self):
        return f"<Atom: {self._id}>"
    
    def __eq__(self, rhs):
        return self.id == rhs.id
    
    def __getitem__(self, key:str):
            
        return self.get(key)
    
    def __setitem__(self, key:str, value:Any):
        self.set(key, value)

class Atom(Item):

    def __init__(self):

        super().__init__()
    

class Bond:

    def __init__(self, itom, jtom):

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

