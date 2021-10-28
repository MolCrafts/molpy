# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-24
# version: 0.0.1

from molpy.base import Item


class AtomType(Item):
    
    _atomTypes_by_name = {}
    
    def __init__(self, typeName, **attr) -> None:
        super().__init__(typeName)
        self._name = typeName
        self._attr = attr
        AtomType._atomTypes_by_name[typeName] = self
        
    def __getitem__(self, attr):
        return self._attr[attr]
    
    @staticmethod
    def getByName(typeName):
        return AtomType._atomTypes_by_name[typeName]
    
    def __eq__(self, at):
        if isinstance(at, AtomType):
            return self._name == at.name
        elif isinstance(at, str):
            return self._name == at
        
    @staticmethod
    def addAtomType(typeName, **attr):
        if typeName not in AtomType._atomTypes_by_name:
            AtomType._atomTypes_by_name[typeName] = AtomType(typeName, **attr)
        else:
            raise ValueError(f'Duplicate AtomType {typeName}')