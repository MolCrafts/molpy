# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-06-18
# version: 0.0.1

class ItemType:

    def __init__(self, name):
        self._name = name
        self._properties = dict()

    def __eq__(self, rhs):
        return self._name == rhs._name
    
    @property
    def name(self):
        return self._name
    
    def __getitem__(self, key:str):
        return self._properties.get(key, None)
    
    def __setitem__(self, key, value):
        self._properties[key] = value
    
class AtomType(ItemType):

    def __init__(self, name):
        super().__init__(name)

class BondType(ItemType):

    def __init__(self, name):
        super().__init__(name)