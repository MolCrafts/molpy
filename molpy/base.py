# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.2

__all__ = ['Node', 'Graph', 'Edge']

from copy import deepcopy

class Item:
    # __slots__ = ("_name", "_itemType", "prop1", "prop2", "__dict__")
    def __init__(self, name, **attr) -> None:
        self.update(attr)
        # self._uuid = id(self)
        self._name = name
        self._itemType = self.__class__.__name__
        
        #TODO: need to redesign class attribute:
        #      type 1: attributes belongs to this python instance, not properties
        #      type 2: attributes are properties but can not be copy
        #      type 3: attributes are properties and to copy
        
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls, self._name)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self._name)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
        
    @property
    def uuid(self):
        return id(self)
    
    @property
    def itemType(self):
        return self._itemType
    
    def __hash__(self) -> int:
        return id(self)
    
    def __repr__(self) -> str:
        return f'< {self._itemType} {self._name} >'
      
    def update(self, attr):
        for k, v in attr.items():
            setattr(self,k, v)
        
    def __eq__(self, o):
        return self.uuid == o.uuid
    
    def __lt__(self, o):
        return self.uuid < o.uuid
    
    @property
    def properties(self):
        return self.__dict__
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        self._name = name
    
    def __call__(self, **attr):
        item = deepcopy(self)
        item.update(attr)
        return item

    
class Node(Item):
    """A Atom DataView class for a molpy Group

    """
    def __init__(self, name, **attr) -> None:
        super().__init__(name, **attr)
                
class Edge(Item):
    def __init__(self, name, **attr) -> None:
        super().__init__(name, **attr)
        
class Graph(Item):
    def __init__(self, name, **attr) -> None:
        super().__init__(name, **attr)