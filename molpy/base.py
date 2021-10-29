# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.2

__all__ = ['Node', 'Graph', 'Edge']

class Item:
    
    def __init__(self, name) -> None:
        self._attr = {}
        self._uuid = id(self)
        self.name = name
        self.itemType = self.__class__.__name__
        
    def __setattr__(self, name, value):
        if not name.startswith('_'):
            self._attr[name] = value
        super().__setattr__(name, value)

    @property
    def uuid(self):
        return self._uuid
    
    def __hash__(self) -> int:
        return self._uuid
    
    def __repr__(self) -> str:
        return f'< {self.__class__.__name__} {self.name} >'
    
    def check_properties(self, **props):
        """ a method to check if the instances has method required properties before method is execute
        For example, if move() method need to check item.position, then add this at the first
        ...: def move(self, x, y, z):
        ...:     self.check_properties(position='required')

        Raises:
            AttributeError: WHEN no required properties
            TypeError: WHEN required property has wrong type
        """
        for k,v in props.items():
            kv = getattr(self, k, None)
            if kv is None:
                AttributeError(f'this method requires {self} has property {k}')
            else:
                if isinstance(kv, v):
                    continue
                else:
                    raise TypeError(f'requires {k} is {v} but {type(kv)}')
        return True
    
    def get(self, property, default=None):
        return getattr(self, property, default)
    
    def set(self, property, value):
        setattr(self, property, value)
        
    def update(self, attr):
        for k, v in attr.items():
            self.set(k, v)
        
    def __eq__(self, o):
        return self.uuid == o.uuid
    
    def __lt__(self, o):
        return self.uuid < o.uuid
    
    @property
    def properties(self):
        return self._attr
    
    def copy(self):
        pass
    
class Node(Item):
    """A Atom DataView class for a molpy Group

    """
    def __init__(self, name) -> None:
        super().__init__(name)
                
class Edge(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)
        
class Graph(Item):
    
    def __init__(self, name) -> None:
        super().__init__(name)