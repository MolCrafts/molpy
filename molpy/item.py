# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

import numpy as np

class Item:
    """base class of the molpy
    """
    def __init__(self, name='') -> None:
        """initialize base class

        Args:
            name (str): the name of instances
        """
        self._uuid = id(self)
        self._name = name
        self._container = []
        self._itemType = self.__class__.__name__
    
    @property
    def properties(self):
        """return properties

        Returns:
            dict: key-value format
        """
        return self.__dict__
    
    @property
    def uuid(self):
        """uuid is the ID assigned by the system is used to distinguish different instances

        Returns:
            int: uuid
        """
        return self._uuid
    
    @property
    def name(self):
        return self._name
    
    @property
    def itemType(self):
        return self._itemType
    
    def __next__(self):
        return next(self._container)
    
    def __iter__(self):
        return iter(self._container)
    
    def __hash__(self):
        return hash(id(self))
    
    def __id__(self):
        return id(self)
    
    def __repr__(self) -> str:
        return f'< {self.__class__.__name__} {self.name} >'
    
    def deserialize(self, o, exclude=[]):
        exclude.append('_format')
        for k, v in o.items():
            if k in exclude:
                continue
            format = o['_format'].get(k, None)
            if format is None:
                setattr(self, k, v)
            else:
                if format == 'ndarray':
                    setattr(self, k, np.array(v))
        return self
    
    def serialize(self, exclude=[]):
        props = {'_format': {}}
        for k, v in self.properties.items():
            if k in exclude:
                continue
            
            if isinstance(v, np.ndarray):
                props[k] = v.tolist()
                props['_format'][k] = 'ndarray'
                
            else:
                props[k] = v
        return props
        
    
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
    
    def get(self, property, default=None):
        """get a property, equivalent to getattr()

        Args:
            property (str): name of property
            default (Any): default to None

        Returns:
            Any: property of this instance
        """
        return getattr(self, property, default)
    
    def set(self, property, value):
        """set a property, equivalent to setattr()

        Args:
            property (str): name of property
            value (Any): value of property
        """
        setattr(self, property, value)
        
    def __eq__(self, o):
        return self.uuid == o.uuid
    
    def __lt__(self, o):
        return self.uuid < o.uuid