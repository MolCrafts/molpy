# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-17
# version: 0.0.1

class Item:
    
    def __init__(self, name) -> None:
        self._uuid = id(self)
        self.name = name
        self._container = []
        self.status = 'new' # 'modified' / 'new'
    
    @property
    def properties(self):
        return self.__dict__
    
    @property
    def uuid(self):
        return self._uuid
    
    def __next__(self):
        return next(self._container)
    
    def __iter__(self):
        return iter(self._container)
    
    def __hash__(self):
        return hash(id(self))
    
    def __repr__(self) -> str:
        return f'< {self.__class__.__name__} {self.name} >'
    
    def deserialize(self, o):
        pass
    
    def serialize(self):
        pass
    
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
        
    def moveTo(self, x, y, z):
        self.check_properties(position='required')
        pass