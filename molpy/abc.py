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