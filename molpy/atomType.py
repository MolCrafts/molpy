# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-24
# version: 0.0.1

class AtomType:
    
    _atomTypes_by_name = {}
    
    def __init__(self, typeName, atomClass=None, element=None, mass=None, **attr) -> None:
        self.name = typeName
        for k, v in attr:
            setattr(self, k, v)
    
    