# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-24
# version: 0.0.1

class AtomType:
    
    def __init__(self, name, **attr) -> None:
        self.name = name
        for k, v in attr:
            setattr(self, k, v)
    
    