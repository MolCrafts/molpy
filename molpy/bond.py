# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-21
# version: 0.0.1

from molpy.base import Edge

class Bond(Edge):
    
    def __init__(self, atom, btom, **attr) -> None:
        self.atom = atom
        self.btom = btom
        self.update(attr)
        super().__init__(self.name)
    
    @property
    def name(self):
        return f'< Bond {self.atom.name}-{self.btom.name} >'
        
    def update(self, attr: dict):
        for k, v in attr.items():
            setattr(self, k, v)

    def __iter__(self):
        return iter((self.atom, self.btom))
    
    def __lt__(self, o):
        return self.uuid < o.uuid