from typing import Callable

class Potential:

    F:Callable|None = None
    E:Callable|None = None
    
    def __init__(self, name:str, type:str):
        self.name = name
        self.type = type

    def __call__(self,input):
        ...

    def energy(self):
        ...

    def forces(self):
        ...

