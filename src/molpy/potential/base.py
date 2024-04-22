from typing import Callable
import numpy as np
from molpy import Alias

class Potential:

    F:Callable|None = None
    E:Callable|None = None

    def __new__(cls, *args, **kwargs):
        if cls.F is None:
            raise NotImplementedError("F method must be implemented")
        return super().__new__(cls)
    
    def __init__(self, name:str, type:str):
        self.name = name
        self.type = type

    def __call__(self,input):
        return self.forward(input)

    def forward(self):
        pass

    def energy(self):
        raise NotImplementedError("energy method must be implemented")

    def forces(self):
        raise NotImplementedError("energy method must be implemented")

