from typing import Callable
from abc import ABC, abstractmethod

class Potential():

    F:Callable|None = None
    E:Callable|None = None

    name: str = ""
    type: str = ""
    registered_params: tuple[str] = ()
    required_inputs: tuple[str] = ()
    outputs: tuple[str] = ()
    
    def __new__(cls, name:str, type:str, registered_params: tuple[str], required_inputs: tuple[str], outputs: tuple[str]):
        cls.name = name
        cls.type = type
        cls.registered_params = registered_params
        cls.required_inputs = required_inputs
        cls.outputs = outputs
        return super().__new__(cls)
    
    @abstractmethod
    def __init__(self, **settings):
        ...

    def __call__(self,input):
        ...

    def energy(self):
        ...

    def forces(self):
        ...

