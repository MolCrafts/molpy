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


    @abstractmethod
    def forward(self, struct, output, **params):
        ...

    def __call__(self, struct, output, **params):
        return self.forward(struct, output, **params)

    def energy(self):
        ...

    def forces(self):
        ...


class PotentialSeq(Potential, list):

    def __new__(cls, *potential:tuple[Potential], name: str = 'PotentialSeq'):
        registered_params = []
        required_inputs = []
        outputs = []
        for p in potential:
            register_params.extend(p.registered_params)
            required_inputs.extend(p.required_inputs)
            outputs.extend(p.outputs)

        register_params = tuple(set(register_params))
        required_inputs = tuple(set(required_inputs))
        outputs = tuple(set(outputs))
        return super().__new__(cls, name, 'container', registered_params, required_inputs, outputs)

    def __init__(self, *potential:tuple[Potential], name: str = 'PotentialSeq'):
        super().__init__()
        self.extend(potential)

    def forward(self, struct, output, **params):
        for p in self:
            struct, output = p(struct, output, **params)
        return struct, output
    
    def energy(self):
        for p in self:
            struct, output = p.energy(struct, output)
        return struct, output
    
    def forces(self):
        for p in self:
            struct, output = p.forces(struct, output)
        return struct, output