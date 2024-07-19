from typing import Callable
from abc import ABC, abstractmethod

class Potential(ABC):

    F:Callable
    E:Callable

    name: str = ""
    type: str = ""
    registered_params: tuple[str] = ()
    required_inputs: tuple[str] = ()
    outputs: tuple[str] = ()
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        ...


class PotentialSeq(list):

    def __init__(self, name: str, *potential:tuple[Potential]):
        super().__init__(potential)
        self.name = name
    
    def forward(self, frame, output: dict, **params):
        for potential in self:
            frame, output = potential.forward(frame, output, **params)
        return frame, output