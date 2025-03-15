from abc import ABC, abstractmethod

class Compute(ABC):

    def __init__(self, kernel):
        self._kernel = kernel

    @abstractmethod
    def compute(self):
        ...
