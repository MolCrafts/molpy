from abc import ABC, abstractmethod

class BaseResult(ABC):
    ...

class BaseCompute(ABC):

    @abstractmethod
    def compute(self, frame) -> BaseResult:
        ...

class Result1D(BaseResult):
    def __init__(self, data):
        self.data = data

    def append(self, data):
        self.data.append(data)

    def reset(self):
        self.data = []