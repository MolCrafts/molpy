# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-26
# version: 0.0.1

from typing import Literal

from numpy import ndarray
from .base import _Modeller


class RandomWalk(_Modeller):

    def __init__(self, strategy:Literal['simple', 'avoid', 'gaussian']):

        self._kernel = None

        # initialize random walk kernel
        if strategy == 'simple':
            self._kernel = self._simple_random_walk()
        elif strategy == 'avoid':
            raise NotImplementedError()
        elif strategy == 'gaussian':
            raise NotImplementedError()
        
        self.strategy = strategy
        
    def linear(self, length:int, start=None) -> ndarray:
        
        positions = self._kernel.linear(length)

        if self.box is not None:
            pass  # do wrap

        return positions

    def graft(self):
        pass

    def __del__(self):
        pass