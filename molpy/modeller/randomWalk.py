# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-26
# version: 0.0.1

from typing import Literal

from numpy import ndarray
from numpy.typing import ArrayLike
from .base import _Modeller
from molpy import molpy_cpp as cmp
import numpy as np

class RandomWalk(_Modeller):

    def __init__(self, region_style, constrain, strategy:Literal['simple', 'avoid', 'gaussian']):

        super().__init__(region_style, constrain)

        self._kernel = None

        # initialize random walk kernel
        if strategy == 'simple':
            self._kernel = cmp.SimpleRandomWalk(*constrain)
        elif strategy == 'avoid':
            raise NotImplementedError()
        elif strategy == 'gaussian':
            raise NotImplementedError()
        
        self.strategy = strategy
        
    def linear(self, length:int, start=None, step_size=1, topo_idx_offset=0) -> ndarray:
        
        if start:
            positions = self._kernel.walk_once(length, step_size, start)
        else:
            positions = self._kernel.walk_once(length, step_size)
        topo = []
        for i in range(topo_idx_offset, length+topo_idx_offset):
            topo.append([i, i+1])

        return positions, topo

    def graft(self, backbone_length:int, graft_point_idx:ArrayLike, graft_length:ArrayLike, step_size=1):
        
        positions, topo = self.linear(backbone_length)
        for i in range(len(graft_point_idx)):
            start = positions[graft_point_idx[i]]
            graft, g_topo = self.linear(graft_length[i]+1, start, step_size)
            topo.append([i, len(positions)])
            topo.extend(g_topo)
            positions.extend(graft)

        return positions, topo


    def __del__(self):
        pass