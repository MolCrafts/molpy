# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

from typing import Literal
import numpy as np

class Cell:
    
    def __init__(self, dimension: Literal[2, 3], boundary_condition: Literal['ppp'], **kwargs) -> None:
        
        
        
        if dimension == 2:
            assert len(boundary_condition) == 2
            
        elif dimension == 3:
            assert len(boundary_condition) == 3
            self._zlo = kwargs['zlo']
            self._zhi = kwargs['zhi']
        
        self._xlo = kwargs['xlo']
        self._xhi = kwargs['xhi']
        self._ylo = kwargs['ylo']
        self._yhi = kwargs['yhi']
        
        self._dimension = dimension
        self._boundary_condition = boundary_condition
        self._x_bc, self._y_bc, self._z_bc = boundary_condition
    
    @property
    def x_boundary_condition(self):
        return self._x_bc
    
    @property
    def xlo(self):
        return self._xlo
    
    @property
    def xhi(self):
        return self._xhi
    
    @property
    def lx(self):
        return self._xhi - self._xlo
        
    @property
    def x_vector(self):
        pass
    
    @property
    def origin(self):
        return