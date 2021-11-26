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
            if "lz" in kwargs:
                self._zlo = 0.0;
                self._zhi = kwargs["lz"]
            else:
                self._zlo = kwargs['zlo']
                self._zhi = kwargs['zhi']
        
        if "lx" in kwargs:
            self._xlo = 0.0
            self._xhi = kwargs['lx']
        else:
            self._xlo = kwargs['xlo']
            self._xhi = kwargs['xhi']
        
        if "ly" in kwargs:
            self._ylo = 0.0
            self._yhi = kwargs['ly']
        else:
            self._ylo = kwargs['ylo']
            self._yhi = kwargs['yhi']
        
        self._dimension = dimension
        self._boundary_condition = boundary_condition
        self._x_bc, self._y_bc, self._z_bc = boundary_condition
        self._pbc = np.asarray([i == "p" for i in boundary_condition])

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
    def ylo(self):
        return self._ylo
    
    @property
    def yhi(self):
        return self._yhi
    
    @property
    def zlo(self):
        return self._zlo
    
    @property
    def zhi(self):
        return self._zhi
    
    @property
    def lx(self):
        return self._xhi - self._xlo
    
    @property
    def ly(self):
        return self._yhi - self._ylo
    
    @property
    def lz(self):
        return self._zhi - self._zlo
        
    @property
    def x_vector(self):
        pass
    
    @property
    def origin(self):
        return

    @property
    def volume(self):
        return self.lx * self.ly * self.lz

    @property
    def matrix(self):
        mat = np.diag([self.lx, self.ly, self.lz])
        return mat
    @property
    def pbc(self):
        return self._pbc