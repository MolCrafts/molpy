import numpy as np

class A:
    
    def __init__(self):
        self.position = np.array([1, 2, 3])

class As:
    
    def __init__(self):
        self._positions = []
        self._as = []

    def add(self, a):
        self._as.append(a)
        self._positions.append(a.position)
        
    @property
    def positions(self):
        return np.asarray(self._positions)