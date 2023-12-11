# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-12-11
# version: 0.0.1

import numpy as np

from typing import Sequence

class Potential:

    def __init__(self, name:str):
        self._name = name

    def __call__(self, *args, **kwargs):
        pass

    def __add__(self, other:'Potential') -> 'Potential':
        if hasattr(self, '_potentials'):
            self._potentials = [self]
        
        if hasattr(other, '_potentials'):
            self._potentials.extend(other._potentials)
        else:
            self._potentials.append(other)

        self.energy = lambda *args, **kwargs: sum([p.energy(*args, **kwargs) for p in self._potentials])
        self.force = lambda *args, **kwargs: sum([p.force(*args, **kwargs) for p in self._potentials])


    @classmethod
    def FromPotentials(cls, potentials:Sequence['Potential']) -> 'Potential':
        
        if len(potentials) == 0:
            raise ValueError('empty potentials')
        if len(potentials) == 1:
            return potentials[0]
        else:
            return potentials[0] + cls.FromPotentials(potentials[1:])

class BondHarmonic(Potential):

    def __init__(self, k:np.ndarray, r0:np.ndarray):
        
        super().__init__('BondHarmonic')
        self._k = k
        self._r0 = r0

    def energy(self, r:np.ndarray) -> np.ndarray:
        return 0.5 * self._k * (r - self._r0)**2
    
    def force(self, r:np.ndarray) -> np.ndarray:
        return -self._k * (r - self._r0)
    
    def __call__(self, r:np.ndarray) -> np.ndarray:
        return self.energy(r)
    

class AngleHarmonic(Potential):

    def __init__(self, k:np.ndarray, theta0:np.ndarray):
        
        super().__init__('AngleHarmonic')
        self._k = k
        self._theta0 = theta0

    def energy(self, theta:np.ndarray) -> np.ndarray:
        return 0.5 * self._k * (theta - self._theta0)**2
    
    def force(self, theta:np.ndarray) -> np.ndarray:
        return -self._k * (theta - self._theta0)
    
    def __call__(self, theta:np.ndarray) -> np.ndarray:
        return self.energy(theta)
    
class PairLJ126(Potential):

    def __init__(self, epsilon:np.ndarray, sigma:np.ndarray):
        
        super().__init__('PairLJ126')
        self._epsilon = epsilon
        self._sigma = sigma

    def energy(self, r:np.ndarray) -> np.ndarray:
        return 4 * self._epsilon * ((self._sigma / r)**12 - (self._sigma / r)**6)
    
    def force(self, r:np.ndarray) -> np.ndarray:
        return 24 * self._epsilon * (2 * (self._sigma / r)**12 - (self._sigma / r)**6) / r
    
    def __call__(self, r:np.ndarray) -> np.ndarray:
        return self.energy(r)