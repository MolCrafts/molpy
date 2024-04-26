import numpy as np
from abc import abstractmethod

class Region:

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def isin(self, xyz)->bool:
        raise NotImplementedError
    
class Sphere(Region):

    def __init__(self, origin, radius, name="Sphere"):
        super().__init__(name)
        self.center = np.array(origin)
        self.radius = radius

    def isin(self, xyz):
        return np.linalg.norm(xyz - self.center) <= self.radius
    
class Cuboid(Region):

    def __init__(self, origin, d, name="Cuboid"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.upper = self.origin + d
        self.d = d
        
    def isin(self, xyz):
        return np.all(self.origin <= xyz) and np.all(xyz <= self.upper)