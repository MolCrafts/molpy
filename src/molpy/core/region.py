import numpy as np
from abc import abstractmethod, ABC

class Region(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def isin(self, xyz)->bool:
        raise NotImplementedError
    
class Cube(Region):

    def __init__(self, origin, side, name="Cube"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.side = side

    def isin(self, xyz):
        return np.all(self.origin <= xyz) and np.all(xyz <= self.origin + self.side)
    
class Sphere(Region):

    def __init__(self, origin, radius, name="Sphere"):
        super().__init__(name)
        self.center = np.array(origin)
        self.radius = radius

    def isin(self, xyz):
        return np.linalg.norm(xyz - self.center) <= self.radius
    
class Cuboid(Region):

    def __init__(self, origin, sides, name="Cuboid"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.upper = self.origin + sides
        self.d = sides
        
    def isin(self, xyz):
        return np.all(self.origin <= xyz) and np.all(xyz <= self.upper)