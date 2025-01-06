import numpy as np
import numpy.typing as npt
from abc import abstractmethod, ABC

class Region(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def isin(self, xyz)->bool:
        raise NotImplementedError
    
    @abstractmethod
    def volumn(self)->float:
        raise NotImplementedError
    
class Boundary(ABC):

    @abstractmethod
    def wrap(self, xyz)->np.ndarray:
        raise NotImplementedError
    
class Cube(Region):

    def __init__(self, origin: npt.ArrayLike, side: int|float, name="Cube"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.side = side

    def isin(self, xyz):
        return np.all(self.origin <= xyz) and np.all(xyz <= self.origin + self.side)
    
    def volumn(self):
        return self.side**3
    
class Sphere(Region):

    def __init__(self, origin: npt.ArrayLike, radius: int|float, name="Sphere"):
        super().__init__(name)
        self.center = np.array(origin)
        self.radius = radius

    def isin(self, xyz):
        return np.linalg.norm(xyz - self.center) <= self.radius
    
    def volumn(self):
        return 4/3*np.pi*self.radius**3
    
class Cuboid(Region):

    def __init__(self, origin: npt.ArrayLike, lengths: npt.ArrayLike, name="Cuboid"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.upper = self.origin + lengths
        self.lengths = lengths
        
    def isin(self, xyz):
        return np.all(self.origin <= xyz) and np.all(xyz <= self.upper)
    
    def volumn(self):
        return np.prod(self.lengths)
    
    def constrain(self, xyz):
        
        upper = xyz - self.upper
        lower = self.origin - xyz
        tmp = np.max(np.concat([upper, lower]))
        return tmp