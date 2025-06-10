import numpy as np
import numpy.typing as npt
from abc import abstractmethod, ABC


class Region(ABC):

    def __init__(self, name):
        self.name = name

    @property
    @abstractmethod
    def bounds(self) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def isin(self, xyz) -> np.ndarray: ...

    def __and__(self, other):
        return AndRegion(self, other)

    def __or__(self, other):
        return OrRegion(self, other)

    def __invert__(self):
        return NotRegion(self)


class AndRegion(Region):
    def __init__(self, r1: Region, r2: Region):
        self.r1 = r1
        self.r2 = r2

    def isin(self, point):
        return self.r1.isin(point) & self.r2.isin(point)

    def __repr__(self):
        return f"<{self.r1} & {self.r2}>"


class OrRegion(Region):
    def __init__(self, r1: Region, r2: Region):
        self.r1 = r1
        self.r2 = r2

    def isin(self, point):
        return self.r1.isin(point) | self.r2.isin(point)

    def __repr__(self):
        return f"<{self.r1} | {self.r2}>"


class NotRegion(Region):
    def __init__(self, region: Region):
        self.region = region

    def isin(self, point):
        return ~self.region.isin(point)

    def __repr__(self):
        return f"<!{self.region}>"


class PeriodicBoundary(ABC):

    @abstractmethod
    def wrap(self, xyz) -> np.ndarray:
        raise NotImplementedError


class Cube(Region):

    def __init__(
        self,
        length: int | float,
        origin: npt.ArrayLike = np.array([0, 0, 0]),
        name="Cube",
    ):
        super().__init__(name)
        self.origin = np.array(origin)
        self.length = length

    def isin(self, xyz):
        return np.logical_and(
            np.all(self.origin <= xyz, axis=1),
            np.all(xyz <= self.origin + self.length, axis=1),
        )

    def __repr__(self):
        return f"<Cube {self.name}: {self.origin} {self.length}>"

    def volumn(self):
        return self.length**3

    @property
    def xlo(self):
        return self.origin[0]

    @property
    def xhi(self):
        return self.origin[0] + self.length

    @property
    def ylo(self):
        return self.origin[1]

    @property
    def yhi(self):
        return self.origin[1] + self.length

    @property
    def zlo(self):
        return self.origin[2]

    @property
    def zhi(self):
        return self.origin[2] + self.length
    
    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        return np.array([
            [self.xlo, self.xhi],
            [self.ylo, self.yhi],
            [self.zlo, self.zhi]
        ]).T


class SphereRegion(Region):

    def __init__(self, radius: int | float, origin: npt.ArrayLike, name="Sphere"):
        super().__init__(name)
        assert isinstance(radius, (int, float)), "Radius must be a number"
        self.origin = np.array(origin)
        assert self.origin.size == 3, "Origin must be a 3D vector"
        self.radius = radius

    def isin(self, xyz):
        return np.linalg.norm(xyz - self.origin, axis=-1) <= self.radius

    def volumn(self):
        return 4 / 3 * np.pi * self.radius**3

    def __repr__(self):
        return f"<Sphere {self.name}: {self.origin} {self.radius}>"


class BoxRegion(Region):

    def __init__(self, lengths: npt.ArrayLike, origin: npt.ArrayLike, name="Box"):
        super().__init__(name)
        self.origin = np.array(origin)
        self.upper = self.origin + lengths
        self.lengths = lengths

    def isin(self, xyz):
        return np.logical_and(
            np.all(self.origin <= xyz, axis=-1), np.all(xyz <= self.upper, axis=-1)
        )

    def volumn(self):
        return np.prod(self.lengths)

    @property
    def bounds(self):
        return np.array([self.origin, self.upper]).T

