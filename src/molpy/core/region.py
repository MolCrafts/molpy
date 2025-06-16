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
        super().__init__(f"({r1.name} & {r2.name})")
        self.r1 = r1
        self.r2 = r2

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """
        Get the bounds of the intersection region.
        
        Returns:
            npt.NDArray[np.float64]: The overlapping bounds of both regions.
        """
        bounds1 = self.r1.bounds
        bounds2 = self.r2.bounds
        # Take the intersection (max of lower bounds, min of upper bounds)
        lower = np.maximum(bounds1[0], bounds2[0])
        upper = np.minimum(bounds1[1], bounds2[1])
        return np.array([lower, upper])

    def isin(self, point):
        return self.r1.isin(point) & self.r2.isin(point)

    def __repr__(self):
        return f"<{self.r1} & {self.r2}>"


class OrRegion(Region):
    def __init__(self, r1: Region, r2: Region):
        super().__init__(f"({r1.name} | {r2.name})")
        self.r1 = r1
        self.r2 = r2

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """
        Get the bounds of the union region.
        
        Returns:
            npt.NDArray[np.float64]: The combined bounds of both regions.
        """
        bounds1 = self.r1.bounds
        bounds2 = self.r2.bounds
        # Take the union (min of lower bounds, max of upper bounds)
        lower = np.minimum(bounds1[0], bounds2[0])
        upper = np.maximum(bounds1[1], bounds2[1])
        return np.array([lower, upper])

    def isin(self, xyz):
        return self.r1.isin(xyz) | self.r2.isin(xyz)

    def __repr__(self):
        return f"<{self.r1} | {self.r2}>"


class NotRegion(Region):
    def __init__(self, region: Region):
        super().__init__(f"(!{region.name})")
        self.region = region

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """
        Get the bounds of the inverted region.
        
        For inverted regions, we return infinite bounds since the complement
        of a finite region is generally infinite.
        
        Returns:
            npt.NDArray[np.float64]: Infinite bounds array.
        """
        inf = np.inf
        return np.array([[-inf, -inf, -inf], [inf, inf, inf]])

    def isin(self, xyz):
        return ~self.region.isin(xyz)

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

    def volume(self):
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

    def volume(self):
        return 4 / 3 * np.pi * self.radius**3

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """
        Get the bounds of the sphere.
        
        Returns:
            npt.NDArray[np.float64]: The bounding box of the sphere.
        """
        return np.array([
            self.origin - self.radius,
            self.origin + self.radius
        ])

    def __repr__(self):
        return f"<Sphere {self.name}: {self.origin} {self.radius}>"


class BoxRegion(Region):

    def __init__(self, lengths: npt.ArrayLike, origin: npt.ArrayLike, name="Box"):
        super().__init__(name)
        self.origin = np.array(origin, dtype=np.float64)
        self.lengths = np.array(lengths, dtype=np.float64)
        self.upper = self.origin + self.lengths

    def isin(self, xyz):
        return np.logical_and(
            np.all(self.origin <= xyz, axis=-1), np.all(xyz <= self.upper, axis=-1)
        )

    def volume(self):
        return np.prod(self.lengths)

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """
        Get the bounds of the box region.
        
        Returns:
            npt.NDArray[np.float64]: The bounds of the box.
        """
        return np.array([self.origin, self.upper])

