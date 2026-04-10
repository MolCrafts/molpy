"""Abstract base classes for force field readers and writers."""

from abc import ABC, abstractmethod
from pathlib import Path

from molpy.core.forcefield import ForceField

PathLike = str | Path


class ForceFieldReader(ABC):
    """Base class for force field file readers."""

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)

    @abstractmethod
    def read(self, forcefield: ForceField | None = None) -> ForceField:
        """Read force field data from file.

        Args:
            forcefield: Optional existing ForceField to populate.

        Returns:
            Populated ForceField object.
        """
        ...


class ForceFieldWriter(ABC):
    """Base class for force field file writers."""

    def __init__(self, path: PathLike) -> None:
        self._path = Path(path)

    @abstractmethod
    def write(self, forcefield: ForceField) -> None:
        """Write force field data to file.

        Args:
            forcefield: ForceField object to serialize.
        """
        ...
