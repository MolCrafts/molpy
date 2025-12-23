from abc import ABC, abstractmethod
from pathlib import Path

from molpy.core.system import FrameSystem

PathLike = str | Path


class ForceFieldReader(ABC):
    """Base class for force field file readers."""

    def __init__(self, path: PathLike, system: FrameSystem | None = None) -> None:
        """
        Initialize force field reader.

        Args:
            path: Path to force field file
            system: Optional existing FrameSystem to populate
        """
        self._path = Path(path)
        self._system = system if system is not None else FrameSystem()

    @abstractmethod
    def read(self, system: FrameSystem | None = None) -> FrameSystem:
        """
        Read force field data into a FrameSystem.

        Args:
            system: Optional existing FrameSystem to populate. If None, uses the one from __init__.

        Returns:
            The populated FrameSystem object
        """
        ...


class ForceFieldWriter(ABC):
    """Base class for force field file writers."""

    def __init__(self, path: PathLike):
        """
        Initialize force field writer.

        Args:
            path: Path to output file
        """
        self._path = Path(path)

    @abstractmethod
    def write(self, system: FrameSystem) -> None:
        """
        Write force field data from a FrameSystem.

        Args:
            system: FrameSystem object containing force field data
        """
        ...
