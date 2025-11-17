from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import IO

from molpy import Frame

PathLike = str | Path


# ─────────────────────────────────────────────────────────────────────
# Shared context-manager boilerplate
# ─────────────────────────────────────────────────────────────────────
class FileBase(ABC):
    """Common logic for Context-manager + lazy file handle."""

    def __init__(self, path: PathLike, mode: str, **open_kwargs):
        self._path = Path(path)
        self._mode = mode
        self._open_kwargs = open_kwargs
        self._fh: IO[str] | None = None

    # ---------- context-manager hooks ---------------------------------
    def __enter__(self):
        self._fh = self._path.open(self._mode, **self._open_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._fh:
            self._fh.close()
            self._fh = None

    # ---------- lazy accessor (works with or without `with`) ----------
    @property
    def fh(self) -> IO[str]:
        if self._fh is None:
            self._fh = self._path.open(self._mode, **self._open_kwargs)
        return self._fh


# ─────────────────────────────────────────────────────────────────────
# DataReader
# ─────────────────────────────────────────────────────────────────────
class DataReader(FileBase, ABC):
    """Base class for data file readers."""

    def __init__(self, path: PathLike, **open_kwargs):
        super().__init__(path, mode="r", **open_kwargs)

    # -- line helpers --------------------------------------------------
    def _iter_nonblank(self) -> Iterator[str]:
        """Iterate over non-blank, stripped lines."""
        self.fh.seek(0)
        for raw in self.fh:
            line = raw.strip()
            if line:
                yield line

    def __iter__(self) -> Iterator[str]:
        """`for line in reader:` yields non-blank, stripped lines."""
        return self._iter_nonblank()

    def read_lines(self) -> list[str]:
        """Return all lines at once."""
        return list(self.fh.readlines())

    # -- high-level parse ---------------------------------------------
    @abstractmethod
    def read(self, frame: Frame | None = None) -> Frame:
        """
        Populate / update a Frame from the underlying file.

        Args:
            frame: Optional existing Frame to populate. If None, creates a new one.

        Returns:
            The populated Frame object
        """
        ...


# ─────────────────────────────────────────────────────────────────────
# DataWriter
# ─────────────────────────────────────────────────────────────────────
class DataWriter(FileBase, ABC):
    """Base class for data file writers."""

    def __init__(self, path: PathLike, **open_kwargs):
        super().__init__(path, mode="w", **open_kwargs)

    @abstractmethod
    def write(self, frame: Frame) -> None:
        """
        Serialize frame into the underlying file.

        Args:
            frame: Frame object to write
        """
        ...
