from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import IO, Generic, TypeVar

from molrs import Frame

from ..base import BaseReader, PathLike


# ─────────────────────────────────────────────────────────────────────
# Shared context-manager boilerplate
# ─────────────────────────────────────────────────────────────────────
class FileBase(BaseReader):
    """Lazy text file handle on top of ``BaseReader``'s path/lifecycle.

    Readers open lazily (no eager existence check — ``must_exist=False`` — so a
    missing file surfaces on first read, preserving historical behavior);
    writers create their target on open.
    """

    def __init__(self, path: PathLike, mode: str, **open_kwargs):
        super().__init__(path, must_exist=False)
        self._path = self.fpath  # retained for subclasses that read self._path
        self._mode = mode
        self._open_kwargs = open_kwargs
        self._fh: IO[str] | None = None

    # ---------- context-manager hooks ---------------------------------
    def __enter__(self):
        self._fh = self._path.open(self._mode, **self._open_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
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
ReadResult = TypeVar("ReadResult", covariant=True)


class DataReader(FileBase, ABC, Generic[ReadResult]):
    """Base class for data file readers."""

    def __init__(self, path: PathLike, **open_kwargs):
        super().__init__(path, mode="r", **open_kwargs)

    # -- line helpers --------------------------------------------------
    def _iter_nonblank(self) -> Iterator[str]:
        """Iterate over non-blank, stripped lines."""
        if self._fh is not None:
            self._fh.seek(0)
            for raw in self._fh:
                line = raw.strip()
                if line:
                    yield line
            return

        # A one-shot read owns its handle. Persistent handles are reserved for
        # an explicit ``with Reader(...)`` lifecycle.
        with self._path.open(self._mode, **self._open_kwargs) as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    yield line

    def __iter__(self) -> Iterator[str]:
        """`for line in reader:` yields non-blank, stripped lines."""
        return self._iter_nonblank()

    def read_lines(self) -> list[str]:
        """Return all lines at once."""
        if self._fh is not None:
            self._fh.seek(0)
            return list(self._fh.readlines())
        with self._path.open(self._mode, **self._open_kwargs) as fh:
            return list(fh.readlines())

    # -- high-level parse ---------------------------------------------
    @abstractmethod
    def read(self, frame: Frame | None = None) -> ReadResult:
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
