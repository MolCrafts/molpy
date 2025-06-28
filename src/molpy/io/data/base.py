from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, List, IO, Any

import molpy as mp


# ─────────────────────────────────────────────────────────────────────
# Shared context-manager boilerplate
# ─────────────────────────────────────────────────────────────────────
class FileBase(ABC):
    """Common logic for Context-manager + lazy file handle."""

    def __init__(self, path: str | Path, mode: str, **open_kwargs):
        self._path = Path(path)
        self._mode = mode
        self._open_kwargs = open_kwargs
        self._fh: IO[Any] | None = None

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
    def fh(self) -> IO[Any]:
        if self._fh is None:
            self._fh = self._path.open(self._mode, **self._open_kwargs)
        return self._fh


# ─────────────────────────────────────────────────────────────────────
# DataReader
# ─────────────────────────────────────────────────────────────────────
class DataReader(FileBase, ABC):
    """Text reader that filters out blank lines."""

    def __init__(self, path: str | Path, **open_kwargs):
        super().__init__(path, mode="r", **open_kwargs)

    # -- line helpers --------------------------------------------------
    def _iter_nonblank(self) -> Iterator[str]:
        self.fh.seek(0)
        for raw in self.fh:
            line = raw.strip()
            if line:
                yield line

    def __iter__(self) -> Iterator[str]:
        """`for line in reader:` yields non-blank, stripped lines."""
        return self._iter_nonblank()

    def read_lines(self) -> List[str]:
        """Return all lines at once."""
        return list(self.fh.readlines())

    # -- high-level parse ---------------------------------------------
    @abstractmethod
    def read(self, frame: mp.Frame) -> mp.Frame:
        """Populate / update a Frame from the underlying text."""
        ...


# ─────────────────────────────────────────────────────────────────────
# DataWriter
# ─────────────────────────────────────────────────────────────────────
class DataWriter(FileBase, ABC):
    """Text writer that guarantees the file is open in write-mode."""

    def __init__(self, path: str | Path, **open_kwargs):
        super().__init__(path, mode="w", **open_kwargs)

    @abstractmethod
    def write(self, frame: mp.Frame) -> None:  # noqa: D401
        """Serialize *frame* into the underlying text file."""
        ...