"""Common base for all molpy IO readers and writers.

``BaseReader`` is the minimal, storage-agnostic ancestor: it normalizes one or
more paths into ``self.fpaths``, optionally validates their existence, and
provides the context-manager lifecycle (``__enter__`` / ``__exit__`` /
``close``). It makes no assumption about *how* a file is read (text handle,
memory map, binary container) — those specifics live in subclasses.
"""

from abc import ABC
from pathlib import Path

PathLike = str | Path


class BaseReader(ABC):
    """Storage-agnostic ancestor for readers/writers.

    Owns path normalization, optional existence validation, and the
    context-manager lifecycle. Subclasses add the actual decoding mechanism
    (text handle, memory map, ...).
    """

    def __init__(self, fpath: PathLike | list[PathLike], *, must_exist: bool = True):
        """Normalize ``fpath`` into ``self.fpaths`` and optionally validate it.

        Args:
            fpath: A single path or a list of paths.
            must_exist: If True (default), every path must already exist or a
                ``FileNotFoundError`` is raised. Writers pass ``False`` since
                their target file is created on write.

        Raises:
            FileNotFoundError: If ``must_exist`` and any path is missing.
        """
        if isinstance(fpath, (str, Path)):
            self.fpaths: list[Path] = [Path(fpath)]
        else:
            self.fpaths = [Path(p) for p in fpath]

        if must_exist:
            for path in self.fpaths:
                if not path.exists():
                    raise FileNotFoundError(f"File not found: {path}")

    @property
    def fpath(self) -> Path:
        """The first (often only) path this reader is bound to."""
        if not self.fpaths:
            raise ValueError("No files available")
        return self.fpaths[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """Release any held resources. No-op by default; subclasses override."""
