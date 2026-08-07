"""AMBER ASCII inpcrd / restrt I/O (molrs-backed).

Parse lives in :func:`molrs.io.read_amber_inpcrd`. This module is a thin
façade that keeps the historical :class:`AmberInpcrdReader` surface, including
optional merge of coordinates into an existing Frame.
"""

from __future__ import annotations

from pathlib import Path

from molrs import Frame

from .base import DataReader


class AmberInpcrdReader(DataReader):
    """Reader for AMBER ASCII ``*.inpcrd`` (old-style) coordinate files.

    * Coordinates: Fortran ``6F12.7``, 6 numbers per line
    * Optional velocities section (same length as coordinates; restart only)
    * Optional final box line (3–6 floats; first three → orthorhombic diagonal)
    """

    __slots__ = ()

    def __init__(self, file: str | Path, **kwargs: object) -> None:
        super().__init__(path=Path(file), **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        """Populate / update a Frame from the inpcrd path.

        Args:
            frame: Optional existing Frame. When it already has an ``"atoms"``
                block, only coordinates (and velocities, if present) are
                replaced in place and non-coordinate columns are preserved.
                Atom count must match. When ``None`` or without atoms, a new
                Frame from molrs is returned.

        Returns:
            The populated Frame.

        Raises:
            ValueError: On parse errors or atom-count mismatch with *frame*.
        """
        import molrs.io

        try:
            loaded = molrs.io.read_amber_inpcrd(str(self._path))
        except OSError as exc:
            raise ValueError(str(exc)) from exc

        if frame is None or "atoms" not in frame:
            return loaded

        n_atoms = loaded["atoms"].nrows
        if frame["atoms"].nrows != n_atoms:
            raise ValueError(
                f"atoms block has {frame['atoms'].nrows} rows, but inpcrd has {n_atoms}"
            )

        atoms = frame["atoms"]
        src = loaded["atoms"]
        atoms["x"] = src["x"]
        atoms["y"] = src["y"]
        atoms["z"] = src["z"]
        if "vel" in src:
            atoms["vel"] = src["vel"]

        frame.box = loaded.box
        # Merge meta (title / timestep) without dropping other keys.
        frame.meta = {**frame.meta, **loaded.meta}
        return frame
