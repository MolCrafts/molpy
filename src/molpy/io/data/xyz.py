import re
import shlex
import numpy as np

from molpy.core import Block, Frame, Box

from .base import DataReader, DataWriter


class XYZReader(DataReader):
    """
    Parse an XYZ file (single model) into an :class:`Frame`.

    Format
    ------
        1. integer `N`  - number of atoms
        2. comment line - stored as   frame["comment"]
        3. N lines: `symbol  x  y  z`
    """

    def read(self, frame: Frame | None = None) -> Frame:
        """
        Parameters
        ----------
        frame
            Optional frame to populate; if *None*, a new one is created.

        Returns
        -------
        Frame
            Frame with:
              * block ``"atoms"``:
                  - ``element``   → (N,)  <U3   array
                  - ``xyz``       → (N,3) float array (Å)
              * metadata ``comment`` (str)
        """
        # --- collect lines ------------------------------------------------
        lines: list[str] = self.read_lines()
        if len(lines) < 2:
            raise ValueError("XYZ file too short")

        natoms = int(lines[0])
        if len(lines) < natoms + 2:
            raise ValueError("XYZ record truncated")

        comment = lines[1]
        records = lines[2 : 2 + natoms]

        # --- parse atom table --------------------------------------------
        symbols: list[str] = []
        coords: list[tuple[float, float, float]] = []
        for rec in records:
            parts = rec.split()
            if len(parts) < 4:
                raise ValueError(f"Bad XYZ line: {rec!r}")
            symbols.append(parts[0])
            x, y, z = parts[1:4]
            coords.append((float(x), float(y), float(z)))

        # --- build / update frame ----------------------------------------
        frame = frame or Frame()
        self._parse_xyz_comment(frame, comment)
        atoms_blk = Block()
        atoms_blk["element"] = np.array(symbols, dtype="U3")
        atoms_blk["xyz"] = np.asarray(coords, dtype=float)

        frame["atoms"] = atoms_blk
        return frame

    def _parse_xyz_comment(self, frame: Frame, comment: str):
        """
        Parse an extended XYZ comment line into a dictionary of key-value pairs.

        Args:
            comment (str): The comment line from an XYZ file.

        Returns:
            dict: Parsed key-value pairs.
        """
        result: dict = {}

        for token in shlex.split(comment):
            if "=" in token:
                key, value = token.split("=", 1)
                if key == "Properties":
                    parts = value.split(":")
                    triples = [(parts[i], parts[i + 1], int(parts[i + 2])) for i in range(0, len(parts), 3)]
                    result[key] = triples
                else:
                    result[key] = value.strip('"')
            else:
                # 单独出现的 key，当作 bool flag
                result[token] = True

        if "Lattice" in result:
            frame.box = Box(np.array([float(x) for x in result.pop("Lattice").split()]).reshape(3, 3))
        print(result)
        frame.metadata.update(result)

        