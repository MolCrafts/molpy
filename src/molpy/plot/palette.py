import csv
import numpy as np
from collections import namedtuple
from pathlib import Path

__all__ = ["default_palette"]

class Palette:
    """
    default palette for plot
    """

    def __init__(self, style: str):
        self.style = style
        self._atoms = None
        self._box = None

    def __repr__(self):
        return f"<Palette: {self.style}>"

    @property
    def atoms(self):
        return self._atoms

    @property
    def box(self):
        return self._box


class DefaultPlatte(Palette):
    """
    default palette for plot
    """

    def __init__(self):
        super().__init__("default")
        self.set_atoms()

    def set_atoms(self):
        with open(Path(__file__).parent / "palettes/default.csv") as csvfile:
            sheet = csv.reader(csvfile)
            header = next(sheet)
            Atom = namedtuple("Atom", header)
            atoms = list(map(Atom._make, sheet))

        self._atoms = atoms

default_palette = DefaultPlatte()