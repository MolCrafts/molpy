# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-03-23
# version: 0.0.1
import numpy as np
from typing import Collection
from copy import deepcopy
from molpy.core.struct import ArrayDict
from molpy.core.topology import Topology
from molpy.core.space import Free, OrthogonalBox, RestrictTriclinicBox

class Frame:

    def __init__(self, name: str = ""):

        self.name = name

        self._atoms = ArrayDict()
        self._bonds = ArrayDict()
        self._angles = ArrayDict()
        self._dihedrals = ArrayDict()
        self._paris = ArrayDict()

        self._box = Free()

    def __repr__(self) -> str:
        return f"<Frame {self.name}: {self.n_atoms} atoms>"

    @classmethod
    def join(cls, structs: Collection["Frame"]) -> "Frame":
        # Type consistency check
        assert all(isinstance(struct, cls) for struct in structs), TypeError(
            "All structs must be of the same type"
        )
        # Create a new struct
        struct = cls()
        for s in structs:
            struct.union(s)
        return struct

    def union(self, other: "Frame") -> "Frame":
        """
        union two structs and return self

        Args:
            other (Frame): the other struct

        Returns:
            Frame: this struct
        """
        self._atoms.concat(other.atoms)
        self._bonds.concat(other.bonds)
        self._angles.concat(other.angles)
        self._dihedrals.concat(other.dihedrals)
        return self
    
    def merge(self, frame: "Frame") -> "Frame":
        """
        merge two frames and return self

        Args:
            frame (Frame): the other frame

        Returns:
            Frame: this frame
        """
        self._atoms.concat(frame.atoms)
        self._bonds.concat(frame.bonds)
        self._angles.concat(frame.angles)
        self._dihedrals.concat(frame.dihedrals)
        return self

    @property
    def n_atoms(self):
        return self._atoms.size

    @property
    def n_bonds(self):
        return self._bonds.size

    @property
    def n_angles(self):
        return self._angles.size

    @property
    def n_dihedrals(self):
        return self._dihedrals.size

    @property
    def atoms(self) -> ArrayDict:
        return self._atoms

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals

    def copy(self):
        return deepcopy(self)

    def __call__(self) -> "Frame":
        return self.copy()
    