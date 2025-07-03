"""System classes for organizing molecular data.

This module defines two types of molecular systems:
- FrameSystem: Uses Frame/Block API for columnar data storage
- AtomisticSystem: Uses Atomistic API for object graph storage
"""

from struct import Struct
import numpy as np

from .frame import Frame
from .forcefield import ForceField
from .box import Box
from .wrapper import Wrapper

class Systemic(Wrapper):

    def __init__(self, wrapped, box: Box | None, forcefield: ForceField | None):
        super().__init__(wrapped)
        self._box = box if box is not None else Box()
        self._forcefield = forcefield if forcefield is not None else ForceField()

    @property   
    def forcefield(self) -> ForceField:
        """Return the force field associated with this system."""
        return self._forcefield
    
    @property
    def box(self) -> Box:
        """Return the simulation box of this system."""
        return self._box


class FrameSystem(Systemic):

    def __init__(self, frame: Frame | None = None, box: Box | None = None, forcefield: ForceField | None = None):
        frame = frame if frame is not None else Frame()
        super().__init__(frame, box, forcefield)
        

class StructSystem(Systemic):
    
    def __init__(self, struct: Struct, box: Box, forcefield: ForceField):
        super().__init__(struct, box, forcefield)