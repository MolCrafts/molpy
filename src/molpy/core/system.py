import numpy as np

from .frame import Frame
from .forcefield import ForceField
from .box import Box
from .atomistic import Struct, AtomicStructure


class System:

    def __init__(self):
        self._forcefield = ForceField()
        self._box = Box()
        self._structs = []

    @property
    def forcefield(self):
        return self._forcefield

    @forcefield.setter
    def forcefield(self, value):
        self._forcefield = value

    @property
    def structs(self):
        return self._structs

    def set_forcefield(self, forcefield: ForceField):
        """Set the forcefield for the system."""
        self.forcefield = forcefield

    def get_forcefield(self):
        """Get the forcefield for the system."""
        if self._forcefield is None:
            raise ValueError("Forcefield not set.")
        return self._forcefield

    def def_box(self, matrix, pbc=np.ones(3, dtype=bool), origin=np.zeros(3)):
        self._box = Box(matrix=matrix, pbc=pbc, origin=origin)

    def add_struct(self, struct: Struct):
        self._structs.append(struct)

    def to_dict(self):
        """Convert system to a dictionary representation."""
        # Convert structures to list of dicts
        struct_dicts = []
        for struct in self._structs:
            if hasattr(struct, 'to_dict'):
                struct_dicts.append(struct.to_dict())
            else:
                # Fallback for basic Struct objects
                struct_dict = dict(struct) if hasattr(struct, 'keys') else {'name': getattr(struct, 'name', 'unknown')}
                struct_dict['__class__'] = struct.__class__.__name__
                struct_dicts.append(struct_dict)
        
        # Handle forcefield
        if hasattr(self._forcefield, 'to_dict'):
            ff_dict = self._forcefield.to_dict()
        else:
            ff_dict = {'name': getattr(self._forcefield, 'name', 'unknown')}
        
        # Handle box
        if hasattr(self._box, 'to_dict'):
            box_dict = self._box.to_dict()
        else:
            box_dict = {
                'matrix': self._box.matrix.tolist() if hasattr(self._box, 'matrix') else [[1,0,0],[0,1,0],[0,0,1]],
                'pbc': self._box.pbc.tolist() if hasattr(self._box, 'pbc') else [True, True, True],
                'origin': self._box.origin.tolist() if hasattr(self._box, 'origin') else [0.0, 0.0, 0.0]
            }
        
        return {
            'forcefield': ff_dict,
            'box': box_dict,
            'structures': struct_dicts,
            'n_structures': len(self._structs)
        }

    def to_frame(self, factory=None):
        """Convert system to a Frame."""
        if not self._structs:
            # Empty system case
            frame = Frame()
        else:
            # Combine all structures into one AtomicStructure
            combined_struct = AtomicStructure("combined_system")
            for struct in self._structs:
                combined_struct.add_struct(struct)
            
            # Convert to frame
            frame = combined_struct.to_frame()
        
        frame.box = self._box
        frame.forcefield = self._forcefield
        return frame

