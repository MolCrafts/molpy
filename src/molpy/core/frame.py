from molpy.core.space import Box
from .struct import ArrayDict
import numpy as np


class Frame:

    def __init__(self):
        self.box = Box()

    def __getitem__(self, key):
        if not hasattr(self, key):
             setattr(self, key, ArrayDict())
        return getattr(self, key)
        
    