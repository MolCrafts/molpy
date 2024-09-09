from molpy.core.space import Box
from .struct import Struct, ArrayDict
import numpy as np


class Frame(Struct):

    def __init__(self):
        super().__init__()
        self.box = Box()

    @classmethod
    def union(cls, *frames: "Frame") -> "Frame":
        frame = Frame()
        for f in frames:
            frame.box = max(frame.box, f.box, key=lambda x: x.volume)

        structs = {}
        for key in frames[0].props:
            for f in frames:

                if key not in structs:
                    structs[key] = [getattr(f, key)]
                else:
                    structs[key].append(getattr(f, key))

        for key, values in structs.items():
            frame[key] = ArrayDict.union(*values)
        return frame

    def __setitem__(self, key, value):
        if key == "box":
            self.box = value
        else:
            super().__setitem__(key, value)
