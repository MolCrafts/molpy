from ..base import Potential
import numpy as np
import molpy as mp

class AnglePotential(Potential):

    @staticmethod
    def or_frame(func):
        def wrapper(self, frame, *args, **kwargs):
            if isinstance(frame, mp.Frame):
                r = frame['atoms'][['x', 'y', 'z']].to_numpy()
                angle_idx = frame['angles'][['i', 'j', 'k']].to_numpy()
                angle_types = frame['angles']['type'].to_numpy()
                return func(self, r, angle_idx, angle_types, *args, **kwargs)
            return func(self, frame, *args, **kwargs)
        return wrapper
