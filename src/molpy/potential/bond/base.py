from ..base import Potential
import numpy as np
import molpy as mp

class BondPotential(Potential):

    @staticmethod
    def or_frame(func):
        def wrapper(self, frame, *args, **kwargs):
            if isinstance(frame, mp.Frame):
                r = frame['atoms'][['x', 'y', 'z']].to_numpy()
                bond_idx = frame['bonds'][['i', 'j']].to_numpy()
                bond_types = frame['bonds']['type'].to_numpy()
                return func(self, r, bond_idx, bond_types, *args, **kwargs)
            return func(self, frame, *args, **kwargs)
        return wrapper

    @or_frame
    def calc_energy(self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray) -> np.ndarray:
        ...

    @or_frame
    def calc_force(self, r: np.ndarray, bond_idx: np.ndarray, bond_types: np.ndarray) -> np.ndarray:
        ...