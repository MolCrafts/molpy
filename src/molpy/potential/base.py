import numpy as np
import molpy as mp

class Potential:
       
    def calc_energy(self, r: np.ndarray) -> np.ndarray:
        pass
    
    def calc_force(self, r: np.ndarray) -> np.ndarray:
        pass

    def get_energy(self, frame: mp.Frame) -> np.ndarray:
        pass
