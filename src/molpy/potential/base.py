import numpy as np
import molpy as mp

class Potential:
       
    def calc_energy(self, r: np.ndarray) -> np.ndarray:
        pass
    
    def calc_force(self, r: np.ndarray) -> np.ndarray:
        pass

    def get_energy(self, frame: mp.Frame) -> np.ndarray:
        pass

class PotentialDict(dict, Potential):

    def calc_energy(self, r_or_frame):
        
        energy = 0
        for pot in self.values():
            energy += pot.calc_energy(r_or_frame)

        return energy
    
    def calc_force(self, r):
        forces = 0
        for pot in self.values():
            forces += pot.calc_force(r)
        return forces