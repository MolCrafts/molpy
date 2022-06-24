# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-12
# version: 0.0.1

from typing import Dict
from molpy.atoms import Atoms

class TrajReader:
    
    def get_frame(self):
        raise NotImplementedError()

    @property
    def nFrames(self):
        raise NotImplementedError()
    
    @staticmethod
    def parse():
        """ parse one frame of traj file
            interface:
            {
                'timestep': int,
                'n_atoms': int,
                'box': {Lx: int, Ly: int, Lz: int, xy: int, xz: int, yx: int, is2D: bool},
                'atoms': {
                    key: value
                },
                'bonds: {
                    'id': ,
                    'type': ,
                    'connect': 
                }
                
            }        
        """
        raise NotImplementedError()
    
class DataReader:
    
    def get_data(self):
        """ parse data file

        Raises:
            NotImplementedError: if not implemented in the derived class
        """
        raise NotImplementedError()


def data2atoms(name:str, data:Dict, withTopo:bool=False, out:Atoms=None):
    
    if out is None:
        out = Atoms(withTopo=False)
        for key, value in data['atoms'].items():
            out.set_atom_values(key, value)
        
    return out
    
    