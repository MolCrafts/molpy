# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import mdtraj as md
import molpy as mp
import numpy as np

class TestSystem:

    def test_load_lammps(self):

        data_path = 'tests/test_io/data/lammps.data'
        
        system = mp.System()
        system.load_data(data_path, 'lammps')

        state = system.state

        
