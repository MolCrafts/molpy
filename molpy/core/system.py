# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import mdtraj as md

from .topology import Topology
from .trajectory import Trajectory
from .frame import Frame, ConcreateFrame
from .box import Box
from .item import Bond

class System:

    def __init__(self, ):

        self._forcefield = None
        self._trajectory = None
        
    def load(self, filename, topo=None):

        traj = md.load(filename, top=topo)
        # 
        mp_topo = Topology()
        for bond in traj.topology.bonds:
            bond = Bond(bond[0].index, bond[1].index, type=bond.type)
            mp_topo.add_bond(bond)

        _trajectory = Trajectory()
        for one_traj in traj:
            # 
            mp_box = Box(*one_traj.unitcell_lengths[0], 0, 0, 0)  # TODO: convert angle to factor
            
            # 
            mp_frame = ConcreateFrame(mp_box, mp_topo, one_traj.time)
            mp_frame.add_property('xyz', one_traj.xyz[0])
            for k, v in one_traj.properties.items():
                mp_frame.add_property(k, v[0])

            
            _trajectory.append(mp_frame)
            self._trajectory = _trajectory
        return _trajectory

    def __iter__(self):
        return iter(self._trajectory)
        

    def iter_load(self, filename, topo=None, chunk=100):

        traj = md.iterload(filename, chunk, topo=topo)

    def sample(self, filename, start, stop, step, topo=None):

        pass
