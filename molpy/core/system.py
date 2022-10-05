# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import mdtraj as md

from molpy.core.nblist import NeighborList

from .topology import Topology
from .trajectory import Trajectory
from .frame import DynamicFrame, Frame, StaticFrame
from .box import Box
from .item import Atom, Bond
from molpy.io import Readers
import numpy as np
from .forcefield import ForceField

class System:

    def __init__(self, ):

        self.forcefield = ForceField()
        self.trajectory = None
        self.frame:Frame = DynamicFrame()
        
    def load(self, filename, topo=None, *args, **kwargs):

        traj = md.load(filename, top=topo, *args, **kwargs)
        # 
        mp_topo = Topology()
        for bond in traj.topology.bonds:
            bond = Bond(bond[0].index, bond[1].index, type=bond.type)
            mp_topo.add_bond(bond)

        trajectory = Trajectory()
        for one_traj in traj:
            # 
            mp_box = Box(*one_traj.unitcell_lengths[0], 0, 0, 0)  # TODO: convert angle to factor
            
            # 
            mp_frame = StaticFrame(mp_box, mp_topo, one_traj.time)
            mp_frame.add_property('xyz', one_traj.xyz[0])
            for k, v in one_traj.properties.items():
                mp_frame.add_property(k, v[0])

            trajectory.append(mp_frame)
            self.trajectory = trajectory

        return trajectory

    def iterload(self, filename, topo=None, chunk=100, stride=0, skip=0, *args, **kwargs):
        
        traj = md.iterload(filename, top=topo, chunk=chunk, stride=stride, skip=skip, *args, **kwargs)

        for one_traj in traj:

            trajectory = Trajectory()
            mp_topo = Topology()
            for bond in one_traj.topology.bonds:
                bond = Bond(bond[0].index, bond[1].index, type=bond.type)
                mp_topo.add_bond(bond)
            for i in range(one_traj.n_frames):

            # 
                mp_box = Box(*one_traj.unitcell_lengths[i], 0, 0, 0)  # TODO: convert angle to factor
            
            # 
                mp_frame = StaticFrame(mp_box, mp_topo, one_traj.time)
                mp_frame.add_property('xyz', one_traj.xyz[i])
                for k, v in one_traj.properties.items():
                    mp_frame.add_property(k, v[i])

                trajectory.append(mp_frame)
            self.trajectory = trajectory
            yield trajectory

    def __iter__(self):
        return iter(self.trajectory)

    def load_traj(self, filename, format, ):

        self._traj = Readers['TrajReaders'][format](filename)
        self.n_frames = self._traj.n_frames

    def get_frame(self, index):
        frame = self._traj.get_one_frame(index)
        return frame

    def sample(self, start, stop, step):

        frame = np.arange(start, stop, step)
        for i in frame:
            yield self.get_frame(i)

    def set_box(self, Lx, Ly, Lz, xy=0, xz=0, yz=0):

        box = Box(Lx, Ly, Lz, xy, xz, yz)
        self.frame.box = box
        return box

    def set_atoms(self, **kwargs):

        for i in zip(*kwargs.values()):
            atom = Atom({k: v for k, v in zip(kwargs.keys(), i)})
            self.frame.add_atom(atom)
