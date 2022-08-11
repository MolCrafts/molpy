# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import mdtraj as md

from .topology import Topology
from .trajectory import Trajectory
from .frame import Frame
from .box import Box

def convert_mdtraj_to_molpy(traj)->Trajectory:

    trajectory = Trajectory()
    topology = Topology()
    # load topology
    _topo = traj.topology
    for bond in _topo.bonds:
        topology.add_bond(bond[0].index, bond[1].index, type=bond.type)

    unitcell_lengths = traj.unitcell_lengths
    unitcell_angles = traj.unitcell_angles

    # get per-frame data
    for i in range(traj.n_frames):
        frame = Frame(topology, traj.time[i])
        xyz = traj.xyz[i]
        n_atoms = xyz.shape[0]
        for n in range(n_atoms):
            frame.add_atom(xyz=xyz[n], **{header: value[i][n] for header, value in traj.properties.items()})

        trajectory.add_frame(frame)

        box = Box(*unitcell_lengths[i], *unitcell_angles[i])

    

    return trajectory

class System:

    def __init__(self, ):

        self._forcefield = None
        self._trajectory = None
        
    def load(self, filename, topo=None):

        traj = md.load(filename, top=topo)

        self._trajectory = convert_mdtraj_to_molpy(traj)