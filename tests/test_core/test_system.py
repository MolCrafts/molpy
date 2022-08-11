# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

import mdtraj as md
import molpy
from molpy.core.item import Bond
from molpy.core.system import System, convert_mdtraj_to_molpy

def test_convert_mdtraj_to_molpy():

    traj = md.load('tests/test_io/data/lammps.dump', top='tests/test_io/data/lammps.data')
    trajectory = convert_mdtraj_to_molpy(traj)

    print(trajectory[0]._atoms)

    assert traj.n_frames == trajectory.n_frames
    assert traj.n_atoms == trajectory[0].n_atoms
    