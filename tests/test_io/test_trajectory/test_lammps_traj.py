import pytest
from molpy.io.trajectory.lammps import LammpsTrajectoryReader
from pathlib import Path


def test_read_frame_fcc_orth(test_data_path):
    reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_orth.dump")
    frame = reader.read_frame(0)

    assert frame["timestep"] == 0
    assert frame.box.matrix.shape == (3, 3)  # type: ignore
    assert len(frame["atoms"].data_vars) > 0  # Check atoms exist
    assert len(frame["atoms"].data_vars) == 5
    expected_box_matrix = [
        [1.5377619196572583, 0, 0],
        [0, 1.5377619196572583, 0],
        [0, 0, 1.5377619196572583],
    ]
    assert (frame["box"].matrix == expected_box_matrix).all()


def test_read_frame_fcc_tric(test_data_path):
    reader = LammpsTrajectoryReader(test_data_path / "trajectory/lammps/fcc_tric.dump")
    frame = reader.read_frame(0)

    assert frame["timestep"] == 0
    assert frame.box.matrix.shape == (3, 3)  # type: ignore
    assert len(frame["atoms"].data_vars) > 0  # Check atoms exist 
    assert len(frame["atoms"].data_vars) == 5

    lx = 1.1922736280710971e+02 - 5.1509317718250820e+01
    ly = 1.1937795684789540e+02 - 5.3089558100317952e+01
    lz = 1.1924193400378044e+02 - 5.2978065996127228e+01
    xy = -2.2480556300586776e-01
    xz = -1.5685139115469910e+00
    yz = 8.7514948257390660e-02
    assert frame["box"].matrix[0, 0] == lx
    assert frame["box"].matrix[1, 1] == ly
    assert frame["box"].matrix[2, 2] == lz
    assert frame["box"].matrix[0, 1] == xy
    assert frame["box"].matrix[0, 2] == xz
    assert frame["box"].matrix[1, 2] == yz

def test_read_multi_traj(test_data_path):
    reader = LammpsTrajectoryReader([test_data_path/"trajectory/lammps/fcc_orth.dump", test_data_path/"trajectory/lammps/fcc_tric.dump"])

    assert reader.n_frames == 2
    frame0 = reader.read_frame(0)
    assert frame0["timestep"] == 0
    assert len(frame0["atoms"].data_vars) > 0  # Check atoms exist
    frame1 = reader.read_frame(1)
    assert frame1["timestep"] == 0
    assert len(frame1["atoms"].data_vars) > 0  # Check atoms exist
