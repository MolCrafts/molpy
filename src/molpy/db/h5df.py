import h5py
import numpy as np
from pathlib import Path
from typing import Union
from nesteddict import ArrayDict
import molpy as mp

def find_index_by_step(group, step: int) -> int|None:
    """
    Find the index of a specific step in the HDF5 group.

    Args:
        group (h5py.Group): The HDF5 group to search.
        step (int): The step to find.

    Returns:
        int|None: The index of the step, or None if not found.
    """
    result = np.where(group["step"][:] == step)[0]
    if result.size == 0:
        return None
    return result[0]
    

class H5DFAdapter:
    """
    H5DFAdapter manages the lifecycle of an HDF5 file, handling opening and closing.
    """

    def __init__(self, path: Union[str, Path], mode: str = "r"):
        """
        Initialize the H5DFAdapter.

        Args:
            path (Union[str, Path]): Path to the HDF5 file.
            mode (str): File access mode ('r', 'r+', 'w', etc.). Defaults to 'r'.
        """
        self._path = Path(path)
        self._mode = mode
        self._file = None

    def __enter__(self):
        self._file = h5py.File(self._path, self._mode)
        return H5DFProxy(self._file)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self._file.close()


class H5DFProxy:
    """
    H5DFProxy provides high-level operations for interacting with an HDF5 file.
    """

    def __init__(self, file: h5py.File):
        """
        Initialize the H5DFProxy with an h5py.File object.

        Args:
            file (h5py.File): The HDF5 file object to manage.
        """
        self._file = file

    @property
    def author(self):
        """
        Retrieve the author attribute from the HDF5 file.

        Returns:
            str: The author's name, or 'Unknown' if not set.
        """
        return bytes(self._file["h5md/author"].attrs.get("name", b"Unknown")).decode("utf-8")

    @property
    def creator(self):
        """
        Retrieve the creator attribute from the HDF5 file.

        Returns:
            str: The creator's name, or 'Unknown' if not set.
        """
        return bytes(self._file["h5md/creator"].attrs.get("name", b"Unknown")).decode("utf-8")

    def get_frame(self, timestep: int, group_name="all", molecule_name="") -> mp.Frame:
        """
        Get a frame from the HDF5 database.

        Args:
            timestep (int): The timestep of the frame to retrieve.
            group_name (str): The name of the group to retrieve the frame from. Defaults to 'all'.
            molecule_name (str): The name of the molecules group to retrieve. Defaults to ''.
        Returns:
            mp.Frame: The frame at the specified timestep.
        """
        f = self._file
        if molecule_name:
            f = self._file[f"{molecule_name}"]
        pgroup = f[f"particles/{group_name}"]
        atoms = {}
        results = {}
        for key in pgroup:
            if key == "box":  # for lammps dump...
                results["box"] = pgroup[key]["edges"]
                continue
            # time-dependent data
            if isinstance(pgroup[key], h5py.Group):  
                index = find_index_by_step(pgroup[key], timestep)
                if index is not None and "value" in pgroup[key]:
                    atoms[key] = pgroup[key]["value"][index]
            # time-independent data
            elif isinstance(pgroup[key], h5py.Dataset):  
                atoms[key] = pgroup[key][:]
        results["atoms"] = ArrayDict(atoms)
        results["step"] = timestep
        return mp.Frame(results)

    def set_frame(self, timestep: int, frame: mp.Frame, group_name="all", molecule_name=""):
        """
        Set a frame in the HDF5 database.

        Args:
            timestep (int): The timestep of the frame to set.
            frame (mp.Frame): The frame to set.
            molecule_name (str): The name of the molecules group to set the frame in. Defaults to ''.
        """
        f = self._file
        if molecule_name:
            f = self._file[f"{molecule_name}"]
        particles_group = f[f"particles/{group_name}"]
        for key, value in frame["atoms"].items():
            if key in particles_group:
                dataset = particles_group[key]
                index = find_index_by_step(dataset, timestep)
                if index is not None:
                    dataset["value"][index] = value
                else:
                    raise ValueError(f"Frame with timestep {timestep} does not exist.")
            else:
                raise KeyError(f"Key {key} not found in HDF5 file.")

    def add_frame(self, frame: ArrayDict, group_name="all", molecule_name=""):
        """
        Add a new frame to the HDF5 database.

        Args:
            timestep (int): The timestep of the new frame.
            frame (ArrayDict): The frame to add.
            group_name (str): The name of the group to add the frame to. Defaults to 'all'.
            molecule_name (str): The name of the molecules group to add the frame to. Defaults to ''.
        """
        f = self._file
        if molecule_name:
            if molecule_name not in f:
                f.create_group(molecule_name)
            f = self._file[f"{molecule_name}"]

        if "particles" not in f:
            f.create_group("particles")

        if group_name not in f["particles"]:
            f.create_group(f"particles/{group_name}")
        particles_group = f[f"particles/{group_name}"]

        for key, value in frame["atoms"].items():
            if key not in particles_group:
                particles_group.create_dataset(key, data=value, maxshape=(None, *value.shape[1:]), chunks=True)
            
    def insert_frame(self, timestep: int, frame: ArrayDict, group_name="all", molecule_name=""):
        """
        Insert a new frame into the HDF5 database.

        Args:
            timestep (int): The timestep of the new frame.
            frame (ArrayDict): The frame to insert.
            group_name (str): The name of the group to insert the frame into. Defaults to 'all'.
            molecule_name (str): The name of the molecules group to insert the frame into. Defaults to ''.
        """
        f = self._file
        if molecule_name:
            if molecule_name not in f:
                f.create_group(molecule_name)
            f = self._file[f"{molecule_name}"]

        if group_name not in f:
            f.create_group(group_name)
        particles_group = f[f"particles/{group_name}"]
        for key, value in frame["atoms"].items():
            if key not in particles_group:
                particles_group.create_dataset(key, data=value, maxshape=(None, *value.shape[1:]), chunks=True)

    def add_trajectory(self, trajectory: mp.Trajectory, group_name="all", molecule_name=""):
        """
        Add a trajectory to the HDF5 database.

        Args:
            trajectory (mp.Trajectory): The trajectory to add.
            group_name (str): The name of the group to add the trajectory to. Defaults to 'all'.
            molecule_name (str): The name of the molecules group to add the trajectory to. Defaults to ''.
        """
        steps = trajectory.timesteps
        frames = trajectory.frames

        probe_