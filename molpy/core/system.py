# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from .frame import DynamicFrame, Frame, StaticFrame
from .box import Box
from molpy.io import Readers
import numpy as np
from .forcefield import Forcefield

class System:
    """
    Top-level class
    """
    def __init__(self, ):

        self.forcefield = Forcefield()
        self.trajectory = None
        self.frame = DynamicFrame()

    @property
    def state(self):
        """
        get the state of the system, which includes the atomic and topological data and box information

        Returns
        -------
        StaticFrame
            state of system
        """
        return self.frame.to_static()

    def __iter__(self):
        return iter(self.trajectory)

    def load_traj(self, filename:str, format:str, ):
        """
        load trajectory from file

        Parameters
        ----------
        filename : str
            the path of file
        format : str
            'lammps'
        """
        trajReader = Readers['TrajReaders'][format]
        self._traj = trajReader(filename)
        self.n_frames = self._traj.n_frames

    def load_data(self, filename:str, format:str, ):
        """
        load data from file

        Parameters
        ----------
        filename : str
            the path of file
        format : str
            'lammps'
        """
        data_reader = Readers['DataReaders'][format]
        with data_reader(filename) as f:
            self.frame = f.get_data()

    def get_frame_by_index(self, index:int)->StaticFrame:
        """
        switch to the frame by index

        Parameters
        ----------
        index : int
            index of the frame

        Returns
        -------
        StaticFrame
            one frame of system
        """
        frame = self._traj.get_one_frame(index)
        return frame

    def sample(self, start, stop, step):
        """
        sample alone the trajectory. This method must invoke after load_traj

        Parameters
        ----------
        start : int
            start index of trajectory
        stop : int
            stop index of trajectory
        step : int
            interval of sampling

        Yields
        ------
        StaticFrame
            one frame of system
        """
        frame = np.arange(start, stop, step)
        for i in frame:
            yield self.get_frame_by_index(i)

    def set_box(self, Lx, Ly, Lz, xy=0, xz=0, yz=0):
        """
        setup Box information for current frame of system

        Parameters
        ----------
        Lx : int
            length of x
        Ly : int
            length of y
        Lz : int
            length of z
        xy : int, optional
            tilte factor, by default 0
        xz : int, optional
            tilte factor, by default 0
        yz : int, optional
            tilte factor, by default 0

        Returns
        -------
        Box
            Box object
        """
        box = Box(Lx, Ly, Lz, xy, xz, yz)
        self.frame.box = box
        return box

    @property
    def box(self):
        return self.frame.box

    def add_atoms(self, **kwargs):
        """
        add atoms to current frame of system
        """
        for i in zip(*kwargs.values()):
            self.frame.add_atom(**{k: v for k, v in zip(kwargs.keys(), i)})
