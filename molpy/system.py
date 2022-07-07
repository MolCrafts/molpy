# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-22
# version: 0.0.1

from molpy.box import Box
from molpy.atoms import Atoms
import numpy as np
from molpy.io import Readers

class System:

    def __init__(self, comment:str=''):

        self.comment = comment
        self.box = None
        self.forcefield = None
        self.atoms:Atoms = Atoms()

        self._traj = None

    # ---= data load interface =---

    def load_data(self, dataFile:str, format:str, method='replace'):
        """load data from file
        Args:
            dataFile (str): path of data file
            format (str): format of data file
            method (str, optional): how to load the data. Defaults to 'replace'.
        """
        data_reader = Readers['DataReaders'][format](dataFile)
        atoms = data_reader.get_atoms()
        if self.atoms is None:
            self.atoms = atoms

        if method == 'replace':
            self.atoms.update(atoms, isBond=False)


    def load_traj(self, trajFile:str, format:str):
        self._traj = Readers['TrajReaders'][format](trajFile)
        return self._traj

    def select_frame(self, nFrame:int, method='replace'):
        frame = self._traj.get_frame(nFrame)
        atoms = self._traj.get_atoms()
        if method == 'replace':
            self.atoms.update(atoms)
        
        box = self._traj.get_box()
        self.set_box(box['Lx'], box['Ly'], box['Lz'], box.get('xy', 0), box.get('xz', 0), box.get('yx', 0), box.get('is2D', False))

    def sample(self, start, stop, interval, method='replace')->int:
        """sample from the trajectory, and inplace update the system. This method is a generator that will return the number of frame.

        Usage:
            # sample every 10 frames
            for nframe in system.sample(0, 100, 10):
                atoms = system.atoms
                positions = atoms['position']

        Args:
            start (int): start frame
            stop (int): stop frame
            interval (int): sample interval
            method (str, optional): how to load and update data. Defaults to 'replace'.

        Yields:
            Iterator[int]: current number of frame
        """
        frame = np.arange(self.nframes)[start:stop:interval]
        for f in frame:
            self.select_frame(f)
            yield f    

    @property
    def nframes(self)->int:
        return self._traj.nframes    

    # ---= forcefield interface =---

    def def_atomtype(self):
        pass

    def def_bondtype(self):
        pass

    def def_angletype(self):
        pass

    def def_dihedraltype(self):
        pass

    # ---= box interface =---

    def get_box(self):
        pass

    def set_box(self, Lx, Ly, Lz=0, xy=0, xz=0, yx=0, is2D=False):
        
        self.box = Box(Lx, Ly, Lz, xy, xz, yx, is2D)    

    # ---= atoms interface =---
    def add_atoms(self):
        pass

    def add_edges(self):
        pass

    def get_angles(self):
        pass

