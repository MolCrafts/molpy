from ..io.log.lammps import LammpsLog
from ..io.script import Script
from .base import Engine

class LammpsScript(Script):
    pass

class Lammps(Engine):
    
    def __init__(self, work_dir:str):
        super().__init__()
        self._work_dir = work_dir
        self._deps = []

    def run(self, command:str, block:bool=False):
        self.check_deps()
        super().run(command, block)

    def get_traj(self, path:str, format:str="lammpstrj"):
        return super().get_traj(path, format)

    def get_log(self, path:str):
        return super().get_log(path, 'lammps')
