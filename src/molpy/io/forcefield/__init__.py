import molpy as mp
from .lammps import LAMMPSForceField

def load_forcefield(filenames:str|list[str], format:str='', forcefield:mp.ForceField|None=None, ):
    if isinstance(filenames, str):
        filenames = [filenames]
    if format == 'lammps':
        return LAMMPSForceField(filenames, forcefield).forcefield