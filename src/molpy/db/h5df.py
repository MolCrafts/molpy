from .base import BaseDatabaseAdapter
from pathlib import Path
import h5py

class H5DF(BaseDatabaseAdapter):
    """
    H5DF is a subclass of BaseDatabaseAdapter that provides an interface for
    interacting with HDF5 databases.
    """

    def __init__(self, path: str|Path, mode: str = "r"):
        self._path = Path(path)
        self._f = h5py.File(self._path, mode)

    @property
    def author(self):
        return bytes(self._f['h5md']['author'].attrs['name']).decode('utf-8')

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._f.close()