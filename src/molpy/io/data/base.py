from abc import ABC, abstractmethod
from pathlib import Path
import molpy as mp


class DataReader:

    def __init__(self, path: Path, *args, **kwargs):
        self._path = Path(path)

    def __enter__(self):
        return self
    
    @abstractmethod
    def read(self, frame: mp.Frame) -> mp.Frame:
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


class DataWriter:

    def __init__(self, path: Path, *args, **kwargs):
        self._path = Path(path)

    def __enter__(self):
        return self

    @abstractmethod
    def write(self, frame: mp.Frame):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...