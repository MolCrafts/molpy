from abc import ABC, abstractmethod
from pathlib import Path
import molpy as mp


class DataReader:

    def __init__(self, path: Path, frame: mp.Frame | None, *args, **kwargs):
        self._path = Path(path)
        self._frame = frame if frame is not None else mp.Frame()

    def __enter__(self):
        return self.read()
    
    @abstractmethod
    def read(self) -> mp.Frame:
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


class DataWriter:

    def __init__(self, path: Path, frame: mp.Frame | None, *args, **kwargs):
        self._path = Path(path)
        self._frame = frame if frame is not None else mp.Frame()

    def __enter__(self):
        return self._frame

    @abstractmethod
    def write(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):

        self.write()

        return super().__exit__(exc_type, exc_value, traceback)