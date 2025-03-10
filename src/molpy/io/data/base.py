from abc import ABC, abstractmethod
from pathlib import Path
import molpy as mp

class FileHandler(ABC):

    def __init__(self, path: Path, *args, **kwargs):
        self._path = Path(path)
        self._file = open(self._path, *args, **kwargs)

    @abstractmethod
    def __enter__(self):
        ...

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        ...

class DataReader(FileHandler):

    def __init__(self, path: Path, system: mp.System | None, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._system = system if system is not None else mp.System()

    def __enter__(self):
        return self.read()
    
    @abstractmethod
    def read(self) -> mp.System:
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        ...


class DataWriter(FileHandler):

    def __init__(self, path: Path, system: mp.System | None, *args, **kwargs):
        super().__init__(path, *args, **kwargs)
        self._system = system if system is not None else mp.System()

    def __enter__(self):
        return self._system

    @abstractmethod
    def write(self):
        ...

    def __exit__(self, exc_type, exc_value, traceback):

        self.write()

        return super().__exit__(exc_type, exc_value, traceback)