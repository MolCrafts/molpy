from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Iterator
import molpy as mp


class DataReader:

    def __init__(self, path: Path, *args, **kwargs):
        self._path = Path(path)
        self._file = None

    def __enter__(self):
        self._file = open(self._path, 'r')
        return self
    
    def read_lines(self) -> List[str]:
        """Read all lines from the file, stripping whitespace and filtering empty lines."""
        if self._file is None:
            with open(self._path, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        else:
            self._file.seek(0)  # Reset file pointer
            return [line.strip() for line in self._file if line.strip()]
    
    def read_lines_iterator(self) -> Iterator[str]:
        """Return an iterator over non-empty lines from the file."""
        if self._file is None:
            with open(self._path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        else:
            self._file.seek(0)  # Reset file pointer
            for line in self._file:
                line = line.strip()
                if line:
                    yield line

    @abstractmethod
    def read(self, frame: mp.Frame) -> mp.Frame:
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self._file.close()
            self._file = None


class DataWriter:

    def __init__(self, path: Path, *args, **kwargs):
        self._path = Path(path)
        self._file = None

    def __enter__(self):
        self._file = open(self._path, 'w')
        return self

    @abstractmethod
    def write(self, frame: mp.Frame):
        ...

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file:
            self._file.close()
            self._file = None