import io
from collections import namedtuple
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Iterator, MutableMapping, Sequence, Union,
                    overload)

import numpy as np

if TYPE_CHECKING:
    import h5py  # type: ignore[import]
    import pyarray as pa  # type: ignore[import]

NestedKey = str | list[str]  # type_check_only

import csv
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py

class ArrayDict(MutableMapping):
    """A dictionary-like object that stores arrays."""

    _data: dict[str, np.ndarray]

    def __init__(self, source: dict = {}):
        source = {k: np.array(v) for k, v in source.items()}
        super().__setattr__("_data", source)

    @classmethod
    def from_dicts(cls, source: list[dict], include: list[str]|None = []) -> "ArrayDict":
        """Create an ArrayDict from a list of dictionaries.

        Args:
            source (list[dict]): A list of dictionaries where each dictionary represents a row.

        Returns:
            ArrayDict: An ArrayDict where keys are the dictionary keys and values are arrays of the corresponding values.
        """

        keys = set(source[0].keys())
        if include:
            keys = keys & set(include)
        data = {key: np.array([row[key] for row in source]) for key in keys}
        return cls(data)
    
    @classmethod
    def from_csv(
        cls,
        source: Path | io.StringIO,
        header: list[str] | None = None,
        seq: str = ",",
        **kwargs
    ) -> "ArrayDict":
        """Create an ArrayDict from a CSV file or CSV data with optional custom header and delimiter.

        Args:
            source (str|list[str]|io.StringIO): If str, path to the CSV file; if list of strings, CSV lines; if StringIO, CSV data.
            header (list[str]|None): Optional list of header fields.
            seq (str): Delimiter for CSV data.

        Returns:
            ArrayDict: An ArrayDict where keys are headers and values are arrays of the corresponding values.
        """
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            f = open(source, "r", encoding="utf-8")
            close_f = True
        elif isinstance(source, io.StringIO):
            f = source
            close_f = False
        else:
            raise TypeError("Unsupported source type.")

        try:
            reader = csv.DictReader(f, fieldnames=header, delimiter=seq, **kwargs)
            rows = list(reader)
        finally:
            if close_f:
                f.close()
        
        return cls.from_dicts(rows)

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    
    @overload
    def __getitem__(self, key: list[str]) -> "ArrayDict": ...
    
    @overload
    def __getitem__(self, key: slice) -> "ArrayDict": ...
    
    @overload
    def __getitem__(self, key: int) -> "ArrayDict": ...
    
    @overload
    def __getitem__(self, key: np.ndarray) -> "ArrayDict": ...

    def __getitem__(self, key: Union[str, list[str], slice, int, np.ndarray]) -> Union[np.ndarray, "ArrayDict"]:
        if isinstance(key, str):
            return self._data[key]
        elif isinstance(key, list):
            return ArrayDict({k: self._data[k] for k in key})
        elif isinstance(key, (slice, int, np.ndarray)):
            return ArrayDict({k: v[key] for k, v in self._data.items()})
        raise KeyError(f"Key {key} not supported in ArrayDict")

    def __setitem__(
        self, key: str | list[str], value: Union[np.ndarray, "ArrayDict"]
    ) -> None:
        if isinstance(key, str) and isinstance(value, (np.ndarray, list)):
            self._data[key] = value
        elif isinstance(key, list) and isinstance(value, ArrayDict):
            for k in key:
                self._data[k] = value[k]
        elif isinstance(key, (slice, int)) and isinstance(value, ArrayDict):
            for k in self._data.keys():
                self._data[k][key] = value[k]
        else:
            raise KeyError(f"set type {type(value)} to '{key}' not support in ArrayDict")

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ArrayDict):
            value = other._data
        elif isinstance(other, dict):
            value = other
        else:
            return False
        return self._data.keys() == value.keys() and all([
            np.allclose(self[k], value[k]) for k in self._data.keys()
        ])

    @property
    def array_length(self) -> int:
        """Get the length of the arrays in the ArrayDict.

        Returns:
            int: The length of the arrays.
        """
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def itertuples(self):
        """Iterate over the rows of the array dictionary.

        Returns:
            Iterator[dict]: An iterator that yields dictionaries representing each row.
        """
        row = namedtuple("Row", self.keys())
        for i in range(self.array_length):
            yield row(*[self[k][i] for k in self.keys()])

    def iterrows(self):
        for i in range(self.array_length):
            yield self[i]

    def iterarrays(self):
        """Iterate over the arrays in the array dictionary.

        Returns:
            Iterator[tuple]: An iterator that yields tuples of the form (key, array).
        """
        for i in zip(*self.values()):
            yield i

    def __repr__(self) -> str:
        return f"<ArrayDict: {' '.join(self._data.keys())}>"

    __str__ = __repr__

    @classmethod
    def concat(cls, others: Sequence["ArrayDict"]) -> "ArrayDict":
        """Concatenate two ArrayDicts.

        Args:
            other (ArrayDict): The ArrayDict to concatenate with.

        Returns:
            ArrayDict: A new ArrayDict containing the concatenated data.
        """
        data = {
            k: np.concatenate([other[k] for other in others]) for k in others[0].keys()
        }
        return cls(data)
    
    def to_dict(self, include: list[str] | None = None, exclude: list[str] | None = None) -> dict[str, np.ndarray]:

        return {
            k: v for k, v in self._data.items()
            if (include is None or k in include) and (exclude is None or k not in exclude)
        }
    
    def to_hdf5(self, path: Path | None = None, include: list[str] | None = None, exclude: list[str] | None = None) -> Path | io.BytesIO:
        """Save the ArrayDict to an HDF5 file.

        Args:
            path (str|Path): The path to the HDF5 file.
            include (list[str]|None): Optional list of keys to include.
            exclude (list[str]|None): Optional list of keys to exclude.
        """
        import h5py
        data = self.to_dict(include=include, exclude=exclude)
        if path is None:
            _path = io.BytesIO()
        else:
            _path = Path(path)

        with h5py.File(_path, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value)
        return _path


    def to_arrow(self, include: list[str] | None = None, exclude: list[str] | None = None) -> "pa.Table":
        """Convert the ArrayDict to a PyArrow Table.

        Args:
            include (list[str]|None): Optional list of keys to include.
            exclude (list[str]|None): Optional list of keys to exclude.

        Returns:
            pa.Table: A PyArrow Table containing the data.
        """
        import pyarrow as pa
        data = self.to_dict(include=include, exclude=exclude)
        return pa.Table.from_pydict(data) 

    def to_numpy(self) -> np.ndarray:
        """Convert the ArrayDict to a NumPy array.

        Returns:
            np.ndarray: A concatenated NumPy array of the values.
        """
        return np.column_stack(list(self.values()))