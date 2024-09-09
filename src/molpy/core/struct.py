from collections.abc import MutableMapping
import numpy as np
from copy import deepcopy

class ArrayDict(MutableMapping[str, np.ndarray]):

    def __init__(self, **kwargs):
        self._data = {k: np.asarray(v) for k, v in kwargs.items()}
    
    def __delitem__(self, key):
        del self._data[key]

    def __getitem__(self, key: str | int) -> np.ndarray | list:
        if isinstance(key, int):
            return [v[key] for v in self._data.values()]
        
        elif isinstance(key, str):
            return self._data[key]
        
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        # assume all arrays have the same length
        return len(next(iter(self._data.values())))
    
    def __setitem__(self, key, value):
        self._data[key] = np.asarray(value)

    def __repr__(self):
        info = {k: f'shape: {v.shape}, dtype: {v.dtype}' for k, v in self._data.items()}
        return f"<ArrayDict {info}>"
    
    @classmethod
    def union(cls, *array_dict: "ArrayDict") -> "ArrayDict":
        ad = ArrayDict()
        for a in array_dict:
            for key, value in a._data.items():
                if key not in ad._data:
                    ad._data[key] = np.atleast_1d(value.copy())
                else:
                    ad._data[key] = np.concatenate([ad._data[key], np.atleast_1d(value)])
        return ad

class Struct:

    def __init__(self):
        
        self.props = []

    def __getitem__(self, key):
        if not hasattr(self, key):
            self[key] = ArrayDict()
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        self.props.append(key)
    
    def copy(self):
        return deepcopy(self)

    @classmethod
    def union(self, *structs: 'Struct') -> 'Struct':
        struct = Struct()
        for s in structs:
            for key, value in s._data.items():
                if key not in struct._data:
                    struct._data[key] = value.copy()
                else:
                    struct._data[key] = np.concatenate([struct._data[key], value])
        return struct