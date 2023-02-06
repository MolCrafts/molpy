# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from typing import Callable, Hashable, List, Optional
from .typing import Dict, NDArray, ArrayLike, Iterable, Any
import numpy as np


class StaticSOA(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._length = self.sentry()

    def sentry(self):
        if len(self) == 0:
            return 0
        max_length = max(self.values())
        min_length = min(self.values())
        if max_length == min_length:
            return max_length
        else:
            raise ValueError("The length of all arrays must be the same.")

    def get_struct(self, keys:Optional[Iterable[str]]=None):

        keys = keys or self.keys()
        field_type = {key: np.array(self[key]).dtype for key in keys}
        field_shape = {key: np.array(self[key]).shape[1:] for key in keys} 

        structured_dtype = np.dtype([(key, field_type[key], field_shape[key]) for key in keys])

        return np.array([x for x in zip(*self.values())], dtype=structured_dtype)

    def __getitem__(self, K):
        if isinstance(K, str):
            return super().__getitem__(K)
        elif isinstance(K, np.ndarray):
            return self.get_struct()[K]
        elif all(map(lambda k: type(k)==str, K)):
            return self.get_struct(K)

    def __setitem__(self, K, V):

        if isinstance(K, str):
            super().__setitem__(K, V)

    @property
    def length(self):
        return self._length


class DynamicSOA:

    def __init__(self, ):

        self.data:Dict[str, List] = {}

    def append(self, key:str, value:ArrayLike)->None:

        if key not in self.data:
            self.data['key'] = []

        self.data[key].append(value)

    def extend(self, key:str, value:ArrayLike)->None:

        if key not in self.data:
            self.data['key'] = []

        self.data[key].extend(value)

    def set_item(self, key:str, value:Optional[List])->None:
        if value:
            self.data[key] = value
        else:
            self.data[key] = []

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data.setdefault(key, [])
        elif isinstance(key, int):
            return {k: v[key] for k, v in self.data.items()}

    def index(self, key, value):
        return self.data[key].index(value)

    def is_aligned(self)->bool:

        lens = list(map(len, self.data.values()))
        max_len = max(lens)
        min_len = min(lens)
        if max_len != min_len:
            return False
        else:
            return True

    def __len__(self):
        if self.is_aligned():
            lens = map(len, self.data.values())
            return max(lens)
        else:
            raise ValueError("DynamicSOA is not aligned.")
        

class AOS:

    def __init__(self,):

        self.data:Dict[Hashable, Any] = {}

    def __setitem__(self, K, V):

        self.data[K] = V
