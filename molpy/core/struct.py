# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from functools import cached_property
import numpy as np

class StructArray(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._length = self.sentry()

    def sentry(self):

        if len(self) == 0:
            return 0

        lengths = np.unique([len(x) for x in self.values()])
        if len(lengths) == 1:
            return lengths[0]
        else:
            raise ValueError("The length of all arrays must be the same.")

    @cached_property
    def concrete(self)->np.ndarray:

        keys = self.keys()
        field_type = {key: np.array(super(StructArray, self).__getitem__(key)).dtype for key in keys}
        field_shape = {key: np.array(super(StructArray, self).__getitem__(key)).shape[1:] for key in keys} 

        structured_dtype = np.dtype([(key, field_type[key], field_shape[key]) for key in keys])

        return np.array([x for x in zip(*self.values())], dtype=structured_dtype)

    def __getitem__(self, K)->np.ndarray:
        return self.concrete[K]

    def __setitem__(self, K, V):

        if isinstance(K, str):
            super().__setitem__(K, V)
            self._length = len(V)
        try:
            self.sentry()
        except ValueError as e:
            del self[K]
            raise e
        finally:
            if 'concrete' in self.__dict__:
                del self.concrete  # clear cache

    @property
    def length(self):
        return self._length