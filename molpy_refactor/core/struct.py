# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

from .typing import Dict, NDArray, ArrayLike, Iterable
import numpy as np

class StructData:

    def __init__(self, ):

        self.data:Dict[str, NDArray] = {}
        self.length = 0

    def get_item(self, key:str)->NDArray:
        return self.data[key]

    def get_items(self, keys:Iterable[str])->NDArray:

        data = self.data
        keys = data.keys()
        field_type = {key: np.array(data[key]).dtype for key in keys}
        field_shape = {key: np.array(data[key]).shape[1:] for key in keys} 

        structured_dtype = np.dtype([(key, field_type[key], field_shape[key]) for key in keys])

        return np.array([x for x in zip(*data.values())], dtype=structured_dtype)

    def set_item(self, key:str, value:ArrayLike)->None:

        self.data[key] = np.array(value)
        self.length = max(self.length, len(value))

    def __getitem__(self, K):
        if isinstance(K, str):
            return self.get_item(K)
        elif isinstance(K, Iterable):
            return self.get_items(K)

    def __setitem__(self, K, V):

        self.set_item(K, V)

    @property
    def size(self):
        return self.length