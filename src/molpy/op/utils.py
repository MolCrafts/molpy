from typing import Callable, Any
from functools import wraps, partial
import molpy as mp
import numpy as np


def get_struct_value_by_path(struct, alias: list[str]):

    values = []
    for al in alias:
        attrs = al.split(".")
        tmp = struct
        for attr in attrs:
            tmp = getattr(tmp, attr)
        values.append(tmp)
    return values


def set_struct_value_by_path(struct, alias: list[str], values:list[Any])->None:

    for al, value in zip(alias, values):
        attrs = al.split(".")
        tmp = struct
        for attr in attrs[:-1]:
            tmp = getattr(tmp, attr)
        setattr(tmp, attrs[-1], value)


class op:

    def __init__(
        self,
        func: Callable = None,
        input_key: list[str] = [],
        output_key: list[str] = [],
    ):
        if func:
            return self.decorate(func)
        self.input_key = input_key
        self.output_key = output_key

    def decorate(self, func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if isinstance(args[0], mp.Struct):
                struct = args[0]
                values = get_struct_value_by_path(struct, self.input_key)
                result = func(*values, *args[len(self.input_key):], **kwargs)
                if len(self.output_key) == 1:
                    set_struct_value_by_path(struct, self.output_key, [result])
                else:
                    set_struct_value_by_path(struct, self.output_key, result)
                return struct
            else:
                return func(*args, **kwargs)

        return wrapper

    def __call__(self, func):
        return self.decorate(func)
