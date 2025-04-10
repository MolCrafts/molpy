from ..base import Potential
import numpy as np
import molpy as mp
from functools import wraps, partial


class PairPotential(Potential):

    @staticmethod
    def or_frame(*args):

        func = args[0]

        def decorator(func=None, extra=None):
            @wraps(func)
            def wrapper(self, frame, *args, **kwargs):
                if isinstance(frame, mp.Frame):
                    r = frame["atoms"][["x", "y", "z"]].to_numpy()
                    pair_idx = frame["pairs"][["i", "j"]].to_numpy()
                    pair_types = frame["pairs"]["type"].to_numpy()
                    if extra:
                        extra_args = [frame[arg].to_numpy() for arg in extra]
                        return func(
                            self, r, pair_idx, pair_types, *extra_args, *args, **kwargs
                        )
                    return func(self, r, pair_idx, pair_types, *args, **kwargs)
                return func(self, frame, *args, **kwargs)

            return wrapper

        if func is not None and callable(func):
            # @or_frame
            # def func
            return decorator(func)
        else:
            return partial(decorator, extra=args)
