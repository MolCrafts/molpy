import molpy as mp
from .constraint import Constraint

class Target:

    def __init__(
        self,
        frame: mp.Frame,
        number: int,
        constraint: Constraint,
        is_fixed: bool = False,
        optimizer=None,
        name: str = "",
    ):
        self.frame = frame
        self.number = number
        self.constraint = constraint
        self.is_fixed = is_fixed
        self.optimizer = optimizer
        self.name = name

    def __repr__(self):
        return f"<Target {self.name}: {self.frame['atoms'].shape[0]} atoms in {self.constraint}>"

    @property
    def n_points(self):
        return len(self.frame["atoms"]) * self.number

    @property
    def points(self):
        return self.frame["atoms"][["x", "y", "z"]]
