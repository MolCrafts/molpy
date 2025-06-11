import molpy as mp
try:
    import molpack as mpk
except ImportError:  # fallback for environments without molpack
    class _MPK:
        class Constraint: ...
    mpk = _MPK()


class Target:

    def __init__(
        self,
        frame: mp.Frame,
        number: int,
        constraint: mpk.Constraint,
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
