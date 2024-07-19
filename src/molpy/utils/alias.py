import numpy as np
from dataclasses import dataclass


@dataclass
class Alias:

    alias: str
    key: str
    type: type
    unit: str
    comment: str

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"<Alias: {self.alias}>"

    def __str__(self) -> str:
        return self.key


class NameSpace(dict):

    _namespace: dict = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self._namespace[name] = self

    @classmethod
    def get(cls, name: str):
        return cls._namespace[name]

    def __repr__(self) -> str:
        return f"<NameSpace: {self.name}>"

    def define(self, alias: Alias):
        self[alias.alias] = alias
        return alias.alias


default = NameSpace("default")

xyz = default.define(
    Alias("xyz", "xyz", np.ndarray, "angstrom", "Cartesian coordinates")
)
name = default.define(
    Alias("name", "name", str, "", "Atom name")
)
atomtype = default.define(
    Alias("atomtype", "atomtype", int, "", "Atom type")
)
charge = default.define(
    Alias("charge", "charge", float, "e", "Atom charge")
)
