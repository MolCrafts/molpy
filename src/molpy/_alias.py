# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from ._utils import Singleton
from typing import Any
import numpy as np
from dataclasses import dataclass

class Aliases(metaclass=Singleton):

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

    _scopes: dict[str, dict] = {'default': {
            "timestep": Alias("timestep", "_ts", int, "fs", "time step"),
            "name": Alias("name", "_name", str, None, "atomic name"),
            "natoms": Alias("natoms", "_natoms", int, None, "number of atoms"),
            "nmolecules": Alias("nmolecules", "_nmolecules", int, None, "number of molecules"),
            "xyz": Alias("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "R": Alias("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "cell": Alias("cell", "_cell", np.ndarray, "angstrom", "unit cell"),
            "energy": Alias("energy", "_energy", float, "meV", "energy"),
            "forces": Alias("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
            "momenta": Alias("momenta", "_momenta", np.ndarray, "eV*fs/angstrom", "momenta"),
            "charge": Alias("charge", "_charge", float, "C", "charge"),
            "mass": Alias("mass", "_mass", float, None, ""),
            "stress": Alias("stress", "_stress", np.ndarray, "GPa", "stress"),
            "idx": Alias("idx", "_idx", int, None, ""),
            "molid": Alias("mol", "_molid", int, None, "molecule index"),
            "Z": Alias("Z", "_atomic_numbers", int, None, "nuclear charge"),
            "atype": Alias("atype", "_atomic_types", int, None, "atomic type"),
            "vdw_radius": Alias("vdw_radius", "_vdw_radius", float, "angstrom", "van der Waals radius"),
            "idx_m": Alias("idx_m", "_idx_m", int, None, "indices of systems"),
            "idx_i": Alias("idx_i", "_idx_i", int, None, "indices of center atoms"),
            "idx_j": Alias("idx_j", "_idx_j", int, None, "indices of neighboring atoms"),
            "idx_i_lr": Alias("idx_i_lr", "_idx_i_lr", int, None, "indices of center atoms for # long-range"),
            "idx_j_lr": Alias("idx_j_lr", "_idx_j_lr", int, None, "indices of neighboring atoms for # long-range"),
            "offsets": Alias("offsets", "_offsets", int, None, "cell offset vectors"),
            "Rij": Alias("Rij", "_Rij", np.ndarray, "angstrom", "vectors pointing from center atoms to neighboring atoms"),
            "dist": Alias("dist", "_dist", np.ndarray, "angstrom", "distances between center atoms and neighboring atoms"),
            "pbc": Alias("pbc", "_pbc", np.ndarray, None, "periodic boundary conditions")
        }}

    def __init__(self, scope_name: str = "default") -> None:
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        self._current = scope_name

    def __getattr__(self, alias: str):
    
        if alias in self._scopes:
            return self(alias)
        elif alias in self._current_scope:
            return self._current_scope[alias].key
        else:
            raise AttributeError(f"alias '{alias}' not found in {self._current} scope")
        
    def __getitem__(self, alias: str):

        if alias in self._current_scope:
            return self._current_scope[alias]
        else:
            raise KeyError(f"alias '{alias}' not found in {self._current} scope")

    def __getstate__(self) -> dict:
        return {
            "_scopes": self._scopes,
            "_current": "default"
        }
    
    def __call__(self, scope_name:str) -> Alias:
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        return self
    
    def __iter__(self):
        yield from self._current_scope
    
    def __contains__(self, alias: str) -> bool:
        return alias in self._current_scope

    def __setstate__(self, value: dict[str, Alias]) -> None:
        self._scopes = value["_scopes"]
        self._current = value["_current"]

    def set(self, alias: str, keyword: str, type: Any, unit: str, comment: str) -> None:
        self._current_scope[alias] = Alias.Alias(alias, keyword, type, unit, comment)

    def alias(self)->list[str]:
        return list(self._current_scope.keys())
    
    def Aliass(self)->list[Alias]:
        return list(self._current_scope.values())
    
    def get_unit(self, alias: str) -> str:
        return self._current_scope[alias].unit

    @property
    def _current_scope(self) -> str:
        return self._scopes[self._current]

    def map(self, alias, key):
        alias.key = key.key
        alias.alias = key.alias
        alias.type = key.type
        alias.unit = key.unit
        alias.comment = key.comment

Alias = Aliases()
