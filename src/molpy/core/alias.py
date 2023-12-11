# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple
from typing import Any
import numpy as np

__all__ = ["Alias"]


class Alias:

    class Item(namedtuple("Item", ["alias", "key", "type", "unit", "comment"])): pass

    _scopes: dict[str, dict] = {'default': {
            "timestep": Item("timestep", "_ts", int, "fs", "time step"),
            "name": Item("name", "_name", str, None, "atomic name"),
            "natoms": Item("natoms", "_natoms", int, None, "number of atoms"),
            "xyz": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "energy": Item("energy", "_energy", float, "meV", "energy"),
            "forces": Item("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
        }}

    def __init__(self, scope_name: str = "default") -> None:
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        self._current = scope_name

    def __getattr__(self, alias: str):
    
        if alias in self._scopes:
            return Alias(alias)
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
    
    def __call__(self, scope_name:str) -> 'Alias':
        if scope_name not in self._scopes:
            self._scopes[scope_name] = {}
        return self
    
    def __iter__(self):
        yield from self._current_scope
    
    def __contains__(self, alias: str) -> bool:
        return alias in self._current_scope

    def __setstate__(self, value: dict[str, Item]) -> None:
        self._scopes = value["_scopes"]
        self._current = value["_current"]

    def set(self, alias: str, keyword: str, type: Any, unit: str, comment: str) -> None:
        self._current_scope[alias] = Alias.Item(alias, keyword, type, unit, comment)

    def alias(self)->list[str]:
        return list(self._current_scope.keys())
    
    def get_unit(self, alias: str) -> str:
        return self._current_scope[alias].unit

    @property
    def _current_scope(self) -> str:
        return self._scopes[self._current]

# keywords = kw = Keywords("_mp_global_")
# kw.set("timestep", "timestep", "fs", "time step")
# kw.set("name", "_mp_name_", None, "atomic name")
# kw.set("atomic_number", "_mp_atomic_number_", None, "atomic number")
# kw.set("charge", "_mp_charge_", None, "charge")
# kw.set("mass", "_mp_mass_", None, "mass")
# kw.set("type", "_mp_type_", None, "atom type")
# kw.set("covalent_radius", "_mp_covalent_radius_", None, "covalekw.setradius")
# kw.set("full_name", "_mp_full_name_", None, "full name")
# kw.set("vdw_radius", "_mp_vdw_radius_", None, "van der Waals radius")
# kw.set("natoms", "_mp_natoms_", None, "number of atoms")
# kw.set("xyz", "_mp_xyz_", None, "atomic coordinates")
# kw.set("positions", "_mp_xyz_", None, "atomic coordinates")
# kw.set("R", "_mp_xyz_", None, "atomic coordinates")
