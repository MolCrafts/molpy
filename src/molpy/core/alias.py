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
            "R": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "energy": Item("energy", "_energy", float, "meV", "energy"),
            "forces": Item("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
            "charge": Item("charge", "_mp_charge_", float, "C", "charge"),
            "masses": Item("masses", "masses", float, None, ""),
            "stress": Item("stress", "_stress", np.ndarray, "GPa", "stress"),
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
    
    def items(self)->list[Item]:
        return list(self._current_scope.values())
    
    def get_unit(self, alias: str) -> str:
        return self._current_scope[alias].unit

    @property
    def _current_scope(self) -> str:
        return self._scopes[self._current]


