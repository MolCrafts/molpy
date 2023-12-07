# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple, defaultdict
from typing import Any
import numpy as np

__all__ = ["Alias"]


class Alias:

    class Item(namedtuple("Item", ["alias", "keyword", "type", "unit", "comment"])):
        pass

    class Scope(dict):

        def __getattr__(self, alias):
            try:
                return self[alias]
            except KeyError:
                return super().__getattr__(alias)
        
        def set(self, alias: str, keyword: str, type: Any, unit: str, comment: str) -> None:
            self[alias] = Alias.Item(alias, keyword, type, unit, comment)

    _scopes: dict[str, Scope] = {'default':Scope({
            "timestep": Item("timestep", "_ts", int, "fs", "time step"),
            "name": Item("name", "_name", str, None, "atomic name"),
            "natoms": Item("natoms", "_natoms", int, None, "number of atoms"),
            "xyz": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "energy": Item("energy", "_energy", float, "meV", "energy"),
            "forces": Item("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
        })}

    def __new__(cls, name: None | str = None):
        if name and name not in cls._scopes:
            cls._scopes[name] = Alias.Scope()
        return super().__new__(cls)

    def __getattr__(self, alias: str) -> Item | Scope:
        if alias not in self._scopes:
            return self._scopes["default"][alias]
        return self._scopes[alias]

    def __getstate__(self) -> dict:
        return {
            "_scopes": self._scopes,
        }

    def __setstate__(self, __value: dict[str, Item]) -> None:
        self._scopes = __value["_scopes"]


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
