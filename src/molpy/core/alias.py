# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple
from typing import Any, Optional

__all__ = ["Aliases"]

class Aliases:
    # Alias = namedtuple("Alias", ["alias", "keyword", "unit", "comment"])
    class Alias(namedtuple("Alias", ["alias", "keyword", "unit", "comment"])):
        pass

    __alias_scopes: dict[str, dict[str, Alias]] = {
        'default': {
            'timestep': Alias('timestep', '_ts', 'fs', 'time step'),
            'name': Alias('name', '_name', None, 'atomic name'),
        }
    }

    def __new__(cls, name: Optional[str]=None):
        if name and name not in cls.__alias_scopes:
            cls.__alias_scopes[name] = {}
        return super().__new__(cls)

    def __init__(self, name: Optional[str]=None):
        self._name = name or 'default'
        self._scope = self.__class__.__alias_scopes[self._name]

    def set(self, alias: str, keyword: str, unit: str, comment: str) -> Alias:
        self._scope[alias] = self.Alias(alias, keyword, unit, comment)

    def get(self, alias: str) -> Alias:
        return self._scope[alias]

    def __getattr__(self, alias: str) -> Alias:
        print(self._scope)
        return self._scope[alias]

    def __getitem__(self, scope: str) -> "Aliases":
        self._scope = self.__class__.__alias_scopes[scope]
        return self

    def __getstate__(self) -> dict[str, Alias]:
        return {
            # "__alias_scopes": self.__class__.__alias_scopes,
            "_name": self._name,
            "_scope": self._scope,
        }

    def __setstate__(self, __value: dict[str, Alias]) -> None:
        # self.__class__.__alias_scopes = __value["__alias_scopes"]
        self._name = __value["_name"]
        self._scope = __value["_scope"]


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
