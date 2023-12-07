# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple, defaultdict
import numpy as np

__all__ = ["Alias"]


class Alias:

    class Item(namedtuple("Item", ["alias", "keyword", "type", "unit", "comment"])):
        pass

    scopes: dict[str, dict[str, Item]] = {'default':{
            "timestep": Item("timestep", "_ts", int, "fs", "time step"),
            "name": Item("name", "_name", str, None, "atomic name"),
            "natoms": Item("natoms", "_natoms", int, None, "number of atoms"),
            "xyz": Item("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "energy": Item("energy", "_energy", float, "meV", "energy"),
            "forces": Item("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
        }}
    
    current = scopes['default']

    def __new__(cls, name: str = 'default'):
        if name not in cls.scopes:
            cls.scopes[name] = {}
        return super().__new__(cls)

    def __init__(self, name: None | str = None):
        self._scope = self.__class__.scopes[name]

    @classmethod
    def __getattr__(cls, name: str) -> Item:
        return cls.current[name]

    def set(self, alias: str, keyword: str, unit: str, comment: str) -> Item:
        self._scope[alias] = self.Item(alias, keyword, unit, comment)

    def get(self, alias: str) -> Item:
        return self._scope[alias]

    def __getattr__(self, alias: str) -> Item:
        print(self._scope)
        return self._scope[alias]

    def __getitem__(self, scope: str) -> Item:
        self._scope = self.__class__.scopes[scope]
        return self

    def __getstate__(self) -> dict[str, Item]:
        return {
            # "scopes": self.__class__.scopes,
            "_name": self._name,
            "_scope": self._scope,
        }

    def __setstate__(self, __value: dict[str, Item]) -> None:
        # self.__class__.scopes = __value["scopes"]
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
