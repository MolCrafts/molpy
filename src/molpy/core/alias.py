# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple
from typing import Optional

__all__ = ["Alias"]

class Aliases:

    Alias = namedtuple("Alias", ["alias", "keyword", "unit", "comment"])

    _aliases = {
        "_global": {}
    }

    def __init__(self, alias:str, keyword:str, unit:Optional[str], comment:Optional[str]):
        self._alias = alias
        self._keyword = keyword
        self._unit = unit
        self._comment = comment
        Alias._aliases[alias] = self

    # def get_keyword(self, alias):
    #     return self._aliases[alias].keyword
    
    # def get_unit(self, alias):
    #     return self._aliases[alias].unit

    # def __getattr__(self, alias: str):
    #     return self.get_keyword(alias)
    
    # def __next__(self):
    #     for alias in self._aliases:
    #         yield from alias

    # def __iter__(self):
    #     return iter(self._aliases)
    
    def __getstate__(self):
        return {
            "_aliases": self._aliases,
            "_alias": self._alias,
            "_keyword": self._keyword,
            "_unit": self._unit,
            "_comment": self._comment,
        }
    
    def __setstate__(self, state):
        self._aliases = state["_aliases"]
        self._alias = state["_alias"]
        self._keyword = state["_keyword"]
        self._unit = state["_unit"]
        self._comment = state["_comment"]
    
    @property
    def alias(self):
        return self._alias
    
    @property
    def keyword(self):
        return self._keyword
    
    @property
    def unit(self):
        return self._unit
    
    @property
    def comment(self):
        return self._comment


timestep = Alias("timestep", "_mp_ts_", "fs", "time step")
name = Alias("name", "_mp_name_", None, "atomic name")
atomic_number = Alias("atomic_number", "_mp_atomic_number_", None, "atomic number")
charge = Alias("charge", "_mp_charge_", None, "charge")
mass = Alias("mass", "_mp_mass_", None, "mass")
type = Alias("type", "_mp_type_", None, "atom type")
covalent_radius = Alias("covalent_radius", "_mp_covalent_radius_", None, "covalekw.setradius")
full_name = Alias("full_name", "_mp_full_name_", None, "full name")
vdw_radius = Alias("vdw_radius", "_mp_vdw_radius_", None, "van der Waals radius")
natoms = Alias("natoms", "_mp_natoms_", None, "number of atoms")
xyz = Alias("xyz", "_mp_xyz_", None, "atomic coordinates")
positions = Alias("positions", "_mp_xyz_", None, "atomic coordinates")
R = Alias("R", "_mp_xyz_", None, "atomic coordinates")
