# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from collections import namedtuple
from typing import Optional


Keyword = namedtuple("Alias", ["alias", "keyword", "unit", "comment"])


class Keywords:
    def __init__(self, name: str):
        self._name = name
        self._keywords: list[Keyword] = []

    def set(self, alias, keyword, unit, comment: Optional[str] = None):
        self._keywords.append(Keyword(alias, keyword, unit, comment))

    def get_keyword(self, alias):
        for kw in self._keywords:
            if kw.alias == alias:
                return kw.keyword
        raise KeyError(f"Keyword {alias} not found in {self._name}.")
    
    def get_alias(self, keyword):
        for kw in self._keywords:
            if kw.keyword == keyword:
                return kw.alias
        raise KeyError(f"Keyword {keyword} not found in {self._name}.")
    
    def get_unit(self, alias):
        for kw in self._keywords:
            if kw.alias == alias:
                return kw.unit
        raise KeyError(f"Keyword {alias} not found in {self._name}.")

    def __getattr__(self, alias: str):
        return self.get_keyword(alias)
    
    def __next__(self):
        for kw in self._keywords:
            yield from kw

    def __iter__(self):
        return iter(self._keywords)


keywords = kw = Keywords("_mp_global_")
kw.set("timestep", "timestep", "fs", "time step")
kw.set("name", "_mp_name_", None, "atomic name")
kw.set("atomic_number", "_mp_atomic_number_", None, "atomic number")
kw.set("charge", "_mp_charge_", None, "charge")
kw.set("mass", "_mp_mass_", None, "mass")
kw.set("type", "_mp_type_", None, "atom type")
kw.set("covalent_radius", "_mp_covalent_radius_", None, "covalekw.setradius")
kw.set("full_name", "_mp_full_name_", None, "full name")
kw.set("vdw_radius", "_mp_vdw_radius_", None, "van der Waals radius")
