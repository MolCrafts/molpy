# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from ._utils import Singleton
from typing import Any
import numpy as np
from dataclasses import dataclass


@dataclass
class AliasEntry:
    alias: str
    key: str
    type: type
    unit: str
    comment: str

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"<Alias: {self.alias}>"


DEFAULT_ALIASES = {
    "timestep": AliasEntry("timestep", "_ts", int, "fs", "time step"),
    "step": AliasEntry("step", "_step", int, None, "simulation step"),
    "name": AliasEntry("name", "_name", str, None, "atomic name"),
    "n_atoms": AliasEntry("n_atoms", "_n_atoms", int, None, "number of atoms"),
    "n_molecules": AliasEntry(
        "n_molecules", "_n_molecules", int, None, "number of molecules"
    ),
    "xyz": AliasEntry("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
    "R": AliasEntry("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
    "cell": AliasEntry("cell", "_cell", np.ndarray, "angstrom", "unit cell"),
    "energy": AliasEntry("energy", "_energy", float, "meV", "energy"),
    "forces": AliasEntry("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
    "momenta": AliasEntry(
        "momenta", "_momenta", np.ndarray, "eV*fs/angstrom", "momenta"
    ),
    "charge": AliasEntry("charge", "_charge", float, "C", "charge"),
    "mass": AliasEntry("mass", "_mass", float, None, ""),
    "stress": AliasEntry("stress", "_stress", np.ndarray, "GPa", "stress"),
    "idx": AliasEntry("idx", "_idx", int, None, ""),
    "molid": AliasEntry("mol", "_molid", int, None, "molecule index"),
    "Z": AliasEntry("Z", "_atomic_numbers", int, None, "nuclear charge"),
    "atype": AliasEntry("atype", "_atomic_types", int, None, "atomic type"),
    "vdw_radius": AliasEntry(
        "vdw_radius", "_vdw_radius", float, "angstrom", "van der Waals radius"
    ),
    "idx_m": AliasEntry("idx_m", "_idx_m", int, None, "indices of systems"),
    "idx_i": AliasEntry("idx_i", "_idx_i", int, None, "indices of center atoms"),
    "idx_j": AliasEntry("idx_j", "_idx_j", int, None, "indices of neighboring atoms"),
    "angle_i": AliasEntry(
        "angle_i", "_angle_i", int, None, "indices of center atoms for angles"
    ),
    "angle_j": AliasEntry(
        "angle_j", "_angle_j", int, None, "indices of neighboring atoms for angles"
    ),
    "angle_k": AliasEntry(
        "angle_k", "_angle_k", int, None, "indices of neighboring atoms for angles"
    ),
    "bond_i": AliasEntry(
        "bond_i", "_bond_i", int, None, "indices of center atoms for bonds"
    ),
    "bond_j": AliasEntry(
        "bond_j", "_bond_j", int, None, "indices of neighboring atoms for bonds"
    ),
    "idx_i_lr": AliasEntry(
        "idx_i_lr", "_idx_i_lr", int, None, "indices of center atoms for # long-range"
    ),
    "idx_j_lr": AliasEntry(
        "idx_j_lr",
        "_idx_j_lr",
        int,
        None,
        "indices of neighboring atoms for # long-range",
    ),
    "offsets": AliasEntry("offsets", "_offsets", int, None, "cell offset vectors"),
    "Rij": AliasEntry(
        "Rij",
        "_Rij",
        np.ndarray,
        "angstrom",
        "vectors pointing from center atoms to neighboring atoms",
    ),
    "dist": AliasEntry(
        "dist",
        "_dist",
        np.ndarray,
        "angstrom",
        "distances between center atoms and neighboring atoms",
    ),
    "pbc": AliasEntry("pbc", "_pbc", np.ndarray, None, "periodic boundary conditions"),
    "dipole_moment": AliasEntry(
        "dipole_moment", "_dipole_moment", np.ndarray, "e*bohr", "dipole moment"
    ),
    "partial_charges": AliasEntry(
        "partial_charges", "_partial_charges", np.ndarray, "e", "partial charges"
    ),
    "polarizability": AliasEntry(
        "polarizability", "_polarizability", np.ndarray, "angstrom^3", "polarizability"
    ),
    "element": AliasEntry("element", "_element", str, None, "element symbol"),
}


class NameSpace(dict):

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"<NameSpace: {self.name}>"

    def set(self, alias: str, key: str, type: type, unit: str, comment: str) -> None:
        self[alias] = AliasEntry(alias, key, type, unit, comment)
        return self[alias]

    def get(self, alias: str) -> AliasEntry:
        return super().__getitem__(alias)

    def load(self, definition: dict[str, AliasEntry]):
        self.update(definition)

    def __getitem__(self, key: str) -> AliasEntry:
        for entry in self.values():
            if entry.key == key:
                return entry
        raise KeyError(f"Key `{key}` not found in namespace {self.name}")

    def __getattr__(self, alias: str) -> str:
        return self[alias].key


class AliasSystem(metaclass=Singleton):

    def __init__(self, **namespace: dict[str, dict]):

        self._namespace: dict[str, NameSpace] = {}

        for name, definition in namespace.items():
            self.new_namespace(name)
            self.namespaces[name].load(definition)

    @property
    def namespaces(self) -> dict[str, NameSpace]:
        return self._namespace

    def __call__(self, namespace: str) -> NameSpace:
        if namespace not in self.namespaces:
            self.new_namespace(namespace)
        return self.namespaces[namespace]

    def __getattr__(self, alias: str) -> NameSpace | str:

        namespace = self.namespaces.get(alias)
        if namespace is not None:
            return namespace
        else:
            if alias in self.namespaces["default"]:
                alias_entry = self.namespaces["default"].get(alias)
            else:
                alias_entry = self.set(alias, alias, type, None, "")

            return alias_entry.key

    def __getitem__(self, key: str) -> AliasEntry | NameSpace:
        if key in self.namespaces:
            return self.namespaces[key]
        else:
            return self.namespaces["default"][key]

    def new_namespace(self, name: str) -> None:
        self.namespaces[name] = NameSpace(name)

    def set(
        self, alias: str, key: str, type: type, unit: str, comment: str
    ) -> AliasEntry:
        return self.namespaces["default"].set(alias, key, type, unit, comment)

    def get(self, alias: str, namespace: str = "default") -> AliasEntry:
        return self.namespaces[namespace].get(alias)

    def list(self) -> dict[str, list[str]]:
        """List all keys in all namespaces."""
        return {
            namespace.name: list(namespace.keys())
            for namespace in self.namespaces.values()
        }

    @classmethod
    def reset(cls):
        """Reset the AliasSystem instance and allow to create a new instance instead of the old one."""
        cls._instances = {}
        return cls()

    def clear(self):
        """remove all namespaces."""
        default = NameSpace("default")
        self.namespaces = {"default": default}
        default.load(DEFAULT_ALIASES)


Alias = AliasSystem(default=DEFAULT_ALIASES)
