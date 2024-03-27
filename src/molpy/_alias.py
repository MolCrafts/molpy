# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-09-29
# version: 0.0.1

from ._utils import Singleton
from typing import Any
import numpy as np
from dataclasses import dataclass

@dataclass
class _Alias:
    alias: str
    key: str
    type: type
    unit: str
    comment: str

    def __hash__(self) -> int:
        return hash(self.key)

    def __repr__(self) -> str:
        return f"<Alias: {self.alias}>"

class _Aliases(dict):

    _scopes: dict[str, "_Aliases"] = {}

    def __init__(self, scope_name: str, definition:dict={}) -> None:
        super().__init__(definition)
        if scope_name not in self._scopes:
            self._scopes[scope_name] = self

    def __getattr__(self, alias: str):

        if alias in self:
            return self[alias].key
        else:
            return self._scopes[alias]
        
    def __call__(self, scope_name: str, definition: dict={}) -> None:
        if scope_name not in self._scopes:
            _Aliases(scope_name, definition)

    def set(self, alias: str, key: str, type: type, unit: str, comment: str) -> None:
        self[alias] = _Alias(alias, key, type, unit, comment)
        
Alias = _Aliases("default", {
            "timestep": _Alias("timestep", "_ts", int, "fs", "time step"),
            "step": _Alias("step", "_step", int, None, "simulation step"),
            "name": _Alias("name", "_name", str, None, "atomic name"),
            "n_atoms": _Alias("n_atoms", "_n_atoms", int, None, "number of atoms"),
            "n_molecules": _Alias("n_molecules", "_n_molecules", int, None, "number of molecules"),
            "xyz": _Alias("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "R": _Alias("xyz", "_xyz", np.ndarray, "angstrom", "atomic coordinates"),
            "cell": _Alias("cell", "_cell", np.ndarray, "angstrom", "unit cell"),
            "energy": _Alias("energy", "_energy", float, "meV", "energy"),
            "forces": _Alias("forces", "_forces", np.ndarray, "eV/angstrom", "forces"),
            "momenta": _Alias("momenta", "_momenta", np.ndarray, "eV*fs/angstrom", "momenta"),
            "charge": _Alias("charge", "_charge", float, "C", "charge"),
            "mass": _Alias("mass", "_mass", float, None, ""),
            "stress": _Alias("stress", "_stress", np.ndarray, "GPa", "stress"),
            "idx": _Alias("idx", "_idx", int, None, ""),
            "molid": _Alias("mol", "_molid", int, None, "molecule index"),
            "Z": _Alias("Z", "_atomic_numbers", int, None, "nuclear charge"),
            "atype": _Alias("atype", "_atomic_types", int, None, "atomic type"),
            "vdw_radius": _Alias("vdw_radius", "_vdw_radius", float, "angstrom", "van der Waals radius"),
            "idx_m": _Alias("idx_m", "_idx_m", int, None, "indices of systems"),
            "idx_i": _Alias("idx_i", "_idx_i", int, None, "indices of center atoms"),
            "idx_j": _Alias("idx_j", "_idx_j", int, None, "indices of neighboring atoms"),
            "idx_i_lr": _Alias("idx_i_lr", "_idx_i_lr", int, None, "indices of center atoms for # long-range"),
            "idx_j_lr": _Alias("idx_j_lr", "_idx_j_lr", int, None, "indices of neighboring atoms for # long-range"),
            "offsets": _Alias("offsets", "_offsets", int, None, "cell offset vectors"),
            "Rij": _Alias("Rij", "_Rij", np.ndarray, "angstrom", "vectors pointing from center atoms to neighboring atoms"),
            "dist": _Alias("dist", "_dist", np.ndarray, "angstrom", "distances between center atoms and neighboring atoms"),
            "pbc": _Alias("pbc", "_pbc", np.ndarray, None, "periodic boundary conditions"),
            "dipole_moment": _Alias("dipole_moment", "_dipole_moment", np.ndarray, "e*bohr", "dipole moment"),
            "partial_charges": _Alias("partial_charges", "_partial_charges", np.ndarray, "e", "partial charges"),
            "polarizability": _Alias("polarizability", "_polarizability", np.ndarray, "angstrom^3", "polarizability"),
        })
