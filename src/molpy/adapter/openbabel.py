"""OpenBabel adapter for MolPy.

This module provides bidirectional synchronization between MolPy's Atomistic
structures and OpenBabel's OBMol objects.

OpenBabel is an optional dependency.
"""

from __future__ import annotations

from typing import Any

from molpy.core.atomistic import Atomistic, Atom, Bond
from molpy.core.element import Element

from .base import Adapter

try:
    import openbabel as ob

    _HAS_OPENBABEL = True
except ImportError:
    _HAS_OPENBABEL = False
    ob = None


class OpenBabelAdapter(Adapter[Atomistic, "ob.OBMol"]):
    """Bridge between MolPy's Atomistic and OpenBabel's OBMol.

    OpenBabel provides built-in force field typing via its FFCalcTypes() function,
    which can be used to compare against our GAFF typifier implementation.
    """

    def __init__(
        self,
        internal: Atomistic | None = None,
        external: "ob.OBMol" | None = None,
    ) -> None:
        if not _HAS_OPENBABEL:
            raise ImportError(
                "OpenBabel is not installed. "
                "Install it with: pip install openbabel-wheel"
            )
        super().__init__(internal, external)
        self._atom_mapper: dict[int, Atom] = {}  # OBAtom id -> MolPy Atom

    @property
    def mol(self) -> "ob.OBMol":
        return self.get_external()

    def _do_sync_to_external(self) -> None:
        """Convert Atomistic to OBMol."""
        if self._internal is None:
            raise ValueError("Cannot sync: internal Atomistic is None")

        atomistic = self._internal
        mol = ob.OBMol()

        # Add atoms
        atom_map: dict[int, int] = {}  # id(MolPy atom) -> OBAtom idx
        for atom in atomistic.atoms:
            symbol = atom.get("symbol", atom.get("element", "C"))
            ob_atom = mol.NewAtom()
            ob_atom.SetAtomicNum(Element(symbol).number)

            # Store formal charge if present
            charge = atom.get("charge", 0)
            if charge != 0:
                ob_atom.SetFormalCharge(int(charge))

            # Store MolPy atom reference for later sync
            self._atom_mapper[ob_atom.GetIndex()] = atom
            atom_map[id(atom)] = ob_atom.GetIndex()

        # Add bonds
        for bond in atomistic.bonds:
            i = atom_map.get(id(bond.itom))
            j = atom_map.get(id(bond.jtom))
            if i is not None and j is not None:
                order = int(bond.get("order", 1))
                # OpenBabel uses 1-indexed atoms
                mol.AddBond(i + 1, j + 1, order)

        # Perceive aromaticity
        ob.SanitizeMol(mol)

        self._external = mol

    def _do_sync_to_internal(self) -> None:
        """Convert OBMol to Atomistic (update coordinates only)."""
        if self._external is None:
            raise ValueError("Cannot sync: external OBMol is None")

        mol = self._external

        # Update coordinates if present
        for mp_atom, ob_idx in zip(self._internal.atoms, self._atom_mapper.keys()):
            ob_atom = mol.GetAtom(ob_idx + 1)  # 1-indexed
            if mol.GetConformer() is not None:
                conf = mol.GetConformer()
                x = conf.GetAtomCoordinates(ob_idx)
                if x is not None:
                    mp_atom["x"] = float(x[0])
                    mp_atom["y"] = float(x[1])
                    mp_atom["z"] = float(x[2])

    def assign_gaff_types(self) -> dict[int, str]:
        """Use OpenBabel's GAFF force field to assign atom types.

        Returns:
            Dictionary mapping atom index (0-based) to GAFF atom type.
        """
        if self._external is None:
            self.sync_to_external()

        mol = self._external

        # Load GAFF force field
        ff = ob.OBForceField.FindForceField("GAFF")
        if ff is None:
            raise RuntimeError(
                "OpenBabel GAFF force field not found. "
                "Ensure OpenBabel is properly installed with GAFF support."
            )

        # Set up force field
        if not ff.Setup(mol):
            raise RuntimeError(
                f"Failed to set up GAFF force field: {ff.GetDescription()}"
            )

        # Extract assigned types
        types: dict[int, str] = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIndex()
            # OpenBabel stores the type in the atom's FFType property
            ff_type = atom.GetType()
            types[idx] = ff_type if ff_type else "???"

        return types

    def get_type_comparison(self) -> dict[str, Any]:
        """Compare OpenBabel GAFF types against MolPy GAFF typing.

        Requires that MolPy Atomistic has already been typed.

        Returns:
            Dictionary with comparison statistics and per-atom results.
        """
        if self._internal is None:
            raise ValueError("MolPy Atomistic must be typed first")

        # Get OpenBabel types
        ob_types = self.assign_gaff_types()

        # Compare with MolPy types
        comparison = {
            "total_atoms": len(list(self._internal.atoms)),
            "agreement": 0,
            "disagreement": 0,
            "molpy_missing": 0,
            "openbabel_missing": 0,
            "details": [],
        }

        for idx, atom in enumerate(self._internal.atoms):
            mp_type = atom.get("type")
            ob_type = ob_types.get(idx, "???")

            sym = atom.get("symbol", "?")

            if mp_type is None:
                comparison["molpy_missing"] += 1
                agreement = False
            elif ob_type == "???":
                comparison["openbabel_missing"] += 1
                agreement = False
            else:
                agreement = mp_type == ob_type
                if agreement:
                    comparison["agreement"] += 1
                else:
                    comparison["disagreement"] += 1

            comparison["details"].append(
                {
                    "index": idx,
                    "symbol": sym,
                    "molpy": mp_type or "???",
                    "openbabel": ob_type,
                    "match": (
                        mp_type == ob_type if (mp_type and ob_type != "???") else None
                    ),
                }
            )

        return comparison
