"""RDKit adapter for MolPy.

This module provides bidirectional synchronization between MolPy's Atomistic
structures and RDKit's Chem.Mol objects.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from molpy.core.atomistic import Atomistic
from molpy.core.wrapper import Wrapper
from molpy.external.base import Adapter

# Stable property tag for bidirectional atom mapping
MP_ID = "mp_id"

# Bond order mappings between MolPy and RDKit
BOND_ORDER_TO_RDKIT: dict[float, Chem.BondType] = {
    1.0: Chem.BondType.SINGLE,
    2.0: Chem.BondType.DOUBLE,
    3.0: Chem.BondType.TRIPLE,
    1.5: Chem.BondType.AROMATIC,
}

RDKIT_TO_BOND_ORDER: dict[Chem.BondType, float] = {
    Chem.BondType.SINGLE: 1.0,
    Chem.BondType.DOUBLE: 2.0,
    Chem.BondType.TRIPLE: 3.0,
    Chem.BondType.AROMATIC: 1.5,
}


def _rdkit_bond_type(order: float) -> Chem.BondType:
    """Convert MolPy bond order to RDKit BondType.

    Raises:
        ValueError: If order is not supported
    """
    order_float = float(order)
    if order_float not in BOND_ORDER_TO_RDKIT:
        raise ValueError(
            f"Bond order {order_float} is not supported. "
            f"Supported orders: {list(BOND_ORDER_TO_RDKIT.keys())}"
        )
    return BOND_ORDER_TO_RDKIT[order_float]


def _order_from_rdkit(bt: Chem.BondType) -> float:
    """Convert RDKit BondType to MolPy bond order.

    Raises:
        ValueError: If bond type is not supported
    """
    if bt not in RDKIT_TO_BOND_ORDER:
        raise ValueError(
            f"RDKit bond type {bt} is not supported. "
            f"Supported types: {list(RDKIT_TO_BOND_ORDER.keys())}"
        )
    return RDKIT_TO_BOND_ORDER[bt]


class _AtomMapper:
    """Manages bidirectional atom mapping between RDKit Mol and Atomistic atoms."""

    def __init__(self, mol: Chem.Mol, atomistic_atoms: list[Any]) -> None:
        """Initialize atom mapper.

        Args:
            mol: RDKit molecule
            atomistic_atoms: List of Atomistic atom entities
        """
        self.mol = mol
        self.atomistic_atoms = atomistic_atoms
        self._forward_map: dict[int, Any] | None = None
        self._reverse_map: dict[int, int] | None = None
        self.build_mapping()

    def build_mapping(self) -> dict[int, Any]:
        """Build atom mapping from RDKit indices to Atomistic atoms.

        Returns:
            Dictionary mapping RDKit atom index to Atomistic atom entity
        """
        if self._forward_map is not None:
            return self._forward_map

        atom_map: dict[int, Any] = {}
        atom_by_id = self._build_atom_id_index()

        mol_atoms = list(self.mol.GetAtoms())

        # STRICT ID-BASED MAPPING ONLY - no position-based fallback
        # Map by MP_ID property
        # Handle both positive (existing atoms) and negative (new atoms) IDs
        for rd_idx, rd_atom in enumerate(mol_atoms):
            if not rd_atom.HasProp(MP_ID):
                raise RuntimeError(
                    f"RDKit atom at index {rd_idx} (symbol={rd_atom.GetSymbol()}) "
                    f"does not have {MP_ID} property. "
                    "All RDKit atoms must have this property for mapping."
                )
            hid = int(rd_atom.GetIntProp(MP_ID))

            # Negative IDs indicate newly added atoms (e.g., from AddHs)
            # These will be created as new atoms in sync_to_internal
            # Do NOT map them here - they don't exist in atomistic yet
            if hid < 0:
                continue

            # STRICT ID-BASED MAPPING - must find atom with matching ID
            if hid not in atom_by_id:
                raise RuntimeError(
                    f"RDKit atom at index {rd_idx} (symbol={rd_atom.GetSymbol()}) "
                    f"has {MP_ID}={hid}, but no Atomistic atom with this id exists. "
                    "This indicates a mapping error. All existing atoms must have matching IDs."
                )
            atom_map[rd_idx] = atom_by_id[hid]

        self._forward_map = atom_map
        self._reverse_map = {id(v): k for k, v in atom_map.items()}
        return atom_map

    def _build_atom_id_index(self) -> dict[int, Any]:
        """Build index of Atomistic atoms by their 'id' property.

        Raises:
            ValueError: If any atom is missing an 'id' attribute
        """
        by_id: dict[int, Any] = {}
        for i, atom in enumerate(self.atomistic_atoms):
            atom_id = atom.get("id")
            if atom_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (symbol={atom.get('symbol')}) "
                    "has no 'id' attribute. All atoms must have an 'id' for mapping."
                )
            atom_id_int = int(atom_id)
            if atom_id_int in by_id:
                raise ValueError(
                    f"Duplicate atom id {atom_id_int} found. "
                    f"Atoms at indices {list(by_id.keys()).index(atom_id_int)} and {i} "
                    "have the same id."
                )
            by_id[atom_id_int] = atom
        return by_id

    def ensure_tags(self) -> None:
        """Ensure MP_ID tags are set on RDKit atoms for reliable mapping.

        Raises:
            ValueError: If atomistic atoms are missing 'id' attributes
        """
        for i, rd_atom in enumerate(self.mol.GetAtoms()):
            if rd_atom.HasProp(MP_ID):
                continue
            if i >= len(self.atomistic_atoms):
                raise RuntimeError(
                    f"RDKit molecule has {self.mol.GetNumAtoms()} atoms, "
                    f"but Atomistic has only {len(self.atomistic_atoms)} atoms. "
                    "Cannot establish mapping."
                )
            ent = self.atomistic_atoms[i]
            ent_id = ent.get("id")
            if ent_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (symbol={ent.get('symbol')}) "
                    "has no 'id' attribute. Cannot set MP_ID tag."
                )
            rd_atom.SetIntProp(MP_ID, int(ent_id))


class RDKitAdapter(Adapter[Atomistic, Chem.Mol]):
    """Bridge between MolPy's atomistic representation and rdkit.Chem.Mol.

    This adapter maintains bidirectional synchronization between MolPy's
    Atomistic structures and RDKit's Chem.Mol objects. It handles conversion
    of atoms, bonds, coordinates, and properties in both directions.

    **Responsibilities:**
    - Synchronize Atomistic ↔ Chem.Mol representations
    - Maintain atom index mapping via internal AtomMapper
    - Handle coordinate transfer

    **Limitations:**
    - Does NOT perform domain operations (3D generation, optimization, etc.)
    - Does NOT execute external tools
    - Pure data synchronization only

    **Usage:**
        >>> adapter = RDKitAdapter(internal=atomistic_structure)
        >>> adapter.sync_to_external()
        >>> mol = adapter.get_external()
        >>> # ... modify mol ...
        >>> adapter.sync_to_internal()
        >>> updated = adapter.get_internal()
    """

    def __init__(
        self,
        internal: Atomistic | None = None,
        external: Chem.Mol | None = None,
    ) -> None:
        """Initialize RDKitAdapter.

        Args:
            internal: Optional Atomistic structure
            external: Optional RDKit Mol

        At least one must be provided, or both can be None and set later.

        If internal is provided and atoms don't have 'id' attributes,
        they will be automatically assigned internal IDs for mapping.
        """
        super().__init__(internal, external)
        self._atom_mapper: _AtomMapper | None = None

        # Auto-assign IDs to atoms if they don't have them
        if internal is not None:
            self._ensure_atom_ids()

    @property
    def internal(self) -> Atomistic:
        """Get the internal Atomistic representation (convenience property)."""
        return self.get_internal()

    @property
    def mol(self) -> Chem.Mol:
        """Get the external Chem.Mol representation (convenience property)."""
        return self.get_external()

    def _ensure_atom_ids(self) -> None:
        """Ensure all atoms in internal Atomistic have 'id' attributes.

        Assigns sequential IDs starting from 0 if atoms don't have them.
        This is required for proper mapping between Atomistic and RDKit.
        """
        if self._internal is None:
            return

        atomistic = self._internal
        if isinstance(atomistic, Wrapper):
            atomistic = atomistic.unwrap()

        # Find the maximum existing ID
        max_id = -1
        atoms_without_id = []

        for atom in atomistic.atoms:
            atom_id = atom.get("id")
            if atom_id is None:
                atoms_without_id.append(atom)
            else:
                try:
                    atom_id_int = int(atom_id)
                    if atom_id_int > max_id:
                        max_id = atom_id_int
                except (ValueError, TypeError):
                    atoms_without_id.append(atom)

        # Assign IDs to atoms without them
        next_id = max_id + 1
        for atom in atoms_without_id:
            atom["id"] = next_id
            next_id += 1

    def sync_to_external(self) -> None:
        """Synchronize external RDKit Mol from internal Atomistic.

        Creates or updates the RDKit Mol based on the current Atomistic structure.
        Handles atoms, bonds, coordinates, and properties.

        Raises:
            ValueError: If internal is None
            RuntimeError: If conversion fails
        """
        super().sync_to_external()

        atomistic = self._internal
        if atomistic is None:
            return

        if isinstance(atomistic, Wrapper):
            atomistic = atomistic.unwrap()

        # Ensure all atoms have IDs (should already be done in __init__, but double-check)
        self._ensure_atom_ids()

        mol = Chem.RWMol()
        atom_map: dict[int, int] = {}

        # Build mapping from atom id to atom object - STRICT ID-BASED MAPPING ONLY
        atom_by_id: dict[int, Any] = {}
        for atom in atomistic.atoms:
            atom_id = atom.get("id")
            if atom_id is None:
                raise RuntimeError(
                    f"Atom {atom} (symbol={atom.get('symbol')}) has no 'id' attribute. "
                    "All atoms must have an 'id' for mapping. "
                    "This should have been assigned in __init__ or _ensure_atom_ids()."
                )
            atom_id_int = int(atom_id)
            if atom_id_int in atom_by_id:
                raise RuntimeError(
                    f"Duplicate atom id {atom_id_int} found. "
                    "Each atom must have a unique 'id'."
                )
            atom_by_id[atom_id_int] = atom

        # Create RDKit atoms and build mapping - ID-BASED ONLY
        for atom in atomistic.atoms:
            atom_id = atom.get("id")
            if atom_id is None:
                raise RuntimeError(
                    f"Atom {atom} (symbol={atom.get('symbol')}) has no 'id' attribute."
                )
            atom_id_int = int(atom_id)

            symbol = atom.get("symbol")
            if symbol is None:
                raise ValueError(
                    f"Atom with id {atom_id_int} has no 'symbol' attribute. "
                    "All atoms must have a symbol."
                )
            rd_atom = Chem.Atom(symbol)

            charge = atom.get("charge")
            if charge is not None:
                rd_atom.SetFormalCharge(int(charge))

            rd_atom.SetIntProp(MP_ID, atom_id_int)
            idx = mol.AddAtom(rd_atom)

            # STRICT ID-BASED MAPPING - no position fallback
            atom_map[id(atom)] = idx

        for bond in atomistic.bonds:
            begin_idx = atom_map.get(id(bond.itom))
            end_idx = atom_map.get(id(bond.jtom))
            if begin_idx is None or end_idx is None:
                continue

            order_val = bond.get("order")
            if order_val is None:
                type_val = bond.get("type", 1.0)
                try:
                    order = float(type_val)
                except Exception:
                    order = 1.0
            else:
                try:
                    order = float(order_val)
                except Exception:
                    order = 1.0

            bt = _rdkit_bond_type(order)
            mol.AddBond(begin_idx, end_idx, bt)

        if any(atom.get("x") is not None for atom in atomistic.atoms):
            conf = Chem.Conformer(mol.GetNumAtoms())
            for atom in atomistic.atoms:
                rdkit_idx = atom_map.get(id(atom))
                if rdkit_idx is None:
                    raise RuntimeError(
                        f"Atom {atom} (symbol={atom.get('symbol')}) not found in atom_map "
                        "when setting coordinates."
                    )
                x = atom.get("x")
                y = atom.get("y")
                z = atom.get("z")
                if x is None or y is None or z is None:
                    raise ValueError(
                        f"Atom {atom} (symbol={atom.get('symbol')}) has incomplete coordinates. "
                        f"x={x}, y={y}, z={z}. All coordinates must be present."
                    )
                conf.SetAtomPosition(rdkit_idx, (float(x), float(y), float(z)))
            mol.AddConformer(conf, assignId=True)

        # Sanitize the molecule to ensure proper atom types and hybridization
        # This is required for many RDKit operations including optimization
        final_mol = mol.GetMol()
        try:
            Chem.SanitizeMol(final_mol)
        except Exception:
            # If sanitization fails, update property cache with less strict settings
            # This allows molecules with unusual valencies to still be used
            final_mol.UpdatePropertyCache(strict=False)

        self._external = final_mol
        self._rebuild_atom_mapper()

    def sync_to_internal(self, update_topology: bool = True) -> None:
        """Synchronize internal Atomistic from external RDKit Mol.

        Creates or updates the Atomistic structure based on the current RDKit Mol.
        Handles atoms, bonds, coordinates, and properties. Preserves existing
        atom references when possible.

        Args:
            update_topology: If True, rebuild bonds from RDKit Mol. If False,
                            only update coordinates and preserve existing bonds.
                            Default is True for full sync, False is useful for
                            geometry optimization where only coordinates change.

        Raises:
            ValueError: If external is None
            RuntimeError: If conversion fails
        """
        super().sync_to_internal()

        mol = self._external
        if mol is None:
            return

        self._rebuild_atom_mapper()
        if self._atom_mapper is None:
            raise RuntimeError("Atom mapper not initialized")

        atom_map = self._atom_mapper.build_mapping()

        if self._internal is None:
            atomistic = Atomistic()
        else:
            atomistic = self._internal
            if isinstance(atomistic, Wrapper):
                atomistic = atomistic.unwrap()

        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None
        existing_atoms = list(atomistic.atoms)
        rdkit_to_atom: dict[int, Any] = {}

        # Find maximum existing atom ID to assign IDs to new atoms
        max_existing_id = -1
        for existing_atom in atomistic.atoms:
            existing_id = existing_atom.get("id")
            if existing_id is not None:
                try:
                    existing_id_int = int(existing_id)
                    if existing_id_int > max_existing_id:
                        max_existing_id = existing_id_int
                except (ValueError, TypeError):
                    pass

        next_new_id = max_existing_id + 1

        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            atom = atom_map.get(rdkit_idx)

            if atom is None:
                # New atom from RDKit (e.g., added hydrogens by AddHs)
                if not rd_atom.HasProp(MP_ID):
                    raise RuntimeError(
                        f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) "
                        f"does not have {MP_ID} property. "
                        "All RDKit atoms must have this property for mapping."
                    )
                atom_id_raw = rd_atom.GetIntProp(MP_ID)

                # Negative IDs are temporary IDs for new atoms (from Generate3D)
                # Assign a proper positive ID
                if atom_id_raw < 0:
                    atom_id = next_new_id
                    next_new_id += 1
                else:
                    # This should not happen if mapping is correct
                    # But handle it anyway
                    atom_id = atom_id_raw

                props: dict[str, Any] = {
                    "symbol": rd_atom.GetSymbol(),
                    "atomic_num": rd_atom.GetAtomicNum(),
                    "id": atom_id,
                }

                if rd_atom.GetFormalCharge() != 0:
                    props["charge"] = rd_atom.GetFormalCharge()

                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    props["x"] = float(pos.x)
                    props["y"] = float(pos.y)
                    props["z"] = float(pos.z)
                    props["xyz"] = [float(pos.x), float(pos.y), float(pos.z)]

                new_atom = atomistic.def_atom(**props)
                rdkit_to_atom[rdkit_idx] = new_atom
            else:
                # Existing atom - update properties
                atom["symbol"] = rd_atom.GetSymbol()
                atom["atomic_num"] = rd_atom.GetAtomicNum()

                if atom.get("id") is None:
                    if not rd_atom.HasProp(MP_ID):
                        raise RuntimeError(
                            f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) "
                            f"does not have {MP_ID} property, and Atomistic atom has no 'id'. "
                            "Cannot establish mapping."
                        )
                    atom["id"] = rd_atom.GetIntProp(MP_ID)

                if rd_atom.GetFormalCharge() != 0:
                    atom["charge"] = rd_atom.GetFormalCharge()

                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    atom["x"] = float(pos.x)
                    atom["y"] = float(pos.y)
                    atom["z"] = float(pos.z)
                    atom["xyz"] = [float(pos.x), float(pos.y), float(pos.z)]

                rdkit_to_atom[rdkit_idx] = atom

        # Only update topology (bonds) if requested
        # For geometry optimization, we want to preserve existing bonds and their properties
        if update_topology:
            existing_bonds = list(atomistic.bonds)
            if existing_bonds:
                atomistic.remove_link(*existing_bonds)

            for rd_bond in mol.GetBonds():
                begin_idx = rd_bond.GetBeginAtomIdx()
                end_idx = rd_bond.GetEndAtomIdx()

                atom1 = rdkit_to_atom.get(begin_idx)
                atom2 = rdkit_to_atom.get(end_idx)

                if atom1 is None:
                    raise RuntimeError(
                        f"RDKit atom at index {begin_idx} not found in rdkit_to_atom mapping. "
                        "This indicates a synchronization error."
                    )
                if atom2 is None:
                    raise RuntimeError(
                        f"RDKit atom at index {end_idx} not found in rdkit_to_atom mapping. "
                        "This indicates a synchronization error."
                    )

                bond_type = rd_bond.GetBondType()
                if bond_type not in RDKIT_TO_BOND_ORDER:
                    raise ValueError(
                        f"RDKit bond type {bond_type} is not supported. "
                        f"Supported types: {list(RDKIT_TO_BOND_ORDER.keys())}"
                    )
                order = RDKIT_TO_BOND_ORDER[bond_type]
                atomistic.def_bond(atom1, atom2, order=order)

        self._internal = atomistic

    def _rebuild_atom_mapper(self) -> None:
        """Rebuild atom mapper after structural changes."""
        if self._external is None:
            self._atom_mapper = None
            return

        mol = self._external
        if self._internal is None:
            atomistic = Atomistic()
            self._atom_mapper = _AtomMapper(mol, list(atomistic.atoms))
        else:
            atomistic = self._internal
            if isinstance(atomistic, Wrapper):
                atomistic = atomistic.unwrap()
            self._atom_mapper = _AtomMapper(mol, list(atomistic.atoms))
            self._atom_mapper.ensure_tags()
