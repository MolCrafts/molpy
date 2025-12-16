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

        # STRICT MP_ID-BASED MAPPING ONLY - no position-based fallback
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
            # STRICT MP_ID-BASED MAPPING - must find atom with matching MP_ID
            if hid not in atom_by_id:
                raise RuntimeError(
                    f"RDKit atom at index {rd_idx} (symbol={rd_atom.GetSymbol()}) "
                    f"has {MP_ID}={hid}, but no Atomistic atom with this MP_ID exists. "
                    "This indicates a mapping error. All existing atoms must have matching MP_IDs."
                )
            atom_map[rd_idx] = atom_by_id[hid]

        self._forward_map = atom_map
        self._reverse_map = {id(v): k for k, v in atom_map.items()}
        return atom_map

    def _build_atom_id_index(self) -> dict[int, Any]:
        """Build index of Atomistic atoms by their MP_ID property.

        Raises:
            ValueError: If any atom is missing an MP_ID attribute
        """
        by_id: dict[int, Any] = {}
        for i, atom in enumerate(self.atomistic_atoms):
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (element={atom.get('element')}) "
                    f"has no '{MP_ID}' attribute. All atoms must have a '{MP_ID}' "
                    "for RDKit mapping."
                )
            mp_id_int = int(mp_id)
            if mp_id_int in by_id:
                raise ValueError(
                    f"Duplicate {MP_ID} {mp_id_int} found. "
                    f"Atoms with this {MP_ID} are not uniquely identifiable."
                )
            by_id[mp_id_int] = atom
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
            ent_mp_id = ent.get(MP_ID)
            if ent_mp_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (element={ent.get('element')}) "
                    f"has no '{MP_ID}' attribute. Cannot set MP_ID tag."
                )
            rd_atom.SetIntProp(MP_ID, int(ent_mp_id))


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
        they will be automatically assigned internal IDs. In addition,
        each atom gets a stable MP_ID (``'mp_id'``) used exclusively
        for RDKit ↔ MolPy atom mapping.
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
        """Ensure all atoms in internal Atomistic have ID and MP_ID attributes.

        - Assigns sequential integer ``'id'`` fields starting from 0 if atoms
          don't have them (for general MolPy tooling).
        - Mirrors these IDs into ``MP_ID`` (``'mp_id'``) for use as the *only*
          key in RDKit ↔ MolPy atom mapping.
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

        # Ensure each atom also has an MP_ID used for RDKit mapping
        for atom in atomistic.atoms:
            if atom.get(MP_ID) is None:
                atom[MP_ID] = int(atom["id"])

    # ------------------------------------------------------------------
    #  Low-level conversion helpers
    # ------------------------------------------------------------------

    def _build_mol_from_atomistic(self, atomistic: Atomistic) -> Chem.Mol:
        """Create a fresh RDKit ``Chem.Mol`` from an ``Atomistic``.

        This constructs a new molecule, populating atoms, bonds and (optionally)
        3D coordinates. Atom indices in RDKit are tagged with MP_ID so that
        subsequent updates can be mapped back to MolPy atoms.
        """
        self._ensure_atom_ids()

        mol = Chem.RWMol()
        atom_map: dict[int, int] = {}

        # Validate and collect unique MP_IDs
        atom_by_mp_id: dict[int, Any] = {}
        for atom in atomistic.atoms:
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise RuntimeError(
                    f"Atom {atom} (element={atom.get('element')}) has no '{MP_ID}' attribute. "
                    "All atoms must have an MP_ID for mapping. "
                    "This should have been assigned in _ensure_atom_ids()."
                )
            mp_id_int = int(mp_id)
            if mp_id_int in atom_by_mp_id:
                raise RuntimeError(
                    f"Duplicate {MP_ID} {mp_id_int} found. "
                    "Each atom must have a unique MP_ID."
                )
            atom_by_mp_id[mp_id_int] = atom

        # Create RDKit atoms and build mapping using MP_ID
        for atom in atomistic.atoms:
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise RuntimeError(
                    f"Atom {atom} (element={atom.get('element')}) has no '{MP_ID}' attribute."
                )
            mp_id_int = int(mp_id)

            # MolPy uses 'element'; RDKit uses symbol
            element = atom.get("element") or atom.get("symbol")
            if element is None:
                raise ValueError(
                    f"Atom with {MP_ID}={mp_id_int} has neither 'element' nor 'symbol' attribute. "
                    "All atoms must have an element."
                )
            rd_atom = Chem.Atom(str(element))

            charge = atom.get("charge")
            if charge is not None:
                rd_atom.SetFormalCharge(int(charge))

            rd_atom.SetIntProp(MP_ID, mp_id_int)
            idx = mol.AddAtom(rd_atom)

            atom_map[id(atom)] = idx

        # Bonds
        for bond in atomistic.bonds:
            begin_idx = atom_map.get(id(bond.itom))
            end_idx = atom_map.get(id(bond.jtom))
            if begin_idx is None or end_idx is None:
                continue

            bt = _rdkit_bond_type(bond.get("order"))
            mol.AddBond(begin_idx, end_idx, bt)

        # Optional coordinates: expect x/y/z on atoms
        if any(atom.get("x") is not None for atom in atomistic.atoms):
            conf = Chem.Conformer(mol.GetNumAtoms())
            for atom in atomistic.atoms:
                rdkit_idx = atom_map.get(id(atom))
                if rdkit_idx is None:
                    raise RuntimeError(
                        f"Atom {atom} (element={atom.get('element')}) not found in atom_map "
                        "when setting coordinates."
                    )
                x = atom.get("x")
                y = atom.get("y")
                z = atom.get("z")
                if x is None or y is None or z is None:
                    raise ValueError(
                        f"Atom {atom} (element={atom.get('element')}) has incomplete coordinates. "
                        f"x={x}, y={y}, z={z}. All coordinates must be present."
                    )
                conf.SetAtomPosition(rdkit_idx, (float(x), float(y), float(z)))
            mol.AddConformer(conf, assignId=True)

        final_mol = mol.GetMol()
        # Let RDKit report sanitization errors directly (no try/except)
        Chem.SanitizeMol(final_mol)
        return final_mol

    def _build_atomistic_from_mol(self, mol: Chem.Mol) -> Atomistic:
        """Create a fresh ``Atomistic`` from a RDKit ``Chem.Mol``.

        A new ``Atomistic`` is allocated and populated from ``mol``. MP_ID tags
        on RDKit atoms are used to assign both ``'id'`` and ``'mp_id'`` on the
        created MolPy atoms.
        """
        atomistic = Atomistic()

        # For pure creation there is no existing Atomistic, so we do NOT use
        # AtomMapper here. Instead we seed IDs directly from RDKit MP_ID tags.
        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None

        max_existing_id = -1
        next_new_id = max_existing_id + 1

        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            if not rd_atom.HasProp(MP_ID):
                raise RuntimeError(
                    f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) "
                    f"does not have {MP_ID} property. "
                    "All RDKit atoms must have this property for mapping."
                )
            atom_id_raw = rd_atom.GetIntProp(MP_ID)

            if atom_id_raw < 0:
                atom_id = next_new_id
                next_new_id += 1
            else:
                atom_id = atom_id_raw

            symbol = rd_atom.GetSymbol()
            props: dict[str, Any] = {
                "element": symbol,
                "symbol": symbol,
                "atomic_num": rd_atom.GetAtomicNum(),
                "id": atom_id,
                MP_ID: atom_id,
            }

            if rd_atom.GetFormalCharge() != 0:
                props["charge"] = rd_atom.GetFormalCharge()

            if conf is not None:
                pos = conf.GetAtomPosition(rdkit_idx)
                props["x"] = float(pos.x)
                props["y"] = float(pos.y)
                props["z"] = float(pos.z)

            atomistic.def_atom(**props)

        # Bonds
        rdkit_to_atom: dict[int, Any] = {}
        for idx, atom in enumerate(atomistic.atoms):
            rdkit_to_atom[idx] = atom

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

        return atomistic

    def _update_mol_from_atomistic(
        self, atomistic: Atomistic, mol: Chem.Mol
    ) -> Chem.Mol:
        """Update an existing RDKit ``Chem.Mol`` from an ``Atomistic``.

        The topology is rebuilt from the Atomistic object; MP_ID tags on atoms
        are preserved and reused for mapping.
        """
        self._ensure_atom_ids()
        # Just rebuild from scratch and return new sanitized molecule
        # while keeping MP_ID values consistent.
        self._external = mol  # keep reference type stable
        new_mol = self._build_mol_from_atomistic(atomistic)
        return new_mol

    def _update_atomistic_from_mol(
        self,
        mol: Chem.Mol,
        atomistic: Atomistic,
        update_topology: bool = True,
    ) -> None:
        """Update an existing ``Atomistic`` from a RDKit ``Chem.Mol``.

        Uses ``_AtomMapper`` and MP_ID tags to preserve existing atom objects
        wherever possible. Optionally rebuilds topology from the RDKit molecule.
        """
        self._external = mol
        self._internal = atomistic
        self._rebuild_atom_mapper()
        if self._atom_mapper is None:
            raise RuntimeError("Atom mapper not initialized")

        atom_map = self._atom_mapper.build_mapping()

        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None
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
                if not rd_atom.HasProp(MP_ID):
                    raise RuntimeError(
                        f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) "
                        f"does not have {MP_ID} property. "
                        "All RDKit atoms must have this property for mapping."
                    )
                atom_id_raw = rd_atom.GetIntProp(MP_ID)

                if atom_id_raw < 0:
                    atom_id = next_new_id
                    next_new_id += 1
                else:
                    atom_id = atom_id_raw

                symbol = rd_atom.GetSymbol()
                props: dict[str, Any] = {
                    "element": symbol,
                    "symbol": symbol,
                    "atomic_num": rd_atom.GetAtomicNum(),
                    "id": atom_id,
                    MP_ID: atom_id,
                }

                if rd_atom.GetFormalCharge() != 0:
                    props["charge"] = rd_atom.GetFormalCharge()

                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    props["x"] = float(pos.x)
                    props["y"] = float(pos.y)
                    props["z"] = float(pos.z)

                new_atom = atomistic.def_atom(**props)
                rdkit_to_atom[rdkit_idx] = new_atom
            else:
                # Existing atom - update properties but keep identity
                symbol = rd_atom.GetSymbol()
                atom["element"] = symbol
                atom["symbol"] = symbol
                atom["atomic_num"] = rd_atom.GetAtomicNum()

                if atom.get("id") is None:
                    if not rd_atom.HasProp(MP_ID):
                        raise RuntimeError(
                            f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) "
                            f"does not have {MP_ID} property, and Atomistic atom has no 'id'. "
                            "Cannot establish mapping."
                        )
                    atom["id"] = rd_atom.GetIntProp(MP_ID)

                if atom.get(MP_ID) is None and rd_atom.HasProp(MP_ID):
                    atom[MP_ID] = rd_atom.GetIntProp(MP_ID)

                if rd_atom.GetFormalCharge() != 0:
                    atom["charge"] = rd_atom.GetFormalCharge()

                if conf is not None:
                    pos = conf.GetAtomPosition(rdkit_idx)
                    atom["x"] = float(pos.x)
                    atom["y"] = float(pos.y)
                    atom["z"] = float(pos.z)

                rdkit_to_atom[rdkit_idx] = atom

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

        # Fresh build / update share the same low-level helper
        new_mol = self._build_mol_from_atomistic(atomistic)
        self._external = new_mol
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

        if self._internal is None:
            atomistic = self._build_atomistic_from_mol(mol)
        else:
            atomistic = self._internal
            if isinstance(atomistic, Wrapper):
                atomistic = atomistic.unwrap()
            self._update_atomistic_from_mol(
                mol, atomistic, update_topology=update_topology
            )

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
