"""RDKit adapter for MolPy.

This module provides bidirectional synchronization between MolPy's Atomistic
structures and RDKit's Chem.Mol objects.

RDKit is an optional dependency.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem

from molpy.core.atomistic import Atomistic

from .base import Adapter

MP_ID = "mp_id"

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
    order_float = float(order)
    if order_float not in BOND_ORDER_TO_RDKIT:
        raise ValueError(
            f"Bond order {order_float} is not supported. "
            f"Supported orders: {list(BOND_ORDER_TO_RDKIT.keys())}"
        )
    return BOND_ORDER_TO_RDKIT[order_float]


def _order_from_rdkit(bt: Chem.BondType) -> float:
    if bt not in RDKIT_TO_BOND_ORDER:
        raise ValueError(
            f"RDKit bond type {bt} is not supported. "
            f"Supported types: {list(RDKIT_TO_BOND_ORDER.keys())}"
        )
    return RDKIT_TO_BOND_ORDER[bt]


class _AtomMapper:
    def __init__(self, mol: Chem.Mol, atomistic_atoms: list[Any]) -> None:
        self.mol = mol
        self.atomistic_atoms = atomistic_atoms
        self._forward_map: dict[int, Any] | None = None
        self._reverse_map: dict[int, int] | None = None
        self.build_mapping()

    def build_mapping(self) -> dict[int, Any]:
        if self._forward_map is not None:
            return self._forward_map

        atom_map: dict[int, Any] = {}
        atom_by_id = self._build_atom_id_index()

        mol_atoms = list(self.mol.GetAtoms())

        for rd_idx, rd_atom in enumerate(mol_atoms):
            if not rd_atom.HasProp(MP_ID):
                raise RuntimeError(
                    f"RDKit atom at index {rd_idx} (symbol={rd_atom.GetSymbol()}) "
                    f"does not have {MP_ID} property. "
                    "All RDKit atoms must have this property for mapping."
                )
            hid = int(rd_atom.GetIntProp(MP_ID))
            if hid < 0:
                continue
            if hid not in atom_by_id:
                # Skip atoms that don't exist in atomistic (new atoms)
                # They will be added in _update_atomistic_from_mol
                continue
            atom_map[rd_idx] = atom_by_id[hid]

        self._forward_map = atom_map
        self._reverse_map = {id(v): k for k, v in atom_map.items()}
        return atom_map

    def _build_atom_id_index(self) -> dict[int, Any]:
        by_id: dict[int, Any] = {}
        for i, atom in enumerate(self.atomistic_atoms):
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (element={atom.get('element')}) "
                    f"has no '{MP_ID}' attribute."
                )
            mp_id_int = int(mp_id)
            if mp_id_int in by_id:
                raise ValueError(f"Duplicate {MP_ID} {mp_id_int} found.")
            by_id[mp_id_int] = atom
        return by_id

    def ensure_tags(self) -> None:
        for i, rd_atom in enumerate(self.mol.GetAtoms()):
            if rd_atom.HasProp(MP_ID):
                continue
            if i >= len(self.atomistic_atoms):
                raise RuntimeError(
                    f"RDKit molecule has {self.mol.GetNumAtoms()} atoms, "
                    f"but Atomistic has only {len(self.atomistic_atoms)} atoms."
                )
            ent = self.atomistic_atoms[i]
            ent_mp_id = ent.get(MP_ID)
            if ent_mp_id is None:
                raise ValueError(
                    f"Atomistic atom at index {i} (element={ent.get('element')}) "
                    f"has no '{MP_ID}' attribute."
                )
            rd_atom.SetIntProp(MP_ID, int(ent_mp_id))


class RDKitAdapter(Adapter[Atomistic, Chem.Mol]):
    """Bridge between MolPy's atomistic representation and rdkit.Chem.Mol."""

    def __init__(
        self,
        internal: Atomistic | None = None,
        external: Chem.Mol | None = None,
    ) -> None:
        super().__init__(internal, external)
        self._atom_mapper: _AtomMapper | None = None

        if internal is not None:
            self._ensure_atom_ids()

    @property
    def internal(self) -> Atomistic:
        return self.get_internal()

    @property
    def mol(self) -> Chem.Mol:
        return self.get_external()

    def _ensure_atom_ids(self) -> None:
        if self._internal is None:
            return

        atomistic = self._internal

        max_id = -1
        atoms_without_id = []
        for atom in atomistic.atoms:
            atom_id = atom.get("id")
            if atom_id is None:
                atoms_without_id.append(atom)
            else:
                try:
                    atom_id_int = int(atom_id)
                except (ValueError, TypeError):
                    atoms_without_id.append(atom)
                    continue
                max_id = max(max_id, atom_id_int)

        next_id = max_id + 1
        for atom in atoms_without_id:
            atom["id"] = next_id
            next_id += 1

        # Ensure all atoms have unique mp_id
        # Check for duplicates and only reassign if necessary
        mp_id_to_atoms: dict[int, list[Any]] = {}
        for atom in atomistic.atoms:
            mp_id = atom.get(MP_ID)
            if mp_id is not None:
                try:
                    mp_id_int = int(mp_id)
                    if mp_id_int not in mp_id_to_atoms:
                        mp_id_to_atoms[mp_id_int] = []
                    mp_id_to_atoms[mp_id_int].append(atom)
                except (ValueError, TypeError):
                    pass

        # If all mp_ids are unique and match ids, keep them
        # Otherwise, reassign to ensure uniqueness
        has_duplicates = any(len(atoms) > 1 for atoms in mp_id_to_atoms.values())
        needs_reassign = has_duplicates

        if not needs_reassign:
            # Check if all atoms have mp_id and they match ids
            for atom in atomistic.atoms:
                atom_id = atom.get("id")
                mp_id = atom.get(MP_ID)
                if mp_id is None:
                    needs_reassign = True
                    break
                if atom_id is not None:
                    try:
                        if int(mp_id) != int(atom_id):
                            needs_reassign = True
                            break
                    except (ValueError, TypeError):
                        needs_reassign = True
                        break

        if needs_reassign:
            # Reassign mp_id to match id (if id exists) or use sequential numbering
            for atom in atomistic.atoms:
                atom_id = atom.get("id")
                if atom_id is not None:
                    try:
                        atom[MP_ID] = int(atom_id)
                    except (ValueError, TypeError):
                        # Fallback to sequential if id is invalid
                        pass

            # Check for duplicates after reassignment and fix if needed
            mp_id_to_atoms = {}
            for atom in atomistic.atoms:
                mp_id = atom.get(MP_ID)
                if mp_id is not None:
                    mp_id_int = int(mp_id)
                    if mp_id_int not in mp_id_to_atoms:
                        mp_id_to_atoms[mp_id_int] = []
                    mp_id_to_atoms[mp_id_int].append(atom)

            # If still have duplicates, use sequential numbering
            if any(len(atoms) > 1 for atoms in mp_id_to_atoms.values()):
                for idx, atom in enumerate(atomistic.atoms):
                    atom[MP_ID] = idx + 1
            else:
                # Fill in missing mp_ids
                max_mp_id = max(
                    (
                        int(a.get(MP_ID))
                        for a in atomistic.atoms
                        if a.get(MP_ID) is not None
                    ),
                    default=0,
                )
                next_mp_id = max_mp_id + 1
                for atom in atomistic.atoms:
                    if atom.get(MP_ID) is None:
                        atom[MP_ID] = next_mp_id
                        next_mp_id += 1
        else:
            # All mp_ids are unique and match ids, just fill in missing ones
            max_mp_id = max(
                (
                    int(a.get(MP_ID))
                    for a in atomistic.atoms
                    if a.get(MP_ID) is not None
                ),
                default=0,
            )
            next_mp_id = max_mp_id + 1
            for atom in atomistic.atoms:
                if atom.get(MP_ID) is None:
                    atom_id = atom.get("id")
                    if atom_id is not None:
                        try:
                            atom[MP_ID] = int(atom_id)
                        except (ValueError, TypeError):
                            atom[MP_ID] = next_mp_id
                            next_mp_id += 1
                    else:
                        atom[MP_ID] = next_mp_id
                        next_mp_id += 1

    # ------------------------------------------------------------------
    #  Low-level conversion helpers
    # ------------------------------------------------------------------

    def _build_mol_from_atomistic(self, atomistic: Atomistic) -> Chem.Mol:
        self._ensure_atom_ids()

        mol = Chem.RWMol()
        atom_map: dict[int, int] = {}

        # Validate and collect unique MP_IDs
        atom_by_mp_id: dict[int, Any] = {}
        for atom in atomistic.atoms:
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise RuntimeError(
                    f"Atom {atom} (element={atom.get('element')}) has no '{MP_ID}' attribute."
                )
            mp_id_int = int(mp_id)
            if mp_id_int in atom_by_mp_id:
                raise RuntimeError(
                    f"Duplicate {MP_ID} {mp_id_int} found. Each atom must have a unique MP_ID."
                )
            atom_by_mp_id[mp_id_int] = atom

        for atom in atomistic.atoms:
            mp_id = atom.get(MP_ID)
            if mp_id is None:
                raise RuntimeError(
                    f"Atom {atom} (element={atom.get('element')}) has no '{MP_ID}' attribute."
                )
            mp_id_int = int(mp_id)

            element = atom.get("element") or atom.get("symbol")
            if element is None:
                raise ValueError(
                    f"Atom with {MP_ID}={mp_id_int} has neither 'element' nor 'symbol' attribute."
                )

            rd_atom = Chem.Atom(str(element))
            charge = atom.get("charge")
            if charge is not None:
                rd_atom.SetFormalCharge(int(charge))

            rd_atom.SetIntProp(MP_ID, mp_id_int)
            idx = mol.AddAtom(rd_atom)
            atom_map[id(atom)] = idx

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
                        f"Atom {atom} (element={atom.get('element')}) not found in atom_map"
                    )
                x = atom.get("x")
                y = atom.get("y")
                z = atom.get("z")
                if x is None or y is None or z is None:
                    raise ValueError(
                        f"Atom {atom} (element={atom.get('element')}) has incomplete coordinates. "
                        f"x={x}, y={y}, z={z}."
                    )
                conf.SetAtomPosition(rdkit_idx, (float(x), float(y), float(z)))
            mol.AddConformer(conf, assignId=True)

        final_mol = mol.GetMol()
        try:
            Chem.SanitizeMol(final_mol)
        except Exception:
            # If sanitization fails (e.g., aromatic bonds not in rings),
            # try with less strict settings
            try:
                final_mol.UpdatePropertyCache(strict=False)
                # For aromatic bonds not in rings, convert to single bonds
                for bond in final_mol.GetBonds():
                    if bond.GetBondType() == Chem.BondType.AROMATIC:
                        # Check if it's in a ring
                        if (
                            not bond.GetBeginAtom().IsInRing()
                            or not bond.GetEndAtom().IsInRing()
                        ):
                            # Convert to single bond if not in ring
                            bond.SetBondType(Chem.BondType.SINGLE)
                final_mol.UpdatePropertyCache(strict=False)
            except Exception:
                # If still fails, just return the molecule without sanitization
                pass
        return final_mol

    def _build_atomistic_from_mol(self, mol: Chem.Mol) -> Atomistic:
        atomistic = Atomistic()

        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None

        mp_ids: list[int] = []
        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            if not rd_atom.HasProp(MP_ID):
                raise RuntimeError(
                    f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) does not have {MP_ID} property."
                )
            mp_ids.append(int(rd_atom.GetIntProp(MP_ID)))

        max_existing_id = max((mid for mid in mp_ids if mid >= 0), default=-1)
        next_new_id = max_existing_id + 1

        created = []
        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            atom_id_raw = int(rd_atom.GetIntProp(MP_ID))

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

            created.append(atomistic.def_atom(**props))

        for rd_bond in mol.GetBonds():
            begin_idx = rd_bond.GetBeginAtomIdx()
            end_idx = rd_bond.GetEndAtomIdx()

            bond_type = rd_bond.GetBondType()
            order = _order_from_rdkit(bond_type)
            atomistic.def_bond(created[begin_idx], created[end_idx], order=order)

        return atomistic

    def _update_atomistic_from_mol(
        self,
        mol: Chem.Mol,
        atomistic: Atomistic,
        update_topology: bool = True,
    ) -> None:
        self._external = mol
        self._internal = atomistic

        # Build initial mapping (before adding new atoms)
        # Use existing mapper if available, otherwise create temporary one
        # This will skip atoms that don't exist in atomistic (new atoms with positive mp_id)
        if self._atom_mapper is not None:
            try:
                atom_map = self._atom_mapper.build_mapping()
            except RuntimeError:
                # If mapping fails (e.g., new atoms), create temporary mapper with existing atoms only
                atomistic_atoms = list(atomistic.atoms) if atomistic is not None else []
                temp_mapper = _AtomMapper(mol, atomistic_atoms)
                atom_map = temp_mapper.build_mapping()
        else:
            atomistic_atoms = list(atomistic.atoms) if atomistic is not None else []
            temp_mapper = _AtomMapper(mol, atomistic_atoms)
            atom_map = temp_mapper.build_mapping()
        conf = mol.GetNumConformers() > 0 and mol.GetConformer() or None
        rdkit_to_atom: dict[int, Any] = {}

        max_existing_id = -1
        for existing_atom in atomistic.atoms:
            existing_id = existing_atom.get("id")
            if existing_id is None:
                continue
            try:
                existing_id_int = int(existing_id)
            except (ValueError, TypeError):
                continue
            max_existing_id = max(max_existing_id, existing_id_int)

        next_new_id = max_existing_id + 1

        for rdkit_idx in range(mol.GetNumAtoms()):
            rd_atom = mol.GetAtomWithIdx(rdkit_idx)
            atom = atom_map.get(rdkit_idx)

            if atom is None:
                if not rd_atom.HasProp(MP_ID):
                    raise RuntimeError(
                        f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) does not have {MP_ID} property."
                    )
                atom_id_raw = int(rd_atom.GetIntProp(MP_ID))
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
                continue

            symbol = rd_atom.GetSymbol()
            atom["element"] = symbol
            atom["symbol"] = symbol
            atom["atomic_num"] = rd_atom.GetAtomicNum()

            if atom.get("id") is None:
                if not rd_atom.HasProp(MP_ID):
                    raise RuntimeError(
                        f"RDKit atom at index {rdkit_idx} (symbol={rd_atom.GetSymbol()}) does not have {MP_ID} property, and Atomistic atom has no 'id'."
                    )
                atom["id"] = int(rd_atom.GetIntProp(MP_ID))

            if atom.get(MP_ID) is None and rd_atom.HasProp(MP_ID):
                atom[MP_ID] = int(rd_atom.GetIntProp(MP_ID))

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
                if atom1 is None or atom2 is None:
                    raise RuntimeError(
                        "RDKit bond references an atom missing from the mapping."
                    )

                order = _order_from_rdkit(rd_bond.GetBondType())
                atomistic.def_bond(atom1, atom2, order=order)

        # Rebuild atom mapper after updating atomistic (new atoms may have been added)
        self._rebuild_atom_mapper()

    def sync_to_external(self) -> None:
        super().sync_to_external()

    def _do_sync_to_external(self) -> None:
        atomistic = self._internal
        if atomistic is None:
            return

        new_mol = self._build_mol_from_atomistic(atomistic)
        self._external = new_mol
        self._rebuild_atom_mapper()

    def sync_to_internal(self, update_topology: bool = True) -> None:
        """Sync from external to internal representation.

        Args:
            update_topology: Whether to update topology when internal already exists.
        """
        if self._external is None:
            raise ValueError(
                "Cannot sync to internal: external representation is None. "
                "Set external using set_external() first."
            )
        self._do_sync_to_internal(update_topology=update_topology)

    def _do_sync_to_internal(self, update_topology: bool = True) -> None:
        mol = self._external
        if mol is None:
            return

        if self._internal is None:
            atomistic = self._build_atomistic_from_mol(mol)
        else:
            self._update_atomistic_from_mol(
                mol, self._internal, update_topology=update_topology
            )
            atomistic = self._internal

        self._internal = atomistic

    def _rebuild_atom_mapper(self) -> None:
        if self._external is None:
            self._atom_mapper = None
            return

        mol = self._external
        if self._internal is None:
            atomistic_atoms: list[Any] = []
        else:
            atomistic_atoms = list(self._internal.atoms)

        self._atom_mapper = _AtomMapper(mol, atomistic_atoms)
        if atomistic_atoms:
            self._atom_mapper.ensure_tags()
