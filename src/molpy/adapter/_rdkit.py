"""RDKit adapter for MolPy.

Bidirectional synchronisation between an :class:`~molpy.Atomistic`
and an :class:`rdkit.Chem.Mol`. RDKit is an optional dependency.

The two representations are joined by one integer tag, :data:`MP_ID`, stored
as an atom component on the molpy side and as an atom property on the RDKit
side. RDKit reorders and adds atoms freely (``AddHs``, ``RemoveHs``), so the
join cannot be positional. An RDKit atom carrying a **negative** tag is one
RDKit created and molpy has not seen yet; it becomes a new atom on the next
sync. An RDKit atom with no tag at all is an error — build the Mol through
:meth:`RdkitAdapter.sync_to_external` or tag it yourself.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

from molrs.core.keys import FORMAL_CHARGE
from molrs.core import Atomistic, BondNumber, BondOrder

from ._adapter import Adapter

#: The join key between an ``Atomistic`` atom and an RDKit atom (see module doc).
MP_ID = "mp_id"

#: RDKit models aromaticity the same way molpy does — as a bond *type*
#: alongside single/double/triple, never as a fractional order. The two
#: alphabets therefore map one-to-one, keyed by the ``bond_type`` column's code
#: (``int(molrs.core.BondOrder.X)``, ``molrs.core.keys.BOND_TYPE``).
BOND_TYPE_TO_RDKIT: dict[int, Chem.BondType] = {
    int(BondOrder.Single): Chem.BondType.SINGLE,
    int(BondOrder.Double): Chem.BondType.DOUBLE,
    int(BondOrder.Triple): Chem.BondType.TRIPLE,
    int(BondOrder.Aromatic): Chem.BondType.AROMATIC,
}
RDKIT_TO_BOND_TYPE: dict[Chem.BondType, int] = {
    rd: mp for mp, rd in BOND_TYPE_TO_RDKIT.items()
}


def _implied_number(bond_type: int) -> int:
    """The localized integer order (``BondNumber`` code) a bond type implies;
    aromatic and unknown bonds carry none, 0 (the Kekulé phase is a separate
    fact, and RDKit keeps its own)."""
    number = BondOrder.from_code(bond_type).implied_number()
    return int(number) if number is not None else int(BondNumber.Unknown)


def _rdkit_bond_type(bond_type: int) -> Chem.BondType:
    code = int(bond_type)
    if code not in BOND_TYPE_TO_RDKIT:
        raise ValueError(
            f"Bond type {code} is not supported. "
            f"Supported types: {sorted(BOND_TYPE_TO_RDKIT)}"
        )
    return BOND_TYPE_TO_RDKIT[code]


def _bond_type_from_rdkit(bt: Chem.BondType) -> int:
    if bt not in RDKIT_TO_BOND_TYPE:
        raise ValueError(
            f"RDKit bond type {bt} is not supported. "
            f"Supported types: {list(RDKIT_TO_BOND_TYPE.keys())}"
        )
    return RDKIT_TO_BOND_TYPE[bt]


class RdkitAdapter(Adapter[Atomistic, Chem.Mol]):
    """Bridge between MolPy's atomistic representation and ``rdkit.Chem.Mol``."""

    def __init__(
        self,
        internal: Atomistic | None = None,
        external: Chem.Mol | None = None,
    ) -> None:
        super().__init__(internal, external)
        if internal is not None:
            self._tag_atoms(internal)

    @property
    def internal(self) -> Atomistic:
        return self.get_internal()

    @property
    def mol(self) -> Chem.Mol:
        return self.get_external()

    def generate_3d(
        self,
        *,
        add_hydrogens: bool = True,
        optimize: bool = True,
    ) -> Atomistic:
        """Add hydrogens, embed 3D coordinates, and optimize geometry via RDKit.

        Returns a new :class:`~molpy.Atomistic` with coordinates;
        this adapter is not mutated. For molpy's native (native) embedder, use
        :class:`molpy.conformer.Conformer` instead.
        """
        working = self.copy()
        if not working.has_external():
            working.sync_to_external()
        mol = Chem.Mol(working.get_external())

        if add_hydrogens:
            mol = _add_hydrogens(mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise RuntimeError(
                f"Sanitization failed: {exc}. "
                "The molecule may have invalid valency or other issues."
            ) from exc
        mol = _embed(mol, 10, 0)
        if optimize:
            mol = _sanitize(mol)
            if mol.GetNumConformers() > 0:
                mol = _optimize_uff(mol, 200, False)

        working.set_external(mol)
        working.sync_to_internal()
        return working.get_internal()

    # ------------------------------------------------------------------
    #  The join key
    # ------------------------------------------------------------------

    @staticmethod
    def _tag_atoms(atomistic: Atomistic) -> None:
        """Give every atom a unique :data:`MP_ID`, keeping the tags it already has."""
        handles = atomistic.entities()
        if not handles:
            return
        if MP_ID not in atomistic.columns():
            for tag, handle in enumerate(handles):
                atomistic.set(handle, MP_ID, tag)
            return
        valid = atomistic.validity(MP_ID)
        if valid.all():
            tags = np.asarray(atomistic.column(MP_ID), dtype=np.int64)
            untagged: list[int] = []
        else:
            tags = np.array(
                [int(atomistic.get(h, MP_ID)) for h, ok in zip(handles, valid) if ok],
                dtype=np.int64,
            )
            untagged = [h for h, ok in zip(handles, valid) if not ok]
        if np.unique(tags).size != tags.size:
            raise ValueError(f"duplicate {MP_ID} tags: every atom needs its own")
        next_tag = int(tags.max()) + 1 if tags.size else 0
        for handle in untagged:
            atomistic.set(handle, MP_ID, next_tag)
            next_tag += 1

    @staticmethod
    def _tag_of(rd_atom: Chem.Atom) -> int:
        if not rd_atom.HasProp(MP_ID):
            raise RuntimeError(
                f"RDKit atom {rd_atom.GetIdx()} ({rd_atom.GetSymbol()}) has no "
                f"{MP_ID} property; build the Mol through sync_to_external() or "
                "tag it (a negative tag marks an atom molpy has not seen)"
            )
        return int(rd_atom.GetIntProp(MP_ID))

    @staticmethod
    def _next_tag(mol: Chem.Mol, *taken: int) -> int:
        """One above every non-negative tag on ``mol`` and in ``taken``."""
        highest = max(taken, default=-1)
        for rd_atom in mol.GetAtoms():
            if rd_atom.HasProp(MP_ID):
                highest = max(highest, int(rd_atom.GetIntProp(MP_ID)))
        return highest + 1

    # ------------------------------------------------------------------
    #  Atomistic -> Mol
    # ------------------------------------------------------------------

    def _build_mol_from_atomistic(self, atomistic: Atomistic) -> Chem.Mol:
        self._tag_atoms(atomistic)
        handles = atomistic.entities()
        elements = atomistic.column("element")  # a hole is a KeyError
        tags = atomistic.column(MP_ID)
        charges = self._formal_charges(atomistic)
        positions = self._positions(atomistic)

        mol = Chem.RWMol()
        for element, tag, charge in zip(elements, tags, charges, strict=True):
            rd_atom = Chem.Atom(str(element))
            if charge:
                rd_atom.SetFormalCharge(int(charge))
            rd_atom.SetIntProp(MP_ID, int(tag))
            mol.AddAtom(rd_atom)

        rd_index = {handle: idx for idx, handle in enumerate(handles)}
        for bond in atomistic.bonds:
            mol.AddBond(
                rd_index[bond.itom.handle],
                rd_index[bond.jtom.handle],
                _rdkit_bond_type(bond.get("bond_type", int(BondOrder.Single))),
            )

        if positions is not None:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for idx, xyz in enumerate(positions):
                conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
            mol.AddConformer(conf, assignId=True)

        final_mol = mol.GetMol()
        try:
            Chem.SanitizeMol(final_mol)
        except Exception as exc:
            raise RuntimeError(
                f"RDKit could not sanitize the molecule: {exc}. Fix the bond "
                "types / formal charges on the Atomistic; they are not repaired."
            ) from exc
        return final_mol

    @staticmethod
    def _formal_charges(atomistic: Atomistic) -> list[int]:
        """Per-atom formal charge, 0 where the component is absent (neutral)."""
        if FORMAL_CHARGE.key not in atomistic.columns():
            return [0] * len(atomistic.entities())
        if atomistic.validity(FORMAL_CHARGE.key).all():
            return [int(q) for q in atomistic.column(FORMAL_CHARGE.key)]
        return [
            int(atomistic.get(h, FORMAL_CHARGE.key) or 0) for h in atomistic.entities()
        ]

    @staticmethod
    def _positions(atomistic: Atomistic) -> np.ndarray | None:
        """``(n, 3)`` coordinates, or ``None`` when the graph carries none at all.

        A graph where only *some* atoms have coordinates raises (``KeyError``
        from the column read): a missing coordinate is not ``0.0``.
        """
        cols = atomistic.columns()
        if not any(k in cols for k in ("x", "y", "z")):
            return None
        # ``column`` raises on a hole; ``atoms["x", "y", "z"]`` would yield None.
        return np.stack(
            [np.asarray(atomistic.column(k)) for k in ("x", "y", "z")],
            axis=1,
        )

    # ------------------------------------------------------------------
    #  Mol -> Atomistic
    # ------------------------------------------------------------------

    @staticmethod
    def _atom_props(
        rd_atom: Chem.Atom, tag: int, position: Any | None
    ) -> dict[str, Any]:
        props: dict[str, Any] = {"element": rd_atom.GetSymbol(), MP_ID: tag}
        if rd_atom.GetFormalCharge() != 0:
            props[FORMAL_CHARGE.key] = rd_atom.GetFormalCharge()
        if position is not None:
            props["x"] = float(position[0])
            props["y"] = float(position[1])
            props["z"] = float(position[2])
        return props

    def _build_atomistic_from_mol(self, mol: Chem.Mol) -> Atomistic:
        atomistic = Atomistic()
        positions = (
            mol.GetConformer().GetPositions() if mol.GetNumConformers() > 0 else None
        )
        next_tag = self._next_tag(mol)

        created = []
        for idx, rd_atom in enumerate(mol.GetAtoms()):
            tag = self._tag_of(rd_atom)
            if tag < 0:
                tag = next_tag
                next_tag += 1
                rd_atom.SetIntProp(MP_ID, tag)
            position = positions[idx] if positions is not None else None
            created.append(
                atomistic.def_atom(**self._atom_props(rd_atom, tag, position))
            )

        for rd_bond in mol.GetBonds():
            bond_type = _bond_type_from_rdkit(rd_bond.GetBondType())
            atomistic.def_bond(
                created[rd_bond.GetBeginAtomIdx()],
                created[rd_bond.GetEndAtomIdx()],
                bond_type=bond_type,
                bond_number=_implied_number(bond_type),
            )
        return atomistic

    def _update_atomistic_from_mol(
        self,
        mol: Chem.Mol,
        atomistic: Atomistic,
        update_topology: bool = True,
    ) -> None:
        """Fold ``mol`` back onto ``atomistic``, joined by :data:`MP_ID`.

        One pass over the RDKit atoms: a known tag updates that atom's element,
        formal charge and coordinates in place; a negative or unknown tag spawns
        a new atom (and writes its tag back onto the RDKit atom so the join
        holds on the next sync). Bonds are rebuilt from RDKit when
        ``update_topology`` is set.
        """
        self._tag_atoms(atomistic)
        by_tag = dict(
            zip(
                (int(t) for t in atomistic.column(MP_ID)),
                atomistic.atoms,
                strict=True,
            )
        )
        next_tag = self._next_tag(mol, *by_tag)
        positions = (
            mol.GetConformer().GetPositions() if mol.GetNumConformers() > 0 else None
        )

        atom_of_rd = []
        for idx, rd_atom in enumerate(mol.GetAtoms()):
            tag = self._tag_of(rd_atom)
            position = positions[idx] if positions is not None else None
            atom = by_tag.get(tag) if tag >= 0 else None
            if atom is None:
                if tag < 0:
                    tag = next_tag
                    next_tag += 1
                    rd_atom.SetIntProp(MP_ID, tag)
                atom = atomistic.def_atom(**self._atom_props(rd_atom, tag, position))
                by_tag[tag] = atom
                atom_of_rd.append(atom)
                continue

            atom["element"] = rd_atom.GetSymbol()
            charge = rd_atom.GetFormalCharge()
            if charge != 0 or FORMAL_CHARGE.key in atom:
                atom[FORMAL_CHARGE.key] = charge
            if position is not None:
                atom["x", "y", "z"] = (float(v) for v in position)
            atom_of_rd.append(atom)

        if update_topology:
            existing = list(atomistic.bonds)
            if existing:
                atomistic.remove_link(*existing)
            for rd_bond in mol.GetBonds():
                bond_type = _bond_type_from_rdkit(rd_bond.GetBondType())
                atomistic.def_bond(
                    atom_of_rd[rd_bond.GetBeginAtomIdx()],
                    atom_of_rd[rd_bond.GetEndAtomIdx()],
                    bond_type=bond_type,
                    bond_number=_implied_number(bond_type),
                )

    # ------------------------------------------------------------------
    #  Adapter protocol
    # ------------------------------------------------------------------

    def _do_sync_to_external(self) -> None:
        if self._internal is None:
            return
        self._external = self._build_mol_from_atomistic(self._internal)

    def sync_to_internal(self, update_topology: bool = True) -> None:
        """Sync from external to internal representation.

        Args:
            update_topology: Whether to rebuild bonds when internal already exists.
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
            self._internal = self._build_atomistic_from_mol(mol)
        else:
            self._update_atomistic_from_mol(
                mol, self._internal, update_topology=update_topology
            )

    def copy(self) -> RdkitAdapter:
        """A new adapter over deep copies of both representations."""
        new_internal = self._internal.copy() if self._internal is not None else None
        new_external = Chem.Mol(self._external) if self._external is not None else None
        return RdkitAdapter(internal=new_internal, external=new_external)


# ---------------------------------------------------------------------------
# RDKit 3D generation / geometry optimization
# ---------------------------------------------------------------------------


def _sanitize(mol: Chem.Mol) -> Chem.Mol:
    """Sanitize *mol* for force-field readiness.

    Tries strict sanitization first; falls back to inferring hybridization
    from connectivity when that fails.

    Returns:
        A sanitized copy (the input is never mutated).

    Raises:
        RuntimeError: If both strict and fallback sanitization fail.
    """
    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        try:
            mol.UpdatePropertyCache(strict=False)
            for atom in mol.GetAtoms():
                if atom.GetHybridization() == Chem.HybridizationType.UNSPECIFIED:
                    degree = atom.GetDegree()
                    hyb = {
                        0: Chem.HybridizationType.S,
                        1: Chem.HybridizationType.SP,
                        2: Chem.HybridizationType.SP2,
                        3: Chem.HybridizationType.SP3,
                    }.get(degree, Chem.HybridizationType.SP3D)
                    atom.SetHybridization(hyb)
            mol.UpdatePropertyCache(strict=False)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to prepare molecule for optimization: {e}. "
                f"Fallback also failed: {e2}. "
                "The molecule may have structural issues."
            ) from e
    return mol


def _optimize_uff(
    mol: Chem.Mol,
    max_iters: int,
    raise_on_failure: bool,
) -> Chem.Mol:
    """Run UFF optimization on a copy of *mol*.

    Returns:
        A new Mol with optimized coordinates.
    """
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)
    before = mol.GetConformer().GetPositions() if mol.GetNumConformers() > 0 else None

    code = rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
    if code != 0:
        msg = (
            f"UFF optimization returned code {code}. "
            f"Code 1 typically means convergence not reached within {max_iters} "
            "iterations. The structure may still be improved."
        )
        if raise_on_failure:
            raise RuntimeError(msg)
        warnings.warn(msg, UserWarning)
    elif before is not None and np.allclose(
        before, mol.GetConformer().GetPositions(), atol=1e-5
    ):
        warnings.warn(
            "UFF optimization left every coordinate unchanged: the structure "
            "is already at a stationary point of UFF, or the optimizer did not run.",
            UserWarning,
        )
    return mol


def _add_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    """Add explicit hydrogens; each new atom gets a negative :data:`MP_ID`.

    Returns:
        A new Mol with explicit hydrogens.
    """
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache(strict=False)
    original_count = mol.GetNumAtoms()
    mol = Chem.AddHs(mol, addCoords=True)
    for idx in range(original_count, mol.GetNumAtoms()):
        rd_atom = mol.GetAtomWithIdx(idx)
        if not rd_atom.HasProp(MP_ID):
            rd_atom.SetIntProp(MP_ID, -(idx - original_count + 1))
    return mol


def _embed(
    mol: Chem.Mol,
    max_attempts: int,
    random_seed: int | None,
) -> Chem.Mol:
    """Embed 3D coordinates into a copy of *mol*.

    Returns:
        A new Mol with 3D coordinates.

    Raises:
        ValueError: If the molecule has no atoms.
        RuntimeError: If embedding fails after *max_attempts*.
    """
    mol = Chem.Mol(mol)
    if mol.GetNumAtoms() == 0:
        raise ValueError("Cannot embed 3D coordinates for empty molecule")

    params = rdDistGeom.ETKDGv3()
    if random_seed is not None:
        params.randomSeed = int(random_seed)
    params.useRandomCoords = True

    embed_result = rdDistGeom.EmbedMolecule(mol, params)
    attempts = 1
    while embed_result == -1 and attempts < max_attempts:
        params.useRandomCoords = True
        if random_seed is not None:
            params.randomSeed = int(random_seed) + attempts
        embed_result = rdDistGeom.EmbedMolecule(mol, params)
        attempts += 1

    if embed_result == -1:
        raise RuntimeError(
            f"3D embedding failed after {max_attempts} attempts. "
            "The molecule may be too large or have structural issues."
        )
    return mol
