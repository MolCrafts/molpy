"""Conversion between an :class:`~molpy.Atomistic` and an :class:`rdkit.Chem.Mol`.

Two pure functions, :func:`to_rdkit` and :func:`from_rdkit`, each building a
new object and leaving its input untouched. There is no cached pair and no join
key: what an atom was on the molpy side travels with the RDKit atom itself.

**What RDKit owns.** Element, formal charge, bond type and coordinates (one
conformer) are written as RDKit's own fields, and on the way back RDKit's values
win: an RDKit step that kekulizes, re-charges or moves an atom is reflected in
the result. A bracket atom's ``h_count`` (SMILES ``[nH]``) also fixes RDKit's
hydrogen count on that atom.

**What rides along.** Every other atom and bond component (``name``,
``res_name``, ``res_id``, ``type``, partial ``charge``, ``mass``, ``stereo``,
``bond_number``, a bond's ``type`` label, …) and every scalar entry of
:attr:`~molpy.Atomistic.props` is stored as an RDKit property named
``"molpy:<key>"`` (:data:`PROP_PREFIX`) and restored unchanged on whatever atom
or bond RDKit kept.

**What does not survive the round trip:**

- Atoms or bonds RDKit removes (``RemoveHs``, fragment stripping) are absent
  from the result; atoms or bonds it creates (``AddHs``) carry only
  RDKit-owned fields.
- Only atoms and bonds cross. Angles, dihedrals, impropers and ports are
  dropped; regenerate them with ``generate_topology``.
- Stereo is not translated. molpy's ``stereo`` component is not turned into
  RDKit chiral tags or bond stereo, and RDKit's chiral tags, bond stereo and CIP
  labels are not turned back. The ``stereo`` component itself rides along
  untouched, so it is stale if RDKit changed that atom's neighbours.
- When RDKit changes a bond's type, the carried ``bond_number`` is replaced by
  the number the new type implies (none for aromatic). Other carried components
  are never recomputed: an atom whose element RDKit changes keeps its old
  ``mass`` and ``type``.
- Only one conformer comes back (``conf_id`` of :func:`from_rdkit`).
- A graph-level prop that is not a bool, int, float or str is refused by
  :func:`to_rdkit` (``TypeError``) rather than dropped.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from molrs.core import Atomistic, BondNumber, BondOrder
from molrs.core.keys import BOND_NUMBER, BOND_TYPE, ELEMENT, FORMAL_CHARGE
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers

#: Namespace of the RDKit properties that carry molpy components across.
PROP_PREFIX = "molpy:"

_H_COUNT = "h_count"
_XYZ = ("x", "y", "z")
#: Atom components RDKit holds as its own fields, not as carried properties.
_RDKIT_OWNED_ATOM_KEYS = frozenset({ELEMENT.key, FORMAL_CHARGE.key, *_XYZ})

_EMBED_ATTEMPTS = 10
_EMBED_SEED = 0
_UFF_MAX_ITERS = 200

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


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def to_rdkit(atomistic: Atomistic) -> Chem.Mol:
    """Build a sanitized RDKit ``Mol`` from ``atomistic``; the input is not touched.

    RDKit atom ``i`` is ``atomistic``'s atom ``i`` and RDKit bond ``k`` its
    bond ``k``. A conformer is added when every atom has coordinates. See the
    module docstring for what is carried and what is lost.

    Args:
        atomistic: The structure to convert. Every atom needs an ``element``.

    Returns:
        A new, sanitized ``rdkit.Chem.Mol``.

    Raises:
        KeyError: If an atom has no ``element``, or only some atoms have
            coordinates (a missing coordinate is not ``0.0``).
        ValueError: If a bond type has no RDKit counterpart.
        TypeError: If a graph-level prop is not a bool, int, float or str.
        RuntimeError: If RDKit cannot sanitize the result.
    """
    atom_fields = [dict(atom.items()) for atom in atomistic.atoms]
    rd_index = {atom.handle: idx for idx, atom in enumerate(atomistic.atoms)}
    bonds = [
        (rd_index[bond.itom.handle], rd_index[bond.jtom.handle], dict(bond.items()))
        for bond in atomistic.bonds
    ]
    n_h_atoms = [0] * len(atom_fields)
    for i, j, _ in bonds:
        n_h_atoms[i] += atom_fields[j].get(ELEMENT.key) == "H"
        n_h_atoms[j] += atom_fields[i].get(ELEMENT.key) == "H"

    mol = Chem.RWMol()
    _set_props(mol, dict(atomistic.props), where="graph prop")
    for idx, fields in enumerate(atom_fields):
        mol.AddAtom(_rdkit_atom(idx, fields, n_h_atoms[idx]))
    for i, j, fields in bonds:
        bond_type = fields.get(BOND_TYPE.key, int(BondOrder.Single))
        mol.AddBond(i, j, _rdkit_bond_type(bond_type))
        _set_props(mol.GetBondBetweenAtoms(i, j), fields, where="bond")

    positions = _positions(atomistic)
    if positions is not None:
        conf = Chem.Conformer(len(atom_fields))
        for idx, xyz in enumerate(positions):
            conf.SetAtomPosition(idx, tuple(float(v) for v in xyz))
        mol.AddConformer(conf, assignId=True)

    out = mol.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception as exc:
        raise RuntimeError(
            f"RDKit could not sanitize the molecule: {exc}. Fix the bond "
            "types / formal charges on the Atomistic; they are not repaired."
        ) from exc
    return out


def from_rdkit(mol: Chem.Mol, *, conf_id: int = -1) -> Atomistic:
    """Build a new :class:`~molpy.Atomistic` from ``mol``; the input is not touched.

    Atom ``i`` of the result is RDKit atom ``i``. Components carried by
    :func:`to_rdkit` come back on the atoms and bonds RDKit kept; RDKit's own
    fields (element, formal charge, bond type, coordinates) override them.
    See the module docstring for what is lost.

    Args:
        mol: Any RDKit molecule — one from :func:`to_rdkit` after RDKit worked
            on it, or one RDKit built itself.
        conf_id: The conformer to read coordinates from; ``-1`` is RDKit's
            default. A ``mol`` with no conformer gives atoms without coordinates.

    Returns:
        A new ``Atomistic`` with atoms and bonds only.

    Raises:
        ValueError: If a bond type has no molpy counterpart (dative, …).
    """
    atomistic = Atomistic(**_carried(mol))
    positions = (
        mol.GetConformer(conf_id).GetPositions() if mol.GetNumConformers() else None
    )
    atoms = [
        atomistic.def_atom(**_atom_fields(rd_atom, positions))
        for rd_atom in mol.GetAtoms()
    ]
    for rd_bond in mol.GetBonds():
        atomistic.def_bond(
            atoms[rd_bond.GetBeginAtomIdx()],
            atoms[rd_bond.GetEndAtomIdx()],
            **_bond_fields(rd_bond),
        )
    return atomistic


def generate_3d(
    atomistic: Atomistic,
    *,
    add_hydrogens: bool = True,
    optimize: bool = True,
) -> Atomistic:
    """Embed 3D coordinates with RDKit's ETKDGv3, optionally UFF-relaxed.

    A round trip through :func:`to_rdkit` and :func:`from_rdkit`: the result is
    a new ``Atomistic`` and ``atomistic`` is not touched. Hydrogens RDKit adds
    carry RDKit-owned fields only. For molpy's native embedder, use
    :class:`molpy.conformer.Conformer` instead.

    Args:
        atomistic: The structure to embed.
        add_hydrogens: Add explicit hydrogens before embedding.
        optimize: Relax the embedded geometry with UFF.

    Returns:
        A new ``Atomistic`` with coordinates.

    Raises:
        ValueError: If ``atomistic`` has no atoms.
        RuntimeError: If RDKit cannot sanitize or embed the molecule.
    """
    mol = to_rdkit(atomistic)
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True)
    mol = _embed(mol)
    if optimize:
        mol = _optimize_uff(mol)
    return from_rdkit(mol)


# ---------------------------------------------------------------------------
#  Atomistic -> Mol
# ---------------------------------------------------------------------------


def _rdkit_atom(idx: int, fields: dict[str, Any], n_h_atoms: int) -> Chem.Atom:
    if ELEMENT.key not in fields:
        raise KeyError(f"atom {idx} has no {ELEMENT.key!r}")
    rd_atom = Chem.Atom(str(fields[ELEMENT.key]))
    rd_atom.SetFormalCharge(int(fields.get(FORMAL_CHARGE.key) or 0))
    if fields.get(_H_COUNT) is not None:
        # A bracket atom's exact hydrogen count (SMILES ``[nH]``, ``[CH2]``).
        # Hydrogens already present as atoms are bonded, so only the rest
        # become RDKit's explicit-H count, and RDKit adds no implicit ones.
        rd_atom.SetNumExplicitHs(max(int(fields[_H_COUNT]) - n_h_atoms, 0))
        rd_atom.SetNoImplicit(True)
    carried = {k: v for k, v in fields.items() if k not in _RDKIT_OWNED_ATOM_KEYS}
    _set_props(rd_atom, carried, where=f"atom {idx}")
    return rd_atom


def _rdkit_bond_type(bond_type: int) -> Chem.BondType:
    code = int(bond_type)
    if code not in BOND_TYPE_TO_RDKIT:
        raise ValueError(
            f"Bond type {code} is not supported. "
            f"Supported types: {sorted(BOND_TYPE_TO_RDKIT)}"
        )
    return BOND_TYPE_TO_RDKIT[code]


def _positions(atomistic: Atomistic) -> np.ndarray | None:
    """``(n, 3)`` coordinates, or ``None`` when the graph carries none at all.

    A graph where only *some* atoms have coordinates raises (``KeyError``
    from the column read): a missing coordinate is not ``0.0``.
    """
    cols = atomistic.columns()
    if not any(k in cols for k in _XYZ):
        return None
    # ``column`` raises on a hole; ``atoms["x", "y", "z"]`` would yield None.
    return np.stack([np.asarray(atomistic.column(k)) for k in _XYZ], axis=1)


def _set_props(target: Any, fields: dict[str, Any], *, where: str) -> None:
    """Store each of ``fields`` on an RDKit atom, bond or Mol as a typed prop."""
    for key, value in fields.items():
        name = PROP_PREFIX + key
        if isinstance(value, bool | np.bool_):
            target.SetBoolProp(name, bool(value))
        elif isinstance(value, int | np.integer):
            target.SetIntProp(name, int(value))
        elif isinstance(value, float | np.floating):
            target.SetDoubleProp(name, float(value))
        elif isinstance(value, str):
            target.SetProp(name, value)
        else:
            raise TypeError(
                f"{where} {key!r} is a {type(value).__name__}; only bool, int, "
                "float and str cross to RDKit"
            )


# ---------------------------------------------------------------------------
#  Mol -> Atomistic
# ---------------------------------------------------------------------------


def _carried(target: Any) -> dict[str, Any]:
    """The ``molpy:``-prefixed props of an RDKit atom, bond or Mol, unprefixed."""
    props = target.GetPropsAsDict(
        includePrivate=False, includeComputed=False, autoConvertStrings=False
    )
    return {
        key[len(PROP_PREFIX) :]: value
        for key, value in props.items()
        if key.startswith(PROP_PREFIX)
    }


def _atom_fields(rd_atom: Chem.Atom, positions: np.ndarray | None) -> dict[str, Any]:
    fields = {
        k: v for k, v in _carried(rd_atom).items() if k not in _RDKIT_OWNED_ATOM_KEYS
    }
    fields[ELEMENT.key] = rd_atom.GetSymbol()
    if rd_atom.GetFormalCharge() != 0:
        fields[FORMAL_CHARGE.key] = rd_atom.GetFormalCharge()
    if positions is not None:
        fields.update(zip(_XYZ, map(float, positions[rd_atom.GetIdx()]), strict=True))
    return fields


def _bond_fields(rd_bond: Chem.Bond) -> dict[str, Any]:
    rd_type = rd_bond.GetBondType()
    if rd_type not in RDKIT_TO_BOND_TYPE:
        raise ValueError(
            f"RDKit bond type {rd_type} is not supported. "
            f"Supported types: {list(RDKIT_TO_BOND_TYPE)}"
        )
    bond_type = RDKIT_TO_BOND_TYPE[rd_type]
    fields = _carried(rd_bond)
    if fields.get(BOND_TYPE.key) != bond_type or BOND_NUMBER.key not in fields:
        number = BondOrder.from_code(bond_type).implied_number()
        fields[BOND_NUMBER.key] = int(
            number if number is not None else BondNumber.Unknown
        )
    fields[BOND_TYPE.key] = bond_type
    return fields


# ---------------------------------------------------------------------------
#  RDKit 3D generation
# ---------------------------------------------------------------------------


def _embed(mol: Chem.Mol) -> Chem.Mol:
    """Embed 3D coordinates into a copy of ``mol`` with ETKDGv3."""
    if mol.GetNumAtoms() == 0:
        raise ValueError("Cannot embed 3D coordinates for empty molecule")
    mol = Chem.Mol(mol)
    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    for attempt in range(_EMBED_ATTEMPTS):
        params.randomSeed = _EMBED_SEED + attempt
        if rdDistGeom.EmbedMolecule(mol, params) != -1:
            return mol
    raise RuntimeError(
        f"3D embedding failed after {_EMBED_ATTEMPTS} attempts. "
        "The molecule may be too large or have structural issues."
    )


def _optimize_uff(mol: Chem.Mol) -> Chem.Mol:
    """Relax a copy of ``mol`` with UFF; warn when it does not converge."""
    mol = Chem.Mol(mol)
    before = mol.GetConformer().GetPositions()
    code = rdForceFieldHelpers.UFFOptimizeMolecule(mol, maxIters=_UFF_MAX_ITERS)
    if code != 0:
        warnings.warn(
            f"UFF optimization returned code {code}: not converged within "
            f"{_UFF_MAX_ITERS} iterations. The structure may still be improved.",
            UserWarning,
            stacklevel=3,
        )
    elif np.allclose(before, mol.GetConformer().GetPositions(), atol=1e-5):
        warnings.warn(
            "UFF optimization left every coordinate unchanged: the structure "
            "is already at a stationary point of UFF.",
            UserWarning,
            stacklevel=3,
        )
    return mol
