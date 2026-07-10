"""Valence completion — fill an atomistic graph's dangling bonds with hydrogens.

Extracting a subgraph (an :class:`~molpy.typifier.affected_region.AffectedRegion`, a
manually sliced fragment, …) leaves the cut atoms under-coordinated: their real
bond-neighbours lay outside the slice. :func:`complete_valence` returns a copy
whose every under-valent atom is capped with hydrogens, so the fragment is a
chemically valid molecule an external tool (antechamber, RDKit, …) can accept.

System-agnostic: it reads element valence, formal charge, and current bond-order
sum, so multiple bonds and common charged atoms are not over-capped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom, Atomistic

#: Typical total valence per element — the bond-order sum a capped atom should reach.
_VALENCE = {
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 2,
    "P": 5,
    "H": 1,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}
#: Cap X–H bond length (Å) keyed by the heavy atom X; a sane default otherwise.
_CAP_LEN = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _orthogonal(v: np.ndarray) -> np.ndarray:
    """A unit vector orthogonal to ``v`` (used when existing bonds cancel out)."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    return _unit(np.cross(v, seed))


def _cap_directions(existing: list[np.ndarray], k: int) -> list[np.ndarray]:
    """``k`` unit directions completing an ~sp3 (tetrahedral) coordination.

    An atom already bonded along ``existing`` gets its remaining ``k`` bonds at
    realistic tetrahedral angles — so cap hydrogens sit where a real neighbour
    would, without clashing. This keeps the sliced fragment's geometry clean
    enough for a charge (sqm) calculation to run directly, no pre-minimisation.
    """
    n = len(existing)
    if n >= 3:
        # The one remaining vertex opposes the sum of the three placed bonds.
        caps = [_unit(-np.sum(existing[:3], axis=0))]
    elif n == 2:
        u1, u2 = existing[0], existing[1]
        bisector = -_unit(u1 + u2)
        normal = (
            _orthogonal(u1)
            if np.linalg.norm(np.cross(u1, u2)) < 1e-6
            else _unit(np.cross(u1, u2))
        )
        half = np.deg2rad(54.75)  # half the 109.47° tetrahedral angle
        c, s = np.cos(half), np.sin(half)
        caps = [_unit(c * bisector + s * normal), _unit(c * bisector - s * normal)]
    elif n == 1:
        # three bonds on a cone 109.47° from the single existing bond, 120° apart.
        u = existing[0]
        e1 = _orthogonal(u)
        e2 = _unit(np.cross(u, e1))
        theta = np.deg2rad(109.47)
        ct, st = np.cos(theta), np.sin(theta)
        caps = [
            _unit(ct * u + st * (np.cos(phi) * e1 + np.sin(phi) * e2))
            for phi in np.deg2rad([0.0, 120.0, 240.0])
        ]
    else:
        # No existing bonds: the four tetrahedral vertices.
        caps = [
            _unit(np.array(v))
            for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])
        ]
    return caps[:k]


def _formal_charge(atom: Atom) -> int:
    """Integer formal charge when present; partial ``charge`` is deliberately ignored."""
    raw = atom.get("formal_charge")
    if raw is None:
        raw = atom.get("formalCharge")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _target_valence(atom: Atom) -> int:
    """Common valence target adjusted for simple formal-charge cases."""
    element = atom.get("element")
    target = _VALENCE.get(element, 0)
    charge = _formal_charge(atom)
    if element == "O" and charge < 0:
        return 1
    if element == "O" and charge > 0:
        return 3
    if element == "N" and charge > 0:
        return 4
    if element == "N" and charge < 0:
        return 2
    return target


def _bond_order(bond) -> float:
    try:
        return float(bond.get("order", 1.0))
    except (TypeError, ValueError):
        return 1.0


def complete_valence(struct: Atomistic) -> Atomistic:
    """Return a copy of ``struct`` with every dangling valence filled by hydrogen.

    An atom is under-coordinated when its current bond-order sum is below its
    element's typical valence (:data:`_VALENCE`); each missing single bond gets a
    hydrogen placed along a valence-completing direction at a standard X-H
    length. Original atoms keep their order (caps are appended), so a downstream
    parameterisation maps position-for-position back onto ``struct`` for its first
    ``struct.n_atoms`` atoms.

    Args:
        struct: the (possibly under-coordinated) atomistic graph to complete.

    Returns:
        A new :class:`~molpy.core.atomistic.Atomistic`; ``struct`` is untouched.
    """
    from molpy.core.atomistic import Atomistic

    mol = Atomistic()
    clone: dict[int, Atom] = {}
    atoms = list(struct.atoms)
    for atom in atoms:
        clone[atom.handle] = mol.def_atom(dict(atom.data))
    for bond in struct.bonds:
        i, j = bond.endpoints
        mol.def_bond(clone[i.handle], clone[j.handle], **dict(bond.data))

    bond_orders = {atom.handle: 0.0 for atom in atoms}
    for bond in struct.bonds:
        order = _bond_order(bond)
        for endpoint in bond.endpoints:
            bond_orders[endpoint.handle] += order

    for atom in atoms:
        element = atom.get("element")
        missing = round(_target_valence(atom) - bond_orders[atom.handle])
        if missing <= 0:
            continue
        p = np.array([atom["x"], atom["y"], atom["z"]], dtype=float)
        existing = [
            _unit(np.array([nb["x"], nb["y"], nb["z"]], dtype=float) - p)
            for nb in struct.get_neighbors(atom)
        ]
        length = _CAP_LEN.get(element, 1.0)
        for direction in _cap_directions(existing, missing):
            hp = p + length * direction
            # No ``name``: AmberTools.parameterize auto-names unnamed atoms
            # ``H<idx>`` — a valid, antechamber-parseable element symbol.
            h = mol.def_atom(
                {
                    "element": "H",
                    "x": float(hp[0]),
                    "y": float(hp[1]),
                    "z": float(hp[2]),
                }
            )
            mol.def_bond(clone[atom.handle], h)
    return mol
