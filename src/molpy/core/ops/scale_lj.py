"""CL&Pol scaleLJ: SAPT-derived Lennard-Jones epsilon scaling.

The non-polarizable CL&P/OPLS-AA epsilon is fit to the *total* condensed-phase
interaction and therefore implicitly contains induction energy. Once explicit
Drude polarization is added (clpol-01), that induction is double-counted, so the
LJ well depth between fragments must be reduced by a SAPT-derived factor

    k_ij = 1 / [ 1 + sum_{non-pol frag} ( C0 r_ij^2 q^2 / alpha + C1 mu^2 / alpha ) ]

with C0 = 0.254952, C1 = 0.106906. The charge-induced-dipole term (q^2/alpha)
carries the r_ij^2 COM-distance prefactor; the dipole-induced-dipole term
(mu^2/alpha) does NOT. Only epsilon is scaled (k_ij <= 1); sigma and atomic
charges are unchanged.

Reference: paduagroup/clandpol scaleLJ + fragment.ff; Goloviznina et al.,
J. Chem. Theory Comput. 15 (2019) 5858, DOI 10.1021/acs.jctc.9b00689.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom
    from molpy.core.forcefield import ForceField

# scaleLJ source constants (paduagroup/clandpol).
C0 = 0.254952
C1 = 0.106906
SIGMA_SCALE = 0.985  # optional sigma scaling for special cases


@dataclass(frozen=True)
class FragmentScaling:
    """SAPT scaling inputs for one molecular fragment.

    Attributes:
        name: Fragment label.
        q: Net charge [e].
        mu: Dipole moment [Debye].
        alpha: Polarizability [A^3] (sum of atomic polarizabilities).
        polarizable: Whether the fragment is treated polarizable; a
            non-polarizable fragment contributes an induction term to k_ij.
    """

    name: str
    q: float
    mu: float
    alpha: float
    polarizable: bool = False


def load_fragment_scaling_data(
    path: str | Path | None = None,
) -> dict[str, FragmentScaling]:
    """Load per-fragment scaling data from a ``clpol_fragments.ff`` file."""
    if path is None:
        from molpy.data.forcefield import get_forcefield_path

        path = get_forcefield_path("clpol_fragments.ff")
    table: dict[str, FragmentScaling] = {}
    for raw in Path(path).read_text().splitlines():
        f = raw.split("#")[0].split()
        if len(f) < 5:
            continue
        name, q, mu, alpha, pol = f[:5]
        table[name] = FragmentScaling(
            name=name,
            q=float(q),
            mu=float(mu),
            alpha=float(alpha),
            polarizable=bool(int(pol)),
        )
    return table


def compute_k_ij(fr_i: FragmentScaling, fr_j: FragmentScaling, r: float) -> float:
    """SAPT epsilon-scaling factor for a fragment pair at COM distance ``r`` [A].

    Mirrors paduagroup/clandpol scaleLJ: a non-polarizable fragment contributes
    an induction term built from the *other* fragment's q/mu and its own alpha.
    """
    if fr_i.alpha <= 0.0 or fr_j.alpha <= 0.0:
        raise ValueError("fragment alpha must be positive (avoids division by zero)")
    denom = 1.0
    if not fr_i.polarizable:
        denom += C0 * r * r * fr_j.q**2 / fr_j.alpha + C1 * fr_j.mu**2 / fr_j.alpha
    if not fr_j.polarizable:
        denom += C0 * r * r * fr_i.q**2 / fr_i.alpha + C1 * fr_i.mu**2 / fr_i.alpha
    return 1.0 / denom


def _com(atoms: Sequence[Atom]) -> tuple[float, float, float]:
    """Mass-weighted centre of mass (geometric centre if masses absent)."""
    total = 0.0
    acc = [0.0, 0.0, 0.0]
    for a in atoms:
        w = a.get("mass") or 1.0
        total += w
        for k, axis in enumerate(("x", "y", "z")):
            acc[k] += w * (a.get(axis) or 0.0)
    return tuple(c / total for c in acc) if total else (0.0, 0.0, 0.0)


def scale_lj(
    ff: ForceField,
    fragments: Mapping[str, Sequence[Atom]],
    frag_data: Mapping[str, FragmentScaling] | None = None,
    *,
    scale_sigma: bool = False,
) -> ForceField:
    """Return a new ForceField with cross-fragment LJ epsilon SAPT-scaled.

    Args:
        ff: Source force field (not mutated).
        fragments: Mapping of fragment label -> its atoms (used for the atom
            types it contains and its centre-of-mass distance).
        frag_data: Per-fragment scaling data; defaults to the built-in
            ``clpol_fragments.ff``.
        scale_sigma: If True, also scale sigma by 0.985 (special cases).

    Returns:
        A deep-copied ForceField whose cross-fragment ``PairType`` epsilons are
        multiplied by k_ij. Intra-fragment pairs, sigma, and charges are
        untouched.
    """
    from molpy.core.forcefield import ForceField, PairType

    if frag_data is None:
        frag_data = load_fragment_scaling_data()

    type_to_frag: dict[str, str] = {}
    coms: dict[str, tuple[float, float, float]] = {}
    for label, atoms in fragments.items():
        if label not in frag_data:
            raise KeyError(f"no scaling data for fragment {label!r}")
        coms[label] = _com(atoms)
        for atom in atoms:
            t = atom.get("type")
            if t is not None:
                type_to_frag[t] = label

    # molrs ForceField is not deep-copyable; clone by merging into a fresh,
    # empty force field (an independent copy of all styles/types).
    new_ff = ForceField(name=ff.name, units=ff.units)
    new_ff.merge(ff)
    for pt in new_ff.get_types(PairType):
        fi = type_to_frag.get(pt.itom.name)
        fj = type_to_frag.get(pt.jtom.name)
        if fi is None or fj is None or fi == fj:
            continue  # only inter-fragment pairs are scaled
        r = sqrt(sum((a - b) ** 2 for a, b in zip(coms[fi], coms[fj])))
        k = compute_k_ij(frag_data[fi], frag_data[fj], r)
        eps = pt.get("epsilon")
        if eps is not None:
            pt["epsilon"] = eps * k
        if scale_sigma and pt.get("sigma") is not None:
            pt["sigma"] = pt.get("sigma") * SIGMA_SCALE
    return new_ff
