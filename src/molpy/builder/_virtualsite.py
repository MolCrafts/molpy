"""Virtual-site augmentation transforms.

A :class:`VirtualSiteBuilder` decorates an existing :class:`~molpy.Atomistic`
with auxiliary particles placed by a rule — Drude shells, TIP4P M-sites, lone
pairs, etc. This is the general pattern (one base class); CL&Pol's polarizer is
just the :class:`DrudeBuilder` instance.

The transform never mutates its input: ``apply`` works on ``copy(struct)`` and
returns the new structure.

Sites are written in molrs's vocabulary: a site is an atom whose ``vsite`` field
names its kind (it carries no ``element``, which is IUPAC-only), a Drude shell
is tied to its core by a ``drudes`` row (``drude/harmonic``: ``k``, ``alpha``,
``thole``) and a constructed site to its frame atoms by a ``virtual_sites`` row
(e.g. ``virtual_site/average3``: ``w1``, ``w2``, ``w3``).
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from copy import copy
from math import sqrt
from typing import Any

import numpy as np
from molrs.core import (
    Atom,
    Atomistic,
    Box,
    Cell,
    DrudeParticle,
    MasslessSite,
    UnitPreset,
    UnitRegistry,
    schema,
)
from molrs.ff.params import clpol_polarizability

# kcal/mol per kJ/mol: alpha.ff states k_D in kJ/mol/Å², molrs is kcal/mol.
_KCAL_PER_KJ = UnitRegistry().quantity(1.0, "kJ/mol").to("kcal/mol").value
# 4πε₀ in e² / (kJ/mol·Å), the paduagroup/clandpol polarizer's unit: the
# inverse of the Coulomb constant (LAMMPS ``real``'s, kcal·Å/(mol·e²)) in kJ.
_FOUR_PI_EPS0 = _KCAL_PER_KJ / UnitPreset("real").coulomb()
# The columns of one CL&Pol ``alpha.ff`` row (``read_clpol_alpha``'s, less
# ``type_name``).
_CLPOL_KEYS = ("m_D", "q_D_sign", "k_D", "alpha", "a_thole")
# TIP4P/2005 O–M distance (Angstrom).
TIP4P_OM = 0.1546

#: A CL&Pol Drude table: ``clpol_polarizability()``'s ``type -> params``
#: mapping, or ``read_clpol_alpha``'s rows (each with a ``type_name``).
Polarizability = Mapping[str, Mapping[str, float]] | Iterable[Mapping[str, Any]]


def _add_relation(
    struct: Atomistic, block: str, nodes: Sequence[Atom], **props: float | str
) -> int:
    """Add a canonical term over live atom references."""
    return struct.add_term(block, nodes, **props).handle


def _polarizability_table(table: Polarizability) -> dict[str, dict[str, float]]:
    """``type -> params`` from either CL&Pol table shape, validated.

    Rows are taken in order, so a type given twice keeps its later row (the
    file's last word, as ``read_clpol_alpha`` documents).
    """
    if isinstance(table, Mapping):
        items = list(table.items())
    else:
        items = []
        for row in table:
            if "type_name" not in row:
                raise ValueError(f"CL&Pol row {dict(row)!r} has no 'type_name'")
            items.append(
                (row["type_name"], {k: v for k, v in row.items() if k != "type_name"})
            )
    normalized: dict[str, dict[str, float]] = {}
    for name, params in items:
        missing = [k for k in _CLPOL_KEYS if k not in params]
        if missing:
            raise ValueError(
                f"CL&Pol parameters for type {name!r} lack {', '.join(missing)}"
            )
        normalized[name] = {k: float(params[k]) for k in _CLPOL_KEYS}
    return normalized


class VirtualSiteBuilder(ABC):
    """Augment a structure with virtual sites placed by a rule.

    Template method :meth:`apply` = copy -> select -> build_sites -> redistribute,
    returning a new :class:`Atomistic` without mutating the input. Subclasses
    implement the three hooks for their specific site scheme.
    """

    def apply(self, struct: Atomistic) -> Atomistic:
        """Return a new structure with virtual sites added; input untouched."""
        work = copy(struct)
        for host in list(self.select(work)):
            sites = self.build_sites(work, host)
            self.redistribute(work, host, sites)
        return work

    @abstractmethod
    def select(self, struct: Atomistic) -> Sequence[Atom]:
        """Return the host atoms that should receive virtual sites."""

    @abstractmethod
    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        """Create live virtual-site particles in ``struct`` for one host."""

    @abstractmethod
    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        """Attach sites to ``struct`` and adjust host charge/mass/topology."""


class DrudeBuilder(VirtualSiteBuilder):
    """CL&Pol polarizer: add a Drude shell to every polarizable heavy atom.

    For each atom whose CL&P type has ``k_D > 0`` in the CL&Pol table
    (``mp.ff.params.clpol_polarizability()``, molrs's ``alpha.ff``), a
    :class:`DrudeParticle` shell is created with charge ``q_D = -sqrt(4*pi*eps0 *
    k_D * alpha)`` and mass ``m_D``, co-located with the core. The core charge and
    mass are reduced by the shell's (``q_core = q_atom - q_D``, ``m_core = m_atom -
    m_D``) so the atom's net charge — and the ion total — is conserved.

    The shell is tied to its core by a molrs ``drudes`` row (``atomi`` the core,
    ``atomj`` the shell) of style ``harmonic`` carrying the
    ``drude/harmonic`` parameters: ``k`` (kcal/mol/Å² in molrs's ``k*r^2``
    form, so ``k = k_D / 2`` against alpha.ff's ``k_D/2 r²``), ``alpha`` (Å³) and
    ``thole`` (CL&Pol's ``a_thole``). Its ``type`` is ``"<core>-<shell>"``.
    The shell carries no ``element``: its ``vsite`` field marks it.
    """

    def __init__(
        self,
        polarizability: Polarizability | None = None,
        *,
        drude_prefix: str = "D",
    ) -> None:
        self.alpha = _polarizability_table(
            polarizability if polarizability is not None else clpol_polarizability()
        )
        # Prefix for the Drude shell atom type, derived from the core type
        # (e.g. core ``NBT`` → shell ``DNBT``). Each polarizable core type thus
        # gets its own shell type, as the LAMMPS DRUDE package expects.
        self.drude_prefix = drude_prefix

    def _is_polarizable(self, atom: Atom) -> bool:
        params = self.alpha.get(atom.get("type"))
        return params is not None and params["k_D"] > 0.0

    def select(self, struct: Atomistic) -> Sequence[Atom]:
        untyped = [
            a for a in struct.atoms if a.get("vsite") is None and a.get("type") is None
        ]
        if untyped:
            raise ValueError(
                f"DrudeBuilder needs a typed graph: {len(untyped)} atom(s) have no "
                "'type', so their CL&Pol polarizability cannot be looked up; "
                "typify the structure first"
            )
        hosts = [a for a in struct.atoms if self._is_polarizable(a)]
        uncharged = [a for a in hosts if a.get("charge") is None]
        if uncharged:
            raise ValueError(
                f"DrudeBuilder needs charged hosts: {len(uncharged)} polarizable "
                "atom(s) have no 'charge' to split with their Drude shell"
            )
        return hosts

    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        params = self.alpha[host["type"]]
        sign = 1.0 if params["q_D_sign"] >= 0 else -1.0
        q_d = sign * sqrt(_FOUR_PI_EPS0 * params["k_D"] * params["alpha"])
        # The shell gets its own force-field type, derived from the core type,
        # so the structure is fully typed before export (no untyped particles).
        attrs: dict[str, Any] = {
            "type": f"{self.drude_prefix}{host['type']}",
            "atomic_number": 0,
            "charge": q_d,
            "mass": params["m_D"],
        }
        # Co-locate with the core if the host carries coordinates.
        for axis in ("x", "y", "z"):
            if host.get(axis) is not None:
                attrs[axis] = host.get(axis)
        return [struct.add_atom(kind=DrudeParticle, **attrs)]

    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        (shell,) = sites
        host["charge"] = host["charge"] - shell["charge"]
        if host.get("mass") is not None:
            host["mass"] = host["mass"] - shell["mass"]
        params = self.alpha[host["type"]]
        _add_relation(
            struct,
            schema.DRUDES,
            (host, shell),
            type=f"{host['type']}-{shell['type']}",
            style="harmonic",
            # alpha.ff's k_D is kJ/mol/Å² in the ½k r² form; molrs's
            # drude/harmonic is k*r² in kcal/mol: k = k_D / 2, in kcal.
            k=params["k_D"] / 2.0 * _KCAL_PER_KJ,
            alpha=params["alpha"],
            thole=params["a_thole"],
        )


class Tip4pBuilder(VirtualSiteBuilder):
    """Place a TIP4P M-site on each water's HOH bisector (proves base generality).

    The massless M-site sits at distance ``d_om`` from O along the HOH bisector;
    the water O charge is moved onto M and O is left neutral. No spring is added —
    contrast with :class:`DrudeBuilder`'s dynamical shell. The O–H vectors are
    minimum-image deltas in ``box`` (a free box when ``None``).

    The construction is recorded as a molrs ``virtual_sites`` row of style
    ``average3`` (``atomi`` = M, ``atomj``/``atomk``/``atoml`` = O, H, H) with
    weights ``w1 = 1 - 2a``, ``w2 = w3 = a``, ``a = d_om / |r_OH1 + r_OH2|``, so
    an engine can rebuild M from the moving O and H. The M-site carries no
    ``element``: its ``vsite`` field marks it.
    """

    def __init__(self, d_om: float = TIP4P_OM, *, box: Box | None = None) -> None:
        self.d_om = d_om
        self.box = box if box is not None else Box(Cell.undefined(), pbc=[False] * 3)

    @staticmethod
    def _xyz(atom: Atom) -> np.ndarray:
        return np.array([atom[axis] for axis in "xyz"], dtype=float)

    @staticmethod
    def _hydrogens(struct: Atomistic, o: Atom) -> list[Atom]:
        return [
            neighbor
            for neighbor, _ in struct.neighbor_bonds(o)
            if neighbor.get("atomic_number") == 1
        ]

    def _bisector_weight(
        self, struct: Atomistic, o: Atom, hydrogens: Sequence[Atom]
    ) -> tuple[np.ndarray, float]:
        """Minimum-image O→H1 + O→H2 and the ``average3`` H weight ``a``."""
        origin = self._xyz(o)
        hs = np.stack([self._xyz(h) for h in hydrogens])
        bis = self.box.delta(np.stack([origin, origin]), hs, minimum_image=True).sum(
            axis=0
        )
        norm = float(np.linalg.norm(bis))
        if norm == 0.0:
            raise ValueError("TIP4P water is linear: its HOH bisector is undefined")
        return bis, self.d_om / norm

    def select(self, struct: Atomistic) -> Sequence[Atom]:
        return [
            a
            for a in struct.atoms
            if a.get("atomic_number") == 8 and len(self._hydrogens(struct, a)) == 2
        ]

    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        bis, a = self._bisector_weight(struct, host, self._hydrogens(struct, host))
        m = self._xyz(host) + a * bis
        return [
            struct.add_atom(
                kind=MasslessSite,
                atomic_number=0,
                charge=host["charge"],
                x=float(m[0]),
                y=float(m[1]),
                z=float(m[2]),
            )
        ]

    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        (msite,) = sites
        hydrogens = self._hydrogens(struct, host)
        _, a = self._bisector_weight(struct, host, hydrogens)
        host["charge"] = 0.0
        _add_relation(
            struct,
            schema.VIRTUAL_SITES,
            (msite, host, *hydrogens),
            style="average3",
            w1=1.0 - 2.0 * a,
            w2=a,
            w3=a,
        )
