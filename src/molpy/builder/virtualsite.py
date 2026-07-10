"""Virtual-site augmentation transforms.

A :class:`VirtualSiteBuilder` decorates an existing :class:`~molpy.core.Atomistic`
with auxiliary particles placed by a rule — Drude shells, TIP4P M-sites, lone
pairs, etc. This is the general pattern (one base class); CL&Pol's polarizer is
just the :class:`DrudeBuilder` instance.

The transform never mutates its input: ``apply`` works on ``struct.copy()`` and
returns the new structure.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

from molpy.core import fields
from molpy.core.atomistic import Atom, Atomistic, Bond, DrudeParticle, MasslessSite

# 4*pi*eps0 in e^2 / (kJ/mol * A), per paduagroup/clandpol polarizer.
FOUR_PI_EPS0 = 0.0007197587
# Canonical CL&Pol Drude spring constant (alpha.ff ships 4184.0 for every type).
# The actual per-type value is read from alpha.ff (params["k_D"]); this constant
# is only a reference for the shipped data. Convention check: alpha.ff defines
# kforce "in the form k/2 r_D^2" and molpy BondHarmonic is U = 1/2 K (r - r0)^2
# (potential/bond/harmonic.py:56) — identical 1/2 prefactor, so k_D maps onto K
# one-to-one with r0 = 0 (no factor-of-2 conversion).
K_DRUDE = 4184.0
# TIP4P/2005 O–M distance (Angstrom).
TIP4P_OM = 0.1546


def load_polarizability(path: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Load per-atom-type Drude/polarizability parameters from an ``alpha.ff`` file.

    Returns a mapping ``type_name -> {m_D, q_D_sign, k_D, alpha, a_thole}``
    transcribed from the paduagroup/clandpol distribution (units: u, e, kJ/mol/A^2,
    A^3, dimensionless). Comment lines (``#``) and blanks are skipped.
    """
    if path is None:
        from molpy.data.forcefield import get_forcefield_path

        path = get_forcefield_path("alpha.ff")
    table: dict[str, dict[str, float]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#")[0].split()
        if len(line) < 6:
            continue
        name, m_d, q_d, k_d, alpha, a_thole = line[:6]
        table[name] = {
            "m_D": float(m_d),
            "q_D_sign": float(q_d),
            "k_D": float(k_d),
            "alpha": float(alpha),
            "a_thole": float(a_thole),
        }
    return table


class VirtualSiteBuilder(ABC):
    """Augment a structure with virtual sites placed by a rule.

    Template method :meth:`apply` = copy -> select -> build_sites -> redistribute,
    returning a new :class:`Atomistic` without mutating the input. Subclasses
    implement the three hooks for their specific site scheme.
    """

    def apply(self, struct: Atomistic) -> Atomistic:
        """Return a new structure with virtual sites added; input untouched."""
        work = struct.copy()
        for host in list(self.select(work)):
            sites = self.build_sites(work, host)
            self.redistribute(work, host, sites)
        return work

    @abstractmethod
    def select(self, struct: Atomistic) -> Sequence[Atom]:
        """Return the host atoms that should receive virtual sites."""

    @abstractmethod
    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        """Construct (unbound) virtual-site particles for one host."""

    @abstractmethod
    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        """Attach sites to ``struct`` and adjust host charge/mass/topology."""


class DrudeBuilder(VirtualSiteBuilder):
    """CL&Pol polarizer: add a Drude shell to every polarizable heavy atom.

    For each atom whose CL&P type has ``k_D > 0`` in ``alpha.ff``, a
    :class:`DrudeParticle` shell is created with charge ``q_D = -sqrt(4*pi*eps0 *
    k_D * alpha)`` and mass ``m_D``, co-located with the core. The core charge and
    mass are reduced by the shell's (``q_core = q_atom - q_D``, ``m_core = m_atom -
    m_D``) so the atom's net charge — and the ion total — is conserved. The shell
    is bound to its core by a harmonic spring (force constant ``K_DRUDE``).
    """

    def __init__(
        self,
        polarizability: dict[str, dict[str, float]] | None = None,
        *,
        drude_prefix: str = "D",
    ) -> None:
        self.alpha = (
            polarizability if polarizability is not None else load_polarizability()
        )
        # Prefix for the Drude shell atom type, derived from the core type
        # (e.g. core ``NBT`` → shell ``DNBT``). Each polarizable core type thus
        # gets its own shell type, as the LAMMPS DRUDE package expects.
        self.drude_prefix = drude_prefix
        # All core–shell springs share one harmonic bond type.
        self.drude_bond_type = "DRUDE"

    def _is_polarizable(self, atom: Atom) -> bool:
        params = self.alpha.get(atom.get("type"))
        return params is not None and params["k_D"] > 0.0

    def select(self, struct: Atomistic) -> Sequence[Atom]:
        return [a for a in struct.atoms if self._is_polarizable(a)]

    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        params = self.alpha[host.get("type")]
        k_d, alpha = params["k_D"], params["alpha"]
        sign = 1.0 if params["q_D_sign"] >= 0 else -1.0
        q_d = sign * sqrt(FOUR_PI_EPS0 * k_d * alpha)
        attrs: dict[str, Any] = {
            "element": "D",
            "charge": q_d,
            "mass": params["m_D"],
            "k_D": k_d,
            "alpha": alpha,
        }
        # Give the shell its own force-field type, derived from the core type, so
        # the structure is fully typed before export (no untyped particles).
        core_type = host.get("type")
        if core_type is not None:
            attrs["type"] = f"{self.drude_prefix}{core_type}"
        # Co-locate with the core if the host carries coordinates.
        for axis in ("x", "y", "z"):
            if host.get(axis) is not None:
                attrs[axis] = host.get(axis)
        return [DrudeParticle(**attrs)]

    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        (shell,) = sites
        q_d = shell.get("charge")
        host.data[fields.CHARGE.key] = host[fields.CHARGE] - q_d
        if host.get("mass") is not None:
            host.data[fields.MASS.key] = host[fields.MASS] - shell[fields.MASS]
        struct.add_atom(shell)
        # Spring constant is the host type's k_D (data-driven, from alpha.ff). The
        # spring carries its own bond type so the augmented structure is fully
        # typed (every core–shell spring shares the one ``DRUDE`` bond type).
        struct.add_bond(
            Bond(
                host,
                shell,
                k=shell.get("k_D"),
                r0=0.0,
                style="drude",
                type=self.drude_bond_type,
            )
        )


class Tip4pBuilder(VirtualSiteBuilder):
    """Place a TIP4P M-site on each water's HOH bisector (proves base generality).

    The massless M-site sits at distance ``d_om`` from O along the HOH bisector;
    the water O charge is moved onto M and O is left neutral. No spring is added —
    contrast with :class:`DrudeBuilder`'s dynamical shell.
    """

    def __init__(self, d_om: float = TIP4P_OM) -> None:
        self.d_om = d_om

    @staticmethod
    def _xyz(atom: Atom) -> tuple[float, float, float]:
        return (atom[fields.POS_X], atom[fields.POS_Y], atom[fields.POS_Z])

    def _hydrogens(self, struct: Atomistic, o: Atom) -> list[Atom]:
        hs: list[Atom] = []
        for bond in struct.bonds:
            a, b = bond.itom, bond.jtom
            if a is o and b.get("element") == "H":
                hs.append(b)
            elif b is o and a.get("element") == "H":
                hs.append(a)
        return hs

    def select(self, struct: Atomistic) -> Sequence[Atom]:
        return [
            a
            for a in struct.atoms
            if a.get("element") == "O" and len(self._hydrogens(struct, a)) == 2
        ]

    def build_sites(self, struct: Atomistic, host: Atom) -> list[Atom]:
        ox, oy, oz = self._xyz(host)
        h1, h2 = self._hydrogens(struct, host)
        b1 = tuple(c - o for c, o in zip(self._xyz(h1), (ox, oy, oz)))
        b2 = tuple(c - o for c, o in zip(self._xyz(h2), (ox, oy, oz)))
        bis = tuple(x + y for x, y in zip(b1, b2))
        norm = sqrt(sum(c * c for c in bis)) or 1.0
        m = tuple(o + self.d_om * c / norm for o, c in zip((ox, oy, oz), bis))
        return [
            MasslessSite(
                element="M",
                charge=host[fields.CHARGE],
                x=m[0],
                y=m[1],
                z=m[2],
            )
        ]

    def redistribute(
        self, struct: Atomistic, host: Atom, sites: Sequence[Atom]
    ) -> None:
        (msite,) = sites
        host.data["charge"] = 0.0
        struct.add_atom(msite)
