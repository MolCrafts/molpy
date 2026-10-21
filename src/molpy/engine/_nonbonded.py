"""The nonbonded treatment an engine deck takes from its frame and force field.

Every engine reads the same two facts, so one policy holds across them:

* **Periodicity** (:func:`is_periodic`): a frame with a periodic box is a
  periodic system; a frame without one (``Atomistic.to_frame()`` carries
  none) is a non-periodic, vacuum system. LAMMPS shrink-wraps it
  (``boundary s s s``) and OpenMM computes it without images (``NoCutoff``
  by default). A method that needs a periodic cell — Ewald / PME / PPPM,
  OpenMM's ``CutoffPeriodic``, a barostat — is refused for it, by name,
  before any file is written; so is GROMACS, whose cutoff scheme simulates
  periodic systems only.
* **Nonbonded settings** (:func:`nonbonded_of`): the force field's own pair
  styles state its cutoffs (each style's ``cutoff``) and its electrostatics
  (``coul/long/pme`` is long-range, ``coul/cut`` / ``coul/charmm`` are cut
  off). Every engine writes those unless the caller overrides them
  explicitly, and falls back to its own default only for what the force
  field leaves unstated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from molrs.core import Frame
    from molrs.ff.forcefield import ForceField

#: Coulomb pair styles priced by Ewald summation (PME / PPPM) in every engine.
LONG_RANGE_COULOMB = frozenset({"coul/long/pme"})
#: Coulomb pair styles truncated at their cutoff.
CUTOFF_COULOMB = frozenset({"coul/cut", "coul/charmm"})
#: Pair styles that are neither dispersion nor the system's electrostatics
#: (Drude dampings): their cutoffs set nothing.
_DAMPING = frozenset({"coul/tt", "thole"})
#: Force-field unit presets whose length unit is the ångström.
_ANGSTROM_UNITS = frozenset({"real", "metal"})


def is_periodic(frame: Frame) -> bool:
    """Whether *frame* is a periodic system: it has a box that is not free."""
    return frame.box is not None and not frame.box.is_free


def refuse_box_free(engine: str, needs: str) -> ValueError:
    """The error for a periodic-only *needs* asked of a box-free frame."""
    return ValueError(
        f"{engine}: {needs} needs a periodic box, and the frame has none (a "
        "frame without one is a non-periodic system). Give it one "
        "(frame.box = mp.Box.cube(...), or molpack's with_periodic_box), or "
        "choose a non-periodic treatment."
    )


@dataclass(frozen=True)
class Nonbonded:
    """The nonbonded settings a force field's pair styles state.

    Attributes:
        coulomb: ``"long"`` (a long-range Coulomb style: Ewald / PME),
            ``"cut"`` (a truncated one), or ``None`` (no Coulomb style).
        coulomb_cutoff: The Coulomb style's ``cutoff``, in the force
            field's length unit; ``None`` when it states none.
        vdw_cutoff: The largest ``cutoff`` of the other pair styles
            (Lennard-Jones, Buckingham, …), in the force field's length unit;
            ``None`` when none states one.
        units: The force field's unit preset.
    """

    coulomb: Literal["long", "cut"] | None
    coulomb_cutoff: float | None
    vdw_cutoff: float | None
    units: str

    @property
    def coulomb_cutoff_nm(self) -> float | None:
        """:attr:`coulomb_cutoff` in nanometres."""
        return self._nm(self.coulomb_cutoff)

    @property
    def vdw_cutoff_nm(self) -> float | None:
        """:attr:`vdw_cutoff` in nanometres."""
        return self._nm(self.vdw_cutoff)

    def _nm(self, length: float | None) -> float | None:
        if length is None:
            return None
        if self.units not in _ANGSTROM_UNITS:
            raise ValueError(
                f"a force field in {self.units!r} units states its cutoffs in no "
                f"length unit molpy can convert to nm; use one of "
                f"{sorted(_ANGSTROM_UNITS)}, or pass the cutoffs explicitly."
            )
        return length / 10.0


def nonbonded_of(forcefield: ForceField) -> Nonbonded:
    """The nonbonded settings *forcefield*'s pair styles state."""
    styles = forcefield.get_styles("pair")
    coulomb = [s for s in styles if s.name in LONG_RANGE_COULOMB | CUTOFF_COULOMB]
    vdw = [
        s for s in styles if s.name not in _DAMPING and not s.name.startswith("coul/")
    ]
    kind: Literal["long", "cut"] | None = None
    if any(s.name in LONG_RANGE_COULOMB for s in coulomb):
        kind = "long"
    elif coulomb:
        kind = "cut"
    return Nonbonded(
        coulomb=kind,
        coulomb_cutoff=_largest_cutoff(coulomb),
        vdw_cutoff=_largest_cutoff(vdw),
        units=forcefield.units,
    )


def _largest_cutoff(styles: list) -> float | None:
    cutoffs = [s.params["cutoff"] for s in styles if "cutoff" in s.params]
    return max(cutoffs) if cutoffs else None
