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
* **Nonbonded settings** (``ForceField.nonbonded``, read from each pair
  style's declared role in molrs): the force field's own pair styles state
  its cutoffs and its electrostatics. Every engine writes those unless the
  caller overrides them explicitly, and falls back to its own default only
  for what the force field leaves unstated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from molpy.core import Frame


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
