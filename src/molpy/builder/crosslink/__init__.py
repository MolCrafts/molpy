"""Offline crosslinking — build a pre-crosslinked network as a graph transform.

A :class:`Crosslinker` takes a Daylight reaction SMARTS and turns one
:class:`~molpy.core.atomistic.Atomistic` into a **new** crosslinked one, leaving
the input untouched. All chemistry (SMARTS matching, the SMIRKS graph edit,
distances, backbone ordering) is delegated to the molrs engine; molpy only
orchestrates *which* sites pair.

- :class:`DeterministicCrosslinker` — exhaustive / ``spacing`` / explicit ``pairs``
- :class:`RandomCrosslinker` — random pairing to a target ``conversion`` (seeded)

Crosslink sites may be pre-marked at modelling time (``site_field``): the one
molrs SMARTS matcher then only pairs marked atoms — no separate site front-end.
"""

from ._crosslinker import Candidate, Crosslinker, SelectionContext
from ._deterministic import DeterministicCrosslinker
from ._random import RandomCrosslinker
from .recipes import crosslink_gel, write_lammps

__all__ = [
    "Candidate",
    "Crosslinker",
    "SelectionContext",
    "DeterministicCrosslinker",
    "RandomCrosslinker",
    "crosslink_gel",
    "write_lammps",
]
