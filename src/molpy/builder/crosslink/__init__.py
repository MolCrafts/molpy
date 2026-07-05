"""Offline crosslinking — build a pre-crosslinked network as a graph transform.

A :class:`Crosslinker` takes a Daylight reaction SMARTS and turns one
:class:`~molpy.core.atomistic.Atomistic` into a **new** crosslinked one, leaving
the input untouched. All chemistry (SMARTS matching, the SMIRKS graph edit,
distances, backbone ordering) is delegated to the molrs engine; molpy only
orchestrates *which* sites pair.

- :class:`DeterministicCrosslinker` — exhaustive / ``spacing`` / explicit ``pairs``
- :class:`RandomCrosslinker` — random pairing to a target ``conversion`` (seeded)
- :class:`PortMatcher` — modelled ``atom["port"]`` markers as an alternate site
  front-end (same occurrence shape as molrs SMARTS matching)
"""

from ._crosslinker import Candidate, Crosslinker
from ._deterministic import DeterministicCrosslinker
from ._port_matcher import PortMatcher
from ._random import RandomCrosslinker
from .recipes import crosslink_gel, write_lammps

__all__ = [
    "Candidate",
    "Crosslinker",
    "DeterministicCrosslinker",
    "PortMatcher",
    "RandomCrosslinker",
    "crosslink_gel",
    "write_lammps",
]
