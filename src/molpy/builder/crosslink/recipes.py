"""PEO-gel recipes (crosslink-03) — compose existing pieces, add no engine.

A crosslinked network is just: build a polymer (any builder), crosslink it with a
:class:`~molpy.builder.crosslink.Crosslinker`, optionally relax the stretched
bonds with an injected minimizer, and export. These helpers wire those existing
steps together; they introduce no new chemistry.

``relax`` is intentionally an injected callable rather than a hard-wired
``molpy.optimize`` call: relaxation needs a fully parameterized force field,
which is the caller's concern, not the crosslinker's.
"""

from __future__ import annotations

from collections.abc import Callable
from os import PathLike

from molpy.core.atomistic import Atomistic

from ._crosslinker import Crosslinker


def crosslink_gel(
    structure: Atomistic,
    crosslinker: Crosslinker,
    *,
    relax: Callable[[Atomistic], Atomistic] | None = None,
) -> Atomistic:
    """Crosslink a pre-built polymer, optionally relax, return the new network.

    The input ``structure`` is never mutated (``crosslinker.apply`` copies it).
    Pass ``relax`` (e.g. a ``molpy.optimize`` minimizer bound to a force field)
    to relax the freshly formed, possibly stretched crosslink bonds.
    """
    gel = crosslinker.apply(structure)
    if relax is not None:
        gel = relax(gel)
    return gel


def write_lammps(
    structure: Atomistic, path: str | PathLike[str], *, atom_style: str = "full"
) -> None:
    """Export a (crosslinked) structure to a LAMMPS data file."""
    from molpy.io import write_lammps_data

    write_lammps_data(path, structure.to_frame(), atom_style=atom_style)
