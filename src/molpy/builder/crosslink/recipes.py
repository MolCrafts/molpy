"""PEO-gel recipes (crosslink-03) — compose existing pieces, add no engine.

A crosslinked network is just: build a polymer (any builder), crosslink it with a
:class:`~molpy.builder.crosslink.Crosslinker`, relax the freshly formed
(often over-stretched) bonds, and export. These helpers wire those existing
steps together; they introduce no new chemistry.

Relaxation composes :class:`molpy.optimize.LBFGS` at the call site — there is no
``minimize`` free function. It runs on the crosslinked :class:`molrs.Frame` with
a soft-vs-force-field fallback: a caller-supplied ``ff`` uses
:class:`~molpy.optimize.ForceFieldPotential`; otherwise the coordinate-only
:class:`~molpy.optimize.SoftPotential` relaxes without needing atom types.
"""

from __future__ import annotations

from os import PathLike

import molrs

from molpy.core.atomistic import Atomistic
from molpy.optimize import ForceFieldPotential, LBFGS, SoftPotential

from ._crosslinker import Crosslinker


def crosslink_gel(
    structure: Atomistic,
    crosslinker: Crosslinker,
    *,
    relax: bool = True,
    ff: molrs.ForceField | None = None,
    fmax: float = 0.05,
    steps: int = 200,
) -> Atomistic:
    """Crosslink a pre-built polymer, relax the new bonds, return the network.

    The input ``structure`` is never mutated (``crosslinker.apply`` copies it),
    and relaxation produces a fresh, independent structure — neither the input
    nor the intermediate crosslinked graph is modified in place.

    Args:
        structure: Pre-built polymer to crosslink.
        crosslinker: Crosslinker defining the reaction and site-selection mode.
        relax: Relax the freshly formed bonds via :class:`~molpy.optimize.LBFGS`.
        ff: Optional molrs force field. When given, relaxation uses
            :class:`~molpy.optimize.ForceFieldPotential`; otherwise it falls back
            to the force-field-free :class:`~molpy.optimize.SoftPotential`.
        fmax: Relaxation force-convergence threshold.
        steps: Maximum relaxation steps.

    Returns:
        The crosslinked (and, when ``relax`` is set, relaxed) network.
    """
    gel = crosslinker.apply(structure)
    if not relax:
        return gel

    potential = ForceFieldPotential(ff) if ff is not None else SoftPotential()
    frame = gel.to_frame()
    result = LBFGS(potential).run(frame, fmax=fmax, steps=steps)
    # Rebuild a fresh structure from the relaxed frame; the crosslinked
    # intermediate ``gel`` is left untouched (immutable recipe).
    return Atomistic.adopt(molrs.Atomistic.from_frame(result.frame))


def write_lammps(
    structure: Atomistic, path: str | PathLike[str], *, atom_style: str = "full"
) -> None:
    """Export a (crosslinked) structure to a LAMMPS data file."""
    from molpy.io import write_lammps_data

    write_lammps_data(path, structure.to_frame(), atom_style=atom_style)
