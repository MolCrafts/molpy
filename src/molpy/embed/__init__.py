"""3D coordinate generation for molpy molecules.

Thin Python wrapper around the molrs Rust ``embed`` pipeline, exposed via the
``molrs`` binary extension. The heavy lifting — fragment / distance-geometry
build, energy minimisation, rotor search, stereo guard — runs inside molrs;
this module only marshals :class:`molpy.Atomistic` across that boundary.

The main-trunk ``molpy.compute.Generate3D`` is a thin :class:`Compute` wrapper
over this function. The RDKit adapter (:mod:`molpy.adapter.rdkit`) and the
RDKit tool (:mod:`molpy.tool.rdkit`) remain available as an optional external
backend.
"""

from __future__ import annotations

import molrs

from molpy.core._molrs import from_molrs, to_molrs
from molpy.core.atomistic import Atomistic

from .report import EmbedReport, StageReport

__all__ = ["EmbedReport", "StageReport", "generate_3d"]


def generate_3d(
    mol: Atomistic,
    *,
    speed: str = "medium",
    add_hydrogens: bool = True,
    rng_seed: int | None = None,
) -> tuple[Atomistic, EmbedReport]:
    """Generate 3D coordinates via the molrs ``embed`` Rust pipeline.

    Parameters
    ----------
    mol:
        Input molecular graph. Element symbols and bond orders are required;
        coordinates may be missing.
    speed:
        Quality preset, one of ``"fast"``, ``"medium"``, or ``"better"``.
    add_hydrogens:
        Add explicit hydrogens before embedding.
    rng_seed:
        Optional deterministic RNG seed.

    Returns
    -------
    tuple[Atomistic, EmbedReport]
        Fresh atomistic structure with generated coordinates plus a per-stage
        report. The input ``mol`` is not mutated.
    """
    if len(list(mol.atoms)) == 0:
        raise ValueError("cannot generate 3D structure for empty molecule")

    rs_in = to_molrs(mol)
    native_opts = molrs.EmbedOptions(
        speed=speed, add_hydrogens=bool(add_hydrogens), seed=rng_seed
    )
    result = molrs.generate_3d(rs_in, native_opts)
    out = from_molrs(result.mol, template=mol)
    report = EmbedReport.from_native(result.report)
    return out, report
