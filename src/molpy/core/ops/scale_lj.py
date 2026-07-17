"""Python argument shaping for molrs' native CL&Pol scaleLJ transform."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import molrs

if TYPE_CHECKING:
    from molpy.core.atomistic import Atom
    from molpy.core.forcefield import ForceField

FragmentScaling = molrs.FragmentScaling
compute_k_ij = molrs.compute_k_ij


def load_fragment_scaling_data(
    path: str | Path | None = None,
) -> dict[str, FragmentScaling]:
    """Return molrs' compiled CL&Pol fragment table.

    Runtime parameter-file parsing was removed; pass explicit
    ``FragmentScaling`` objects to :func:`scale_lj` for custom data.
    """
    if path is not None:
        raise ValueError(
            "runtime CL&Pol table parsing was removed; pass frag_data explicitly"
        )
    return molrs.fragment_scaling_data()


def scale_lj(
    ff: ForceField,
    fragments: Mapping[str, Sequence[Atom]],
    frag_data: Mapping[str, FragmentScaling] | None = None,
    *,
    scale_sigma: bool = False,
) -> ForceField:
    """Shape atom views and delegate COM/formula/FF rewriting to molrs."""
    payload = {
        label: (
            [str(atom.get("type") or "") for atom in atoms],
            [
                (
                    float(atom.get("x") or 0.0),
                    float(atom.get("y") or 0.0),
                    float(atom.get("z") or 0.0),
                )
                for atom in atoms
            ],
            [float(atom.get("mass") or 1.0) for atom in atoms],
        )
        for label, atoms in fragments.items()
    }
    return molrs.scale_lj(ff, payload, frag_data, scale_sigma)


__all__ = [
    "FragmentScaling",
    "compute_k_ij",
    "load_fragment_scaling_data",
    "scale_lj",
]
