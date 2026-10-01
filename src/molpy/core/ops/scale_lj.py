"""Python argument shaping for the native CL&Pol scaleLJ transform."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import molrs

if TYPE_CHECKING:
    from molrs import Atom
    from molrs.ff import ForceField, FragmentScaling


def scale_lj(
    ff: ForceField,
    fragments: Mapping[str, Sequence[Atom]],
    frag_data: Mapping[str, FragmentScaling] | None = None,
    *,
    scale_sigma: bool = False,
) -> ForceField:
    """Shape atom views and delegate COM/formula/FF rewriting to the native core."""
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
    return molrs.ff.scale_lj(ff, payload, frag_data, scale_sigma)


__all__ = ["scale_lj"]
