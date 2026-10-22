"""Compiled kernels pushed into ``mp.ff.potential.Potentials`` by hand.

That ``mp.ff`` is ``molrs.ff`` submodule by submodule is ``test_public_paths.py``'s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import molpy as mp


def test_a_kernel_built_by_hand_is_pushed_into_potentials() -> None:
    pos = np.array([0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 1.6, 1.2, 0.0])
    pots = mp.ff.potential.Potentials()
    pots.push(
        mp.ff.compile.ExplicitTerms(
            "bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.5
        ).compile()
    )
    pots.push(
        mp.ff.compile.ExplicitTerms(
            "angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=120.0
        ).compile()
    )
    energy, _ = pots.evaluate(pos)
    want = 300.0 * (0.1**2 + 0.3**2) + 50.0 * (math.pi / 2 - math.radians(120.0)) ** 2
    assert energy == pytest.approx(want, rel=1e-12)
