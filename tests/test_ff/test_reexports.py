"""``molpy.ff`` is ``molrs.ff``, submodule by submodule, by identity."""

from __future__ import annotations

import math

import molrs
import numpy as np
import pytest

import molpy as mp

SUBMODULES = (
    "charge",
    "clpol_scaling",
    "compile",
    "forcefield",
    "ir",
    "params",
    "potential",
    "style_registry",
)


def test_the_submodules_are_molrs_s() -> None:
    assert sorted(mp.ff.__all__) == sorted(molrs.ff.__all__)


@pytest.mark.parametrize("sub", SUBMODULES)
def test_every_native_name_is_the_molrs_object(sub: str) -> None:
    native = getattr(molrs.ff, sub)
    mine = getattr(mp.ff, sub)
    assert sorted(mine.__all__) == sorted(native.__all__)
    for name in native.__all__:
        assert getattr(mine, name) is getattr(native, name), (sub, name)


def test_the_typifier_adds_only_the_ambertools_typifiers() -> None:
    native = molrs.ff.typifier
    for name in native.__all__:
        assert getattr(mp.ff.typifier, name) is getattr(native, name)
    assert sorted(set(mp.ff.typifier.__all__) - set(native.__all__)) == [
        "AntechamberTypifier",
        "TleapTypifier",
    ]


def test_a_kernel_built_by_hand_is_pushed_into_potentials() -> None:
    pos = np.array([0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 1.6, 1.2, 0.0])
    pots = mp.ff.potential.Potentials()
    pots.push(
        mp.ff.compile.compile_explicit_terms(
            "bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.5
        )
    )
    pots.push(
        mp.ff.compile.compile_explicit_terms(
            "angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=120.0
        )
    )
    energy, _ = pots.calc_energy_forces(pos)
    want = 300.0 * (0.1**2 + 0.3**2) + 50.0 * (math.pi / 2 - math.radians(120.0)) ** 2
    assert energy == pytest.approx(want, rel=1e-12)
