"""``molpy.potential`` is molrs's potentials and force-field IR, by identity."""

from __future__ import annotations

import math

import molrs
import numpy as np
import pytest

import molpy as mp

FROM_IR = (
    "IrError",
    "Param",
    "StyleSpec",
    "categories",
    "evaluate",
    "register_category",
    "register_style",
    "styles",
    "unregister",
)
FROM_POTENTIAL = ("LJCut", "Potential", "kernel")


@pytest.mark.parametrize("name", FROM_IR)
def test_ir_names_are_molrs_objects(name: str) -> None:
    assert getattr(mp.potential, name) is getattr(molrs.ff.ir, name)


@pytest.mark.parametrize("name", FROM_POTENTIAL)
def test_potential_names_are_molrs_objects(name: str) -> None:
    assert getattr(mp.potential, name) is getattr(molrs.ff.potential, name)


def test_the_namespace_is_exactly_the_re_exports() -> None:
    assert sorted(mp.potential.__all__) == sorted(FROM_IR + FROM_POTENTIAL)


def test_md_defines_no_potential() -> None:
    for name in ("LJCut", "Potential", "Potentials"):
        assert not hasattr(mp.md, name)


def test_a_kernel_built_by_hand_is_pushed_into_potentials() -> None:
    pos = np.array([0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 1.6, 1.2, 0.0])
    pots = mp.Potentials()
    pots.push(
        mp.potential.kernel("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.5)
    )
    pots.push(
        mp.potential.kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=120.0)
    )
    energy, _ = pots.calc_energy_forces(pos)
    want = 300.0 * (0.1**2 + 0.3**2) + 50.0 * (math.pi / 2 - math.radians(120.0)) ** 2
    assert energy == pytest.approx(want, rel=1e-12)
