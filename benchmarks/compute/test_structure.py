"""molpy.compute structure benchmarks: structure factor, bond order, PMFT.

StaticStructureFactorDebye (frame only), BondOrder and PMFTXY (frame + nlist) —
thin shells over the molrs diffraction / environment / pmft kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from molpy.compute import BondOrder, PMFTXY, StaticStructureFactorDebye

pytestmark = pytest.mark.benchmark


def test_static_structure_factor(benchmark, cmp_frame) -> None:
    k = np.linspace(0.5, 8.0, 40)
    op = StaticStructureFactorDebye(k)
    out = benchmark(op, cmp_frame)
    assert isinstance(out, list) and len(out) >= 1


def test_bond_order(benchmark, cmp_frame, cmp_nlist) -> None:
    op = BondOrder(n_theta=6, n_phi=6)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1


def test_pmft_xy(benchmark, cmp_frame, cmp_nlist) -> None:
    op = PMFTXY(x_max=5.0, y_max=5.0, n_x=20, n_y=20)
    out = benchmark(op, cmp_frame, cmp_nlist)
    assert isinstance(out, list) and len(out) >= 1
