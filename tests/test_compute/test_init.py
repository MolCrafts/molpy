"""``molpy.compute`` package facade — the Compute contract is molrs-owned."""

from __future__ import annotations

from molrs.compute import Compute as MolrsCompute

from molpy.compute import Compute


def test_compute_protocol_is_the_molrs_one():
    assert Compute is MolrsCompute


def test_time_correlation_family_is_the_molrs_one():
    import molrs

    import molpy as mp

    for name in ("TimeCorrelation", "Spectrum", "Acf", "Oacf", "Vacf", "Msd"):
        assert getattr(mp.compute, name) is getattr(molrs.compute, name)
