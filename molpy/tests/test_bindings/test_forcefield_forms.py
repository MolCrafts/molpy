"""Form conversions across the FFI seam: ``ForceField.canonical``,
``ForceField.to_form`` and ``ForceField.fit_form``.

The algebra, its exactness on random parameters × random configurations and
the residual's monotonicity are tested in Rust (``ff::ir::form``,
``ff::ir::torsion``); these tests prove the Python surface: each
method returns a new force field that prices as the old one (exactly, or with
the residual it reports), and a refusal is a ``ValueError`` naming the type
and the condition.
"""

from __future__ import annotations

import math

import molpy as mp
import numpy as np
import pytest

XYZ = np.array([[0.1, 1.2, -0.3], [0.0, 0.0, 0.0], [1.4, 0.1, 0.2], [1.9, 1.1, 0.9]])


def _opls() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("opls", units="real")
    atoms = ff.def_style("atom", "full")
    ct = atoms.def_type("CT", mass=12.011)
    ff.def_style("dihedral", "opls").def_type(
        "CT-CT-CT-CT", ct, ct, ct, ct, k1=1.3, k2=-0.05, k3=0.2, k4=0.0
    )
    return ff


def _periodic(phase: float) -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("p", units="real")
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
    ff.def_style("dihedral", "periodic").def_type(
        "CT-CT-CT-CT", ct, ct, ct, ct, k1=1.0, periodicity1=2, phase1=phase
    )
    return ff


def _frame() -> mp.core.Frame:
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, XYZ[:, d].copy())
    atoms.insert("type", ["CT"] * 4)
    dihedrals = mp.core.Block()
    for key, atom in (("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)):
        dihedrals.insert(key, np.array([atom], dtype=np.uint32))
    dihedrals.insert("type", ["CT-CT-CT-CT"])
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame["dihedrals"] = dihedrals
    return frame


def _energy(ff: mp.ff.forcefield.ForceField) -> float:
    frame = _frame()
    return mp.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)[0]


def test_canonical_is_dihedral_periodic_with_the_same_energy() -> None:
    ff = _opls()
    canonical = ff.canonical()
    assert canonical is not ff
    assert [s.name for s in canonical.get_styles("dihedral")] == ["periodic"]
    assert math.isclose(_energy(canonical), _energy(ff), rel_tol=1e-12)
    assert ff.get_style("dihedral", "opls") is not None, "the source is unchanged"


def test_to_form_is_exact_or_raises_naming_the_condition() -> None:
    ff = _opls()
    rb = ff.to_form("dihedral", "rb")
    (t,) = rb.get_types("dihedral")
    assert math.isclose(_energy(rb), _energy(ff), rel_tol=1e-12)
    assert set(t.params) >= {"c0", "c1", "c2", "c3", "c4", "c5"}
    with pytest.raises(
        mp.ff.ir.OutOfImageError, match=r"CT-CT-CT-CT.*sin\(2φ\)"
    ) as err:
        _periodic(30.0).to_form("dihedral", "rb")
    assert (err.value.type, err.value.to) == ("CT-CT-CT-CT", "dihedral rb")
    with pytest.raises(mp.ff.ir.NoFormError, match="no form family") as err:
        ff.to_form("improper", "harmonic")
    assert (err.value.category, err.value.style) == ("improper", "harmonic")


def test_fit_form_reports_its_residual() -> None:
    phi = np.linspace(-math.pi, math.pi, 36, endpoint=False)
    out, residual = _periodic(30.0).fit_form("dihedral", "opls", phi, offset=True)
    assert out.get_style("dihedral", "opls") is not None
    assert residual["rms"] > 0.0 and set(residual) == {
        "sum_sq",
        "rms",
        "max_abs",
        "types",
    }
    (row,) = residual["types"]
    assert (row["style"], row["type"], row["exact"]) == (
        "periodic",
        "CT-CT-CT-CT",
        False,
    )
    # Heavier weights never lower the minimised Σ w r².
    _, heavier = _periodic(30.0).fit_form(
        "dihedral", "opls", phi, np.full(phi.size, 2.0), offset=True
    )
    assert heavier["sum_sq"] >= residual["sum_sq"]
    # In the image, the fit is the exact conversion.
    _, exact = _periodic(180.0).fit_form("dihedral", "multi/harmonic", phi)
    assert exact["types"][0]["exact"] and exact["max_abs"] < 1e-12
