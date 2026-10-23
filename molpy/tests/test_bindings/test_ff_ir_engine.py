"""Engine forms of force-field IR styles from Python (``ff-ir-02-protocol``
§8): a style registered with ``lammps="positional"`` writes and reads as
its LAMMPS style with nothing else written; one without a LAMMPS form is
refused by name."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import molpy as mp
import numpy as np
import pytest
from molpy.ff import ir, style_registry


def _lammps_include(
    ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame, **options: object
) -> str:
    """The LAMMPS force-field include ``ff`` writes for ``frame``."""
    buf = bytearray()
    mp.io.lammps.LammpsForcefieldWriter(frame, **options).write(buf, ff)  # type: ignore[arg-type]
    return buf.decode()


FENE = (
    "-0.5*k*r0^2*log(1-(r/r0)^2)"
    "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)"
)
FENE_PARAMS = {"k": "E/L^2", "r0": "L", "epsilon": "E", "sigma": "L"}
CHAIN = np.array([[0.0, 0.0, 0.0], [0.97, 0.0, 0.0], [0.97, 1.06, 0.0]])


@pytest.fixture
def registered() -> Iterator[list[tuple[str, str]]]:
    names: list[tuple[str, str]] = []
    yield names
    for category, name in names:
        if any(s.name == name for s in style_registry.styles(category)):
            style_registry.unregister_style(category, name)


def chain(style: str) -> tuple[mp.ff.forcefield.ForceField, mp.core.Frame]:
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, CHAIN[:, d].copy())
    atoms.insert("type", ["B"] * len(CHAIN))
    bonds = mp.core.Block()
    bonds.insert("atomi", np.array([0, 1], dtype=np.uint32))
    bonds.insert("atomj", np.array([1, 2], dtype=np.uint32))
    bonds.insert("type", ["B-B", "B-B"])
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    ff = mp.ff.forcefield.ForceField("beads", units="real")
    ff.set_special_bonds([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    b = ff.def_style("atom", "full").def_type("B", mass=1.0)
    ff.def_style("bond", style).def_type(
        "B-B", b, b, k=30.0, r0=1.5, epsilon=1.0, sigma=1.0
    )
    return ff, frame


def energy(ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame) -> float:
    return mp.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)[0]


def test_a_positional_style_writes_and_reads_as_its_lammps_style(
    registered, tmp_path: Path
) -> None:
    style_registry.register_style(
        "bond", "fene/py", params=FENE_PARAMS, expression=FENE, lammps="positional:fene"
    )
    registered.append(("bond", "fene/py"))
    (info,) = [s for s in style_registry.styles("bond") if s.name == "fene/py"]
    assert info.lammps == "positional:fene"
    ff, frame = chain("fene/py")
    text = _lammps_include(ff, frame)
    assert "bond_style fene\n" in text
    assert "bond_coeff B-B 30.000000 1.500000 1.000000 1.000000\n" in text
    path = tmp_path / "fene.ff"
    path.write_text(text, encoding="utf-8")
    back = mp.io.lammps.LammpsForcefieldReader().read(path)
    assert back.get_style("bond", "fene/py") is not None
    assert energy(back, frame) == pytest.approx(energy(ff, frame), rel=1e-12)


def test_built_ins_name_their_lammps_forms() -> None:
    forms = {(s.category, s.name): s.lammps for s in style_registry.styles()}
    assert forms[("bond", "harmonic")] == "positional"
    assert forms[("dihedral", "periodic")] == "custom:fourier"
    assert forms[("improper", "periodic")] == "custom:cvff"
    assert forms[("dihedral", "class2")] == "custom:class2"
    assert forms[("dihedral", "rb")] is None


def test_without_a_lammps_form_lammps_refuses_by_name(registered) -> None:
    style_registry.register_style(
        "bond", "fene/none", params=FENE_PARAMS, expression=FENE
    )
    registered.append(("bond", "fene/none"))
    ff, frame = chain("fene/none")
    with pytest.raises(
        ValueError, match=r"LAMMPS has no form for bond `fene/none`.*LEPTON"
    ):
        _lammps_include(ff, frame)
    # A form given afterwards: the style writes.
    style_registry.register_engine_form(
        "lammps", "bond", "fene/none", "positional:fene"
    )
    assert "bond_style fene\n" in _lammps_include(ff, frame)


def test_register_engine_form_refuses_by_variant(registered) -> None:
    style_registry.register_style("bond", "fene/x", params=FENE_PARAMS, expression=FENE)
    registered.append(("bond", "fene/x"))
    with pytest.raises(ir.NoEngineFormError) as e:
        style_registry.register_engine_form("gromacs", "bond", "fene/x", "positional")
    assert e.value.engine == "GROMACS"
    with pytest.raises(ir.SealedError):
        style_registry.register_engine_form(
            "lammps", "bond", "harmonic", "positional:other"
        )
    with pytest.raises(ir.NoKernelError):
        style_registry.register_engine_form(
            "lammps", "bond", "nothing/here", "positional"
        )
    with pytest.raises(ValueError, match="positional"):
        style_registry.register_engine_form("lammps", "bond", "fene/x", "lepton")
    # A positional form the spec cannot have: a style parameter LAMMPS's
    # line has no place for.
    with pytest.raises(ir.NoEngineFormError, match="style parameter `width`"):
        style_registry.register_style(
            "pair",
            "soft/x",
            params={"a": "E"},
            style_params={"cutoff": "L", "width": "L"},
            expression="a*(1+cos(3.141592653589793*r/cutoff))+0*width",
            lammps="positional",
        )


def test_a_style_spec_class_takes_its_lammps_form(registered) -> None:
    class Fene(style_registry.StyleDeclaration):
        category = "bond"
        name = "fene/cls"
        params = FENE_PARAMS
        expression = FENE
        lammps = "positional:fene"

    registered.append(("bond", "fene/cls"))
    (info,) = [s for s in style_registry.styles("bond") if s.name == "fene/cls"]
    assert info.lammps == "positional:fene"
