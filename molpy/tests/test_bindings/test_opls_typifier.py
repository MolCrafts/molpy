"""Python-binding coverage for the native OPLS-AA typifier."""

import math

import molpy as mp
import numpy as np
import pytest


def _ethane() -> "mp.core.Atomistic":
    """Ethane (C2H6) with explicit hydrogens and a plausible geometry."""
    mol = mp.core.Atomistic()
    c1 = mol.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    c2 = mol.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0)
    hpos = [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]
    for c, (x, y, z) in hpos:
        h = mol.add_atom(atomic_number=1, x=x, y=y, z=z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol


def test_typify_assigns_atom_types():
    """typify() returns a typed Atomistic graph."""
    typifier = mp.ff.typifier.OplsAaTypifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, mp.core.Atomistic)
    frame = typed.to_frame()
    atoms = frame["atoms"]
    assert atoms.n_rows == 8
    types = atoms["type"]
    # Every atom typed (no empty / null type label).
    assert all(str(t) != "" for t in types)


def test_typify_and_compose_potentials():
    """typify() adds bonded blocks; compose path yields finite energy (no build())."""
    typifier = mp.ff.typifier.OplsAaTypifier()
    mol = _ethane()
    typed = typifier.typify(mol)
    frame = typed.to_frame()
    assert frame["bonds"].n_rows == 7
    assert frame["angles"].n_rows == 12
    assert frame["dihedrals"].n_rows == 9
    # Every pair within three bonds of ethane's eight atoms: 7 + 12 + 9.
    assert frame["pairs"].n_rows == 28

    pots = mp.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
    energy, forces = pots.evaluate(frame)
    assert math.isfinite(energy)
    assert np.isfinite(np.asarray(forces)).all()


def test_xml_source_constructs():
    """The constructor accepts OPLS-AA XML text (``mp.io.Text``), its
    bytes, or a path; a plain ``str`` is ambiguous and refused."""
    # The embedded canonical set is also reachable via the reader; round-trip a
    # minimal well-formed OPLS-AA forcefield document.
    xml = (
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>"
    )
    typifier = mp.ff.typifier.OplsAaTypifier(mp.io.Text(xml))
    assert typifier is not None
    assert mp.ff.typifier.OplsAaTypifier(xml.encode()) is not None
    with pytest.raises(TypeError, match="molpy.io.Text"):
        mp.ff.typifier.OplsAaTypifier(xml)


def test_a_path_source_constructs(tmp_path):
    path = tmp_path / "opls.xml"
    path.write_text(
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>",
        encoding="utf-8",
    )
    assert mp.ff.typifier.OplsAaTypifier(path) is not None


def test_invalid_xml_raises_not_panics():
    """Malformed input raises a Python exception rather than aborting."""
    with pytest.raises((ValueError, RuntimeError)):
        mp.ff.typifier.OplsAaTypifier(mp.io.Text("<not valid xml <<<"))


def test_oplsaa_rejects_coarse_grain():
    typifier = mp.ff.typifier.OplsAaTypifier()
    cg = mp.core.CoarseGrain()
    with pytest.raises(TypeError):
        typifier.typify(cg)
