"""Python-binding coverage for the native OPLS-AA typifier."""

import math

import molrs
import numpy as np
import pytest


def _ethane() -> "molrs.core.Atomistic":
    """Ethane (C2H6) with explicit hydrogens and a plausible geometry."""
    mol = molrs.core.Atomistic()
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


def test_opls_typifier_is_exposed():
    """OplsAaTypifier exists and constructs from embedded OPLS-AA."""
    assert "OplsAaTypifier" in molrs.ff.typifier.__all__
    assert molrs.ff.typifier.OplsAaTypifier is molrs.ff.typifier.OplsAaTypifier
    typifier = molrs.ff.typifier.OplsAaTypifier()
    assert isinstance(typifier, molrs.ff.typifier.Typifier)


def test_typify_assigns_atom_types():
    """typify() returns a typed Atomistic graph."""
    typifier = molrs.ff.typifier.OplsAaTypifier()
    typed = typifier.typify(_ethane())
    assert isinstance(typed, molrs.core.Atomistic)
    frame = typed.to_frame()
    atoms = frame["atoms"]
    assert atoms.n_rows == 8
    types = atoms["type"]
    # Every atom typed (no empty / null type label).
    assert all(str(t) != "" for t in types)


def test_typify_and_compose_potentials():
    """typify() adds bonded blocks; compose path yields finite energy (no build())."""
    typifier = molrs.ff.typifier.OplsAaTypifier()
    mol = _ethane()
    typed = typifier.typify(mol)
    frame = typed.to_frame()
    assert frame["bonds"].n_rows == 7
    assert frame["angles"].n_rows == 12
    assert frame["dihedrals"].n_rows == 9

    pairs = molrs.ff.potential.intramolecular_pairs(frame)
    frame["pairs"] = pairs
    pots = molrs.ff.compile.PotentialCompiler(typifier.forcefield()).compile(frame)
    energy, forces = pots.evaluate(frame)
    assert math.isfinite(energy)
    assert np.isfinite(np.asarray(forces)).all()


def test_xml_source_constructs():
    """The constructor accepts OPLS-AA XML text (``molrs.io.Text``), its
    bytes, or a path; a plain ``str`` is ambiguous and refused."""
    # The embedded canonical set is also reachable via the reader; round-trip a
    # minimal well-formed OPLS-AA forcefield document.
    xml = (
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>"
    )
    typifier = molrs.ff.typifier.OplsAaTypifier(molrs.io.Text(xml))
    assert typifier is not None
    assert molrs.ff.typifier.OplsAaTypifier(xml.encode()) is not None
    with pytest.raises(TypeError, match="molrs.io.Text"):
        molrs.ff.typifier.OplsAaTypifier(xml)


def test_a_path_source_constructs(tmp_path):
    path = tmp_path / "opls.xml"
    path.write_text(
        "<ForceField><AtomTypes>"
        '<Type name="opls_135" class="CT" element="C" mass="12.011"/>'
        "</AtomTypes></ForceField>",
        encoding="utf-8",
    )
    assert molrs.ff.typifier.OplsAaTypifier(path) is not None


def test_invalid_xml_raises_not_panics():
    """Malformed input raises a Python exception rather than aborting."""
    with pytest.raises((ValueError, RuntimeError)):
        molrs.ff.typifier.OplsAaTypifier(molrs.io.Text("<not valid xml <<<"))


def test_oplsaa_rejects_coarse_grain():
    typifier = molrs.ff.typifier.OplsAaTypifier()
    cg = molrs.core.CoarseGrain()
    with pytest.raises(TypeError):
        typifier.typify(cg)
