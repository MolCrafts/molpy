"""``write_openmm_system``: a force field plus the residue templates of a
typed frame, which OpenMM's ``createSystem`` needs for the PDB
``write_pdb`` writes of the same frame.

The structure tests run everywhere. The OpenMM tests run only where
``openmm`` is importable (binding tests never require it): they build the
System from the two written files, as a generated deck does, and price the
fixtures' molecules (``molrs/src/ff/testdata/openmm``), whose OpenMM
energies the fixtures pin.
"""

from __future__ import annotations

from conftest import capture

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import molrs
import numpy as np
import pytest

FIXTURES = Path(__file__).resolve().parents[2] / "molrs/src/ff/testdata/openmm"
OFFSET = 50.0  # Å between copies


def _molecules(name: str, copies: int) -> tuple[molrs.core.Frame, dict]:
    """``copies`` of the ``name`` fixture's molecule, typed and charged, with
    no names and no residue columns, each copy ``OFFSET`` Å further along x."""
    case = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    n = len(case["types"])
    positions = np.concatenate(
        [np.asarray(case["positions"]) + [OFFSET * c, 0.0, 0.0] for c in range(copies)]
    )
    atoms = molrs.core.Block()
    for k, key in enumerate("xyz"):
        atoms.insert(key, positions[:, k].copy())
    atoms.insert("type", case["types"] * copies)
    atoms.insert("element", case["elements"] * copies)
    atoms.insert("charge", np.tile(np.asarray(case["charges"]), copies))
    bonds = np.asarray(case["bonds"], dtype=np.uint64)
    bonds = np.concatenate([bonds + c * n for c in range(copies)])
    block = molrs.core.Block()
    block.insert("atomi", bonds[:, 0].copy())
    block.insert("atomj", bonds[:, 1].copy())
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = block
    return frame, case


def _forcefield() -> molrs.ff.forcefield.ForceField:
    return molrs.io.openmm_xml.OpenmmXmlReader().read(FIXTURES / "opls.xml")


def test_one_template_per_molecule_named_as_the_pdb_names_it(tmp_path: Path) -> None:
    frame, case = _molecules("opls", 2)
    text = capture(
        molrs.io.openmm_xml.OpenmmXmlWriter().write_system, _forcefield(), frame
    )
    path = tmp_path / "system.xml"
    molrs.io.openmm_xml.OpenmmXmlWriter().write_system(path, _forcefield(), frame)
    assert path.read_text(encoding="utf-8") == text

    (residue,) = ET.fromstring(text).iter("Residue")
    assert residue.get("name") == "MOL"
    atoms = residue.findall("Atom")
    assert [a.get("type") for a in atoms] == case["types"]
    assert [float(a.get("charge")) for a in atoms] == case["charges"]
    assert len(residue.findall("Bond")) == len(case["bonds"])
    assert residue.findall("ExternalBond") == []

    pdb = molrs.io.read_pdb(molrs.io.Text(capture(molrs.io.write_pdb, frame)))
    names = [a.get("name") for a in atoms]
    assert list(pdb["atoms"]["name"]) == names * 2
    assert list(pdb["atoms"]["res_id"]) == [1] * 12 + [2] * 12


def test_an_atom_type_the_force_field_lacks_is_a_value_error() -> None:
    frame, case = _molecules("opls", 1)
    frame["atoms"]["type"] = ["opls_999"] + case["types"][1:]
    with pytest.raises(ValueError, match="opls_999"):
        capture(
            molrs.io.openmm_xml.OpenmmXmlWriter().write_system, _forcefield(), frame
        )


def _openmm_system(tmp_path: Path, name: str, copies: int):
    """The OpenMM System of ``copies`` of the ``name`` fixture molecule, built
    as a deck builds it: ``createSystem`` on the topology of the PDB molrs
    writes, with the force field and templates molrs writes."""
    from openmm.app import ForceField, NoCutoff, PDBFile

    frame, case = _molecules(name, copies)
    molrs.io.write_pdb(tmp_path / "system.pdb", frame)
    ff = molrs.io.openmm_xml.OpenmmXmlReader().read(FIXTURES / f"{name}.xml")
    molrs.io.openmm_xml.OpenmmXmlWriter().write_system(
        tmp_path / "system.xml", ff, frame
    )
    pdb = PDBFile(str(tmp_path / "system.pdb"))
    system = ForceField(str(tmp_path / "system.xml")).createSystem(
        pdb.topology, nonbondedMethod=NoCutoff
    )
    return system, pdb.topology, frame, case


@pytest.mark.parametrize("name", ["amber", "charmm", "opls"])
def test_openmm_prices_the_written_system_as_the_fixture(
    name: str, tmp_path: Path
) -> None:
    """The fixture's energy is OpenMM's on its own force field and residue
    template (OPLS's with foyer's geometric mixing applied to the System);
    the written deck reproduces it, OPLS's through the geometric-mixing
    ``<Script>`` ``createSystem`` runs (it ignores ``combining_rule``)."""
    pytest.importorskip("openmm")
    from openmm import Context, Platform, VerletIntegrator, unit

    system, _, frame, case = _openmm_system(tmp_path, name, 1)
    context = Context(
        system, VerletIntegrator(0.001), Platform.getPlatformByName("Reference")
    )
    # The fixture's positions (not the PDB's 3 decimals), in nm.
    xyz = np.column_stack([np.asarray(frame["atoms"][k]) for k in "xyz"])
    context.setPositions(xyz * 0.1)
    energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalorie_per_mole)
    )
    assert energy == pytest.approx(case["openmm_kcal"]["total"], rel=1e-9)


def test_openmm_matches_one_template_to_every_molecule(tmp_path: Path) -> None:
    pytest.importorskip("openmm")
    from openmm import NonbondedForce, unit

    system, topology, _, case = _openmm_system(tmp_path, "opls", 2)
    assert topology.getNumResidues() == 2
    assert system.getNumParticles() == 24
    (nonbonded,) = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    charges = [
        nonbonded.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        for i in range(24)
    ]
    np.testing.assert_allclose(charges, case["charges"] * 2, atol=1e-12)


def test_the_geometric_mixing_follows_the_run_time_method(tmp_path: Path) -> None:
    """The long-range method is ``createSystem``'s: under PME the geometric
    Lennard-Jones is periodic at the run's cutoff, the charges stay in the
    PME ``NonbondedForce`` (at epsilon 0), and LJPME, which mixes its
    dispersion otherwise, is refused."""
    pytest.importorskip("openmm")
    from openmm import CustomNonbondedForce, NonbondedForce, Vec3, unit
    from openmm.app import LJPME, PME, ForceField, PDBFile

    frame, case = _molecules("opls", 2)
    frame.box = molrs.core.Box.cube(80.0)
    molrs.io.write_pdb(tmp_path / "system.pdb", frame)
    molrs.io.openmm_xml.OpenmmXmlWriter().write_system(
        tmp_path / "system.xml", _forcefield(), frame
    )
    pdb = PDBFile(str(tmp_path / "system.pdb"))
    assert pdb.topology.getPeriodicBoxVectors()[0] == Vec3(8.0, 0, 0) * unit.nanometer
    ff = ForceField(str(tmp_path / "system.xml"))
    cutoff = 1.2 * unit.nanometer
    system = ff.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=cutoff)
    (nonbonded,) = [f for f in system.getForces() if isinstance(f, NonbondedForce)]
    (lj,) = [f for f in system.getForces() if isinstance(f, CustomNonbondedForce)]
    assert nonbonded.getNonbondedMethod() == NonbondedForce.PME
    assert lj.getName() == "GeometricLennardJones"
    assert lj.getNonbondedMethod() == CustomNonbondedForce.CutoffPeriodic
    assert lj.getCutoffDistance() == cutoff
    params = [nonbonded.getParticleParameters(i) for i in range(24)]
    assert all(e.value_in_unit(unit.kilojoule_per_mole) == 0.0 for _, _, e in params)
    np.testing.assert_allclose(
        [q.value_in_unit(unit.elementary_charge) for q, _, _ in params],
        case["charges"] * 2,
        atol=1e-12,
    )
    assert lj.getNumExclusions() == nonbonded.getNumExceptions()
    with pytest.raises(ValueError, match="LJPME"):
        ff.createSystem(pdb.topology, nonbondedMethod=LJPME, nonbondedCutoff=cutoff)
