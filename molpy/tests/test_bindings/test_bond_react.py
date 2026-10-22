"""LAMMPS ``fix bond/react`` file sets: ``BondReactTemplate``,
``write_lammps_bond_react_map`` and ``write_lammps_bond_react_system``.

The map-file format follows https://docs.lammps.org/fix_bond_react.html; when
a LAMMPS binary is on PATH the system's data, force-field and molecule files
are also read by LAMMPS itself (``fix bond/react`` needs the REACTION package;
the files it reads are checked with ``read_data`` / ``molecule``).
"""

from __future__ import annotations


import molpy as mp
import pytest


def _forcefield() -> mp.ff.forcefield.ForceField:
    """Two atom types and two bond types; ``c3-oh`` is used only by the template."""
    ff = mp.ff.forcefield.ForceField("hand")
    atoms = ff.def_style("atom", "full")
    c3 = atoms.def_type("c3", mass=12.011)
    oh = atoms.def_type("oh", mass=15.999)
    pairs = ff.def_style("pair", "lj/cut", {"cutoff": 9.0})
    pairs.def_type("c3", c3, epsilon=0.1, sigma=3.4)
    pairs.def_type("oh", oh, epsilon=0.2, sigma=3.0)
    bonds = ff.def_style("bond", "harmonic")
    bonds.def_type("c3-c3", c3, c3, k=600.0, r0=1.53)
    bonds.def_type("c3-oh", c3, oh, k=640.0, r0=1.41)
    return ff


def _system() -> mp.core.Frame:
    """Two bonded ``c3`` atoms; each states its ``mass``, which the data file's
    ``Masses`` section reads (the writer derives none from the species)."""
    mol = mp.core.Atomistic()
    a = mol.add_atom(
        atomic_number=6,
        type="c3",
        mass=12.011,
        x=0.0,
        y=0.0,
        z=0.0,
        charge=0.0,
        mol_id=1,
    )
    b = mol.add_atom(
        atomic_number=6,
        type="c3",
        mass=12.011,
        x=1.53,
        y=0.0,
        z=0.0,
        charge=0.0,
        mol_id=1,
    )
    mol.add_bond(a, b, type="c3-c3")
    frame = mol.to_frame()
    frame.box = mp.core.Box(mp.core.Cell.lengths([20.0, 20.0, 20.0]))
    return frame


def _template() -> mp.io.lammps.BondReactTemplate:
    """c3 + oh → c3-oh: the new bond type exists only in the post template.

    ``oh`` is a type only the template uses, so the system's data file takes
    its mass from the template atoms' ``mass``."""
    c3 = {"atomic_number": 6, "type": "c3", "mass": 12.011, "charge": 0.0}
    oh = {"atomic_number": 8, "type": "oh", "mass": 15.999, "charge": 0.0}
    pre = mp.core.Atomistic()
    c_pre = pre.add_atom(**c3, x=0.0, y=0.0, z=0.0, react_id=1)
    o_pre = pre.add_atom(**oh, x=3.0, y=0.0, z=0.0, react_id=2)
    post = mp.core.Atomistic()
    c_post = post.add_atom(**c3, x=0.0, y=0.0, z=0.0, react_id=1)
    o_post = post.add_atom(**oh, x=1.41, y=0.0, z=0.0, react_id=2)
    post.add_bond(c_post, o_post, type="c3-oh")
    return mp.io.lammps.BondReactTemplate(
        pre=pre, post=post, initiator_atoms=[c_pre, o_pre]
    )


def test_the_template_keeps_its_objects_and_writes_a_map(tmp_path):
    template = _template()
    assert len(template.initiator_atoms) == 2
    assert list(template.edge_atoms) == []
    mp.io.write_lammps_bond_react_map(tmp_path / "rxn1.map", template)
    lines = (tmp_path / "rxn1.map").read_text(encoding="utf-8").splitlines()
    assert "2 equivalences" in lines
    assert "0 edgeIDs" in lines and "0 deleteIDs" in lines
    sections = [
        lines.index(s) for s in ("InitiatorIDs", "EdgeIDs", "DeleteIDs", "Equivalences")
    ]
    assert sections == sorted(sections)
    assert lines[lines.index("Equivalences") + 2 :] == ["1   1", "2   2"]
    buf = bytearray(b"stale")
    mp.io.write_lammps_bond_react_map(buf, template)
    assert bytes(buf) == (tmp_path / "rxn1.map").read_bytes()


def test_react_ids_may_be_given_directly_and_must_match():
    pre = mp.core.Atomistic()
    pre.add_atom(atomic_number=6, type="c3", react_id=1)
    pre.add_atom(atomic_number=6, type="c3", react_id=2)
    post = mp.core.Atomistic()
    post.add_atom(atomic_number=6, type="c3", react_id=1)
    template = mp.io.lammps.BondReactTemplate(pre, post, [1, 2])
    with pytest.raises(ValueError, match="different atoms"):
        mp.io.write_lammps_bond_react_map(bytearray(), template)
    with pytest.raises(ValueError, match="exactly 2"):
        mp.io.write_lammps_bond_react_map(
            bytearray(), mp.io.lammps.BondReactTemplate(pre, pre, [1])
        )


def test_the_system_covers_template_only_types(tmp_path):
    workdir = tmp_path / "rxn"
    mp.io.write_lammps_bond_react_system(
        workdir, _system(), _forcefield(), {"rxn1": _template()}
    )
    coeff_lines = [
        line.split()[:3]
        for line in (workdir / "rxn.ff").read_text(encoding="utf-8").splitlines()
        if line.startswith(("pair_coeff", "bond_coeff"))
    ]
    assert ["bond_coeff", "c3-oh"] in [line[:2] for line in coeff_lines]
    assert ["pair_coeff", "oh", "oh"] in coeff_lines
    data = (workdir / "rxn.data").read_text(encoding="utf-8")
    assert "Bond Type Labels\n\n1 c3-c3\n2 c3-oh\n" in data
    post = (workdir / "rxn1_post.mol").read_text(encoding="utf-8")
    assert "Bonds\n\n1 2 1 2\n" in post
    assert (workdir / "rxn1.map").exists()


def test_a_forcefield_missing_a_label_writes_nothing(tmp_path):
    workdir = tmp_path / "rxn"
    ff = mp.ff.forcefield.ForceField("hand")
    atoms = ff.def_style("atom", "full")
    c3 = atoms.def_type("c3", mass=12.011)
    ff.def_style("pair", "lj/cut", {"cutoff": 9.0}).def_type(
        "c3", c3, epsilon=0.1, sigma=3.4
    )
    with pytest.raises(ValueError):
        mp.io.write_lammps_bond_react_system(
            workdir, _system(), ff, {"rxn1": _template()}
        )
    assert not workdir.exists()


def test_a_sequence_of_templates_is_named_rxn_n(tmp_path):
    workdir = tmp_path / "seq"
    mp.io.write_lammps_bond_react_system(
        workdir, _system(), _forcefield(), [_template()]
    )
    assert (workdir / "rxn1_pre.mol").exists()
