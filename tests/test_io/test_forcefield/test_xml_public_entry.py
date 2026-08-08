"""XML force-field public entry lives only on molpy.io (not package root)."""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.data import get_forcefield_path
from molpy.io.forcefield.xml import read_xml_forcefield as xml_mod_read


def test_xml_forcefield_not_on_package_root():
    """I/O is mp.io only — no mp.read_* duals on the facade."""
    assert "read_xml_forcefield" not in mp.__all__
    assert not hasattr(mp, "read_xml_forcefield")
    for leaked in (
        "read_forcefield_xml",
        "read_opls_xml",
        "read_pdb",
        "write_pdb",
        "read_lammps_forcefield",
    ):
        assert leaked not in mp.__all__
        assert not hasattr(mp, leaked)


def test_read_xml_forcefield_is_only_on_io():
    assert mp.io.read_xml_forcefield is xml_mod_read


def test_read_xml_forcefield_requires_explicit_path():
    with pytest.raises(FileNotFoundError):
        mp.io.read_xml_forcefield("tip3p.xml")


def test_read_xml_forcefield_loads_packaged_tip3p_via_data_helper():
    ff = mp.io.read_xml_forcefield(get_forcefield_path("tip3p.xml"))
    assert ff is not None
    from molpy.core.forcefield import AtomType

    assert len(list(ff.get_types(AtomType))) >= 2
