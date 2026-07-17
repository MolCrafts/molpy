"""Unit tests for :mod:`molpy.builder.assembly._library`."""

import pytest

import molpy as mp
from molpy.builder.assembly import MonomerLibrary
from molpy.core import fields
from molpy.parser.smiles import parse_cgsmiles


class TestMonomerLibrary:
    def test_empty_library_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            MonomerLibrary({})

    def test_template_without_a_site_is_rejected(self):
        naked = mp.Atomistic()
        naked.def_atom(element="C")
        with pytest.raises(ValueError, match="marks no reaction site"):
            MonomerLibrary({"X": naked})

    def test_expand_stamps_contiguous_residue_identity(self, eo_factory):
        topology = parse_cgsmiles("{[#EO]|3}").base_graph
        world = MonomerLibrary({"EO": eo_factory()}).expand(topology)
        assert sorted({int(atom[fields.RES_ID]) for atom in world.atoms}) == [1, 2, 3]
        assert {str(atom[fields.RES_NAME]) for atom in world.atoms} == {"EO"}

    def test_unknown_topology_label_is_rejected(self, eo_factory):
        topology = parse_cgsmiles("{[#ZZ]|2}").base_graph
        with pytest.raises(ValueError, match="lacks"):
            MonomerLibrary({"EO": eo_factory()}).expand(topology)

    def test_template_snapshots_cannot_stale_builder_caches(self, eo_factory):
        template = eo_factory()
        library = MonomerLibrary({"EO": template})
        assert "EO" in library
        exposed = library["EO"]
        assert exposed is not template
        original_types = [atom.get(fields.TYPE) for atom in exposed.atoms]

        template.atoms[0][fields.TYPE] = "caller-mutation"
        exposed.atoms[0][fields.TYPE] = "returned-copy-mutation"
        topology = parse_cgsmiles("{[#EO]}").base_graph
        expanded = library.expand(topology)
        assert [atom.get(fields.TYPE) for atom in expanded.atoms] == original_types
