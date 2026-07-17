"""Unit tests for :mod:`molpy.builder.assembly._sites`."""

import pytest

from molpy.builder.assembly import SiteMap
from molpy.core import fields


class TestSiteMap:
    def test_struct_property_preserves_input_identity(self, eo_factory):
        struct = eo_factory()
        assert SiteMap(struct).struct is struct

    def test_label_returns_self_for_chaining(self, eo_factory):
        struct = eo_factory()
        sites = SiteMap(struct)
        atom = struct.atoms[0]
        assert sites.label(atom, "x") is sites
        assert atom[fields.SITE] == "x"

    def test_label_elements_marks_in_handle_order(self, eo_factory):
        struct = eo_factory()
        marked = SiteMap(struct).label_elements("O", "left", "right")
        assert [atom[fields.SITE] for atom in marked] == ["left", "right"]
        assert [atom.handle for atom in marked] == sorted(
            atom.handle for atom in marked
        )

    def test_label_elements_rejects_too_few_atoms(self, eo_factory):
        with pytest.raises(ValueError, match="need 3"):
            SiteMap(eo_factory()).label_elements("O", "a", "b", "c")

    def test_every_nth_marks_site_and_leaving_hydrogen(self, eo_factory):
        struct = eo_factory()
        oxygens = [atom for atom in struct.atoms if atom.get(fields.ELEMENT) == "O"]
        marked = SiteMap(struct).every_nth(
            oxygens, 1, "x", leaving="h", fold_charge=False
        )
        assert marked == oxygens
        assert sum(atom.get(fields.SITE) == "h" for atom in struct.atoms) == 2

    def test_clear_removes_only_site_labels(self, eo_factory):
        struct = eo_factory()
        sites = SiteMap(struct)
        sites.clear()
        assert all(atom.get(fields.SITE) in (None, "") for atom in struct.atoms)
        assert all(atom.get(fields.ELEMENT) for atom in struct.atoms)
