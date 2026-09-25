"""Wiring: molpy labels sites through the native marker.

The labelling and leaving-group behaviour itself belongs to the native core and
is tested there; what this file owns is that molpy exposes that marker by
identity and that it accepts molpy atom views.
"""

import molrs

from molpy.builder.assembly import SiteMap
from molpy.core import fields


class TestSiteMap:
    def test_the_native_marker_is_the_one_molpy_exposes(self):
        assert SiteMap is molrs.SiteMap

    def test_label_atoms_accepts_molpy_atom_views(self, eo_factory):
        mol = eo_factory()
        atoms = list(mol.atoms)[:2]

        marked = SiteMap(mol).label_atoms(atoms, "a")

        assert marked == [atoms[0].handle]
        assert atoms[0].get(fields.SITE) == "a"
