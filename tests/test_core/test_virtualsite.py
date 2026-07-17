"""Virtual sites are live refs identified by their persisted marker."""

import pytest

from molpy.core import Atom, Atomistic, DrudeParticle, MasslessSite, VirtualSite


def test_class_hierarchy() -> None:
    assert issubclass(VirtualSite, Atom)
    assert issubclass(DrudeParticle, VirtualSite)
    assert issubclass(MasslessSite, VirtualSite)


@pytest.mark.parametrize(
    ("cls", "marker"),
    [
        (VirtualSite, "virtual"),
        (DrudeParticle, "drude"),
        (MasslessSite, "massless"),
    ],
)
def test_factory_sets_marker(cls: type[VirtualSite], marker: str) -> None:
    struct = Atomistic()
    site = struct.def_virtual_site(kind=cls, element="D")

    assert site.get("vsite") == marker
    assert site.is_virtual
    assert struct.atoms[0] is site


def test_marker_survives_copy_roundtrip() -> None:
    struct = Atomistic()
    struct.def_atom(element="C", charge=0.5)
    struct.def_virtual_site(kind=DrudeParticle, element="D", charge=-1.0)
    clone = struct.copy()

    assert sorted(atom.get("vsite") or "" for atom in clone.atoms) == ["", "drude"]
    assert [atom.is_virtual for atom in clone.atoms].count(True) == 1


def test_virtual_site_has_no_detached_form() -> None:
    with pytest.raises(TypeError):
        DrudeParticle(element="D")  # type: ignore[call-arg]
