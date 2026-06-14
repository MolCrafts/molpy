"""Tests for the virtual-site data model (VirtualSite/DrudeParticle/MasslessSite).

Key design fact (verified): molrs re-materialises nodes as plain ``Atom`` (the
intern cache is weak and ``_intern_atom`` uses ``_entity_cls``), so the Python
subclass identity is NOT preserved across a ``.copy()`` / round-trip. Identity
is therefore carried by the persistent ``vsite`` marker field, exposed via
``Atom.is_virtual``.
"""

from molpy.core import (
    Atom,
    Atomistic,
    DrudeParticle,
    MasslessSite,
    VirtualSite,
)


# --- ac-001: class hierarchy --------------------------------------------------
def test_class_hierarchy():
    assert issubclass(VirtualSite, Atom)
    assert issubclass(DrudeParticle, VirtualSite)
    assert issubclass(MasslessSite, VirtualSite)


def test_construction_sets_marker():
    assert VirtualSite().get("vsite") == "virtual"
    assert DrudeParticle().get("vsite") == "drude"
    assert MasslessSite().get("vsite") == "massless"


def test_is_virtual_property():
    assert DrudeParticle().is_virtual is True
    assert MasslessSite().is_virtual is True
    assert Atom(element="C").is_virtual is False


# --- ac-002: injectable via add_atom, marker survives molrs round-trip --------
def test_add_atom_injection_and_retrieval():
    asm = Atomistic()
    asm.add_atom(Atom(element="C"))
    asm.add_atom(DrudeParticle(element="D", charge=-1.5))
    vsites = [a for a in asm.atoms if a.is_virtual]
    assert len(vsites) == 1
    assert vsites[0].get("vsite") == "drude"
    assert vsites[0].get("charge") == -1.5


def test_marker_survives_copy_roundtrip():
    asm = Atomistic()
    asm.add_atom(Atom(element="C", charge=0.5))
    asm.add_atom(DrudeParticle(element="D", charge=-1.0))
    clone = asm.copy()
    # Subclass identity is lost (re-interned as Atom) but the marker field
    # persists — identification must use the field, not isinstance.
    kinds = sorted(a.get("vsite") or "" for a in clone.atoms)
    assert kinds == ["", "drude"]
    assert [a.is_virtual for a in clone.atoms].count(True) == 1
