"""molrs is an implementation detail.

Every molrs symbol a user needs must be reachable from ``molpy``, and for bare
identity re-exports it must be the **same object** — not a wrapper layer.

Molpy-owned exceptions (subclass with real additions, or a richer package) are
listed in ``MOLPY_OWNED``.
"""

import molrs
import numpy as np

import molpy as mp
from molpy import fields

# molpy keeps a richer type or package under the same public name.
MOLPY_OWNED = frozenset(
    {
        "Atomistic",  # subclass — graph views / adopt
        "Box",  # subclass
        "CoarseGrain",  # subclass
        "Conformer",  # subclass — Atomistic marshalling
        "Trajectory",  # subclass
        "Region",  # molpy MaskPredicate sugar over native regions
        "io",  # molpy.io package
        "compute",  # molpy.compute package
        "typifier",  # molpy.typifier package
        "scale_lj",  # molpy.core.ops wrapper
    }
)


def test_every_molrs_public_name_is_reachable_from_molpy():
    for name in molrs.__all__:
        assert hasattr(mp, name), (
            f"molpy.{name} is missing; user code would have to import molrs"
        )


def test_reexports_are_identity_except_molpy_owned():
    for name in molrs.__all__:
        if name in MOLPY_OWNED:
            continue
        assert getattr(mp, name) is getattr(molrs, name), (
            f"molpy.{name} is not molrs.{name}: a wrapper layer was introduced"
        )


def test_molpy_owned_are_still_usable_from_molpy():
    for name in MOLPY_OWNED:
        assert hasattr(mp, name), f"molpy-owned {name} missing from facade"
    assert issubclass(mp.Atomistic, molrs.Atomistic)
    assert issubclass(mp.Box, molrs.Box)
    assert issubclass(mp.CoarseGrain, molrs.CoarseGrain)
    assert issubclass(mp.Trajectory, molrs.Trajectory)
    assert issubclass(mp.Conformer, molrs.Conformer)
    assert mp.Conformer is not molrs.Conformer
    assert mp.Region is not molrs.Region
    assert mp.io.__name__ == "molpy.io"
    assert mp.compute.__name__ == "molpy.compute"
    assert mp.typifier.__name__ == "molpy.typifier"


def test_storage_primitives_are_public_on_molpy():
    for name in ("Frame", "Block", "Element", "MetaValue"):
        assert name in mp.__all__
        assert getattr(mp, name) is getattr(molrs, name)


def test_core_types_are_top_level_not_only_under_core():
    """Users write molpy.Atomistic / molpy.SphereRegion, not molpy.core.*."""
    assert mp.Atomistic is not None
    assert mp.SphereRegion is not None
    assert mp.BondHarmonicStyle is not None
    assert mp.fields is fields
    assert "Atomistic" in mp.__all__
    assert "SphereRegion" in mp.__all__
    assert "BondHarmonicStyle" in mp.__all__
    assert "fields" in mp.__all__


def test_frame_is_usable_without_importing_molrs():
    frame = mp.Frame()
    assert isinstance(frame, molrs.Frame)


def test_reaction_is_usable_without_importing_molrs():
    rxn = mp.Reaction("[N:1].[O:2]>>[N:1][O:2]")
    assert rxn.forming_bonds == [(1, 2)]


# --------------------------------------------------------------------------
# fields.SITE — a FieldSpec, not a string knob
# --------------------------------------------------------------------------


def test_site_is_a_registered_field_spec():
    assert isinstance(fields.SITE, fields.FieldSpec)
    assert fields.SITE.key == "site"
    assert fields.SITE.dtype == np.dtype("U16")
    assert "SITE" in fields.__all__


def test_site_is_a_sparse_atom_annotation():
    """Only the atoms that may react are marked; the rest are left alone."""
    eo = mp.Atomistic()
    atoms = [
        eo.def_atom(element=e, x=float(i), y=0.0, z=0.0) for i, e in enumerate("OCCO")
    ]
    for a, b in zip(atoms, atoms[1:], strict=False):
        eo.def_bond(a, b)

    eo.atoms[0][fields.SITE] = "a"
    eo.atoms[3][fields.SITE] = "b"

    assert eo.atoms[0].get(fields.SITE) == "a"
    assert eo.atoms[3].get(fields.SITE) == "b"
    # unmarked atoms were never assigned; "" / absent both mean "not a site"
    assert not eo.atoms[1].get(fields.SITE)
    assert not eo.atoms[2].get(fields.SITE)
