"""molrs is an implementation detail (graph-assembler-02-kernel, ac-021 / ac-019).

Every molrs symbol a user needs must be reachable from ``molpy``, and it must be
the **same object** — a re-export, not a wrapper layer. A class that holds a
molrs object and forwards each method is the façade this rules out.
"""

import molrs
import numpy as np

import molpy as mp
from molpy.core import fields

REEXPORTED = [
    "Reaction",
    "SmartsPattern",
    "NeighborQuery",
    "Graph",
    "perceive_aromaticity",
    "find_rings",
]


def test_engine_primitives_are_reachable_from_molpy():
    for name in REEXPORTED:
        assert hasattr(mp, name), (
            f"molpy.{name} is missing; user code would import molrs"
        )


def test_reexports_are_the_same_object_not_a_wrapper():
    for name in REEXPORTED:
        assert getattr(mp, name) is getattr(molrs, name), (
            f"molpy.{name} is not molrs.{name}: a wrapper layer was introduced"
        )


def test_reexports_are_public():
    for name in REEXPORTED:
        assert name in mp.__all__


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
