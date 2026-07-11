"""Handle-view contract tests for the ECS-backed structure layer.

Replaces the removed standalone-``Entity``/``TypeBucket`` value model with the
handle-view model from spec ``molgraph-ecs-03-molpy``: identity via interning,
stable removal (no reindex), zero-copy ``adopt``, column reads through the molrs
world, and no ``"X"`` placeholder for a missing element.
"""

import numpy as np
import pytest

import molrs

from molpy import Atom, Atomistic, Bond
from molpy.core import fields
from molpy.core.entity import Entities, Entity


class TestEntitiesColumnAccess:
    """Column-style access over a sequence of handle views."""

    def test_column_read_simple(self):
        s = Atomistic()
        s.def_atom(element="C", mass=12.0)
        s.def_atom(element="N", mass=14.0)
        s.def_atom(element="O", mass=16.0)

        assert np.array_equal(s.atoms["element"], ["C", "N", "O"])
        assert np.array_equal(s.atoms["mass"], [12.0, 14.0, 16.0])

    def test_column_read_missing_keys(self):
        s = Atomistic()
        s.def_atom(element="C")
        s.def_atom(element="N", charge=-1.0)
        s.def_atom()

        assert np.array_equal(s.atoms["element"], ["C", "N", None])
        assert np.array_equal(s.atoms["charge"], [None, -1.0, None])

    def test_integer_and_slice_indexing(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        b = s.def_atom(element="N")
        c = s.def_atom(element="O")

        assert s.atoms[0] is a
        assert s.atoms[1] is b
        sliced = s.atoms[1:]
        assert len(sliced) == 2
        assert sliced[0] is b
        assert sliced[1] is c

    def test_xyz_column_is_vectors(self):
        s = Atomistic()
        s.def_atom(element="C", xyz=[0.0, 0.0, 0.0])
        s.def_atom(element="N", xyz=[1.5, 0.0, 0.0])

        positions = s.atoms["xyz"]
        assert len(positions) == 2
        assert np.array_equal(positions[0], [0.0, 0.0, 0.0])
        assert np.array_equal(positions[1], [1.5, 0.0, 0.0])

    def test_empty(self):
        s = Atomistic()
        assert len(s.atoms["element"]) == 0
        assert isinstance(s.atoms, Entities)


class TestIdentityInterning:
    """Identity preserved by interning on the stable handle."""

    def test_def_atom_returns_same_bound_view(self):
        s = Atomistic()
        a = Atom(element="C", xyz=[0, 0, 0])
        result = s.add_atom(a)
        assert result is a  # same object becomes bound

    def test_bond_endpoints_are_interned_atoms(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        b = s.def_atom(element="O")
        bd = s.def_bond(a, b)

        assert bd.itom is a
        assert bd.jtom is b
        assert a in s.atoms
        assert list(s.atoms)[0] is a

    def test_hash_stable_across_iteration(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        h1 = hash(a)
        # re-iterate: same interned object, same hash
        again = list(s.atoms)[0]
        assert again is a
        assert hash(again) == h1

    def test_membership_is_identity(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        other = Atom(element="C")
        assert a in s.atoms
        assert other not in s.atoms


class TestRemovalStability:
    """Removing a middle atom keeps the others (and bonds) valid; no reindex."""

    def test_remove_middle_keeps_others(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        b = s.def_atom(element="N")
        c = s.def_atom(element="O")
        bd = s.def_bond(a, c)

        s.del_atom(b)

        assert len(s.atoms) == 2
        assert a in s.atoms
        assert c in s.atoms
        # the a--c bond still resolves to the same interned endpoints
        assert bd.itom is a
        assert bd.jtom is c
        assert a.get("element") == "C"
        assert c.get("element") == "O"

    def test_remove_atom_drops_incident_bonds(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        b = s.def_atom(element="N")
        s.def_bond(a, b)
        s.del_atom(b)
        assert len(s.bonds) == 0


class TestAdoptZeroCopy:
    """``Atomistic.adopt`` takes over a molrs-produced graph zero-copy."""

    def test_adopt_empties_source(self):
        g = molrs.Atomistic()
        ha = g.add_atom("C", 0, 0, 0)
        hb = g.add_atom("O", 1, 1, 1)
        g.add_bond(ha, hb)

        m = Atomistic.adopt(g)

        assert g.n_atoms == 0  # source emptied (moved, not copied)
        assert len(m.atoms) == 2
        assert len(m.bonds) == 1
        assert set(m.atoms["element"]) == {"C", "O"}

    def test_from_molrs_graph_removed(self):
        assert not hasattr(Atomistic, "from_molrs_graph")


class TestColumnZeroCopy:
    """Scalar columns route through the molrs world; ``column`` is a view."""

    def test_charge_column_is_numpy_view(self):
        s = Atomistic()
        s.def_atom(element="C", charge=0.0)
        s.def_atom(element="N", charge=-0.5)

        col = s.column(fields.CHARGE.key)
        assert isinstance(col, np.ndarray)
        assert np.allclose(col, [0.0, -0.5])

    def test_x_column_writethrough(self):
        s = Atomistic()
        a = s.def_atom(element="C", xyz=[1.0, 2.0, 3.0])
        col = s.column(fields.X.key)
        col[0] = 9.0
        assert a.get("x") == 9.0


class TestNoPlaceholder:
    """An atom without an element reads ``None`` — never the old ``"X"``."""

    def test_missing_element_is_none(self):
        s = Atomistic()
        a = s.def_atom()  # no chemical identity
        assert a.get(fields.ELEMENT.key) is None
        assert s.get(a.handle, fields.ELEMENT.key) is None

    def test_missing_field_getitem_raises(self):
        s = Atomistic()
        a = s.def_atom(element="C")
        with pytest.raises(KeyError):
            _ = a["nonexistent_field"]


class TestEntitySubclassing:
    def test_subclass_instantiable(self):
        class S(Atomistic):
            pass

        s = S()
        s.def_atom(element="C")
        assert len(s.atoms) == 1
        assert isinstance(s, molrs.Atomistic)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
