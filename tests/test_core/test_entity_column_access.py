"""
Test column-style access for Entities and TypeBucket.
Tests the feature: atomistic.atoms["symbol"] = ['C', 'C']
"""

import numpy as np
import pytest

from molpy import Atom, Atomistic
from molpy.core.entity import Entities, Entity, TypeBucket


class TestEntitiesColumnAccess:
    """Test Entities list with column-style access."""

    def test_column_read_simple(self):
        """Test reading column data with string key."""
        a1 = Entity({"symbol": "C", "mass": 12.0})
        a2 = Entity({"symbol": "N", "mass": 14.0})
        a3 = Entity({"symbol": "O", "mass": 16.0})

        ents = Entities[Entity]([a1, a2, a3])

        # Column access - now returns numpy arrays
        symbols = ents["symbol"]
        masses = ents["mass"]

        assert np.array_equal(symbols, ["C", "N", "O"])
        assert np.array_equal(masses, [12.0, 14.0, 16.0])

    def test_column_read_missing_keys(self):
        """Test column access when some entities lack the key."""
        a1 = Entity({"symbol": "C"})
        a2 = Entity({"symbol": "N", "charge": -1})
        a3 = Entity({})

        ents = Entities[Entity]([a1, a2, a3])

        # get() returns None for missing keys
        symbols = ents["symbol"]
        charges = ents["charge"]

        assert np.array_equal(symbols, ["C", "N", None])
        assert np.array_equal(charges, [None, -1, None])

    def test_integer_indexing(self):
        """Test that integer indexing still works."""
        a1 = Entity({"symbol": "C"})
        a2 = Entity({"symbol": "N"})

        ents = Entities[Entity]([a1, a2])

        assert ents[0] is a1
        assert ents[1] is a2

    def test_slice_indexing(self):
        """Test that slice indexing still works."""
        a1 = Entity({"symbol": "C"})
        a2 = Entity({"symbol": "N"})
        a3 = Entity({"symbol": "O"})

        ents = Entities[Entity]([a1, a2, a3])

        sliced = ents[1:]
        assert len(sliced) == 2
        assert sliced[0] is a2
        assert sliced[1] is a3

    def test_empty_entities(self):
        """Test column access on empty Entities."""
        ents = Entities[Entity]()
        symbols = ents["symbol"]
        assert len(symbols) == 0


class TestTypeBucketWithEntities:
    """Test TypeBucket returning Entities for column access."""

    def test_bucket_returns_entities(self):
        """Test that bucket() returns Entities instance."""
        tb = TypeBucket[Entity]()
        e1 = Entity({"type": "A"})
        e2 = Entity({"type": "B"})

        tb.add(e1)
        tb.add(e2)

        result = tb.bucket(Entity)

        # Should be Entities instance
        assert isinstance(result, Entities)
        assert len(result) == 2

    def test_bucket_column_access(self):
        """Test column access on bucket result."""
        tb = TypeBucket[Entity]()
        e1 = Entity({"symbol": "C", "mass": 12.0})
        e2 = Entity({"symbol": "N", "mass": 14.0})

        tb.add(e1)
        tb.add(e2)

        entities = tb.bucket(Entity)
        symbols = entities["symbol"]
        masses = entities["mass"]

        assert np.array_equal(symbols, ["C", "N"])
        assert np.array_equal(masses, [12.0, 14.0])

    def test_exact_bucket_returns_entities(self):
        """Test that exact_bucket() returns Entities."""
        tb = TypeBucket[Entity]()
        e1 = Entity({"x": 1})

        tb.add(e1)
        result = tb.exact_bucket(Entity)

        assert isinstance(result, Entities)
        assert len(result) == 1

    def test_all_returns_entities(self):
        """Test that all() returns Entities."""
        tb = TypeBucket[Entity]()
        e1 = Entity({"a": 1})
        e2 = Entity({"b": 2})

        tb.add(e1)
        tb.add(e2)

        all_ents = tb.all()
        assert isinstance(all_ents, Entities)
        assert len(all_ents) == 2


class TestAtomisticColumnAccess:
    """Test the target use case: atomistic.atoms["symbol"]"""

    def test_atoms_column_read(self):
        """Test reading atom symbols via column access."""
        atomistic = Atomistic()

        a1 = Atom({"symbol": "C", "xyz": [0, 0, 0]})
        a2 = Atom({"symbol": "C", "xyz": [1, 0, 0]})
        a3 = Atom({"symbol": "H", "xyz": [0, 1, 0]})

        atomistic.add_entity(a1, a2, a3)

        # Target syntax - returns numpy array
        symbols = atomistic.atoms["symbol"]

        assert np.array_equal(symbols, ["C", "C", "H"])

    def test_atoms_column_read_positions(self):
        """Test reading positions via column access."""
        atomistic = Atomistic()

        a1 = Atom({"symbol": "C", "xyz": [0.0, 0.0, 0.0]})
        a2 = Atom({"symbol": "N", "xyz": [1.5, 0.0, 0.0]})

        atomistic.add_entity(a1, a2)

        positions = atomistic.atoms["xyz"]

        assert len(positions) == 2
        assert np.array_equal(positions[0], [0.0, 0.0, 0.0])
        assert np.array_equal(positions[1], [1.5, 0.0, 0.0])

    def test_atoms_property_returns_entities(self):
        """Ensure atomistic.atoms returns Entities, not plain list."""
        atomistic = Atomistic()

        a1 = Atom({"symbol": "C"})
        atomistic.add_entity(a1)

        atoms = atomistic.atoms

        # Should be Entities to support column access
        assert isinstance(atoms, Entities)

    def test_mixed_attribute_access(self):
        """Test accessing different attributes from same entity set."""
        atomistic = Atomistic()

        a1 = Atom({"symbol": "C", "mass": 12.0, "charge": 0.0})
        a2 = Atom({"symbol": "N", "mass": 14.0, "charge": -0.5})

        atomistic.add_entity(a1, a2)

        symbols = atomistic.atoms["symbol"]
        masses = atomistic.atoms["mass"]
        charges = atomistic.atoms["charge"]

        assert np.array_equal(symbols, ["C", "N"])
        assert np.array_equal(masses, [12.0, 14.0])
        assert np.array_equal(charges, [0.0, -0.5])

    def test_column_access_with_subclasses(self):
        """Test column access works with Entity subclasses."""

        class SpecialAtom(Atom):
            pass

        atomistic = Atomistic()

        a1 = SpecialAtom({"symbol": "C", "special": True})
        a2 = Atom({"symbol": "N", "special": False})

        atomistic.add_entity(a1, a2)

        # Should get all atoms (including subclass)
        symbols = atomistic.atoms["symbol"]
        specials = atomistic.atoms["special"]

        # Order may vary based on bucket implementation
        assert set(symbols) == {"C", "N"}
        assert set(specials) == {True, False}


class TestTypeBucketWithSubclasses:
    """Test TypeBucket behavior with inheritance."""

    def test_subclass_bucketing(self):
        """Test that subclasses are returned in parent bucket."""

        class SpecialEntity(Entity):
            pass

        tb = TypeBucket[Entity]()
        e1 = Entity({"type": "base"})
        e2 = SpecialEntity({"type": "special"})

        tb.add(e1)
        tb.add(e2)

        # bucket(Entity) should return both
        all_ents = tb.bucket(Entity)
        assert len(all_ents) == 2

        # Column access should work on mixed types
        types = all_ents["type"]
        assert np.array_equal(types, ["base", "special"])

    def test_exact_bucket_excludes_subclasses(self):
        """Test exact_bucket only returns exact type."""

        class SpecialEntity(Entity):
            pass

        tb = TypeBucket[Entity]()
        e1 = Entity({"x": 1})
        e2 = SpecialEntity({"x": 2})

        tb.add(e1)
        tb.add(e2)

        # exact_bucket should only return base Entity
        exact = tb.exact_bucket(Entity)
        assert len(exact) == 1
        assert exact["x"] == [1]


class TestTypeBucketModification:
    """Test that TypeBucket modifications work with Entities."""

    def test_add_many(self):
        """Test add_many with column access."""
        tb = TypeBucket[Entity]()
        entities = [
            Entity({"id": 1}),
            Entity({"id": 2}),
            Entity({"id": 3}),
        ]

        tb.add_many(entities)

        result = tb.bucket(Entity)
        ids = result["id"]
        assert np.array_equal(ids, [1, 2, 3])

    def test_remove_maintains_entities(self):
        """Test that remove keeps Entities structure."""
        tb = TypeBucket[Entity]()
        e1 = Entity({"name": "first"})
        e2 = Entity({"name": "second"})

        tb.add(e1)
        tb.add(e2)

        tb.remove(e1)

        result = tb.bucket(Entity)
        names = result["name"]
        assert names == ["second"]

    def test_len_counts_all_entities(self):
        """Test that len() works correctly."""
        tb = TypeBucket[Entity]()
        tb.add(Entity({"x": 1}))
        tb.add(Entity({"x": 2}))
        tb.add(Entity({"x": 3}))

        assert len(tb) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
