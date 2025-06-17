"""
Test suite for molpy.core.protocol module.

This test suite provides comprehensive coverage for the protocol classes
including Entity, SpatialMixin, HierarchyMixin, and Entities.
"""

import pytest
import numpy as np
from molpy.core.protocol import Entity, SpatialMixin, HierarchyMixin, Entities


class TestEntity:
    def test_dict_behavior(self):
        e = Entity(name="foo", bar=123)
        assert e["name"] == "foo"
        e["baz"] = 456
        assert e["baz"] == 456
        d = e.to_dict()
        assert d["bar"] == 123
        
    def test_clone(self):
        e = Entity(name="foo", bar=123)
        e2 = e.clone(bar=999)
        assert e2["bar"] == 999
        assert e["bar"] == 123
        assert e2 is not e
        
    def test_call(self):
        e = Entity(name="foo", bar=123)
        e2 = e(bar=888)
        assert e2["bar"] == 888
        assert e["bar"] == 123

    def test_hash_and_equality(self):
        e1 = Entity(name="foo")
        e2 = Entity(name="foo")
        assert e1 == e1  # Same object
        assert e1 != e2  # Different objects
        assert hash(e1) == id(e1)
        assert hash(e2) == id(e2)

    def test_comparison(self):
        e1 = Entity(name="foo")
        e2 = Entity(name="bar")
        # Just test that comparison works (order depends on memory addresses)
        assert (e1 < e2) or (e2 < e1)


class ConcreteSpatial(Entity, SpatialMixin):
    """Concrete implementation of SpatialMixin for testing."""
    
    def __init__(self, xyz=None, **kwargs):
        super().__init__(**kwargs)
        if xyz is not None:
            self.xyz = xyz
    
    @property
    def xyz(self):
        return np.array(self.get("xyz", [0.0, 0.0, 0.0]))
    
    @xyz.setter
    def xyz(self, value):
        self["xyz"] = np.asarray(value).tolist()


class TestSpatialMixin:
    def test_distance_calculation(self):
        s1 = ConcreteSpatial(xyz=[0, 0, 0])
        s2 = ConcreteSpatial(xyz=[3, 4, 0])
        assert abs(s1.distance_to(s2) - 5.0) < 1e-10
        
    def test_move(self):
        s = ConcreteSpatial(xyz=[1, 2, 3])
        s.move([2, 3, 4])
        np.testing.assert_array_almost_equal(s.xyz, [3, 5, 7])
        
    def test_rotate(self):
        s = ConcreteSpatial(xyz=[1, 0, 0])
        # Rotate 90 degrees around z-axis
        s.rotate([0, 0, 1], np.pi/2)
        expected = [0, 1, 0]
        np.testing.assert_array_almost_equal(s.xyz, expected, decimal=10)


class ConcreteHierarchy(Entity, HierarchyMixin):
    """Concrete implementation of HierarchyMixin for testing."""
    pass


class TestHierarchyMixin:
    def test_parent_child_relationship(self):
        parent = ConcreteHierarchy(name="parent")
        child = ConcreteHierarchy(name="child")
        
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children
        assert parent.is_root
        assert child.depth == 1
        
    def test_hierarchy_properties(self):
        root = ConcreteHierarchy(name="root")
        child1 = ConcreteHierarchy(name="child1")
        child2 = ConcreteHierarchy(name="child2")
        grandchild = ConcreteHierarchy(name="grandchild")
        
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        
        assert root.is_root
        assert not child1.is_leaf
        assert child2.is_leaf
        assert grandchild.depth == 2
        assert grandchild.get_root() is root
        
    def test_descendants(self):
        root = ConcreteHierarchy(name="root")
        child = ConcreteHierarchy(name="child")
        grandchild = ConcreteHierarchy(name="grandchild")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        descendants = root.get_descendants()
        assert len(descendants) == 2
        assert child in descendants
        assert grandchild in descendants
        
    def test_find_by_condition(self):
        root = ConcreteHierarchy(name="root")
        child = ConcreteHierarchy(name="target", special=True)
        grandchild = ConcreteHierarchy(name="grandchild")
        
        root.add_child(child)
        child.add_child(grandchild)
        
        found = root.find_by_condition(lambda x: x.get("special", False))
        assert found is child
        
        not_found = root.find_by_condition(lambda x: x.get("nonexistent", False))
        assert not_found is None
        
    def test_remove_child(self):
        parent = ConcreteHierarchy(name="parent")
        child = ConcreteHierarchy(name="child")
        
        parent.add_child(child)
        assert child in parent.children
        
        parent.remove_child(child)
        assert child not in parent.children
        assert child.parent is None


class TestEntities:
    def test_basic_operations(self):
        entities = Entities()
        e1 = Entity(name="first")
        e2 = Entity(name="second")
        
        entities.add(e1)
        entities.add(e2)
        
        assert len(entities) == 2
        assert e1 in entities
        assert e2 in entities
        assert entities[0] is e1
        
    def test_get_by_name(self):
        entities = Entities()
        e1 = Entity(name="first")
        e2 = Entity(name="second")
        
        entities.add(e1)
        entities.add(e2) 
        
        found = entities.get_by_name("first")
        assert found is e1
        
        not_found = entities.get_by_name("nonexistent")
        assert not_found is None
        
    def test_filter_and_get_by(self):
        entities = Entities()
        e1 = Entity(name="first", type="A")
        e2 = Entity(name="second", type="B")
        e3 = Entity(name="third", type="A")
        
        entities.extend([e1, e2, e3])
        
        # Test get_by
        found = entities.get_by(lambda x: x.get("type") == "B")
        assert found is e2
        
        # Test filter_by
        filtered = entities.filter_by(lambda x: x.get("type") == "A")
        assert len(filtered) == 2
        assert e1 in filtered
        assert e3 in filtered
        
    def test_remove_operations(self):
        entities = Entities()
        e1 = Entity(name="first")
        e2 = Entity(name="second")
        
        entities.extend([e1, e2])
        
        # Remove by object
        entities.remove(e1)
        assert e1 not in entities
        assert len(entities) == 1
        
        # Remove by name
        entities.remove("second")
        assert len(entities) == 0
        
    def test_clear_and_to_list(self):
        entities = Entities()
        e1 = Entity(name="first")
        e2 = Entity(name="second")
        
        entities.extend([e1, e2])
        assert len(entities) == 2
        
        # Test to_list
        as_list = entities.to_list()
        assert len(as_list) == 2
        assert isinstance(as_list, list)
        
        # Test clear
        entities.clear()
        assert len(entities) == 0
        
    def test_repr(self):
        entities = Entities()
        entities.add(Entity(name="test"))
        
        repr_str = repr(entities)
        assert "Entities" in repr_str
        assert "1 items" in repr_str
