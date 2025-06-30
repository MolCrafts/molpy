"""
Test suite for molpy.core.struct module.

This test suite provides tests for the base Struct class and MolecularStructure.
Atomic structure tests have been moved to test_atoms.py.
"""

import pytest
import numpy as np
from molpy.core.atomistic import Struct, Atom, Bond, Angle, Dihedral, Atomistic


class TestEntity:
    def test_dict_behavior(self):
        e = Struct(name="foo", bar=123)
        assert e["name"] == "foo"
        e["baz"] = 456
        assert e["baz"] == 456
        d = e.to_dict()
        assert d["bar"] == 123
    
    def test_clone(self):
        e = Struct(name="foo", bar=123)
        e2 = e.clone(bar=999)
        assert e2["bar"] == 999
        assert e["bar"] == 123
        assert e2 is not e
    
    def test_call(self):
        e = Struct(name="foo", bar=123)
        e2 = e(bar=888)
        assert e2["bar"] == 888
        assert e["bar"] == 123

class TestStruct:
    def test_init_basic(self):
        struct = Struct(name="test_struct")
        assert struct["name"] == "test_struct"
        assert "test_struct" in repr(struct)
        unnamed = Struct()
        assert "name" not in repr(unnamed)
    
    def test_clone(self):
        struct = Struct(name="foo")
        struct2 = struct.clone(name="bar")
        assert struct2["name"] == "bar"
        assert struct["name"] == "foo"
        assert struct2 is not struct


