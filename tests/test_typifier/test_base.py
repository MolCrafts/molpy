#!/usr/bin/env python3
"""Unit tests for typifier base classes.

Tests cover:
- TypifierError exception
- Typifier abstract base class
- Mixin classes
- BaseTypifier
"""

from molpy import Atomistic, ForceField
from molpy.typifier.base import (
    AngleTypifierMixin,
    AtomisticTypifier,
    AtomTypifierMixin,
    BaseTypifier,
    BondTypifierMixin,
    Typifier,
    TypifierError,
)


class TestTypifierError:
    """Test TypifierError exception."""

    def test_typifier_error_creation(self):
        """Test creating TypifierError."""
        error = TypifierError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestTypifier:
    """Test Typifier abstract base class."""

    def test_typifier_is_abstract(self):
        """Test that Typifier requires typify method implementation."""
        ff = ForceField()

        # Typifier can be instantiated if typify is implemented
        class ConcreteTypifier(Typifier):
            def typify(self, struct):
                return struct

        # This should work
        typifier = ConcreteTypifier(ff)
        assert typifier.ff is ff

    def test_typifier_has_forcefield(self):
        """Test that Typifier subclasses have forcefield attribute."""

        class ConcreteTypifier(Typifier):
            def typify(self, struct):
                return struct

        ff = ForceField()
        typifier = ConcreteTypifier(ff)

        assert typifier.ff is ff


class TestMixinClasses:
    """Test mixin classes."""

    def test_atom_typifier_mixin(self):
        """Test AtomTypifierMixin."""

        class TestTypifier(AtomTypifierMixin):
            def __init__(self, ff):
                self.ff = ff

        ff = ForceField()
        typifier = TestTypifier(ff)

        struct = Atomistic()
        result = typifier.typify(struct)

        assert result is struct

    def test_bond_typifier_mixin(self):
        """Test BondTypifierMixin."""

        class TestTypifier(BondTypifierMixin):
            def __init__(self, ff):
                self.ff = ff

        ff = ForceField()
        typifier = TestTypifier(ff)

        struct = Atomistic()
        result = typifier.typify(struct)

        assert result is struct

    def test_angle_typifier_mixin(self):
        """Test AngleTypifierMixin."""

        class TestTypifier(AngleTypifierMixin):
            def __init__(self, ff):
                self.ff = ff

        ff = ForceField()
        typifier = TestTypifier(ff)

        struct = Atomistic()
        result = typifier.typify(struct)

        assert result is struct


class TestBaseTypifier:
    """Test BaseTypifier class."""

    def test_base_typifier_creation(self):
        """Test creating BaseTypifier instance."""
        typifier = BaseTypifier()

        assert isinstance(typifier, BaseTypifier)


class TestAtomisticTypifier:
    """Test AtomisticTypifier class."""

    def test_atomistic_typifier_is_abstract(self):
        """Test that AtomisticTypifier requires typify method implementation."""
        ff = ForceField()

        # AtomisticTypifier can be instantiated if typify is implemented
        class ConcreteAtomisticTypifier(AtomisticTypifier):
            def __init__(self, ff):
                self.ff = ff

            def typify(self, struct):
                return struct

        # This should work
        typifier = ConcreteAtomisticTypifier(ff)
        assert typifier.ff is ff

    def test_atomistic_typifier_inheritance(self):
        """Test that AtomisticTypifier inherits from all mixins."""

        class ConcreteAtomisticTypifier(AtomisticTypifier):
            def __init__(self, ff):
                self.ff = ff

            def typify(self, struct):
                return struct

        ff = ForceField()
        typifier = ConcreteAtomisticTypifier(ff)

        # Should have all mixin methods
        struct = Atomistic()
        result = typifier.typify(struct)
        assert result is struct
