"""Unit tests for :mod:`molpy.builder.assembly._finalize`."""

from molpy.builder.assembly import AssemblyFinalizer


class TestAssemblyFinalizer:
    def test_enables_aromaticity_perception_for_assembled_products(self):
        assert AssemblyFinalizer().perceive_aromaticity is True
