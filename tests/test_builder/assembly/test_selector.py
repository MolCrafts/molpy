"""Unit tests for :mod:`molpy.builder.assembly._selector`."""

import inspect

import pytest

from molpy.builder.assembly import Selector


class TestSelector:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Selector()

    def test_atoms_of_returns_all_named_handles_once(self):
        assert Selector._atoms_of({1: 7, 2: 7, 3: 11}) == frozenset({7, 11})

    def test_select_is_the_only_required_extension_point(self):
        assert inspect.isabstract(Selector)
        assert Selector.__abstractmethods__ == frozenset({"select"})
