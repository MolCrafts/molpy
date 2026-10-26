"""Verify that Python subclasses of ``mp.core.Box`` are allowed.

molpy's ``Box`` subclasses the native one. The other core data classes
(``Frame``, ``Block``, ``Atomistic``, ``ForceField``, …) are subclassable too
(see test_frame.py, test_block.py, test_views.py, test_forcefield_builder.py).
"""

import pickle

import molpy as mp
import numpy as np
import pytest


class _PickledBox(mp.core.Box):
    """Module level, so pickle can find it."""


class TestBoxSubclass:
    """``class Sub(mp.core.Box)`` must instantiate and inherit base methods."""

    def test_subclass_can_be_defined(self):
        class Sub(mp.core.Box):
            pass

        assert issubclass(Sub, mp.core.Box)

    def test_subclass_instance_is_a_box(self):
        class Sub(mp.core.Box):
            pass

        instance = Sub(mp.core.Cell.matrix(np.eye(3) * 10.0))
        assert isinstance(instance, mp.core.Box)
        assert isinstance(instance, Sub)

    def test_subclass_inherits_methods(self):
        class Sub(mp.core.Box):
            pass

        instance = Sub(mp.core.Cell.matrix(np.eye(3) * 10.0))
        # Inherited from molpy.core.Box.
        assert instance.volume() == pytest.approx(1000.0)

    def test_subclass_can_add_python_attributes(self):
        # PyO3 `#[new]` is the binding constructor, so a subclass that wants
        # additional kwargs must override __new__ to strip them before
        # delegating. Subsequent Python attribute assignment is unrestricted.
        class Sub(mp.core.Box):
            def __new__(cls, cell, *, label):
                instance = super().__new__(cls, cell)
                instance.label = label
                return instance

        instance = Sub(mp.core.Cell.matrix(np.eye(3) * 5.0), label="cube-5")
        assert instance.label == "cube-5"
        assert instance.volume() == pytest.approx(125.0)

    def test_subclass_can_override_repr(self):
        class Sub(mp.core.Box):
            def __repr__(self):
                return "<Sub>"

        instance = Sub(mp.core.Cell.matrix(np.eye(3) * 2.0))
        assert repr(instance) == "<Sub>"

    def test_subclass_pickles_as_itself_with_its_attributes(self):
        instance = _PickledBox(mp.core.Cell.matrix(np.eye(3) * 3.0))
        instance.label = "cube-3"
        back = pickle.loads(pickle.dumps(instance))
        assert type(back) is _PickledBox
        assert back.label == "cube-3"
        assert back.volume() == pytest.approx(27.0)
