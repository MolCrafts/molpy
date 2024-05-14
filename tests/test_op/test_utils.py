import pytest
from molpy.op.utils import op
import molpy as mp
import numpy as np
import numpy.testing as npt


class TestOpDecorator:

    def test_get_set_struct_value_by_path(self):
        mp.Alias.set("order", "order", int, None, "")
        struct = mp.StaticStruct()
        struct.atoms.xyz = np.array([1, 2, 3])
        struct.bonds.order = np.array([1, 2, 3])

        values = mp.op.utils.get_struct_value_by_path(
            struct, ["atoms.xyz", "bonds.order"]
        )
        npt.assert_equal(values[0], np.array([1, 2, 3]))
        npt.assert_equal(values[1], np.array([1, 2, 3]))

        mp.op.utils.set_struct_value_by_path(
            struct,
            ["atoms.xyz", "bonds.order"],
            [np.array([3, 2, 1]), np.array([3, 2, 1])],
        )

        npt.assert_equal(struct.atoms.xyz, np.array([3, 2, 1]))
        npt.assert_equal(struct.bonds.order, np.array([3, 2, 1]))

    def test_op_decorator(self):

        @op(input_key=["a", "b"], output_key=["c"])
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

        struct = mp.StaticStruct()
        struct["a"] = 1
        struct["b"] = 2
        assert add(struct)["c"] == 3

    def test_op_level(self):

        @op(input_key=["atoms.xyz"], output_key=["atoms.xyz"])
        def add(a, b):
            return a + b

        struct = mp.StaticStruct()
        struct.atoms.xyz = np.array([1, 2, 3])
        npt.assert_equal(
            add(struct, np.array([2, 3, 4])).atoms.xyz, np.array([3, 5, 7])
        )
