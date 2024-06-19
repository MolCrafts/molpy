import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

from molpy.core.struct import BaseStructure

from ..utils import random_array

class TestBaseStructure:

    @pytest.fixture(name="struct")
    def test_init(self):

        struct = BaseStructure(name="struct1")
        assert struct.name == "struct1"

class TestStruct:

    @pytest.fixture(name="struct")
    def test_init(self):

        struct = mp.Struct(name="struct1")
        struct.atoms.xyz = random_array((3, 3))
        assert struct.name == "struct1"
        yield struct

    def test_clone(self, struct: mp.Struct):
        struct2 = struct.clone()
        assert struct2.name == "struct1"
        assert struct2.n_atoms == 3