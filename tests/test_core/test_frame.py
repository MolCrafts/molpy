import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt

class TestFrame:

    @pytest.fixture(name="frame")
    def test_init(self):

        frame = mp.Frame(name="frame")
        assert frame.name == "frame"

        yield frame

    def test_add_struct(self, frame):

        struct = mp.Struct(name="struct1")
        frame.add_struct(struct)

        assert frame.n_atoms == 10