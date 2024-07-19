import pytest
import molpy as mp
import numpy as np
import numpy.testing as npt
from pathlib import Path

class TestFrame:

    @pytest.fixture(name="frame")
    def test_init(self):

        frame = mp.Frame(name="frame")
        assert frame.name == "frame"

        yield frame

    def test_add_struct(self, test_data_path: Path):

        tfsi1 = mp.io.load_frame(test_data_path / "data/pdb/tfsi.pdb")
        tfsi2 = tfsi1.copy()

        li_ion1 = mp.io.load_frame(test_data_path / "data/pdb/li+.pdb")
        li_ion2 = li_ion1.copy()

        frame = mp.Frame(name="frame")
        frame.merge(tfsi1)
        frame.merge(tfsi2)
        frame.merge(li_ion1)
        frame.merge(li_ion2)

        assert frame.n_atoms == 15 * 2 + 2
        assert frame.topology.n_bonds == 14 * 2
        print(frame.topology.bonds)
