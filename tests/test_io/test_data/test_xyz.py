"""
Tests for XYZReader and XYZWriter using chemfiles-testcases/data/xyz files.
"""
import pytest
import numpy as np
from pathlib import Path
from molpy.io.data.xyz import XYZReader
from molpy.core import Box

@pytest.fixture
def xyz_test_files(TEST_DATA_DIR) -> Path:
    return TEST_DATA_DIR / "data/xyz/extended.xyz"

class TestXYZReader:

    def test_extend_format(self, xyz_test_files):
        reader = XYZReader(xyz_test_files)
        frame = reader.read()

        box = frame.box
        assert isinstance(box, Box)
        assert box.matrix.shape == (3, 3)
        assert np.allclose(box.matrix, np.array([8.43116035, 0.0, 0.0, 0.158219155128, 14.5042431863, 0.0, 1.16980663624, 4.4685149855, 14.9100096405]).reshape(3, 3))

        pytest.approx(frame.metadata["ENERGY"], -2069.84934116)

        assert frame["atoms", "xyz"].shape == (192, 3)