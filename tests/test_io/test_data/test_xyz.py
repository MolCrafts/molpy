"""
Tests for XYZReader and XYZWriter using chemfiles-testcases/xyz files.
"""

from pathlib import Path

import numpy as np
import pytest

from molpy.core import Box
from molpy.io.data.xyz import XYZReader


@pytest.fixture
def xyz_test_files(TEST_DATA_DIR) -> Path:
    return TEST_DATA_DIR / "xyz/extended.xyz"


class TestXYZReader:
    def test_extend_format(self, xyz_test_files):
        reader = XYZReader(xyz_test_files)
        frame = reader.read()

        box = frame.box
        # Check that coordinates are stored as separate x, y, z fields
        assert "x" in frame["atoms"]
        assert "y" in frame["atoms"]
        assert "z" in frame["atoms"]
        assert frame["atoms"]["x"].shape == (192,)
        assert frame["atoms"]["y"].shape == (192,)
        assert frame["atoms"]["z"].shape == (192,)
        assert isinstance(box, Box)
        assert box.matrix.shape == (3, 3)
        assert np.allclose(
            box.matrix,
            np.array(
                [
                    8.43116035,
                    0.0,
                    0.0,
                    0.158219155128,
                    14.5042431863,
                    0.0,
                    1.16980663624,
                    4.4685149855,
                    14.9100096405,
                ]
            ).reshape(3, 3),
        )

        pytest.approx(frame.metadata["ENERGY"], -2069.84934116)

