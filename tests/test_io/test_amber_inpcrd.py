"""Tests for AMBER inpcrd file reading."""

from textwrap import dedent

import numpy as np
import pytest

from molpy.core.frame import Frame
from molpy.io.data.amber import AmberInpcrdReader


@pytest.fixture
def tmp_inpcrd_dir(tmp_path):
    """Create temporary directory for inpcrd test files."""
    return tmp_path / "inpcrd_tests"


def test_inpcrd_basic_coords_only(tmp_inpcrd_dir):
    """Test reading inpcrd with only coordinates (no velocities, no box)."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test.inpcrd"

    # 3 atoms: need (3*3 + 5) // 6 = 2 lines of coordinates
    content = dedent(
        """\
        Simple 3-atom system
          3
          0.000000    1.000000    2.000000    3.000000    4.000000    5.000000
          6.000000    7.000000    8.000000
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert "atoms" in frame
    assert frame["atoms"].nrows == 3
    assert frame.metadata["title"] == "Simple 3-atom system"

    # Check coordinates
    np.testing.assert_array_almost_equal(
        frame["atoms"]["x"],
        [0.0, 3.0, 6.0],
    )
    np.testing.assert_array_almost_equal(
        frame["atoms"]["y"],
        [1.0, 4.0, 7.0],
    )
    np.testing.assert_array_almost_equal(
        frame["atoms"]["z"],
        [2.0, 5.0, 8.0],
    )


def test_inpcrd_with_time(tmp_inpcrd_dir):
    """Test reading inpcrd with timestamp."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_time.inpcrd"

    content = dedent(
        """\
        Test with time
          2   100.5
          1.0    2.0    3.0    4.0    5.0    6.0
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame["atoms"].nrows == 2
    assert frame.metadata["timestep"] == 100


def test_inpcrd_with_velocities(tmp_inpcrd_dir):
    """Test reading inpcrd with coordinates and velocities.

    Velocities only appear in restart files (those with a timestamp).
    """
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_vel.inpcrd"

    # 2 atoms: (2*3 + 5) // 6 = 1 line each for coords and velocities
    # Timestamp (25.0) signals this is a restart file with velocities
    content = dedent(
        """\
        Test with velocities
          2   25.0
          1.0    2.0    3.0    4.0    5.0    6.0
          0.1    0.2    0.3    0.4    0.5    0.6
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame["atoms"].nrows == 2
    assert "vel" in frame["atoms"]

    # Check velocities
    expected_vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    np.testing.assert_array_almost_equal(frame["atoms"]["vel"], expected_vel)


def test_inpcrd_with_box(tmp_inpcrd_dir):
    """Test reading inpcrd with box dimensions."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_box.inpcrd"

    content = dedent(
        """\
        Test with box
          2
          1.0    2.0    3.0    4.0    5.0    6.0
         10.0   20.0   30.0   90.0   90.0   90.0
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame.box is not None
    # Box should use first 3 values as diagonal
    expected_box = np.diag([10.0, 20.0, 30.0])
    np.testing.assert_array_almost_equal(frame.box.matrix, expected_box)


def test_inpcrd_with_velocities_and_box(tmp_inpcrd_dir):
    """Test reading inpcrd with all optional sections."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_full.inpcrd"

    content = dedent(
        """\
        Full test: coords + vels + box
          2   50.0
          1.0    2.0    3.0    4.0    5.0    6.0
          0.1    0.2    0.3    0.4    0.5    0.6
         15.0   15.0   15.0
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame["atoms"].nrows == 2
    assert "vel" in frame["atoms"]
    assert frame.box is not None
    assert frame.metadata["timestep"] == 50


def test_inpcrd_update_existing_frame(tmp_inpcrd_dir):
    """Test updating coordinates in existing frame."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_update.inpcrd"

    content = dedent(
        """\
        Update test
          2
          9.0    8.0    7.0    6.0    5.0    4.0
    """
    )
    inpcrd_file.write_text(content)

    # Create pre-existing frame with atoms
    existing_frame = Frame()
    existing_frame["atoms"] = {
        "id": np.array([1, 2]),
        "name": np.array(["CA", "CB"]),
        "x": np.array([0.0, 0.0]),
        "y": np.array([0.0, 0.0]),
        "z": np.array([0.0, 0.0]),
    }

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read(existing_frame)

    # Should update coordinates but preserve name
    assert frame["atoms"]["name"][0] == "CA"
    np.testing.assert_array_almost_equal(frame["atoms"]["x"], [9.0, 6.0])
    np.testing.assert_array_almost_equal(frame["atoms"]["y"], [8.0, 5.0])
    np.testing.assert_array_almost_equal(frame["atoms"]["z"], [7.0, 4.0])


def test_inpcrd_too_short_file(tmp_inpcrd_dir):
    """Test error handling for file that's too short."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_short.inpcrd"

    content = "Only title\n"
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    with pytest.raises(ValueError, match="too short"):
        reader.read()


def test_inpcrd_insufficient_coords(tmp_inpcrd_dir):
    """Test error handling for insufficient coordinate lines."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_insufficient.inpcrd"

    # Claims 5 atoms but doesn't provide enough data
    # Need (5*3 + 5) // 6 = 3 lines, but only provide 1
    content = dedent(
        """\
        Insufficient data
          5
          1.0    2.0    3.0    4.0    5.0    6.0
    """
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    with pytest.raises(ValueError, match="Not enough lines"):
        reader.read()


def test_inpcrd_atom_count_mismatch(tmp_inpcrd_dir):
    """Test error with existing frame that has wrong atom count."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_mismatch.inpcrd"

    # 3 atoms need (3*3 + 5) // 6 = 2 lines of coordinates
    content = dedent(
        """\
        Mismatch test
          3
          1.0    2.0    3.0    4.0    5.0    6.0
          7.0    8.0    9.0
    """
    )
    inpcrd_file.write_text(content)

    # Create frame with wrong number of atoms
    existing_frame = Frame()
    existing_frame["atoms"] = {
        "id": np.array([1, 2]),  # Only 2 atoms
        "x": np.zeros(2),
        "y": np.zeros(2),
        "z": np.zeros(2),
    }

    reader = AmberInpcrdReader(inpcrd_file)
    with pytest.raises(ValueError, match="atoms block has 2.*but inpcrd has 3"):
        reader.read(existing_frame)


def test_inpcrd_large_system(tmp_inpcrd_dir):
    """Test reading a larger system (10 atoms) to verify line counting."""
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_large.inpcrd"

    # 10 atoms: (10*3 + 5) // 6 = 5 lines of coordinates
    coords = []
    for i in range(30):  # 30 coordinate values
        coords.append(f"{i:12.7f}")

    # Split into lines of 6 values each
    lines = [
        "Large system test",
        " 10",
    ]
    for i in range(0, 30, 6):
        lines.append("".join(coords[i : i + 6]))

    inpcrd_file.write_text("\n".join(lines))

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame["atoms"].nrows == 10
    # Check first and last atoms
    np.testing.assert_array_almost_equal(frame["atoms"]["x"][0], 0.0)
    np.testing.assert_array_almost_equal(frame["atoms"]["z"][9], 29.0)


def test_inpcrd_fixed_width_no_whitespace(tmp_inpcrd_dir):
    """Negative signs can abut the previous value in Fortran 6F12.7 format.

    For example ``50.5413286-100.7101036`` must be split as two values
    (50.5413286, -100.7101036) instead of raising a parse error.
    """
    tmp_inpcrd_dir.mkdir()
    inpcrd_file = tmp_inpcrd_dir / "test_abutting.inpcrd"

    # 2 atoms, 6 values, each 12 chars wide.
    # Construct a line where negative signs touch the prior value.
    content = (
        "Abutting negatives\n"
        "  2\n"
        "  50.5413286-100.7101036  12.3456789 -44.5678901  88.8888888  -0.1234567\n"
    )
    inpcrd_file.write_text(content)

    reader = AmberInpcrdReader(inpcrd_file)
    frame = reader.read()

    assert frame["atoms"].nrows == 2
    np.testing.assert_allclose(frame["atoms"]["x"], [50.5413286, -44.5678901])
    np.testing.assert_allclose(frame["atoms"]["y"], [-100.7101036, 88.8888888])
    np.testing.assert_allclose(frame["atoms"]["z"], [12.3456789, -0.1234567])
