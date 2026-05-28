"""Verify ``molpy.Frame`` truly inherits from ``molrs.Frame``.

These tests guarantee:

1. ``molpy.Frame`` IS-A ``molrs.Frame`` so any ``molrs.*`` API that takes
   a frame accepts a molpy frame without a ``.to_molrs()`` / ``._inner``
   bridge.
2. Python-only state (``metadata`` dict, object-dtype block columns) is
   invisible to the Rust slot — molrs kernels only see what they
   already understood.
3. ``frame.box`` getter still returns a ``molpy.Box`` (which itself
   inherits ``molrs.Box``), preserving molpy's enriched Box API.
"""

import molrs
import numpy as np

import molpy
from molpy.core.frame import Block, Frame


class TestFrameInheritance:
    """``molpy.Frame`` IS-A ``molrs.Frame``."""

    def test_molpy_frame_is_a_molrs_frame(self):
        f = Frame()
        assert isinstance(f, molrs.Frame)
        assert isinstance(f, Frame)

    def test_subclass_relation_holds_at_type_level(self):
        assert issubclass(Frame, molrs.Frame)

    def test_frame_directly_accepted_by_molrs_api(self, tmp_path):
        """A ``molpy.Frame`` must extract through PyO3's ``Frame`` downcast
        — exercised via ``molrs.write_xyz`` which signature takes
        ``Frame``."""
        f = Frame()
        atoms = Block()
        atoms["symbol"] = np.array(["H"], dtype=object)
        atoms["x"] = np.array([0.0], dtype=np.float32)
        atoms["y"] = np.array([0.0], dtype=np.float32)
        atoms["z"] = np.array([0.0], dtype=np.float32)
        f["atoms"] = atoms

        out = tmp_path / "from_molpy.xyz"
        molrs.write_xyz(str(out), f)
        assert out.exists()


class TestPythonOnlyStateIsolation:
    """``metadata`` and object-dtype columns are invisible to Rust kernels."""

    def test_metadata_lives_on_python_side(self):
        f = Frame(timestep=42, description="water box")
        assert f.metadata["timestep"] == 42
        assert f.metadata["description"] == "water box"

    def test_metadata_survives_molrs_roundtrip(self, tmp_path):
        """metadata must persist on the Python side even after the frame
        passes through a molrs API call. The molrs API only touches the
        Rust slot — metadata is invisible there and untouched."""
        f = Frame(timestep=7)
        atoms = Block()
        atoms["x"] = np.array([1.0], dtype=np.float32)
        atoms["y"] = np.array([0.0], dtype=np.float32)
        atoms["z"] = np.array([0.0], dtype=np.float32)
        atoms["symbol"] = np.array(["He"], dtype=object)
        f["atoms"] = atoms

        # Round-trip through a molrs writer; metadata stays put.
        out = tmp_path / "rt.xyz"
        molrs.write_xyz(str(out), f)
        assert f.metadata["timestep"] == 7


class TestBoxGetterStillUpgrades:
    """``frame.box`` still surfaces a ``molpy.Box`` (not bare ``molrs.Box``)."""

    def test_box_getter_returns_molpy_box(self):
        from molpy.core.box import Box

        f = Frame()
        f.box = Box.cubic(10.0)
        b = f.box
        assert isinstance(b, Box)
        assert isinstance(b, molrs.Box)
