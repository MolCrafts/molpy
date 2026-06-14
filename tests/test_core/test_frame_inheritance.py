"""Verify ``molpy.Frame`` collapses onto the canonical ``molrs.Frame``.

After the ``frame-block-sink`` cutover, molpy no longer subclasses the molrs
column store. These tests pin the new contract:

1. ``molpy.core.frame.Frame is molrs.Frame`` — identity, not a subclass.
2. The frame carries the rich molrs API (metadata dict, blocks, to_dict) and is
   accepted directly by every ``molrs.*`` API.
3. ``frame.box`` returns the molrs box carrying the enriched accessors that were
   sunk into molrs (``is_free`` / ``style`` / ``volume``).
4. Object / None columns are rejected fail-fast (numpy-only Store contract).
"""

import molrs
import numpy as np
import pytest

from molpy.core.frame import Block, Frame


class TestFrameIdentityCollapse:
    """``molpy.Frame`` IS ``molrs.Frame`` (no intermediate subclass)."""

    def test_frame_is_molrs_frame(self):
        assert Frame is molrs.Frame

    def test_instance_is_molrs_frame(self):
        assert isinstance(Frame(), molrs.Frame)

    def test_frame_accepted_by_molrs_api(self, tmp_path):
        f = Frame()
        atoms = Block()
        atoms["symbol"] = np.array(["H"])  # native str column (numpy-only)
        atoms["x"] = np.array([0.0], dtype=np.float64)
        atoms["y"] = np.array([0.0], dtype=np.float64)
        atoms["z"] = np.array([0.0], dtype=np.float64)
        f["atoms"] = atoms
        out = tmp_path / "from_molpy.xyz"
        molrs.write_xyz(str(out), f)
        assert out.exists()


class TestRichFrameSurface:
    """The collapsed Frame exposes the molrs rich API."""

    def test_metadata(self):
        f = Frame(timestep=42, description="water box")
        assert f.metadata["timestep"] == 42
        assert f.metadata["description"] == "water box"

    def test_metadata_survives_molrs_roundtrip(self, tmp_path):
        f = Frame(timestep=7)
        atoms = Block()
        atoms["x"] = np.array([1.0], dtype=np.float64)
        atoms["y"] = np.array([0.0], dtype=np.float64)
        atoms["z"] = np.array([0.0], dtype=np.float64)
        atoms["symbol"] = np.array(["He"])
        f["atoms"] = atoms
        molrs.write_xyz(str(tmp_path / "rt.xyz"), f)
        assert f.metadata["timestep"] == 7


class TestBoxEnrichmentSunkIntoMolrs:
    """``frame.box`` is the molrs box, now carrying is_free / style / volume."""

    def test_box_is_molrs_box_with_enriched_api(self):
        f = Frame()
        f.box = molrs.Box.cube(10.0)
        b = f.box
        assert isinstance(b, molrs.Box)
        assert b.is_free is False
        assert b.style == "orthogonal"
        assert b.volume() == pytest.approx(1000.0, abs=1.0)


class TestNumpyOnlyContract:
    """Object / None columns are rejected at write time."""

    def test_object_column_rejected(self):
        f = Frame()
        atoms = Block()
        with pytest.raises(molrs.BlockDtypeError):
            atoms["bad"] = np.array(["a", 1, None], dtype=object)
        f["atoms"] = atoms
