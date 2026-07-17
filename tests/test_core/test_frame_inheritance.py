"""Verify the canonical ``molrs.Frame`` surface used by molpy.

These tests pin the direct dependency contract:

1. The frame carries the rich molrs API (metadata dict, blocks, to_dict) and is
   accepted directly by every ``molrs.*`` API.
2. ``frame.simbox`` returns the molrs box carrying the enriched accessors that were
   sunk into molrs (``is_free`` / ``style`` / ``volume``).
3. Object / None columns are rejected fail-fast (numpy-only Store contract).
"""

import molrs
import numpy as np
import pytest

from molrs import Block, Frame, MetaValue


class TestCanonicalFrame:
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

    def test_typed_meta(self):
        f = Frame(
            meta={
                "timestep": MetaValue("i64", 42),
                "description": MetaValue("string", "water box"),
            }
        )
        assert f.meta["timestep"].dtype == "i64"
        assert f.meta["timestep"].value == 42
        assert f.meta["description"].value == "water box"

    def test_typed_meta_survives_molrs_roundtrip(self, tmp_path):
        f = Frame(meta={"timestep": MetaValue("i64", 7)})
        atoms = Block()
        atoms["x"] = np.array([1.0], dtype=np.float64)
        atoms["y"] = np.array([0.0], dtype=np.float64)
        atoms["z"] = np.array([0.0], dtype=np.float64)
        atoms["symbol"] = np.array(["He"])
        f["atoms"] = atoms
        molrs.write_xyz(str(tmp_path / "rt.xyz"), f)
        assert f.meta["timestep"].value == 7


class TestBoxEnrichmentSunkIntoMolrs:
    """``frame.simbox`` is the molrs box, now carrying is_free / style / volume."""

    def test_box_is_molrs_box_with_enriched_api(self):
        f = Frame()
        f.simbox = molrs.Box.cube(10.0)
        b = f.simbox
        assert isinstance(b, molrs.Box)
        assert b.is_free is False
        assert b.style == "orthogonal"
        assert b.volume() == pytest.approx(1000.0, abs=1.0)

    def test_removed_box_alias_is_absent(self):
        frame = Frame()
        assert not hasattr(frame, "box")
        with pytest.raises(AttributeError):
            getattr(frame, "box")

    def test_removed_untyped_metadata_alias_is_absent(self):
        frame = Frame()
        assert not hasattr(frame, "metadata")
        with pytest.raises(AttributeError):
            getattr(frame, "metadata")


class TestNumpyOnlyContract:
    """Object / None columns are rejected at write time."""

    def test_object_column_rejected(self):
        f = Frame()
        atoms = Block()
        with pytest.raises(molrs.BlockDtypeError):
            atoms["bad"] = np.array(["a", 1, None], dtype=object)
        f["atoms"] = atoms
