"""Tests for Packmol packer and Molpack interface.

Unit tests (TestPackmolUnit, TestMolpackUnit) run without any external binary.
Integration tests (*Integration classes) require Packmol in PATH and are
skipped automatically when unavailable.
"""

import shutil

import numpy as np
import pytest

import molpy.pack as mpk
from molpy import Script
from molpy.core import Block, Frame
from molpy.pack.molpack import Molpack
from molpy.pack.packer.packmol import Packmol

PACKMOL_AVAILABLE = shutil.which("packmol") is not None
needs_packmol = pytest.mark.skipif(
    not PACKMOL_AVAILABLE, reason="Packmol executable not found in PATH"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def water_frame() -> Frame:
    """3-atom water molecule with id, element, and separate x/y/z columns."""
    xyz = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    return Frame(
        {
            "atoms": Block(
                {
                    "id": np.array([1, 2, 3], dtype=int),
                    "element": np.array(["O", "H", "H"]),
                    "x": xyz[:, 0],
                    "y": xyz[:, 1],
                    "z": xyz[:, 2],
                }
            )
        }
    )


@pytest.fixture
def box20() -> mpk.InsideBoxConstraint:
    """20 Å cubic box at origin."""
    return mpk.InsideBoxConstraint(
        length=np.array([20.0, 20.0, 20.0]),
        origin=np.array([0.0, 0.0, 0.0]),
    )


@pytest.fixture
def water_target(water_frame, box20) -> mpk.Target:
    return mpk.Target(frame=water_frame, number=5, constraint=box20)


def _fake_optimized(n_atoms: int) -> Frame:
    """Fake Packmol output: linearly-spaced x/y/z coordinates."""
    coords = np.arange(n_atoms * 3, dtype=float).reshape(n_atoms, 3)
    return Frame(
        {
            "atoms": Block(
                {
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "z": coords[:, 2],
                }
            )
        }
    )


# ---------------------------------------------------------------------------
# Unit tests — no Packmol binary needed
# ---------------------------------------------------------------------------


class TestPackmolUnit:
    # --- init ---------------------------------------------------------------

    def test_init_defaults(self):
        p = Packmol()
        assert p.executable is None
        assert p.workdir is None

    def test_init_with_workdir(self, tmp_path):
        p = Packmol(workdir=tmp_path)
        assert p.workdir == tmp_path

    # --- target management --------------------------------------------------

    def test_add_target(self, water_target):
        p = Packmol()
        p.add_target(water_target)
        assert len(p.targets) == 1
        assert p.targets[0] is water_target

    def test_def_target(self, water_frame, box20):
        p = Packmol()
        t = p.def_target(frame=water_frame, number=10, constraint=box20)
        assert len(p.targets) == 1
        assert t.frame is water_frame
        assert t.number == 10

    def test_no_targets_raises(self):
        with pytest.raises(ValueError, match="No targets"):
            Packmol()()

    # --- constraint formatting ----------------------------------------------

    def test_constraint_inside_box(self, box20):
        cmd = Packmol()._constraint_to_packmol(box20)
        assert cmd.startswith("inside box")
        assert "0.000000" in cmd
        assert "20.000000" in cmd

    def test_constraint_outside_box(self):
        cmd = Packmol()._constraint_to_packmol(
            mpk.OutsideBoxConstraint(
                origin=np.array([0.0, 0.0, 0.0]),
                lengths=np.array([5.0, 5.0, 5.0]),
            )
        )
        assert cmd.startswith("outside box")

    def test_constraint_inside_sphere(self):
        cmd = Packmol()._constraint_to_packmol(
            mpk.InsideSphereConstraint(radius=5.0, center=np.array([0.0, 0.0, 0.0]))
        )
        assert cmd.startswith("inside sphere")
        assert "5.000000" in cmd

    def test_constraint_outside_sphere(self):
        cmd = Packmol()._constraint_to_packmol(
            mpk.OutsideSphereConstraint(radius=3.0, center=np.array([1.0, 1.0, 1.0]))
        )
        assert cmd.startswith("outside sphere")

    def test_constraint_min_distance(self):
        cmd = Packmol()._constraint_to_packmol(mpk.MinDistanceConstraint(dmin=2.5))
        assert "discale" in cmd
        assert "2.500000" in cmd

    def test_constraint_combined(self, box20):
        combined = mpk.AndConstraint(
            box20,
            mpk.OutsideSphereConstraint(radius=2.0, center=np.array([5.0, 5.0, 5.0])),
        )
        cmd = Packmol()._constraint_to_packmol(combined)
        assert "inside box" in cmd
        assert "outside sphere" in cmd

    # --- input script generation --------------------------------------------

    def test_generate_input_script_content(self, water_target, tmp_path):
        script = Packmol(workdir=tmp_path).generate_input_script(
            targets=[water_target], max_steps=1000, seed=42
        )
        assert isinstance(script, Script)
        text = script.text
        assert "tolerance 2.0" in text
        assert "filetype pdb" in text
        assert "output .optimized.pdb" in text
        assert "nloop 1000" in text
        assert "seed 42" in text
        assert "number 5" in text
        assert "inside box" in text

    def test_generate_input_only_creates_file(self, water_target, tmp_path):
        inp = Packmol(workdir=tmp_path).generate_input_only(
            targets=[water_target], max_steps=500, seed=7
        )
        assert inp.exists()
        content = inp.read_text()
        assert "nloop 500" in content
        assert "seed 7" in content

    # --- _build_final_frame -------------------------------------------------

    def test_build_final_frame_copies_optimized_coords(self, water_target):
        n_total = 5 * 3  # 5 molecules × 3 atoms
        optimized = _fake_optimized(n_total)
        result = Packmol()._build_final_frame([water_target], optimized)

        assert isinstance(result, Frame)
        assert len(result["atoms"]["x"]) == n_total
        assert np.allclose(result["atoms"]["x"], optimized["atoms"]["x"])
        assert np.allclose(result["atoms"]["y"], optimized["atoms"]["y"])
        assert np.allclose(result["atoms"]["z"], optimized["atoms"]["z"])

    def test_build_final_frame_cumulative_bond_angle_offsets(self, box20):
        """Index offsets in bonds/angles must accumulate across all instances."""
        frame_a = Frame(
            {
                "atoms": Block(
                    {
                        "id": np.array([1, 2, 3], dtype=int),
                        "x": np.zeros(3),
                        "y": np.zeros(3),
                        "z": np.zeros(3),
                    }
                ),
                "bonds": Block(
                    {
                        "id": np.array([1], dtype=int),
                        "atomi": np.array([0], dtype=int),
                        "atomj": np.array([1], dtype=int),
                    }
                ),
                "angles": Block(
                    {
                        "id": np.array([1], dtype=int),
                        "atomi": np.array([0], dtype=int),
                        "atomj": np.array([1], dtype=int),
                        "atomk": np.array([2], dtype=int),
                    }
                ),
            }
        )
        frame_b = Frame(
            {
                "atoms": Block(
                    {
                        "id": np.array([1, 2], dtype=int),
                        "x": np.zeros(2),
                        "y": np.zeros(2),
                        "z": np.zeros(2),
                    }
                ),
                "bonds": Block(
                    {
                        "id": np.array([1], dtype=int),
                        "atomi": np.array([0], dtype=int),
                        "atomj": np.array([1], dtype=int),
                    }
                ),
            }
        )
        # 2 × frame_a (3 atoms each) + 1 × frame_b (2 atoms) = 8 atoms
        result = Packmol()._build_final_frame(
            targets=[
                mpk.Target(frame=frame_a, number=2, constraint=box20, name="A"),
                mpk.Target(frame=frame_b, number=1, constraint=box20, name="B"),
            ],
            optimized_frame=_fake_optimized(8),
        )

        # Atom IDs reassigned 1–8
        np.testing.assert_array_equal(result["atoms"]["id"], np.arange(1, 9, dtype=int))
        # Molecule IDs: three instances numbered 1, 2, 3
        np.testing.assert_array_equal(
            result["atoms"]["mol_id"], np.array([1, 1, 1, 2, 2, 2, 3, 3], dtype=int)
        )
        # Bonds: A-inst0 (0→1), A-inst1 (3→4), B-inst0 (6→7)
        np.testing.assert_array_equal(
            result["bonds"]["atomi"], np.array([0, 3, 6], dtype=int)
        )
        np.testing.assert_array_equal(
            result["bonds"]["atomj"], np.array([1, 4, 7], dtype=int)
        )
        # Angles: only from frame_a instances → (0,1,2) and (3,4,5)
        np.testing.assert_array_equal(
            result["angles"]["atomi"], np.array([0, 3], dtype=int)
        )
        np.testing.assert_array_equal(
            result["angles"]["atomj"], np.array([1, 4], dtype=int)
        )
        np.testing.assert_array_equal(
            result["angles"]["atomk"], np.array([2, 5], dtype=int)
        )

    def test_build_final_frame_improper_offsets(self, box20):
        frame = Frame(
            {
                "atoms": Block(
                    {
                        "id": np.array([1, 2, 3, 4], dtype=int),
                        "x": np.zeros(4),
                        "y": np.zeros(4),
                        "z": np.zeros(4),
                    }
                ),
                "impropers": Block(
                    {
                        "id": np.array([1], dtype=int),
                        "atomi": np.array([0], dtype=int),
                        "atomj": np.array([1], dtype=int),
                        "atomk": np.array([2], dtype=int),
                        "atoml": np.array([3], dtype=int),
                    }
                ),
            }
        )
        result = Packmol()._build_final_frame(
            targets=[mpk.Target(frame=frame, number=2, constraint=box20, name="X")],
            optimized_frame=_fake_optimized(8),
        )
        np.testing.assert_array_equal(
            result["impropers"]["atomi"], np.array([0, 4], dtype=int)
        )
        np.testing.assert_array_equal(
            result["impropers"]["atoml"], np.array([3, 7], dtype=int)
        )

    def test_build_final_frame_rejects_legacy_bond_keys(self, box20):
        """Bonds using legacy i/j keys (without 'atom' prefix) must raise KeyError."""
        legacy = Frame(
            {
                "atoms": Block(
                    {
                        "id": np.array([1, 2], dtype=int),
                        "x": np.zeros(2),
                        "y": np.zeros(2),
                        "z": np.zeros(2),
                    }
                ),
                "bonds": Block(
                    {
                        "id": np.array([1], dtype=int),
                        "i": np.array([0], dtype=int),
                        "j": np.array([1], dtype=int),
                    }
                ),
            }
        )
        with pytest.raises(KeyError):
            Packmol()._build_final_frame(
                targets=[mpk.Target(frame=legacy, number=1, constraint=box20)],
                optimized_frame=_fake_optimized(2),
            )


# ---------------------------------------------------------------------------
# Molpack unit tests — no Packmol binary needed
# ---------------------------------------------------------------------------


class TestMolpackUnit:
    def test_init(self, tmp_path):
        mp = Molpack(workdir=tmp_path)
        assert mp.workdir == tmp_path
        assert len(mp.targets) == 0

    def test_add_target(self, water_frame, box20, tmp_path):
        mp = Molpack(workdir=tmp_path)
        t = mp.add_target(water_frame, number=5, constraint=box20)
        assert len(mp.targets) == 1
        assert t.number == 5


# ---------------------------------------------------------------------------
# Integration tests — require Packmol executable
# ---------------------------------------------------------------------------


@pytest.mark.external
@needs_packmol
class TestPackmolIntegration:
    def test_pack_single_target(self, water_target, tmp_path):
        result = Packmol(workdir=tmp_path)(
            targets=[water_target], max_steps=100, seed=42
        )
        assert isinstance(result, Frame)
        assert len(result["atoms"]["x"]) == 15  # 5 molecules × 3 atoms

    def test_pack_multiple_targets(self, water_frame, tmp_path):
        result = Packmol(workdir=tmp_path)(
            targets=[
                mpk.Target(
                    frame=water_frame,
                    number=3,
                    constraint=mpk.InsideBoxConstraint(
                        length=np.array([5.0, 5.0, 5.0]),
                        origin=np.array([0.0, 0.0, 0.0]),
                    ),
                ),
                mpk.Target(
                    frame=water_frame,
                    number=2,
                    constraint=mpk.InsideBoxConstraint(
                        length=np.array([5.0, 5.0, 5.0]),
                        origin=np.array([5.0, 5.0, 5.0]),
                    ),
                ),
            ],
            max_steps=100,
            seed=42,
        )
        assert len(result["atoms"]["x"]) == 15  # (3 + 2) × 3 atoms

    def test_pack_method(self, water_target, tmp_path):
        result = Packmol(workdir=tmp_path).pack(
            targets=[water_target], max_steps=100, seed=42
        )
        assert isinstance(result, Frame)
        assert len(result["atoms"]["x"]) == 15


@pytest.mark.external
@needs_packmol
class TestMolpackIntegration:
    def test_optimize(self, water_frame, box20, tmp_path):
        mp = Molpack(workdir=tmp_path)
        mp.add_target(water_frame, number=5, constraint=box20)
        result = mp.optimize(max_steps=100, seed=42)
        assert isinstance(result, Frame)
        assert len(result["atoms"]["x"]) == 15  # 5 molecules × 3 atoms
