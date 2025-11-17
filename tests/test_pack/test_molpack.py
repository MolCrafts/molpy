"""
Tests for Packmol packer and Molpack interface.

These tests require packmol executable to be available in PATH.
If packmol is not found, all tests will be skipped.
"""

import shutil

import numpy as np
import pytest

import molpy.pack as mpk
from molpy.core import Block, Frame
from molpy.pack.molpack import Molpack
from molpy.pack.packer.packmol import Packmol

# Check if packmol is available
PACKMOL_AVAILABLE = shutil.which("packmol") is not None

# Skip all tests if packmol is not available
pytestmark = pytest.mark.skipif(
    not PACKMOL_AVAILABLE, reason="Packmol executable not found in PATH"
)


@pytest.fixture
def simple_water_frame() -> Frame:
    """Create a simple water molecule frame."""
    atoms = Block(
        {
            "element": np.array(["O", "H", "H"]),
            "xyz": np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        }
    )
    return Frame({"atoms": atoms})


@pytest.fixture
def simple_target(simple_water_frame):
    """Create a simple packing target."""
    box_constraint = mpk.InsideBoxConstraint(
        length=np.array([10.0, 10.0, 10.0]), origin=np.array([0.0, 0.0, 0.0])
    )
    return mpk.Target(frame=simple_water_frame, number=5, constraint=box_constraint)


class TestPackmol:
    """Test Packmol packer class."""

    def test_init(self):
        """Test Packmol initialization."""
        packer = Packmol(executable="packmol")
        assert packer.executable == "packmol"
        assert packer.workdir is None

    def test_init_with_workdir(self, tmp_path):
        """Test Packmol initialization with workdir."""
        packer = Packmol(workdir=tmp_path)
        assert packer.workdir == tmp_path

    def test_add_target(self, simple_target):
        """Test adding targets."""
        packer = Packmol()
        packer.add_target(simple_target)
        assert len(packer.targets) == 1
        assert packer.targets[0] == simple_target

    def test_def_target(self, simple_water_frame):
        """Test defining target via convenience method."""
        packer = Packmol()
        box_constraint = mpk.InsideBoxConstraint(
            length=np.array([10.0, 10.0, 10.0]), origin=np.array([0.0, 0.0, 0.0])
        )
        target = packer.def_target(
            frame=simple_water_frame, number=10, constraint=box_constraint
        )
        assert len(packer.targets) == 1
        assert target.frame == simple_water_frame
        assert target.number == 10

    def test_call_no_targets(self):
        """Test that calling without targets raises error."""
        packer = Packmol()
        with pytest.raises(ValueError, match="No targets provided"):
            packer()

    def test_generate_input_script(self, simple_target, tmp_path):
        """Test generate_input_script method returns Script object."""
        packer = Packmol(workdir=tmp_path)
        script = packer.generate_input_script(
            targets=[simple_target], max_steps=1000, seed=42
        )

        # Check it's a Script object
        from molpy import Script

        assert isinstance(script, Script)
        assert script.name == "packmol_input"
        assert script.language == "other"

        # Check content
        text = script.text
        assert "tolerance 2.0" in text
        assert "filetype pdb" in text
        assert "output .optimized.pdb" in text
        assert "nloop 1000" in text
        assert "seed 42" in text
        assert "structure" in text
        assert "number 5" in text
        assert "inside box" in text

        # Test preview
        preview = script.preview(max_lines=10)
        assert "tolerance" in preview
        assert "structure" in preview

    def test_generate_input_only(self, simple_target, tmp_path):
        """Test generate_input_only method."""
        packer = Packmol(workdir=tmp_path)
        input_file = packer.generate_input_only(
            targets=[simple_target], max_steps=1000, seed=42
        )

        assert input_file.exists()
        content = input_file.read_text()
        assert "tolerance 2.0" in content
        assert "filetype pdb" in content
        assert "output .optimized.pdb" in content
        assert "nloop 1000" in content
        assert "seed 42" in content
        assert "structure" in content
        assert "number 5" in content
        assert "inside box" in content

    def test_constraint_to_packmol_box(self):
        """Test constraint conversion for box constraints."""
        packer = Packmol()

        box_constraint = mpk.InsideBoxConstraint(
            length=np.array([10.0, 10.0, 10.0]), origin=np.array([0.0, 0.0, 0.0])
        )
        cmd = packer._constraint_to_packmol(box_constraint)
        assert "inside box" in cmd
        assert "0.000000" in cmd
        assert "10.000000" in cmd

        outside_box = mpk.OutsideBoxConstraint(
            origin=np.array([0.0, 0.0, 0.0]), lengths=np.array([5.0, 5.0, 5.0])
        )
        cmd = packer._constraint_to_packmol(outside_box)
        assert "outside box" in cmd

    def test_constraint_to_packmol_sphere(self):
        """Test constraint conversion for sphere constraints."""
        packer = Packmol()

        sphere_constraint = mpk.InsideSphereConstraint(
            radius=5.0, center=np.array([0.0, 0.0, 0.0])
        )
        cmd = packer._constraint_to_packmol(sphere_constraint)
        assert "inside sphere" in cmd
        assert "5.000000" in cmd

        outside_sphere = mpk.OutsideSphereConstraint(
            radius=3.0, center=np.array([1.0, 1.0, 1.0])
        )
        cmd = packer._constraint_to_packmol(outside_sphere)
        assert "outside sphere" in cmd

    def test_constraint_to_packmol_min_distance(self):
        """Test constraint conversion for minimum distance constraint."""
        packer = Packmol()

        min_dist = mpk.MinDistanceConstraint(dmin=2.5)
        cmd = packer._constraint_to_packmol(min_dist)
        assert "discale" in cmd
        assert "2.500000" in cmd

    def test_constraint_to_packmol_combined(self):
        """Test constraint conversion for combined constraints."""
        packer = Packmol()

        box = mpk.InsideBoxConstraint(length=np.array([10.0, 10.0, 10.0]))
        sphere = mpk.OutsideSphereConstraint(
            radius=2.0, center=np.array([5.0, 5.0, 5.0])
        )
        combined = mpk.AndConstraint(box, sphere)

        cmd = packer._constraint_to_packmol(combined)
        assert "inside box" in cmd
        assert "outside sphere" in cmd

    def test_build_final_frame(self, simple_target):
        """Test building final frame from packing results."""
        packer = Packmol()

        # Create optimized frame with coordinates
        n_atoms_total = 5 * 3  # 5 water molecules * 3 atoms
        optimized_frame = Frame(
            {
                "atoms": Block(
                    {
                        "xyz": np.random.rand(n_atoms_total, 3) * 10.0,
                    }
                )
            }
        )

        result = packer._build_final_frame([simple_target], optimized_frame)

        assert isinstance(result, Frame)
        assert "atoms" in result
        assert len(result["atoms"]["xyz"]) == n_atoms_total
        # Check that coordinates match
        assert np.allclose(result["atoms"]["xyz"], optimized_frame["atoms"]["xyz"])

    def test_real_packing_small(self, simple_target, tmp_path):
        """Test real packing with packmol."""
        packer = Packmol(workdir=tmp_path)

        # Use small number of molecules and steps for quick test
        result = packer(
            targets=[simple_target],
            max_steps=100,  # Small number for quick test
            seed=42,
            tolerance=2.0,
        )

        assert isinstance(result, Frame)
        assert "atoms" in result
        # Should have 5 molecules * 3 atoms = 15 atoms
        assert len(result["atoms"]["xyz"]) == 15

    def test_real_packing_multiple_targets(self, simple_water_frame, tmp_path):
        """Test real packing with multiple targets."""
        packer = Packmol(workdir=tmp_path)

        # Create two different targets
        box1 = mpk.InsideBoxConstraint(
            length=np.array([5.0, 5.0, 5.0]), origin=np.array([0.0, 0.0, 0.0])
        )
        target1 = mpk.Target(frame=simple_water_frame, number=3, constraint=box1)

        box2 = mpk.InsideBoxConstraint(
            length=np.array([5.0, 5.0, 5.0]), origin=np.array([5.0, 5.0, 5.0])
        )
        target2 = mpk.Target(frame=simple_water_frame, number=2, constraint=box2)

        result = packer(
            targets=[target1, target2],
            max_steps=100,
            seed=42,
            tolerance=2.0,
        )

        assert isinstance(result, Frame)
        assert "atoms" in result
        # Should have (3 + 2) * 3 = 15 atoms
        assert len(result["atoms"]["xyz"]) == 15

    def test_pack_method(self, simple_target, tmp_path):
        """Test pack() method with real packmol."""
        packer = Packmol(workdir=tmp_path)
        result = packer.pack(targets=[simple_target], max_steps=100, seed=42)

        assert isinstance(result, Frame)
        assert "atoms" in result
        assert len(result["atoms"]["xyz"]) == 15


class TestMolpack:
    """Test Molpack high-level interface."""

    def test_init(self, tmp_path):
        """Test Molpack initialization."""
        packer = Molpack(workdir=tmp_path, packer="packmol")
        assert packer.workdir == tmp_path
        assert len(packer.targets) == 0

    def test_add_target(self, simple_water_frame, tmp_path):
        """Test adding target via Molpack."""
        packer = Molpack(workdir=tmp_path, packer="packmol")
        box_constraint = mpk.InsideBoxConstraint(
            length=np.array([10.0, 10.0, 10.0]), origin=np.array([0.0, 0.0, 0.0])
        )
        target = packer.add_target(
            simple_water_frame, number=5, constraint=box_constraint
        )
        assert len(packer.targets) == 1
        assert target.number == 5

    def test_optimize(self, simple_water_frame, tmp_path):
        """Test optimize method with real packmol."""
        packer = Molpack(workdir=tmp_path, packer="packmol")
        box_constraint = mpk.InsideBoxConstraint(
            length=np.array([10.0, 10.0, 10.0]), origin=np.array([0.0, 0.0, 0.0])
        )
        packer.add_target(simple_water_frame, number=5, constraint=box_constraint)
        result = packer.optimize(max_steps=100, seed=42)
        assert isinstance(result, Frame)
        assert "atoms" in result
        assert len(result["atoms"]["xyz"]) == 15
