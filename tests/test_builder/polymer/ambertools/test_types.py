"""Unit tests for :mod:`molpy.builder.polymer.ambertools.types`."""

from pathlib import Path
from unittest.mock import MagicMock

from molpy.builder.polymer.ambertools import AmberBuildResult


class TestAmberBuildResult:
    def test_carries_structure_parameters_and_output_paths(self):
        result = AmberBuildResult(
            frame=MagicMock(),
            forcefield=MagicMock(),
            prmtop_path=Path("polymer.prmtop"),
            inpcrd_path=Path("polymer.inpcrd"),
            pdb_path=Path("polymer.pdb"),
            monomer_count=10,
            cgsmiles="{[#M]|10}",
        )
        assert result.monomer_count == 10
        assert result.cgsmiles == "{[#M]|10}"
        assert result.pdb_path == Path("polymer.pdb")
