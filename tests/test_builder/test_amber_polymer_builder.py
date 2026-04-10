"""Tests for AmberPolymerBuilder.

These tests require AmberTools (antechamber, parmchk2, prepgen, tleap) to be installed.
They are marked as external tests and will be skipped if AmberTools is not available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from molpy.builder.polymer.ambertools import (
    AmberBuildResult,
    AmberPolymerBuilder,
)


class TestAmberPolymerBuilderInit:
    """Unit tests for AmberPolymerBuilder defaults."""

    def test_default_values(self):
        builder = AmberPolymerBuilder(library={})
        assert builder.force_field == "gaff2"
        assert builder.charge_method == "bcc"
        assert builder.work_dir == Path("amber_work")
        assert builder.keep_intermediates is True

    def test_custom_values(self, tmp_path: Path):
        builder = AmberPolymerBuilder(
            library={},
            force_field="gaff",
            charge_method="gas",
            work_dir=tmp_path / "custom_work",
            keep_intermediates=False,
        )
        assert builder.force_field == "gaff"
        assert builder.charge_method == "gas"
        assert builder.work_dir == tmp_path / "custom_work"
        assert builder.keep_intermediates is False


class TestAmberBuildResult:
    """Unit tests for AmberBuildResult."""

    def test_fields(self, tmp_path: Path):
        """Test that AmberBuildResult can be instantiated with all fields."""
        result = AmberBuildResult(
            frame=MagicMock(),
            forcefield=MagicMock(),
            prmtop_path=tmp_path / "polymer.prmtop",
            inpcrd_path=tmp_path / "polymer.inpcrd",
            pdb_path=tmp_path / "polymer.pdb",
            monomer_count=10,
            cgsmiles="{[#EO]|10}",
        )

        assert result.monomer_count == 10
        assert result.cgsmiles == "{[#EO]|10}"
        assert result.pdb_path == tmp_path / "polymer.pdb"


class TestAmberPolymerBuilderValidation:
    """Unit tests for AmberPolymerBuilder validation (no external tools)."""

    def test_empty_library(self):
        """Test that empty library raises error on build."""
        builder = AmberPolymerBuilder(library={})

        with pytest.raises(ValueError, match="Labels .* not found in library"):
            builder.build("{[#EO]|5}")

    def test_missing_label_in_library(self):
        """Test that using undefined label raises error."""
        # Create a mock Atomistic with proper port markers
        mock_monomer = MagicMock()
        mock_atom_head = MagicMock()
        mock_atom_head.get.side_effect = lambda k: "<" if k == "port" else None
        mock_atom_head.__getitem__ = lambda self, k: "<" if k == "port" else None
        mock_atom_tail = MagicMock()
        mock_atom_tail.get.side_effect = lambda k: ">" if k == "port" else None
        mock_atom_tail.__getitem__ = lambda self, k: ">" if k == "port" else None
        mock_monomer.atoms = [mock_atom_head, mock_atom_tail]

        builder = AmberPolymerBuilder(library={"PS": mock_monomer})

        with pytest.raises(ValueError, match="Labels .* not found in library"):
            builder.build("{[#EO]|5}")  # EO not in library

    def test_missing_port_markers(self):
        """Test that monomer without port markers raises error."""
        mock_monomer = MagicMock()
        mock_atom = MagicMock()
        mock_atom.get.return_value = None  # No port marker
        mock_monomer.atoms = [mock_atom]

        builder = AmberPolymerBuilder(library={"EO": mock_monomer})

        with pytest.raises(ValueError, match="has no port annotations"):
            builder.build("{[#EO]|5}")


class TestAmberPolymerBuilderSequence:
    """Unit tests for CGSmiles to tleap sequence translation."""

    def _create_mock_builder_with_prepared_monomers(self, labels: list[str]):
        """Create a builder with mock prepared monomers."""
        from molpy.builder.polymer.ambertools.amber_builder import _PreparedMonomer

        mock_monomer = MagicMock()
        mock_head = MagicMock()
        mock_head.get.side_effect = lambda k: "<" if k == "port" else None
        mock_head.__getitem__ = lambda self, k: (
            "<" if k == "port" else "HA" if k == "name" else None
        )
        mock_tail = MagicMock()
        mock_tail.get.side_effect = lambda k: ">" if k == "port" else None
        mock_tail.__getitem__ = lambda self, k: (
            ">" if k == "port" else "HT" if k == "name" else None
        )
        mock_monomer.atoms = [mock_head, mock_tail]

        library = {label: mock_monomer for label in labels}
        builder = AmberPolymerBuilder(library=library)

        # Mock prepared monomers
        for label in labels:
            builder._prepared_monomers[label] = _PreparedMonomer(
                label=label,
                frcmod_file=Path(f"{label}.frcmod"),
                head_prepi=Path(f"H{label}.prepi"),
                chain_prepi=Path(f"{label}.prepi"),
                tail_prepi=Path(f"T{label}.prepi"),
                head_resname=f"H{label[:2].upper()}",
                chain_resname=label[:3].upper(),
                tail_resname=f"T{label[:2].upper()}",
            )

        return builder

    def test_single_monomer_sequence(self):
        """Test sequence for single monomer."""
        from molpy.parser.smiles import parse_cgsmiles

        builder = self._create_mock_builder_with_prepared_monomers(["EO"])
        ir = parse_cgsmiles("{[#EO]}")

        sequence = builder._build_sequence(ir.base_graph)

        # Single monomer uses chain variant
        assert sequence == "EO"

    def test_repeat_monomer_sequence(self):
        """Test sequence for repeated monomer ({[#EO]|3})."""
        from molpy.parser.smiles import parse_cgsmiles

        builder = self._create_mock_builder_with_prepared_monomers(["EO"])
        ir = parse_cgsmiles("{[#EO]|3}")

        sequence = builder._build_sequence(ir.base_graph)

        # First=HEAD, middle=CHAIN, last=TAIL
        assert sequence == "HEO EO TEO"

    def test_five_monomer_sequence(self):
        """Test sequence for 5 repeated monomers."""
        from molpy.parser.smiles import parse_cgsmiles

        builder = self._create_mock_builder_with_prepared_monomers(["EO"])
        ir = parse_cgsmiles("{[#EO]|5}")

        sequence = builder._build_sequence(ir.base_graph)

        # First=HEAD, middle 3=CHAIN, last=TAIL
        assert sequence == "HEO EO EO EO TEO"
