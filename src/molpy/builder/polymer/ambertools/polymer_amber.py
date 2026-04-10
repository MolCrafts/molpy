"""Amber-based polymer builder.

This module provides AmberPolymerBuilder, which uses Amber's toolchain
(antechamber, prepgen, tleap) for polymer assembly.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from molpy.core.atomistic import Atomistic
from molpy.core.frame import Frame
from molpy.io.readers import read_amber_prmtop, read_pdb
from molpy.parser.smiles import parse_cgsmiles
from .amber_leap import generate_leap_script

from .amber_utils import configure_amber_wrappers
from molpy.builder.polymer.core import PolymerBuildResult, TypifierProtocol
from molpy.builder.polymer.residue_manager import ResidueManager

if TYPE_CHECKING:
    from molpy.builder.polymer.connectors import Connector

logger = logging.getLogger(__name__)


class AmberPolymerBuilder:
    """Build polymers using Amber's residue-based workflow.

    Pipeline:
    1. Convert monomers to Amber prep files (via antechamber + prepgen)
    2. Parse CGSmiles to extract sequence
    3. Generate tleap script to build sequence
    4. Execute tleap to create polymer
    5. Convert Amber output (prmtop/inpcrd) to Atomistic
    """

    def __init__(
        self,
        library: Mapping[str, Atomistic],
        connector: AmberConnector,
        typifier: TypifierProtocol | None = None,
        placer=None,
        force_field: str = "gaff2",
        conda_env: str = "AmberTools25",
        workdir: Path | None = None,
    ):
        self.library = library
        self.connector = connector
        self.typifier = typifier
        self.placer = placer
        self.force_field = force_field
        self.conda_env = conda_env

        if workdir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.workdir = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self.workdir = Path(workdir)
            self.workdir.mkdir(parents=True, exist_ok=True)

        self.antechamber, self.prepgen, self.tleap = configure_amber_wrappers(
            self.workdir, self.conda_env
        )
        self._check_tools()

        self.residue_manager = ResidueManager(
            self.workdir, self.antechamber, self.prepgen
        )
        self._monomer_to_prep: dict[str, Path] = {}

    def _check_tools(self) -> None:
        """Verify Amber tools are available."""
        for wrapper in [self.antechamber, self.prepgen, self.tleap]:
            wrapper.check()

    def _convert_monomers_to_residues(self) -> None:
        """Convert all library monomers to Amber prep files."""
        for label, monomer in self.library.items():
            if label in self._monomer_to_prep:
                continue

            head_atom = None
            tail_atom = None
            left_conn_atom = None
            right_conn_atom = None

            for idx, atom in enumerate(monomer.atoms):
                port = atom.get("port", "")
                if ">" in str(port):
                    head_atom = idx
                    left_conn_atom = atom
                if "<" in str(port):
                    tail_atom = idx
                    right_conn_atom = atom

            if head_atom is None and tail_atom is None:
                raise ValueError(
                    f"Monomer '{label}' has no port marker. "
                    "Define [>] and/or [<] in monomer SMILES."
                )

            if head_atom is not None and tail_atom is not None:
                variant = "chain"
            elif tail_atom is not None:
                variant = "head"
            else:
                variant = "tail"

            left_leaving_atoms = None
            right_leaving_atoms = None

            if (
                hasattr(self.connector, "get_leaving_atoms")
                and left_conn_atom
                and right_conn_atom
            ):
                left_leaving_atoms, right_leaving_atoms = (
                    self.connector.get_leaving_atoms(
                        monomer, left_conn_atom, right_conn_atom
                    )
                )
            else:
                left_sel = getattr(self.connector, "left_leaving_selector", None)
                if left_conn_atom and left_sel is not None:
                    left_leaving_atoms = left_sel(monomer, left_conn_atom)
                right_sel = getattr(self.connector, "right_leaving_selector", None)
                if right_conn_atom and right_sel is not None:
                    right_leaving_atoms = right_sel(monomer, right_conn_atom)

            prep_file = self.residue_manager.create_residue(
                residue_name=label,
                monomer=monomer,
                head_atom=head_atom,
                tail_atom=tail_atom,
                variant=variant,
                left_leaving_atoms=left_leaving_atoms,
                right_leaving_atoms=right_leaving_atoms,
                charge_method="bcc",
                atom_type=(
                    self.force_field if self.force_field.startswith("gaff") else "gaff2"
                ),
                net_charge=0,
            )

            self._monomer_to_prep[label] = prep_file

    def _parse_cgsmiles_sequence(self, cgsmiles: str) -> list[str]:
        """Extract monomer sequence from CGSmiles string."""
        ir = parse_cgsmiles(cgsmiles)
        nodes = ir.base_graph.nodes if hasattr(ir, "base_graph") else []
        sequence = [node.label for node in nodes]
        if not sequence:
            raise ValueError(f"Could not extract sequence from CGSmiles: {cgsmiles}")
        return sequence

    def _build_via_tleap(self, sequence: list[str], output_prefix: str) -> Frame:
        """Build polymer using tleap."""
        prep_files = []
        seen_labels: set[str] = set()
        for label in sequence:
            if label not in seen_labels:
                prep_file = self._monomer_to_prep[label]
                prep_file_rel = prep_file.relative_to(self.workdir)
                prep_files.append(prep_file_rel)
                seen_labels.add(label)

        script_content = generate_leap_script(
            force_field=self.force_field,
            prep_files=prep_files,
            sequence=sequence,
            output_prefix=output_prefix,
        )

        script_file = self.workdir / f"{output_prefix}_leap.in"
        script_file.write_text(script_content)

        try:
            self.tleap.run_from_script(
                script_text=script_content,
                script_name=script_file.name,
            )
        except Exception as e:
            raise RuntimeError(f"TLeap execution failed: {e}") from e

        prmtop_path = self.workdir / f"{output_prefix}.prmtop"
        inpcrd_path = self.workdir / f"{output_prefix}.inpcrd"

        if not prmtop_path.exists():
            raise RuntimeError(
                f"TLeap did not produce {prmtop_path}. Check {script_file} for errors."
            )

        pdb_path = self.workdir / f"{output_prefix}.pdb"
        if not pdb_path.exists():
            frame, forcefield = read_amber_prmtop(prmtop_path, inpcrd_path)
            return frame

        frame = read_pdb(pdb_path)
        return frame

    def build(self, cgsmiles: str) -> PolymerBuildResult:
        """Build polymer from CGSmiles notation."""
        self._convert_monomers_to_residues()

        sequence = self._parse_cgsmiles_sequence(cgsmiles)

        undefined = set(sequence) - set(self.library.keys())
        if undefined:
            raise ValueError(
                f"Sequence contains undefined monomers: {undefined}. "
                f"Available: {set(self.library.keys())}"
            )

        output_prefix = "polymer"
        self._build_via_tleap(sequence, output_prefix)

        prmtop_path = self.workdir / f"{output_prefix}.prmtop"
        inpcrd_path = self.workdir / f"{output_prefix}.inpcrd"
        pdb_path = self.workdir / f"{output_prefix}.pdb"

        logger.info("TLeap succeeded: prmtop=%s, inpcrd=%s", prmtop_path, inpcrd_path)

        polymer = Atomistic()
        polymer._output_files = {
            "prmtop": str(prmtop_path),
            "inpcrd": str(inpcrd_path),
            "pdb": str(pdb_path),
        }

        return PolymerBuildResult(
            polymer=polymer,
            connection_history=[],
            total_steps=len(sequence) - 1,
        )

    def __del__(self):
        """Cleanup temporary directory if created."""
        if hasattr(self, "_temp_dir") and self._temp_dir is not None:
            self._temp_dir.cleanup()
