"""
Packmol - Packer implementation using Packmol binary.

This module provides a packer implementation that uses Packmol binary
for actual packing execution.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import molpy.io as mp_io
from molpy import Frame, Script

from ..target import Target
from .base import Packer


class Packmol(Packer):
    """
    Packer implementation using Packmol binary.

    This class provides a packer that uses Packmol binary for packing molecules.

    Usage:
        packer = Packmol(executable="packmol")
        result = packer(targets, max_steps=1000, seed=4628)
    """

    def __init__(
        self,
        executable: str | None = None,
        workdir: Path | None = None,
    ):
        """
        Initialize Packmol packer.

        Args:
            executable: Path to packmol executable. If None, uses:
                       1. PACKMOL_BIN environment variable
                       2. "packmol" on PATH
            workdir: Working directory for temporary files.
                    If None, creates a temporary directory.
        """
        Packer.__init__(self)
        self.executable = executable
        self.workdir = workdir

    def __call__(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int | None = None,
        workdir: Path | None = None,
        tolerance: float = 2.0,
        cleanup: bool = True,
        **kwargs,
    ) -> Frame:
        """
        Pack molecules using Packmol backend.

        Args:
            targets: List of packing targets. If None, uses stored targets.
            max_steps: Maximum optimization steps
            seed: Random seed for packing. If None, uses default (4628).
            workdir: Optional working directory (overrides default)
            tolerance: Distance tolerance in Angstroms
            cleanup: Whether to clean up intermediate files after success
            **kwargs: Additional arguments (ignored for now)

        Returns:
            Packed molecular system as Frame
        """
        # Use provided targets or stored targets
        if targets is None:
            targets = self.targets

        if not targets:
            raise ValueError("No targets provided for packing")

        if seed is None:
            seed = 4628

        # Use provided workdir or default
        workdir = workdir if workdir is not None else self.workdir

        # Locate packmol executable
        packmol_path = self._locate_packmol()

        # Setup workdir
        temp_workdir = False
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="molpack_"))
            temp_workdir = True
        else:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate input file
            input_file = self._generate_input(
                targets, max_steps, seed, tolerance, workdir
            )

            # Run Packmol
            self._execute_packmol(packmol_path, input_file, workdir)

            # Read output
            output_file = workdir / ".optimized.pdb"
            if not output_file.exists():
                raise FileNotFoundError(
                    f"Packmol output file not found: {output_file}. "
                    f"Check Packmol output for errors."
                )

            optimized_frame = mp_io.read_pdb(output_file)

            # Cleanup if requested
            if cleanup:
                self._cleanup_intermediate_files(workdir)

            # Build final frame with topology from targets
            return self._build_final_frame(targets, optimized_frame)
        finally:
            # Cleanup temporary workdir if we created it
            if temp_workdir and workdir.exists():
                shutil.rmtree(workdir)

    def _locate_packmol(self) -> str:
        """Locate packmol executable."""
        executable = self.executable
        if executable is None:
            executable = os.environ.get("PACKMOL_BIN", "packmol")

        packmol_path = shutil.which(executable)

        if packmol_path is None:
            raise FileNotFoundError(
                f"Packmol executable '{executable}' not found in PATH. "
                f"Set PACKMOL_BIN environment variable or ensure packmol is in PATH."
            )

        return packmol_path

    def _generate_input_script(
        self,
        targets: list[Target],
        max_steps: int,
        seed: int,
        tolerance: float,
        workdir: Path,
    ) -> Script:
        """
        Generate Packmol input script as a Script object.

        Args:
            targets: List of packing targets
            max_steps: Maximum optimization steps
            seed: Random seed
            tolerance: Distance tolerance
            workdir: Working directory

        Returns:
            Script object containing Packmol input
        """
        # Create script with header
        script = Script.from_text(
            name="packmol_input",
            text="",
            language="other",
            description="Packmol input file",
        )

        # Add global settings
        script.append(f"tolerance {tolerance}")
        script.append("filetype pdb")
        script.append("output .optimized.pdb")
        script.append(f"nloop {max_steps}")
        script.append(f"seed {seed}")
        script.append("")  # Empty line for readability

        # Write structure files and add to input
        for i, target in enumerate(targets):
            frame = target.frame
            number = target.number
            constraint = target.constraint

            # Create structure file
            struct_file = workdir / f"structure_{i}.pdb"
            mp_io.write_pdb(struct_file, frame)

            # Add structure block - use relative path (filename only) since packmol runs in workdir
            script.append(f"structure {struct_file.name}")
            script.append(f"  number {number}")
            constraint_def = self._constraint_to_packmol(constraint)
            if constraint_def:
                # Handle multi-line constraints (e.g., AndConstraint)
                for line in constraint_def.split("\n"):
                    script.append(f"  {line}")
            script.append("end structure")
            script.append("")  # Empty line between structures

        return script

    def _generate_input(
        self,
        targets: list[Target],
        max_steps: int,
        seed: int,
        tolerance: float,
        workdir: Path,
    ) -> Path:
        """
        Generate Packmol input file.

        Args:
            targets: List of packing targets
            max_steps: Maximum optimization steps
            seed: Random seed
            tolerance: Distance tolerance
            workdir: Working directory

        Returns:
            Path to generated input file
        """
        script = self._generate_input_script(
            targets, max_steps, seed, tolerance, workdir
        )

        # Save script to file
        input_file = workdir / ".packmol.inp"
        script.path = input_file
        script.save()

        return input_file

    def _execute_packmol(
        self, packmol_path: str, input_file: Path, workdir: Path
    ) -> None:
        """Execute Packmol process."""
        output_file = workdir / ".packmol.out"
        log_file = workdir / ".packmol.log"

        try:
            with (
                open(input_file) as infile,
                open(output_file, "w") as outfile,
                open(log_file, "w") as log,
            ):
                result = subprocess.run(
                    [packmol_path],
                    stdin=infile,
                    stdout=outfile,
                    stderr=log,
                    cwd=workdir,
                    check=False,  # We'll check manually to provide better errors
                )

            if result.returncode != 0:
                # Read error log
                error_msg = ""
                if log_file.exists():
                    with open(log_file) as f:
                        error_msg = f.read()

                raise RuntimeError(
                    f"Packmol failed with exit code {result.returncode}.\n"
                    f"Check {log_file} for details.\n"
                    f"Error output:\n{error_msg[:500]}"
                )

        except FileNotFoundError:
            raise FileNotFoundError(f"Packmol executable not found: {packmol_path}")
        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to execute Packmol: {e}")

    def _cleanup_intermediate_files(self, workdir: Path) -> None:
        """Clean up intermediate files."""
        patterns = [
            ".packmol.inp",
            ".packmol.out",
            ".packmol.log",
            "structure_*.pdb",
        ]

        for pattern in patterns:
            if "*" in pattern:
                # Glob pattern
                for file in workdir.glob(pattern):
                    file.unlink()
            else:
                # Direct file
                file = workdir / pattern
                if file.exists():
                    file.unlink()

    def _constraint_to_packmol(self, constraint) -> str:
        """Convert molpy constraint to Packmol input format."""
        from ..constraint import (
            AndConstraint,
            InsideBoxConstraint,
            InsideSphereConstraint,
            MinDistanceConstraint,
            OutsideBoxConstraint,
            OutsideSphereConstraint,
        )

        def box_cmd(origin, upper, not_flag: bool) -> str:
            flag = "outside" if not_flag else "inside"
            return (
                f"{flag} box {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f} "
                f"{upper[0]:.6f} {upper[1]:.6f} {upper[2]:.6f}"
            )

        def sphere_cmd(center, radius, not_flag: bool) -> str:
            flag = "outside" if not_flag else "inside"
            return (
                f"{flag} sphere {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
                f"{radius:.6f}"
            )

        if isinstance(constraint, AndConstraint):
            cmd1 = self._constraint_to_packmol(constraint.a)
            cmd2 = self._constraint_to_packmol(constraint.b)
            return f"{cmd1}\n  {cmd2}"

        if isinstance(constraint, InsideBoxConstraint):
            return box_cmd(constraint.origin, constraint.upper, False)
        elif isinstance(constraint, InsideSphereConstraint):
            return sphere_cmd(constraint.center, constraint.radius, False)
        elif isinstance(constraint, OutsideBoxConstraint):
            return box_cmd(constraint.origin, constraint.upper, True)
        elif isinstance(constraint, OutsideSphereConstraint):
            return sphere_cmd(constraint.center, constraint.radius, True)
        elif isinstance(constraint, MinDistanceConstraint):
            # Packmol uses "discale" for minimum distance
            return f"discale {constraint.dmin:.6f}"
        else:
            raise NotImplementedError(
                f"Packmol does not support constraint type {type(constraint)}"
            )

    def _build_final_frame(
        self, targets: list[Target], optimized_frame: Frame
    ) -> Frame:
        """
        Build final frame by expanding target frames according to instance counts,
        then writing optimized coordinates back.

        Process:
        1. Expand each target frame by its number of instances
        2. Offset topology indices (bonds, angles, dihedrals) for each instance
        3. Concatenate all expanded frames
        4. Write optimized coordinates back to the expanded frame
        """
        # Step 1: Expand target frames according to instance counts
        # Count atoms per instance to compute offsets
        target_atoms_count = []
        for target in targets:
            atoms_block = target.frame["atoms"]
            n_atoms = len(atoms_block.get("id", atoms_block.get("xyz", [])))
            for _ in range(target.number):
                target_atoms_count.append(n_atoms)

        # Compute cumulative offsets for atom indices
        atom_offsets = np.concatenate([[0], np.cumsum(target_atoms_count)])

        # Prepare lists for collecting expanded blocks
        all_atoms = []
        all_bonds = []
        all_angles = []
        all_dihedrals = []

        # Counters for topology IDs
        current_instance = 0
        bond_id_counter = 0
        angle_id_counter = 0
        dihedral_id_counter = 0

        # Step 2: Expand each target frame by its number of instances
        for target in targets:
            for _inst in range(target.number):
                offset = atom_offsets[current_instance]

                # Expand atoms block - preserve ALL fields including type and charge
                atoms = target.frame["atoms"].copy()
                n = len(atoms.get("id", atoms.get("xyz", [])))

                # Set molecule ID for this instance
                atoms["mol"] = np.full(n, current_instance + 1, dtype=int)

                # Reassign atom IDs sequentially (1-indexed)
                if "id" in atoms:
                    atoms["id"] = np.arange(offset + 1, offset + n + 1, dtype=int)

                # Ensure charge field is available as 'q' for LAMMPS compatibility
                # LAMMPS uses 'q' but we store as 'charge' in atoms
                if "charge" in atoms and "q" not in atoms:
                    atoms["q"] = atoms["charge"]
                elif "q" in atoms and "charge" not in atoms:
                    atoms["charge"] = atoms["q"]

                all_atoms.append(atoms)

                # Expand bonds block
                if "bonds" in target.frame:
                    bonds = target.frame["bonds"].copy()
                    m = len(bonds.get("id", bonds.get("i", bonds.get("atom_i", []))))

                    # Reassign bond IDs sequentially
                    if "id" in bonds:
                        bonds["id"] = np.arange(
                            bond_id_counter + 1, bond_id_counter + m + 1, dtype=int
                        )
                        bond_id_counter += m

                    # Offset atom indices in bonds
                    for end in ("i", "j", "atom_i", "atom_j"):
                        if end in bonds:
                            bonds[end] = bonds[end] + offset

                    all_bonds.append(bonds)

                # Expand angles block
                if "angles" in target.frame:
                    angles = target.frame["angles"].copy()
                    p = len(angles.get("id", angles.get("i", angles.get("atom_i", []))))

                    # Reassign angle IDs sequentially
                    if "id" in angles:
                        angles["id"] = np.arange(
                            angle_id_counter + 1, angle_id_counter + p + 1, dtype=int
                        )
                        angle_id_counter += p

                    # Offset atom indices in angles
                    for end in ("i", "j", "k", "atom_i", "atom_j", "atom_k"):
                        if end in angles:
                            angles[end] = angles[end] + offset

                    all_angles.append(angles)

                # Expand dihedrals block
                if "dihedrals" in target.frame:
                    dihedrals = target.frame["dihedrals"].copy()
                    q = len(
                        dihedrals.get(
                            "id", dihedrals.get("i", dihedrals.get("atom_i", []))
                        )
                    )

                    # Reassign dihedral IDs sequentially
                    if "id" in dihedrals:
                        dihedrals["id"] = np.arange(
                            dihedral_id_counter + 1,
                            dihedral_id_counter + q + 1,
                            dtype=int,
                        )
                        dihedral_id_counter += q

                    # Offset atom indices in dihedrals
                    for end in (
                        "i",
                        "j",
                        "k",
                        "l",
                        "atom_i",
                        "atom_j",
                        "atom_k",
                        "atom_l",
                    ):
                        if end in dihedrals:
                            dihedrals[end] = dihedrals[end] + offset

                    all_dihedrals.append(dihedrals)

                current_instance += 1

        # Step 3: Concatenate all expanded blocks
        def concat_blocks(blocks):
            """Concatenate list of block dicts into a single block dict, preserving all fields."""
            if not blocks:
                return {}
            # Collect all keys from all blocks (in case some blocks have different fields)
            all_keys = set()
            for blk in blocks:
                all_keys.update(blk.keys())

            combined = {}
            for key in all_keys:
                # Concatenate arrays for this key from all blocks
                arrays = []
                for blk in blocks:
                    if key in blk:
                        arrays.append(blk[key])
                    else:
                        # If a block doesn't have this key, use None or default value
                        # This shouldn't happen if all blocks come from the same source
                        raise ValueError(
                            f"Inconsistent fields in blocks: key '{key}' missing in some blocks"
                        )
                combined[key] = np.concatenate(arrays, axis=0)
            return combined

        # Build expanded frame with all topology
        final_frame = Frame()
        final_frame["atoms"] = concat_blocks(all_atoms)
        if all_bonds:
            final_frame["bonds"] = concat_blocks(all_bonds)
        if all_angles:
            final_frame["angles"] = concat_blocks(all_angles)
        if all_dihedrals:
            final_frame["dihedrals"] = concat_blocks(all_dihedrals)

        # Validate that all required fields are present in final frame
        required_atom_fields = ["type", "charge"]
        for field in required_atom_fields:
            if field not in final_frame["atoms"]:
                raise ValueError(
                    f"Required field '{field}' missing in final packed frame. "
                    f"Available fields: {list(final_frame['atoms'].keys())}"
                )

        # Step 4: Write optimized coordinates back to expanded frame
        # Packmol returns coordinates in optimized_frame, overwrite them
        if "xyz" in optimized_frame["atoms"]:
            final_frame["atoms"]["xyz"] = optimized_frame["atoms"]["xyz"]
        elif (
            "x" in optimized_frame["atoms"]
            and "y" in optimized_frame["atoms"]
            and "z" in optimized_frame["atoms"]
        ):
            # Handle separate x, y, z columns
            final_frame["atoms"]["x"] = optimized_frame["atoms"]["x"]
            final_frame["atoms"]["y"] = optimized_frame["atoms"]["y"]
            final_frame["atoms"]["z"] = optimized_frame["atoms"]["z"]
        else:
            raise ValueError(
                "Optimized frame must contain 'xyz' or 'x', 'y', 'z' coordinates"
            )

        return final_frame

    def generate_input_script(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int = 4628,
        tolerance: float = 2.0,
        workdir: Path | None = None,
    ) -> Script:
        """
        Generate Packmol input script as a Script object without saving.

        This allows you to preview, edit, or format the script before saving.

        Args:
            targets: List of packing targets. If None, uses stored targets.
            max_steps: Maximum optimization steps
            seed: Random seed for packing
            tolerance: Distance tolerance in Angstroms
            workdir: Optional working directory (for structure file paths)

        Returns:
            Script object containing Packmol input

        Raises:
            ValueError: If no targets provided
        """
        if targets is None:
            targets = self.targets

        if not targets:
            raise ValueError("No targets provided")

        workdir = workdir if workdir is not None else self.workdir
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="molpack_"))
        else:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        return self._generate_input_script(targets, max_steps, seed, tolerance, workdir)

    def generate_input_only(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int = 4628,
        tolerance: float = 2.0,
        workdir: Path | None = None,
    ) -> Path:
        """
        Generate Packmol input file without running the packing.

        Args:
            targets: List of packing targets. If None, uses stored targets.
            max_steps: Maximum optimization steps
            seed: Random seed for packing
            tolerance: Distance tolerance in Angstroms
            workdir: Optional working directory

        Returns:
            Path to generated input file
        """
        if targets is None:
            targets = self.targets

        if not targets:
            raise ValueError("No targets provided")

        workdir = workdir if workdir is not None else self.workdir
        if workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="molpack_"))
        else:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        input_file = self._generate_input(targets, max_steps, seed, tolerance, workdir)
        return input_file

    def pack(
        self,
        targets: list[Target] | None = None,
        max_steps: int = 1000,
        seed: int | None = None,
    ) -> Frame:
        """Pack molecules using Packmol backend.

        This method implements the abstract pack() method from Packer base class.
        It delegates to __call__() for the actual implementation.
        """
        return self(targets, max_steps=max_steps, seed=seed)
