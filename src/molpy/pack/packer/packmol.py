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
from molpy.core.frame import Frame
from molpy.core.script import Script

from ..constraint import InsideBoxConstraint
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
        pbc: np.ndarray | list[float] | None = None,
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
            pbc: Periodic boundary conditions. 3 values ``[lx, ly, lz]`` for
                 a box with origin at zero, or 6 values
                 ``[xmin, ymin, zmin, xmax, ymax, zmax]``.
                 Requires Packmol >= 20.15.0.
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
                targets, max_steps, seed, tolerance, workdir, pbc=pbc
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
        pbc: np.ndarray | list[float] | None = None,
    ) -> Script:
        """
        Generate Packmol input script as a Script object.

        Args:
            targets: List of packing targets
            max_steps: Maximum optimization steps
            seed: Random seed
            tolerance: Distance tolerance
            workdir: Working directory
            pbc: Periodic boundary conditions (3 or 6 floats), or None.

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
        if pbc is not None:
            pbc_vals = np.asarray(pbc, dtype=float).ravel()
            if pbc_vals.size not in (3, 6):
                raise ValueError(f"pbc must have 3 or 6 values, got {pbc_vals.size}")
            script.append("pbc " + " ".join(f"{v:.6f}" for v in pbc_vals))
        script.append("")  # Empty line for readability

        # Write structure files and add to input
        for i, target in enumerate(targets):
            frame = target.frame
            number = target.number
            constraint = target.constraint

            # Frame should already have x, y, z fields (never use xyz)

            # Create structure file
            struct_file = workdir / f"structure_{i}.pdb"
            mp_io.write_pdb(struct_file, frame)

            # Add structure block - use relative path (filename only) since packmol runs in workdir
            script.append(f"structure {struct_file.name}")
            script.append(f"  number {number}")
            # When PBC is active, Packmol already confines molecules to the
            # periodic box — skip redundant InsideBoxConstraint.  Other
            # constraints (plane, sphere, …) are still written.
            skip = pbc is not None and isinstance(constraint, InsideBoxConstraint)
            if not skip:
                constraint_def = self._constraint_to_packmol(constraint)
                if constraint_def:
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
        pbc: np.ndarray | list[float] | None = None,
    ) -> Path:
        """
        Generate Packmol input file.

        Args:
            targets: List of packing targets
            max_steps: Maximum optimization steps
            seed: Random seed
            tolerance: Distance tolerance
            workdir: Working directory
            pbc: Periodic boundary conditions (3 or 6 floats), or None.

        Returns:
            Path to generated input file
        """
        script = self._generate_input_script(
            targets, max_steps, seed, tolerance, workdir, pbc=pbc
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

    # Per-section atom-index columns to offset when replicating topology.
    _INDEX_ENDS = {
        "bonds": ("atomi", "atomj"),
        "angles": ("atomi", "atomj", "atomk"),
        "dihedrals": ("atomi", "atomj", "atomk", "atoml"),
        "impropers": ("atomi", "atomj", "atomk", "atoml"),
    }

    def _build_final_frame(
        self, targets: list[Target], optimized_frame: Frame
    ) -> Frame:
        """
        Build the packed frame by replicating each target's topology to its copy
        count and stamping in the optimized coordinates.

        Every source column is materialised from the molrs Block **once per
        target** and then tiled with numpy (``np.tile`` for values, ``np.repeat``
        for per-copy index offsets).  The earlier implementation copied the molrs
        Block once *per instance* (e.g. 20 chains + 108 ions = 128×) and re-read
        its string columns each time; on a ~30k-atom system that O(copies) string
        re-materialisation dominated wall-clock (tens of minutes).  Materialising
        once per target makes it O(atoms).

        Process:
        1. For each target, tile its atoms/bonds/angles/... columns ``number``
           times, offsetting atom indices per copy.
        2. Concatenate the per-target sections.
        3. Reassign global 1-based ids and stamp optimized x/y/z back.
        """

        def materialize(block) -> dict:
            """Read every column of a molrs Block into numpy exactly once."""
            return {k: np.asarray(block[k]) for k in block.keys()}

        atoms_parts: list[dict] = []
        topo_parts: dict[str, list[dict]] = {name: [] for name in self._INDEX_ENDS}

        atom_offset = 0  # running atom-index base across all copies of all targets
        mol_base = 0  # running molecule-id base
        for target in targets:
            number = target.number
            src_atoms = materialize(target.frame["atoms"])
            n = target.frame["atoms"].nrows
            # atom-index base for each copy of this target: (number,)
            copy_offsets = atom_offset + np.arange(number) * n

            atoms = {k: np.tile(col, number) for k, col in src_atoms.items()}
            atoms["mol_id"] = np.repeat(
                np.arange(mol_base + 1, mol_base + number + 1), n
            ).astype(int)
            # 'q' mirror kept for LAMMPS-full compatibility (charge is canonical)
            if "charge" in atoms and "q" not in atoms:
                atoms["q"] = atoms["charge"]
            elif "q" in atoms and "charge" not in atoms:
                atoms["charge"] = atoms["q"]
            atoms_parts.append(atoms)

            for name, ends in self._INDEX_ENDS.items():
                if name not in target.frame:
                    continue
                src = materialize(target.frame[name])
                if name == "bonds" and ("atomi" not in src or "atomj" not in src):
                    raise KeyError(
                        "Bond block must contain 'atomi' and 'atomj' columns. "
                        f"Got: {list(src.keys())}"
                    )
                m = len(next(iter(src.values())))
                row_off = np.repeat(copy_offsets, m)  # (number*m,)
                exp = {}
                for k, col in src.items():
                    tiled = np.tile(col, number)
                    exp[k] = tiled + row_off if k in ends else tiled
                topo_parts[name].append(exp)

            atom_offset += number * n
            mol_base += number

        def concat_parts(parts: list[dict]) -> dict:
            """Concatenate numpy-dict parts, filling any missing key per part."""
            if not parts:
                return {}
            all_keys: set[str] = set()
            for p in parts:
                all_keys.update(p.keys())
            ref_dtype = {
                key: next(p[key].dtype for p in parts if key in p) for key in all_keys
            }

            def part_len(p: dict) -> int:
                for v in p.values():
                    return len(v)
                return 0

            combined = {}
            for key in all_keys:
                arrays = []
                for p in parts:
                    if key in p:
                        arrays.append(p[key])
                    else:
                        n_rows = part_len(p)
                        dtype = ref_dtype[key]
                        if np.issubdtype(dtype, np.floating):
                            arrays.append(np.full(n_rows, np.nan, dtype=dtype))
                        elif np.issubdtype(dtype, np.integer):
                            arrays.append(np.zeros(n_rows, dtype=dtype))
                        else:
                            arrays.append(np.full(n_rows, None, dtype=object))
                combined[key] = np.concatenate(arrays, axis=0)
            return combined

        final_frame = Frame()
        atoms = concat_parts(atoms_parts)
        atoms["id"] = np.arange(1, len(next(iter(atoms.values()))) + 1, dtype=int)
        final_frame["atoms"] = atoms
        for name in self._INDEX_ENDS:
            if topo_parts[name]:
                section = concat_parts(topo_parts[name])
                section["id"] = np.arange(
                    1, len(next(iter(section.values()))) + 1, dtype=int
                )
                final_frame[name] = section

        # Stamp optimized coordinates back (packmol writes copies target-major,
        # matching the tiling order above).
        opt = optimized_frame["atoms"]
        if "x" in opt and "y" in opt and "z" in opt:
            final_frame["atoms"]["x"] = np.asarray(opt["x"])
            final_frame["atoms"]["y"] = np.asarray(opt["y"])
            final_frame["atoms"]["z"] = np.asarray(opt["z"])
        else:
            raise ValueError("Optimized frame must contain 'x', 'y', 'z' coordinates")

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
