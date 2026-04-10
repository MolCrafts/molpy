"""Residue management for Amber-based polymer building.

This module handles conversion of Atomistic monomers to Amber residue
templates (prep files) using antechamber and prepgen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from molpy.core.atomistic import Atomistic
from molpy.io.writers import write_pdb
from molpy.wrapper import AntechamberWrapper, PrepgenWrapper
from molpy.io.data.amber_prep import PrepResidue, read_prep
from molpy.wrapper.prepgen import write_prepgen_control_file


class ResidueManager:
    """Manages conversion of monomers to Amber residue templates.

    This class handles the pipeline:
    Atomistic → PDB → antechamber (mol2) → prepgen → prep file

    Attributes:
        workdir: Working directory for intermediate files.
        antechamber: Configured AntechamberWrapper.
        prepgen: Configured PrepgenWrapper.
        residue_cache: Cache mapping monomer hash → prep file path.
    """

    def __init__(
        self,
        workdir: Path,
        antechamber: AntechamberWrapper,
        prepgen: PrepgenWrapper,
    ):
        """Initialize residue manager.

        Args:
            workdir: Working directory for files.
            antechamber: Antechamber wrapper instance.
            prepgen: Prepgen wrapper instance.
        """
        self.workdir = Path(workdir)
        self.antechamber = antechamber
        self.prepgen = prepgen
        self.residue_cache: dict[str, Path] = {}

        # Create subdirectories
        self.prep_dir = self.workdir / "prep_files"
        self.prep_dir.mkdir(parents=True, exist_ok=True)

        # Update prepgen workdir to prep_dir for relative path support
        # This is needed because conda run has issues with absolute paths
        self.prepgen.workdir = self.prep_dir

    def _compute_monomer_hash(self, monomer: Atomistic) -> str:
        """Compute a hash for monomer to use as cache key.

        Args:
            monomer: Atomistic monomer structure.

        Returns:
            Hex digest string.
        """
        # Create a string representation of the monomer
        # Include atom elements and connectivity
        atoms_str = "".join(
            f"{atom['element']}{atom.get('x', 0):.3f}{atom.get('y', 0):.3f}{atom.get('z', 0):.3f}"
            for atom in monomer.atoms
        )
        bonds_str = "".join(
            f"{bond.get('i', 0)}-{bond.get('j', 0)}" for bond in monomer.bonds
        )

        content = atoms_str + bonds_str
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def create_residue(
        self,
        residue_name: str,
        monomer: Atomistic,
        head_atom: int | None = None,
        tail_atom: int | None = None,
        variant: Literal["chain", "head", "tail"] = "chain",
        left_leaving_atoms: list | None = None,
        right_leaving_atoms: list | None = None,
        charge_method: str = "bcc",
        atom_type: str = "gaff2",
        net_charge: int = 0,
    ) -> Path:
        """Convert Atomistic monomer to Amber prep file.

        Pipeline:
        1. Write monomer to PDB
        2. Run antechamber to assign atom types and charges (→ mol2)
        3. Run prepgen to create prep file with connection atoms
        4. Cache the prep file

        Args:
            residue_name: Name for the residue (e.g., "EO" for ethylene oxide).
            monomer: Atomistic monomer structure.
            head_atom: Index of head connection atom (0-based, left connection).
                Required for "chain" and "tail" variants.
            tail_atom: Index of tail connection atom (0-based, right connection).
                Required for "chain" and "head" variants.
            variant: Prepgen residue variant:
                - "chain": both head and tail connection points
                - "head": chain start residue (tail connection only)
                - "tail": chain end residue (head connection only)
            left_leaving_atoms: List of atoms to remove at left connection point.
                If None, automatically detects H atoms bonded to head_atom.
            right_leaving_atoms: List of atoms to remove at right connection point.
                If None, automatically detects H atoms bonded to tail_atom.
            charge_method: Antechamber charge method ("bcc", "gas", etc.).
            atom_type: Atom typing scheme ("gaff", "gaff2").
            net_charge: Net charge of the monomer.

        Returns:
            Path to the generated prep file.

        Raises:
            ValueError: If required connection atoms are missing or out of range.
            RuntimeError: If antechamber or prepgen fails.
        """
        # Validate connection atoms against selected variant
        n_atoms = len(monomer.atoms)
        if variant not in ("chain", "head", "tail"):
            raise ValueError(f"Unsupported prepgen variant: {variant}")

        if variant in ("chain", "tail") and head_atom is None:
            raise ValueError(f"Variant '{variant}' requires head_atom")
        if variant in ("chain", "head") and tail_atom is None:
            raise ValueError(f"Variant '{variant}' requires tail_atom")

        if head_atom is not None and not (0 <= head_atom < n_atoms):
            raise ValueError(
                f"Head atom index {head_atom} out of range (0-{n_atoms - 1})"
            )
        if tail_atom is not None and not (0 <= tail_atom < n_atoms):
            raise ValueError(
                f"Tail atom index {tail_atom} out of range (0-{n_atoms - 1})"
            )

        # Check cache
        monomer_hash = self._compute_monomer_hash(monomer)
        cache_key = f"{residue_name}_{monomer_hash}"

        if cache_key in self.residue_cache:
            cached_prep = self.residue_cache[cache_key]
            if cached_prep.exists():
                return cached_prep

        # File paths
        pdb_file = self.prep_dir / f"{residue_name}_{monomer_hash}.pdb"
        ac_file = self.prep_dir / f"{residue_name}_{monomer_hash}.ac"
        prep_file = self.prep_dir / f"{residue_name}_{monomer_hash}.prep"

        # Step 1: Write PDB
        # Convert Atomistic to Frame
        frame = monomer.to_frame()
        write_pdb(pdb_file, frame)

        # Step 2: Run antechamber
        # Output ac format (Antechamber format) which prepgen expects
        # Use absolute paths for conda run compatibility
        try:
            result = self.antechamber.atomtype_assign(
                input_file=pdb_file.absolute(),
                output_file=ac_file.absolute(),
                input_format="pdb",
                output_format="ac",
                charge_method=charge_method,
                atom_type=atom_type,
                net_charge=net_charge,
            )

            # Print antechamber output for debugging
            if result.returncode != 0:
                print(f"Antechamber warning (return code {result.returncode})")
                if result.stdout:
                    print(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    print(f"STDERR:\n{result.stderr}")

        except Exception as e:
            raise RuntimeError(
                f"Antechamber failed for residue {residue_name}: {e}"
            ) from e

        if not ac_file.exists():
            error_msg = f"Antechamber did not produce output file {ac_file}"
            if result.stdout:
                error_msg += f"\n\nAntechamber STDOUT:\n{result.stdout}"
            if result.stderr:
                error_msg += f"\n\nAntechamber STDERR:\n{result.stderr}"
            raise RuntimeError(error_msg)

        # Step 3: Run prepgen
        # Get atom names from ac file
        # Parse ac file to get atom names
        ac_content = ac_file.read_text()
        lines = ac_content.split("\n")

        # Find ATOM section in ac file
        atom_section_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ATOM"):
                atom_section_start = i
                break

        if atom_section_start is None:
            raise RuntimeError(f"Cannot find ATOM section in {ac_file}")

        # Read atom names from ac file (format: ATOM index name type ...)
        # Example: ATOM      1  C   UNK  ...
        atom_names = []
        for i in range(atom_section_start, len(lines)):
            line = lines[i].strip()
            if not line or not line.startswith("ATOM"):
                break
            parts = line.split()
            if len(parts) >= 3:
                atom_names.append(parts[2])  # Atom name is third field (index 2)

        if head_atom is not None and head_atom >= len(atom_names):
            raise ValueError(f"Head atom index {head_atom} out of range for ac atoms")
        if tail_atom is not None and tail_atom >= len(atom_names):
            raise ValueError(f"Tail atom index {tail_atom} out of range for ac atoms")

        # Get atom names (not indices)
        head_name = atom_names[head_atom] if head_atom is not None else None
        tail_name = atom_names[tail_atom] if tail_atom is not None else None

        # Find atoms to omit using leaving group lists or automatic detection
        omit_names = self._find_all_omit_atoms(
            monomer,
            ac_file,
            head_atom,
            tail_atom,
            atom_names,
            left_leaving_atoms,
            right_leaving_atoms,
        )

        # Create control file for prepgen
        # NOTE: We do NOT specify PRE_HEAD_TYPE/POST_TAIL_TYPE
        # User must ensure monomer port atoms have correct chemical environment
        control_file = self.prep_dir / f"{residue_name}_{monomer_hash}.mc"
        write_prepgen_control_file(
            control_file,
            variant=variant,
            head_name=head_name,
            tail_name=tail_name,
            omit_names=omit_names,
            charge=net_charge,
        )

        print(
            f"Prepgen: residue={residue_name}, variant={variant}, "
            f"head={head_name}, tail={tail_name}, omit={omit_names}"
        )

        # Use relative paths because conda run has issues with absolute paths
        # Prepgen workdir is set to prep_dir in __init__
        ac_rel = ac_file.name
        prep_rel = prep_file.name
        control_rel = control_file.name

        try:
            result = self.prepgen.generate_residue(
                input_file=ac_rel,
                output_file=prep_rel,
                control_file=control_rel,
                residue_name=residue_name,
            )

            # Print prepgen output for debugging
            if result.returncode != 0:
                print(f"Prepgen warning (return code {result.returncode})")
            if result.stdout:
                print(f"Prepgen STDOUT:\n{result.stdout}")
            if result.stderr:
                print(f"Prepgen STDERR:\n{result.stderr}")

        except Exception as e:
            raise RuntimeError(f"Prepgen failed for residue {residue_name}: {e}") from e

        if not prep_file.exists():
            error_msg = f"Prepgen did not produce output file {prep_file}"
            if result.stdout:
                error_msg += f"\n\nPrepgen STDOUT:\n{result.stdout}"
            if result.stderr:
                error_msg += f"\n\nPrepgen STDERR:\n{result.stderr}"
            raise RuntimeError(error_msg)

        # Cache and return
        self.residue_cache[cache_key] = prep_file
        return prep_file

    def _find_all_omit_atoms(
        self,
        monomer: Atomistic,
        ac_file: Path,
        head_atom: int | None,
        tail_atom: int | None,
        atom_names: list[str],
        left_leaving_atoms: list | None = None,
        right_leaving_atoms: list | None = None,
    ) -> list[str]:
        """Find all atoms to omit during polymerization.

        Uses provided leaving atom lists or automatic detection:
        1. If left_leaving_atoms provided, use it for left side
        2. If right_leaving_atoms provided, use it for right side
        3. Otherwise, automatically detect H atoms bonded to connection points

        Args:
            monomer: Original monomer structure.
            ac_file: Path to ac file with bond information.
            head_atom: Index of head connection atom (left connection).
            tail_atom: Index of tail connection atom (right connection).
            atom_names: List of atom names from ac file.
            left_leaving_atoms: Optional list of Atom entities to omit on left side.
            right_leaving_atoms: Optional list of Atom entities to omit on right side.

        Returns:
            List of atom names to omit during polymerization.
        """
        omit_names = []
        atoms_list = list(monomer.atoms)

        # Convert atom entities to indices and then to names
        if left_leaving_atoms is not None:
            for atom in left_leaving_atoms:
                try:
                    idx = atoms_list.index(atom)
                    if idx < len(atom_names):
                        omit_names.append(atom_names[idx])
                except ValueError:
                    pass  # Atom not in list

        if right_leaving_atoms is not None:
            for atom in right_leaving_atoms:
                try:
                    idx = atoms_list.index(atom)
                    if idx < len(atom_names):
                        omit_names.append(atom_names[idx])
                except ValueError:
                    pass  # Atom not in list

        # If no explicit leaving atoms provided, use automatic detection
        if left_leaving_atoms is None and right_leaving_atoms is None:
            # Fall back to automatic H detection
            leaving_groups = self._find_leaving_groups(
                ac_file, head_atom, tail_atom, atom_names
            )
            omit_names.extend(leaving_groups)

        # Remove duplicates while preserving order
        seen = set()
        unique_omit_names = []
        for name in omit_names:
            if name not in seen:
                seen.add(name)
                unique_omit_names.append(name)

        return unique_omit_names

    def _find_leaving_groups(
        self,
        ac_file: Path,
        head_atom: int | None,
        tail_atom: int | None,
        atom_names: list[str],
    ) -> list[str]:
        """Find leaving groups (typically H atoms) bonded to head/tail atoms.

        Args:
            ac_file: Path to ac file with bond information.
            head_atom: Index of head connection atom.
            tail_atom: Index of tail connection atom.
            atom_names: List of atom names.

        Returns:
            List of atom names to omit during polymerization.
        """
        omit_names = []

        # Parse ac file to find bonds
        ac_content = ac_file.read_text()
        lines = ac_content.split("\n")

        # Find BOND section
        bond_section_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith("BOND"):
                bond_section_start = i + 1
                break

        if bond_section_start is None:
            return omit_names  # No bonds found

        # Read bonds (format: BOND index1 index2)
        bonds = []
        for i in range(bond_section_start, len(lines)):
            line = lines[i].strip()
            if not line or not line.startswith("BOND"):
                break
            parts = line.split()
            if len(parts) >= 3:
                # AC file uses 1-based indexing
                atom1 = int(parts[1]) - 1  # Convert to 0-based
                atom2 = int(parts[2]) - 1
                bonds.append((atom1, atom2))

        # Find H atoms bonded to declared connection atoms
        connection_atoms = {idx for idx in (head_atom, tail_atom) if idx is not None}
        if not connection_atoms:
            return omit_names

        for atom1, atom2 in bonds:
            if atom1 in connection_atoms:
                # Check if atom2 is hydrogen
                if atom2 < len(atom_names) and atom_names[atom2].startswith("H"):
                    omit_names.append(atom_names[atom2])
            elif atom2 in connection_atoms:
                # Check if atom1 is hydrogen
                if atom1 < len(atom_names) and atom_names[atom1].startswith("H"):
                    omit_names.append(atom_names[atom1])

        return omit_names

    def get_residue(self, residue_name: str, monomer_hash: str) -> Path | None:
        """Get cached prep file if available.

        Args:
            residue_name: Residue name.
            monomer_hash: Monomer hash string.

        Returns:
            Path to prep file if cached, None otherwise.
        """
        cache_key = f"{residue_name}_{monomer_hash}"
        return self.residue_cache.get(cache_key)

    def load_residue(self, prep_file: Path) -> PrepResidue:
        """Load a prep file.

        Args:
            prep_file: Path to prep file.

        Returns:
            PrepResidue object.
        """
        return read_prep(prep_file)
