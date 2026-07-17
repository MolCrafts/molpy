"""MolTemplate (.lt) reader over ``molpy.parser.moltemplate``."""

from __future__ import annotations

from pathlib import Path

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField
from molpy.parser.moltemplate import build_forcefield, build_system, parse_file


def _resolve(file_path: str | Path) -> Path:
    return Path(file_path)


class MolTemplateReader:
    """Reader for MolTemplate (.lt) files."""

    def read(self, file_path: str | Path) -> ForceField:
        """Parse a .lt file and return the force-field component."""
        resolved = _resolve(file_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved}"
            )
        doc = parse_file(resolved)
        return build_forcefield(doc, base_dir=resolved.parent)

    def read_molecule(self, file_path: str | Path) -> Atomistic:
        """Parse a .lt file and return the full assembled system."""
        atomistic, _ = self.read_system(file_path)
        return atomistic

    def read_all_molecules(self, file_path: str | Path) -> list[Atomistic]:
        """Return every molecule (list of Atomistic) from the .lt file.

        With the native parser this is equivalent to ``[read_molecule()]``
        because ``new`` statements are pre-merged into the combined system.
        """
        atomistic, _ = self.read_system(file_path)
        return [atomistic]

    def read_system(self, file_path: str | Path) -> tuple[Atomistic, ForceField]:
        """Parse a .lt file and return both ``(Atomistic, ForceField)``."""
        resolved = _resolve(file_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved}"
            )
        doc = parse_file(resolved)
        return build_system(doc, base_dir=resolved.parent)


def read_moltemplate(file_path: str | Path) -> ForceField:
    """Convenience: parse a .lt file and return its ForceField."""
    return MolTemplateReader().read(file_path)


def read_moltemplate_molecule(file_path: str | Path) -> Atomistic:
    """Convenience: parse a .lt file and return the assembled Atomistic."""
    return MolTemplateReader().read_molecule(file_path)


def read_moltemplate_molecules(file_path: str | Path) -> list[Atomistic]:
    """Convenience: parse a .lt file and return ``[Atomistic]``."""
    return MolTemplateReader().read_all_molecules(file_path)


def read_moltemplate_system(
    file_path: str | Path,
) -> tuple[Atomistic, ForceField]:
    """Convenience: parse a .lt file and return ``(Atomistic, ForceField)``."""
    return MolTemplateReader().read_system(file_path)
