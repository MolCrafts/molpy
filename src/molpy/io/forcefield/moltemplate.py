"""MolTemplate (``.lt``) force-field / system reader."""

from __future__ import annotations

from pathlib import Path

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField
from molpy.parser.moltemplate import build_forcefield, build_system, parse_file


def _resolve(file_path: str | Path) -> Path:
    return Path(file_path)


class MolTemplateReader:
    """Reader for MolTemplate (``.lt``) files."""

    def read(self, file_path: str | Path) -> ForceField:
        """Parse a ``.lt`` file and return the force-field component."""
        resolved = _resolve(file_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved}"
            )
        doc = parse_file(resolved)
        return build_forcefield(doc, base_dir=resolved.parent)

    def read_molecule(self, file_path: str | Path) -> Atomistic:
        """Parse a ``.lt`` file and return the assembled molecule."""
        atomistic, _ = self.read_system(file_path)
        return atomistic

    def read_all_molecules(self, file_path: str | Path) -> list[Atomistic]:
        """Return the assembled system as a one-element list of ``Atomistic``."""
        atomistic, _ = self.read_system(file_path)
        return [atomistic]

    def read_system(self, file_path: str | Path) -> tuple[Atomistic, ForceField]:
        """Parse a ``.lt`` file and return ``(Atomistic, ForceField)``."""
        resolved = _resolve(file_path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"MolTemplate file not found: {file_path} -> {resolved}"
            )
        doc = parse_file(resolved)
        return build_system(doc, base_dir=resolved.parent)


def read_moltemplate(file_path: str | Path) -> ForceField:
    """Read the force field from a MolTemplate ``.lt`` file."""
    return MolTemplateReader().read(file_path)


def read_moltemplate_molecule(file_path: str | Path) -> Atomistic:
    """Read the assembled molecule from a MolTemplate ``.lt`` file."""
    return MolTemplateReader().read_molecule(file_path)


def read_moltemplate_molecules(file_path: str | Path) -> list[Atomistic]:
    """Read the assembled system as a list of molecules from a ``.lt`` file."""
    return MolTemplateReader().read_all_molecules(file_path)


def read_moltemplate_system(
    file_path: str | Path,
) -> tuple[Atomistic, ForceField]:
    """Read a MolTemplate ``.lt`` file as ``(Atomistic, ForceField)``."""
    return MolTemplateReader().read_system(file_path)
