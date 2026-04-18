"""MolTemplate (.lt) reader — thin shim over ``molpy.parser.moltemplate``.

Preserves the historical public API (``MolTemplateReader``,
``read_moltemplate``, ``read_moltemplate_molecule``,
``read_moltemplate_molecules``) while delegating to the new native parser.
Adds a new ``read_system()`` method that returns ``(Atomistic, ForceField)``
for end-to-end moltemplate execution.
"""

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

    def read_molecule(
        self,
        file_path: str | Path,
        molecule_name: str | None = None,  # kept for backward compat
    ) -> Atomistic:
        """Parse a .lt file and return the first instantiated molecule.

        ``molecule_name`` is accepted for backward compatibility but ignored
        -- the new parser returns the full assembled system.
        """
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


def read_moltemplate_molecule(
    file_path: str | Path, molecule_name: str | None = None
) -> Atomistic:
    """Convenience: parse a .lt file and return the assembled Atomistic."""
    return MolTemplateReader().read_molecule(file_path, molecule_name)


def read_moltemplate_molecules(file_path: str | Path) -> list[Atomistic]:
    """Convenience: parse a .lt file and return ``[Atomistic]``."""
    return MolTemplateReader().read_all_molecules(file_path)


def read_moltemplate_system(
    file_path: str | Path,
) -> tuple[Atomistic, ForceField]:
    """Convenience: parse a .lt file and return ``(Atomistic, ForceField)``."""
    return MolTemplateReader().read_system(file_path)
