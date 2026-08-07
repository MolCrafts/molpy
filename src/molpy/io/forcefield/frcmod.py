"""AMBER FRCMOD I/O (molrs-backed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.io.utils import ensure_parent_dir


def read_frcmod(file: str | Path) -> dict[str, Any]:
    """Read an AMBER FRCMOD file into a section dictionary.

    Returns keys: ``remark``, ``mass``, ``bond``, ``angle``, ``dihe``,
    ``improper``, ``nonbon``, ``raw_text``.
    """
    import molrs.io

    path = Path(file)
    if not path.is_file():
        raise FileNotFoundError(path)
    return molrs.io.read_frcmod(str(path))


def write_frcmod(file: str | Path, sections: dict[str, Any]) -> None:
    """Write FRCMOD sections to *file*."""
    import molrs.io

    path = Path(file)
    ensure_parent_dir(path)
    payload = {k: str(v) for k, v in sections.items() if v is not None}
    molrs.io.write_frcmod(str(path), payload)
