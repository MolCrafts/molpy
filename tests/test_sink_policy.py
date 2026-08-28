"""Permanent anti-duplication: molrs-covered capabilities stay in molrs."""

from __future__ import annotations

from pathlib import Path

_IO = Path(__file__).resolve().parents[1] / "src" / "molpy" / "io"

# Formats molrs already exposes on the Python surface. molpy readers must
# delegate (contain ``molrs.io`` / ``molrs.ff``), not grow a second parser.
_SUNK = {
    "data/pdb.py": "molrs.io",
    "data/xyz.py": "molrs.io",
    "data/gro.py": "molrs.io",
}

# molpy-native extensions: molrs does not provide these on the Python surface.
_NATIVE = {
    "data/ac.py",
    "forcefield/moltemplate.py",
    "data/lammps_bond_react.py",
    "emit/openmm.py",
}


def test_sunk_formats_delegate_to_molrs() -> None:
    for rel, token in _SUNK.items():
        text = (_IO / rel).read_text(encoding="utf-8")
        assert token in text, f"{rel} must delegate via {token}"


def test_native_extensions_are_not_required_to_delegate() -> None:
    for rel in _NATIVE:
        path = _IO / rel
        assert path.exists(), f"native extension missing: {rel}"
