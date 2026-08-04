"""ac-008 — no hand-picked fixture subset without an in-file reason.

Scans ``tests/test_typifier/`` for the historical failure mode: asserting on a
tiny ad-hoc list (e.g. only ethane) with no ``# subset reason:`` comment.

A full-repo directory-scanned fixture harness is not required for every module;
this gate covers the typifier surface where the chem-perceive chain was burned.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_TYPI_DIR = Path(__file__).resolve().parent
_REASON = re.compile(r"#\s*subset reason\s*:", re.I)
# Forbidden: a one-element string list that looks like a cherry-picked fixture name
_CHERRY = re.compile(
    r"""\[\s*["']e_ethane["']\s*\]|"""
    r"""only\s*=\s*\[\s*["'][^"']+["']\s*\]|"""
    r"""fixtures\s*=\s*\[\s*["']e_ethane["']\s*\]""",
    re.I,
)


def test_typifier_tests_declare_subset_reason_when_cherry_picking():
    """Any cherry-pick pattern in this package must sit under a subset reason."""
    offenders: list[str] = []
    for path in sorted(_TYPI_DIR.glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if not _CHERRY.search(text):
            continue
        if not _REASON.search(text):
            offenders.append(path.name)
    assert not offenders, (
        "cherry-picked fixture list without '# subset reason:' in: "
        + ", ".join(offenders)
    )


def test_atd_and_mmff_goldens_state_their_subset_policy():
    """Methane/ethane goldens must document why not the full 37-molecule matrix."""
    atd = (_TYPI_DIR / "test_atd.py").read_text(encoding="utf-8")
    mmff = (_TYPI_DIR / "test_mmff.py").read_text(encoding="utf-8")
    assert _REASON.search(atd), "test_atd.py needs '# subset reason:'"
    # mmff uses locked RDKit ethane geometry — require reason or "RDKit-locked"
    assert _REASON.search(mmff) or "RDKit" in mmff and "hardcoded" in mmff.lower(), (
        "test_mmff.py must document that goldens are offline-locked (not a silent subset)"
    )
