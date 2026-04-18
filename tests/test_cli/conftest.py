"""Shared fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from molpy.cli import main

FIXTURES = (
    Path(__file__).parent.parent / "test_parser" / "test_moltemplate" / "fixtures"
)


@pytest.fixture
def tip3p_lt() -> Path:
    return FIXTURES / "tip3p.lt"


@pytest.fixture
def run_cli(capsys):
    """Run ``molpy ...`` and return (exit_code, stdout, stderr)."""

    def _run(*argv: str) -> tuple[int, str, str]:
        rc = main(list(argv))
        captured = capsys.readouterr()
        return int(rc or 0), captured.out, captured.err

    return _run
